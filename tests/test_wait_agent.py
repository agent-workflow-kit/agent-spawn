"""``agent-spawn wait`` against seeded agent records and the fake claude-session.

No test depends on timing: end states are driven by writing the record (or ending a tracked
process) and the waiting process is observed only through events it causes.
"""

from __future__ import annotations

import json
import signal
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from agent_spawn import procs
from agent_spawn.ids import new_agent_id
from agent_spawn.lifecycle import lease_key
from agent_spawn.store import Store
from tests import schema_registry
from tests.conftest import Isolation
from tests.fakebin.harness import Fakebin
from tests.fixtures import sample_outcome
from tests.launch_support import Env, result_json, script_live_run
from tests.procguard import ProcTracker

pytestmark = pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="signal readiness is read from /proc"
)


@pytest.fixture
def env(fakebin: Fakebin, isolation_env: Isolation) -> Env:
    return Env(fakebin, isolation_env)


def seed(
    env: Env,
    *,
    status: str = "parked",
    reads_third_party: bool | None = False,
    run: int | None = 1,
    **fields: Any,
) -> str:
    """Insert an agent record (and its run row) straight into the store."""
    values: dict[str, Any] = {
        "status": status,
        "name": "seeded",
        "mode": "headless",
        "launch_kind": "implement",
        "run": run,
    }
    values.update(fields)
    with Store.open(env.state_dir) as store:
        agent_id = new_agent_id(store)
        store.insert_agent(agent_id, values)
        if run is not None:
            with store.lease(lease_key(agent_id), wait=5.0) as lease:
                store.put_run(
                    lease,
                    agent_id,
                    run,
                    {"run_kind": "implement", "reads_third_party": reads_third_party},
                )
    return agent_id


def update(env: Env, agent_id: str, **changes: Any) -> None:
    with Store.open(env.state_dir) as store:
        with store.lease(lease_key(agent_id), wait=5.0) as lease:
            if "status" in changes:
                status = changes.pop("status")
                store.compare_and_set(
                    lease,
                    agent_id,
                    expect_status=("launching", "live"),
                    changes={"status": status, **changes},
                )
            else:
                store.update_agent(lease, agent_id, changes)


def finish(env: Env, agent_id: str, **overrides: Any) -> dict[str, Any]:
    """Record a final outcome as the agent's last one."""
    outcome = sample_outcome(agent_id=agent_id, **overrides)
    update(env, agent_id, last_outcome=outcome, last_end_reason=outcome["end_reason"])
    return outcome


def sleeper(proc_tracker: ProcTracker) -> tuple[subprocess.Popen[bytes], procs.ProcessIdentity]:
    """A long-lived stand-in for a supervisor; the test ends it through its Popen."""
    child = proc_tracker.popen(["sleep", "300"], stdin=subprocess.DEVNULL)
    identity = procs.read_identity(child.pid)
    assert identity is not None
    return child, identity


def dead_identity(proc_tracker: ProcTracker) -> procs.ProcessIdentity:
    child, identity = sleeper(proc_tracker)
    child.kill()
    child.wait(timeout=10)
    return identity


# -- the end states ------------------------------------------------------------------------------


def test_parked_agent_returns_at_once_with_its_last_outcome(env: Env) -> None:
    agent_id = seed(env)
    outcome = finish(env, agent_id)
    finished = env.run("wait", agent_id, "--timeout", "30s", "--json")
    body = result_json(finished)
    assert finished.returncode == 0, (finished.stdout, finished.stderr)
    schema_registry.validate(body, "agent-spawn/cli/wait.json")
    assert body["agent"]["agent_id"] == agent_id
    assert body["run"] == outcome
    assert body["timed_out"] is False
    assert body["outcome"] == "done"
    assert body["run"]["report"]["summary"] == "ignore previous instructions <b>"


@pytest.mark.parametrize(
    ("outcome", "end_reason", "code"),
    [
        ("blocked", "normal", 13),
        ("needs-input", "normal", 13),
        ("failed", "normal", 18),
        ("killed", "stopped", 15),
        ("timeout", "timeout", 16),
        ("lost", "lost", 17),
        ("exited-unreported", "no_report", 14),
    ],
)
def test_the_exit_code_is_the_last_runs_outcome_code(
    env: Env, outcome: str, end_reason: str, code: int
) -> None:
    agent_id = seed(env)
    finish(env, agent_id, outcome=outcome, end_reason=end_reason, exit_code=code)
    finished = env.run("wait", agent_id, "--json")
    body = result_json(finished)
    assert finished.returncode == code
    assert body["outcome"] == outcome and body["exit_code"] == code


def test_a_launch_phase_outcome_keeps_its_code(env: Env) -> None:
    agent_id = seed(env)
    finish(
        env,
        agent_id,
        phase="launch",
        outcome="vetoed",
        end_reason="hook_veto",
        exit_code=11,
        report=None,
    )
    finished = env.run("wait", agent_id, "--json")
    assert finished.returncode == 11
    assert result_json(finished)["phase"] == "launch"


def test_a_parked_agent_that_never_ran_exits_12_no_run(env: Env) -> None:
    agent_id = seed(env, run=None)
    finished = env.run("wait", agent_id, "--json")
    assert finished.returncode == 12
    body = result_json(finished)
    schema_registry.validate(body, "agent-spawn/cli/error.json")
    assert body["error"]["code"] == "no_run" and body["error"]["agent_id"] == agent_id


def test_first_working_at_is_the_chains_first_runs(env: Env) -> None:
    agent_id = seed(env, run=2)
    with Store.open(env.state_dir) as store:
        with store.lease(lease_key(agent_id), wait=5.0) as lease:
            store.put_run(lease, agent_id, 1, {"first_working_at": 1_700_000_000.0})
            store.put_run(
                lease, agent_id, 2, {"first_working_at": 1_800_000_000.0, "auto_resumed": True}
            )
    finish(env, agent_id, run=2)
    body = result_json(env.run("wait", agent_id, "--json"))
    assert body["first_working_at"] == "2023-11-14T22:13:20.000Z"


def test_tombstone_with_a_run_returns_its_last_outcome(env: Env) -> None:
    agent_id = seed(env, status="torn_down")
    finish(env, agent_id)
    finished = env.run("wait", agent_id, "--json")
    assert finished.returncode == 0, finished.stderr
    assert result_json(finished)["agent"]["status"] == "torn_down"


def test_tombstone_without_a_run_exits_10(env: Env) -> None:
    agent_id = seed(env, status="torn_down", run=None)
    finished = env.run("wait", agent_id, "--json")
    assert finished.returncode == 10
    error = result_json(finished)["error"]
    assert error["code"] == "not_found" and error["details"]["status"] == "torn_down"


def test_a_tombstone_prefix_is_not_found(env: Env) -> None:
    agent_id = seed(env, status="torn_down")
    finish(env, agent_id)
    assert env.run("wait", agent_id[:7], "--json").returncode == 10


def test_unknown_ambiguous_and_missing_state(env: Env) -> None:
    assert env.run("wait", "ag-aaaaaaaa", "--json").returncode == 10  # no state db at all
    first = seed(env)
    finish(env, first)
    assert env.run("wait", "ag-aaaaaaaa", "--json").returncode == 10
    sibling = first[:7] + ("a" if first[7] != "a" else "b") + "bcdef"[: len(first) - 8]
    with Store.open(env.state_dir) as store:
        store.insert_agent(sibling, {"status": "parked", "name": "x"})
    ambiguous = env.run("wait", first[:7], "--json")
    assert ambiguous.returncode == 21
    assert sorted(result_json(ambiguous)["error"]["candidates"]) == sorted([first, sibling])
    assert env.run("wait", "ag-ab", "--json").returncode == 2


# -- waiting ---------------------------------------------------------------------------------------


def test_timeout_on_a_live_agent_exits_22(env: Env, proc_tracker: ProcTracker) -> None:
    _child, identity = sleeper(proc_tracker)
    agent_id = seed(env, status="live", supervisor=identity)
    finished = env.run("wait", agent_id, "--timeout", "300ms", "--json")
    body = result_json(finished)
    assert finished.returncode == 22
    schema_registry.validate(body, "agent-spawn/cli/wait.json")
    assert body["timed_out"] is True and body["run"] is None and body["outcome"] is None
    assert body["agent"]["status"] == "live"
    assert procs.is_alive(identity)  # wait did not touch the agent


def test_a_parked_agent_whose_supervisor_is_alive_is_still_waited_on(
    env: Env, proc_tracker: ProcTracker
) -> None:
    _child, identity = sleeper(proc_tracker)
    agent_id = seed(env, supervisor=identity)
    finish(env, agent_id)
    assert env.run("wait", agent_id, "--timeout", "300ms", "--json").returncode == 22


def test_a_pending_promotion_is_waited_on(env: Env, proc_tracker: ProcTracker) -> None:
    _child, identity = sleeper(proc_tracker)
    agent_id = seed(env, status="tearing_down", promotion_pending=True, supervisor=identity)
    finish(env, agent_id)
    assert env.run("wait", agent_id, "--timeout", "300ms", "--json").returncode == 22


def test_tearing_down_with_a_live_supervisor_is_waited_on(
    env: Env, proc_tracker: ProcTracker
) -> None:
    _child, identity = sleeper(proc_tracker)
    agent_id = seed(env, status="tearing_down", supervisor=identity)
    finish(env, agent_id)
    assert env.run("wait", agent_id, "--timeout", "300ms", "--json").returncode == 22


def test_tearing_down_with_a_dead_supervisor_returns_the_outcome(
    env: Env, proc_tracker: ProcTracker
) -> None:
    agent_id = seed(env, status="tearing_down", supervisor=dead_identity(proc_tracker))
    finish(env, agent_id)
    finished = env.run("wait", agent_id, "--json")
    assert finished.returncode == 0, (finished.stdout, finished.stderr)
    assert result_json(finished)["agent"]["status"] == "tearing_down"


def test_an_ephemeral_agent_torn_down_after_done_exits_0(env: Env) -> None:
    agent_id = seed(env, status="torn_down", ephemeral=True)
    finish(env, agent_id)
    finished = env.run("wait", agent_id, "--json")
    assert finished.returncode == 0, (finished.stdout, finished.stderr)
    body = result_json(finished)
    assert body["agent"]["status"] == "torn_down" and body["outcome"] == "done"


@pytest.mark.parametrize("status", ["parked", "tearing_down"])
def test_a_pending_promotion_with_a_dead_supervisor_is_reported_not_waited_on_forever(
    env: Env, proc_tracker: ProcTracker, status: str
) -> None:
    # Every park clears a pending promotion, so this row is damaged: reported, not waited on.
    agent_id = seed(
        env, status=status, promotion_pending=True, supervisor=dead_identity(proc_tracker)
    )
    finish(env, agent_id)
    finished = env.run("wait", agent_id, "--json")
    assert finished.returncode == 12
    assert result_json(finished)["error"]["code"] == "failed"


def test_a_pending_promotion_without_any_supervisor_is_reported(env: Env) -> None:
    agent_id = seed(env, status="parked", promotion_pending=True)
    finish(env, agent_id)
    assert env.run("wait", agent_id, "--json").returncode == 12


def test_a_parked_agents_report_artifact_is_returned_and_schema_valid(env: Env) -> None:
    agent_id = seed(env)
    artifact = {"name": "plan.md", "path": "/x/artifacts/plan.md", "sha256": "ab" * 32, "size": 7}
    outcome = sample_outcome(agent_id=agent_id)
    outcome["report"]["artifact"] = artifact
    update(env, agent_id, last_outcome=outcome, last_end_reason="normal")
    finished = env.run("wait", agent_id, "--json")
    body = result_json(finished)
    assert finished.returncode == 0, (finished.stdout, finished.stderr)
    schema_registry.validate(body, "agent-spawn/cli/wait.json")
    assert body["run"]["report"]["artifact"] == artifact


def test_a_launching_agent_with_a_live_launcher_is_waited_on(
    env: Env, proc_tracker: ProcTracker
) -> None:
    _child, identity = sleeper(proc_tracker)
    agent_id = seed(env, status="launching", launcher=identity, run=None)
    assert env.run("wait", agent_id, "--timeout", "300ms", "--json").returncode == 22


def test_wait_returns_once_the_agent_parks_and_the_supervisor_exits(
    env: Env, proc_tracker: ProcTracker
) -> None:
    supervisor, identity = sleeper(proc_tracker)
    agent_id = seed(env, status="live", supervisor=identity)
    waiter = proc_tracker.popen(
        env.argv("wait", agent_id, "--json"),
        env=env.env(),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    update(env, agent_id, status="parked", park_reason="done")
    finish(env, agent_id)
    # parked, but the supervisor still runs: only its exit ends the wait
    supervisor.kill()
    supervisor.wait(timeout=10)
    out, err = waiter.communicate(timeout=60)
    assert waiter.returncode == 0, (out, err)
    body = json.loads(out)
    assert body["agent"]["status"] == "parked" and body["outcome"] == "done"


def seed_live(env: Env, proc_tracker: ProcTracker, **fields: Any) -> str:
    """A live agent with a session and a dead supervisor."""
    env.fakebin.seed_session("cs-abcdefgh", adopted=True, final=False)
    return seed(
        env,
        status="live",
        supervisor=dead_identity(proc_tracker),
        csession_id="cs-abcdefgh",
        claude_session_path=str(env.fakebin.bin_dir / "claude-session"),
        **fields,
    )


def test_a_dead_supervisor_with_a_final_run_is_parked_by_wait(
    env: Env, proc_tracker: ProcTracker
) -> None:
    agent_id = seed_live(env, proc_tracker)
    finished = env.run("wait", agent_id, "--json")  # the fake's status says: done
    assert finished.returncode == 0, (finished.stdout, finished.stderr)
    body = result_json(finished)
    assert body["agent"]["status"] == "parked" and body["outcome"] == "done"
    assert env.fakebin.argvs("claude-session") != []
    assert [a for a in env.fakebin.argvs("claude-session") if a[0] in ("stop", "start")] == []


def test_a_dead_supervisor_with_a_live_session_is_flagged_and_waited_on(
    env: Env, proc_tracker: ProcTracker
) -> None:
    script_live_run(env.fakebin)
    agent_id = seed_live(env, proc_tracker)
    finished = env.run("wait", agent_id, "--timeout", "2500ms", "--json")
    assert finished.returncode == 22  # the agent stays live: `stop` is the remedy
    row = env.agent(agent_id)
    assert row["status"] == "live" and row["supervisor_lost"] is True
    assert row["needs_attention"] == "lost"


def test_a_launching_agent_without_launcher_or_supervisor_is_reported(env: Env) -> None:
    agent_id = seed(env, status="launching", run=None)
    assert env.run("wait", agent_id, "--json").returncode == 12


# -- signals -------------------------------------------------------------------------------------


def _handlers_installed(pid: int) -> bool:
    """SIGTERM and SIGHUP are caught (Python catches only SIGINT before ``main`` runs)."""
    for line in Path(f"/proc/{pid}/status").read_text().splitlines():
        if line.startswith("SigCgt:"):
            caught = int(line.split()[1], 16)
            return bool(caught & (1 << (signal.SIGTERM - 1))) and bool(
                caught & (1 << (signal.SIGHUP - 1))
            )
    return False


@pytest.mark.parametrize(
    ("sig", "code"),
    [(signal.SIGTERM, 143), (signal.SIGHUP, 130), (signal.SIGINT, 130)],
)
def test_a_signal_stops_the_wait_and_never_reaches_the_agent(
    env: Env, proc_tracker: ProcTracker, tmp_path: Path, sig: signal.Signals, code: int
) -> None:
    supervisor, identity = sleeper(proc_tracker)
    agent_id = seed(env, status="live", supervisor=identity)
    result = tmp_path / "result.json"
    waiter = proc_tracker.popen(
        env.argv("wait", agent_id, "--json", "--result-file", str(result)),
        env=env.env(),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    env.wait_for(lambda: _handlers_installed(waiter.pid), "the waiter's signal handlers")
    waiter.send_signal(sig)  # the exact Popen child
    out, err = waiter.communicate(timeout=60)
    assert waiter.returncode == code, (out, err)
    assert out == b"" and not result.exists()  # nothing printed, nothing written
    assert supervisor.poll() is None and procs.is_alive(identity)  # never forwarded
    assert env.agent(agent_id)["status"] == "live"


# -- --result-file -------------------------------------------------------------------------------


def test_result_file_receives_exactly_what_json_prints(env: Env, tmp_path: Path) -> None:
    agent_id = seed(env)
    finish(env, agent_id)
    result = tmp_path / "out.json"
    finished = env.run("wait", agent_id, "--json", "--result-file", str(result))
    assert result.read_text() == finished.stdout
    assert stat.S_IMODE(result.stat().st_mode) == 0o600


def test_result_file_also_receives_a_timeout_result(
    env: Env, proc_tracker: ProcTracker, tmp_path: Path
) -> None:
    _child, identity = sleeper(proc_tracker)
    agent_id = seed(env, status="live", supervisor=identity)
    result = tmp_path / "out.json"
    finished = env.run(
        "wait", agent_id, "--timeout", "200ms", "--json", "--result-file", str(result)
    )
    assert finished.returncode == 22
    assert json.loads(result.read_text())["timed_out"] is True


def test_result_file_receives_the_error_envelope_of_a_failed_wait(env: Env, tmp_path: Path) -> None:
    agent_id = seed(env, run=None)
    result = tmp_path / "out.json"
    finished = env.run("wait", agent_id, "--json", "--result-file", str(result))
    assert finished.returncode == 12
    assert result.read_text() == finished.stdout


def test_result_file_never_writes_through_a_symlink(env: Env, tmp_path: Path) -> None:
    agent_id = seed(env)
    finish(env, agent_id)
    victim = tmp_path / "victim"
    victim.write_text("keep")
    link = tmp_path / "link.json"
    link.symlink_to(victim)
    finished = env.run("wait", agent_id, "--json", "--result-file", str(link))
    assert finished.returncode == 0
    assert victim.read_text() == "keep"  # the link is replaced, never written through


def _third_party_agent(env: Env, cwd: Path) -> str:
    agent_id = seed(env, reads_third_party=True, cwd=str(cwd))
    finish(env, agent_id)
    return agent_id


def test_result_file_inside_a_third_party_runs_cwd_exits_2_and_writes_nothing(
    env: Env, tmp_path: Path
) -> None:
    cwd = tmp_path / "throwaway"
    cwd.mkdir()
    agent_id = _third_party_agent(env, cwd)
    for target in (cwd / "r.json", cwd / "sub" / "r.json"):
        finished = env.run("wait", agent_id, "--json", "--result-file", str(target))
        assert finished.returncode == 2, (finished.stdout, finished.stderr)
        assert result_json(finished)["error"]["code"] == "usage"
        assert not target.exists()
    assert list(cwd.iterdir()) == []  # not even a temporary file


def test_result_file_beside_but_not_inside_the_cwd_is_allowed(env: Env, tmp_path: Path) -> None:
    cwd = tmp_path / "work"
    cwd.mkdir()
    agent_id = _third_party_agent(env, cwd)
    sibling = tmp_path / "work-results"
    sibling.mkdir()
    finished = env.run("wait", agent_id, "--json", "--result-file", str(sibling / "r.json"))
    assert finished.returncode == 0, finished.stderr
    assert (sibling / "r.json").exists()


def test_result_file_through_a_symlink_into_the_cwd_is_refused(env: Env, tmp_path: Path) -> None:
    cwd = tmp_path / "work"
    cwd.mkdir()
    agent_id = _third_party_agent(env, cwd)
    link = tmp_path / "alias"
    link.symlink_to(cwd)
    finished = env.run("wait", agent_id, "--json", "--result-file", str(link / "r.json"))
    assert finished.returncode == 2
    assert list(cwd.iterdir()) == []


def test_an_unclassified_run_counts_as_third_party_reading(env: Env, tmp_path: Path) -> None:
    cwd = tmp_path / "work"
    cwd.mkdir()
    agent_id = seed(env, reads_third_party=None, cwd=str(cwd))
    finish(env, agent_id)
    finished = env.run("wait", agent_id, "--json", "--result-file", str(cwd / "r.json"))
    assert finished.returncode == 2


def test_a_trusted_run_may_write_into_its_cwd(env: Env, tmp_path: Path) -> None:
    cwd = tmp_path / "work"
    cwd.mkdir()
    agent_id = seed(env, reads_third_party=False, cwd=str(cwd))
    finish(env, agent_id)
    finished = env.run("wait", agent_id, "--json", "--result-file", str(cwd / "r.json"))
    assert finished.returncode == 0
    assert (cwd / "r.json").exists()


def test_without_a_result_file_a_third_party_run_is_waited_on_normally(
    env: Env, tmp_path: Path
) -> None:
    agent_id = _third_party_agent(env, tmp_path)
    assert env.run("wait", agent_id, "--json").returncode == 0


LAUNCH = ("launch", "--mode", "headless", "--skill", "do-it", "--json", "--wait")


def _mark_third_party(env: Env, agent_id: str) -> None:
    with Store.open(env.state_dir) as store:
        with store.lease(lease_key(agent_id), wait=5.0) as lease:
            store.put_run(lease, agent_id, 1, {"reads_third_party": True})


def test_result_file_inside_a_third_party_runs_session_dir_exits_2(
    env: Env, tmp_path: Path
) -> None:
    first = result_json(env.run(*LAUNCH))
    agent_id = first["agent"]["agent_id"]
    _mark_third_party(env, agent_id)
    session_dir = env.iso.state / "claude-session" / "sessions" / first["agent"]["csession_id"]
    inside = session_dir / "work" / "r.json"  # the fake creates no directory: the path decides
    refused = env.run("wait", agent_id, "--json", "--result-file", str(inside))
    assert refused.returncode == 2, (refused.stdout, refused.stderr)
    assert not inside.exists()
    allowed = env.run("wait", agent_id, "--json", "--result-file", str(tmp_path / "ok.json"))
    assert allowed.returncode == 0
    assert (tmp_path / "ok.json").exists()


def test_the_session_directory_check_uses_the_agents_stored_claude_session(
    env: Env, tmp_path: Path
) -> None:
    first = result_json(env.run(*LAUNCH))
    agent_id = first["agent"]["agent_id"]
    _mark_third_party(env, agent_id)
    session_dir = env.iso.state / "claude-session" / "sessions" / first["agent"]["csession_id"]
    inside = session_dir / "work" / "r.json"
    # claude-session is no longer on PATH: only the path stored on the agent finds it
    refused = env.run(
        "wait",
        agent_id,
        "--json",
        "--result-file",
        str(inside),
        extra_env={"PATH": "/usr/bin:/bin"},
    )
    assert refused.returncode == 2, (refused.stdout, refused.stderr)
    assert not inside.exists()


@pytest.mark.parametrize(("code", "error_code"), [(1, "internal"), (12, "failed"), (20, "refused")])
def test_an_unreadable_session_status_fails_closed_for_a_third_party_run(
    env: Env, fakebin: Fakebin, tmp_path: Path, code: int, error_code: str
) -> None:
    agent_id = result_json(env.run(*LAUNCH))["agent"]["agent_id"]
    _mark_third_party(env, agent_id)
    fakebin.script(
        "claude-session", "status", exit=code, error={"code": error_code, "message": "boom"}
    )
    target = tmp_path / "r.json"
    finished = env.run("wait", agent_id, "--json", "--result-file", str(target))
    assert finished.returncode == code, (finished.stdout, finished.stderr)
    assert result_json(finished)["error"]["code"] == error_code
    assert not target.exists()  # nothing is written, not even the error envelope


# -- after a real launch -------------------------------------------------------------------------


def test_wait_after_launch_returns_the_same_outcome(env: Env) -> None:
    launched = env.run(*LAUNCH)
    first = result_json(launched)
    agent_id = first["agent"]["agent_id"]
    waited = env.run("wait", agent_id, "--json")
    second = result_json(waited)
    assert waited.returncode == launched.returncode == 0
    assert second["run"] == first["run"]
    assert second["first_working_at"] == first["first_working_at"]
    assert second["agent"]["agent_id"] == agent_id
    schema_registry.validate(second, "agent-spawn/cli/wait.json")


def test_wait_changes_nothing_in_the_record(env: Env) -> None:
    agent_id = result_json(env.run(*LAUNCH))["agent"]["agent_id"]
    before = env.agent(agent_id)
    env.run("wait", agent_id, "--json")
    assert env.agent(agent_id) == before


# -- review round 1 ------------------------------------------------------------------------------


def _inproc(env: Env, agent_id: str, result_file: Any = None) -> Any:
    from types import SimpleNamespace

    from agent_spawn.wait_agent import _Signals, _Wait  # pyright: ignore[reportPrivateUsage]

    inv = SimpleNamespace(
        result_file=result_file,
        args=SimpleNamespace(agent_id=agent_id, timeout=None),
        warnings=[],
        decision_ids=[],
        state_dir=str(env.state_dir),
        config=None,
    )
    return _Wait(inv, _Signals())  # type: ignore[arg-type]


def _target(path: Path) -> Any:
    from agent_spawn import output

    return output.resolve_result_file(str(path))


def test_the_result_file_is_checked_again_when_the_run_number_changes(
    env: Env, tmp_path: Path
) -> None:
    from agent_spawn.output import CliError

    cwd = tmp_path / "work"
    cwd.mkdir()
    agent_id = seed(env, reads_third_party=False, cwd=str(cwd))
    waiter = _inproc(env, agent_id, _target(cwd / "r.json"))
    with Store.open(env.state_dir) as store:
        row = store.get_agent(agent_id)
        assert row is not None
        waiter.check_result_file(store, row)  # trusted run 1: allowed, remembered
        assert waiter.checked_run == (agent_id, 1)
        with store.lease(lease_key(agent_id), wait=5.0) as lease:
            store.put_run(lease, agent_id, 2, {"reads_third_party": True})
            store.update_agent(lease, agent_id, {"run": 2})
        row = store.get_agent(agent_id)
        assert row is not None
        with pytest.raises(CliError) as excinfo:
            waiter.check_result_file(store, row)  # run 2 reads third-party content
    assert excinfo.value.exit_code == 2
    assert waiter.inv.result_file is None  # nothing is written, not even the error


def test_a_record_that_disappears_mid_wait_exits_10(
    env: Env, proc_tracker: ProcTracker, monkeypatch: pytest.MonkeyPatch
) -> None:
    from agent_spawn.output import CliError
    from agent_spawn.wait_agent import _Wait  # pyright: ignore[reportPrivateUsage]

    _child, identity = sleeper(proc_tracker)
    agent_id = seed(env, status="live", supervisor=identity)
    waiter = _inproc(env, agent_id)
    gone = False
    original = _Wait.holder

    def holder(row: Any) -> Any:
        nonlocal gone
        gone = True  # the record vanishes after this look
        return original(row)

    class Vanishing:
        def __init__(self, store: Store) -> None:
            self.store = store

        def get_agent(self, key: str) -> Any:
            return None if gone else self.store.get_agent(key)

        def __getattr__(self, name: str) -> Any:
            return getattr(self.store, name)

    monkeypatch.setattr(_Wait, "holder", staticmethod(holder))
    with Store.open(env.state_dir) as store:
        with pytest.raises(CliError) as excinfo:
            waiter.poll(Vanishing(store), agent_id, None)
    assert excinfo.value.exit_code == 10 and excinfo.value.code == "not_found"


def test_a_gone_session_has_no_directory_to_protect(
    env: Env, fakebin: Fakebin, tmp_path: Path
) -> None:
    first = result_json(env.run(*LAUNCH))
    agent_id = first["agent"]["agent_id"]
    _mark_third_party(env, agent_id)
    session_dir = env.iso.state / "claude-session" / "sessions" / first["agent"]["csession_id"]
    fakebin.script(
        "claude-session", "status", error={"code": "not_found", "message": "gone"}, exit=10
    )
    inside = session_dir / "work" / "r.json"
    inside.parent.mkdir(parents=True, exist_ok=True)
    finished = env.run("wait", agent_id, "--json", "--result-file", str(inside))
    assert finished.returncode == 0, (finished.stdout, finished.stderr)
    assert inside.exists()  # the wait proceeded: the session is gone


def test_check_result_file_drops_the_file_on_any_exception(env: Env, tmp_path: Path) -> None:
    cwd = tmp_path / "work"
    cwd.mkdir()
    agent_id = seed(env, reads_third_party=True, cwd=str(cwd))
    waiter = _inproc(env, agent_id, _target(tmp_path / "r.json"))

    class Broken:
        def get_run(self, *_a: Any) -> Any:
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        waiter.check_result_file(Broken(), {"agent_id": agent_id, "run": 1})
    assert waiter.inv.result_file is None  # cli.main writes nothing to an unchecked path


@pytest.mark.parametrize("raises", [False, True])
def test_a_signal_during_the_result_file_check_exits_via_the_signal(
    env: Env, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, raises: bool
) -> None:
    from agent_spawn.output import CliError, PayloadKind
    from agent_spawn.wait_agent import _Wait  # pyright: ignore[reportPrivateUsage]

    agent_id = seed(env)
    finish(env, agent_id)
    waiter = _inproc(env, agent_id, _target(tmp_path / "r.json"))

    def check(self: Any, _store: Any, _row: Any) -> None:
        self.signals.first = signal.SIGTERM
        if raises:
            raise CliError.usage("refused")

    monkeypatch.setattr(_Wait, "check_result_file", check)
    with Store.open(env.state_dir) as store:
        outcome = waiter.poll(store, agent_id, None)
    assert outcome.exit_code == 143 and outcome.kind is PayloadKind.NONE


def test_wait_falls_back_when_the_stored_claude_session_is_gone(
    env: Env, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The result-file check looks the session up with the configured claude-session when the
    one stored with the agent was removed by an upgrade."""
    from types import SimpleNamespace

    from agent_spawn import wait_agent
    from agent_spawn.output import CliError

    configured = tmp_path / "configured-claude-session"
    configured.write_text("#!/bin/sh\nexit 0\n")
    configured.chmod(0o755)
    commands: list[Any] = []

    class Recording:
        def __init__(self, command: Any, **_kwargs: Any) -> None:
            commands.append(list(command))

        def status(self, _session: str) -> Any:
            raise CliError("not_found", "gone")  # exit 10: no directory to protect

    monkeypatch.setattr(wait_agent, "SessionBackend", Recording)
    agent_id = seed(env)
    waiter = _inproc(env, agent_id)
    waiter.inv.config = SimpleNamespace(claude_session_cmd=[str(configured)])
    row = {
        "csession_id": "cs-abcdefgh",
        "claude_session_path": str(tmp_path / "uv-cache" / "claude-session"),
    }
    assert waiter.run_directories(row) == []
    assert commands == [[str(configured)]]
