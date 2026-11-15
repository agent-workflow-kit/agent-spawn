"""``stop``: the action chosen per state, signals to verified processes only, the wait, the
audit record and the races (process tests against the fakes; real stand-in processes are
recorded and killed by exact identity)."""

from __future__ import annotations

import json
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from agent_spawn import lifecycle, procs
from agent_spawn.procs import ProcessIdentity
from agent_spawn.stop import Plan, Stop
from agent_spawn.store import Store
from tests import schema_registry
from tests.conftest import Isolation
from tests.fakebin.harness import Fakebin
from tests.launch_support import Env, result_json, script_live_run
from tests.procguard import ProcTracker

AGENT = "ag-abcdefgh"
SESSION = "cs-abcdefgh"
AUDIT_ARGS = ("--actor", "auto:tester", "--decision-id", "d-1")

CHILD = """
import signal, sys, time
marker, ready, exits = sys.argv[1], sys.argv[2], sys.argv[3] == "exit"
def on_term(signum, frame):
    open(marker, "w").write("term")
    if exits:
        sys.exit(0)
signal.signal(signal.SIGTERM, on_term)
open(ready, "w").write("ready")
for _ in range(1200):
    time.sleep(0.05)
"""


@pytest.fixture
def env(fakebin: Fakebin, isolation_env: Isolation) -> Env:
    return Env(fakebin, isolation_env)


def seed(env: Env, fakebin: Fakebin, **fields: Any) -> None:
    row: dict[str, Any] = {
        "name": "agent test",
        "mode": "headless",
        "status": "live",
        "run": 1,
        "claude_session_path": str(fakebin.bin_dir / "claude-session"),
    }
    row.update(fields)
    with Store.open(env.state_dir) as store:
        store.insert_agent(AGENT, row)


def stand_in(
    env: Env, proc_tracker: ProcTracker, tmp_path: Path, *, exits: bool = True
) -> tuple[ProcessIdentity, Path]:
    """A process that records a SIGTERM in a marker file (and exits unless ``exits`` is false);
    ready once it has a handler."""
    where = tmp_path / f"proc{len(list(tmp_path.glob('proc*')))}"
    where.mkdir()
    script = where / "stand_in.py"
    script.write_text(CHILD)
    marker, ready = where / "marker", where / "ready"
    child = proc_tracker.popen(
        [sys.executable, str(script), str(marker), str(ready), "exit" if exits else "stay"],
        stdin=subprocess.DEVNULL,
    )
    env.wait_for(ready.exists, "the stand-in process")
    identity = procs.read_identity(child.pid)
    assert identity is not None
    return identity, marker


def dead_identity(env: Env, proc_tracker: ProcTracker, tmp_path: Path) -> ProcessIdentity:
    identity, marker = stand_in(env, proc_tracker, tmp_path)
    handle = procs.open_verified(identity)
    assert handle is not None
    with handle:
        assert handle.send(signal.SIGTERM)
        assert handle.wait_exit(10)
    return identity


def agent_row(env: Env) -> dict[str, Any]:
    return env.agent(AGENT)


def park(env: Env, **fields: Any) -> None:
    """What the supervisor (or the launcher) does after it was told to stop."""
    with Store.open(env.state_dir, create=False) as store:
        values: dict[str, Any] = {"park_reason": "killed", "needs_attention": None, **fields}
        assert lifecycle.park(store, AGENT, **values) is not None


def delete_record(env: Env) -> None:
    with Store.open(env.state_dir, create=False) as store:
        with store.lease(lifecycle.lease_key(AGENT)) as lease:
            assert store.delete_agent(lease, AGENT)


def stop_calls(fakebin: Fakebin) -> list[list[str]]:
    return [a for a in fakebin.argvs("claude-session") if a[0] == "stop"]


def audits(env: Env) -> list[dict[str, Any]]:
    path = env.state_dir / "audit.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def check_stop(payload: dict[str, Any]) -> None:
    schema_registry.validate(payload, "agent-spawn/cli/stop.json")


def check_error(payload: dict[str, Any]) -> None:
    schema_registry.validate(payload, "agent-spawn/cli/error.json")


def start_stop(env: Env, proc_tracker: ProcTracker, *extra: str) -> subprocess.Popen[bytes]:
    return proc_tracker.popen(
        env.argv("stop", AGENT, "--json", *AUDIT_ARGS, *extra),
        env=env.env(),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def finish(proc: subprocess.Popen[bytes]) -> tuple[int, Any]:
    out, _err = proc.communicate(timeout=60)
    return proc.returncode, json.loads(out)


# --- no-ops and refusals ------------------------------------------------------------------


def test_a_parked_agent_is_a_no_op(env: Env, fakebin: Fakebin) -> None:
    seed(env, fakebin, status="parked", park_reason="done", csession_id=SESSION)
    done = env.run("stop", AGENT, "--json", *AUDIT_ARGS)
    body = result_json(done)
    assert done.returncode == 0
    check_stop(body)
    assert body["action"] == "none" and body["already_parked"] is True
    assert body["agent"]["status"] == "parked" and body["session_stop"] is None
    assert stop_calls(fakebin) == []
    assert audits(env)[-1]["action"] == "stop" and audits(env)[-1]["result"] == "ok"


def test_an_unknown_agent_exits_10(env: Env, fakebin: Fakebin) -> None:
    seed(env, fakebin, status="parked")
    done = env.run("stop", "ag-zzzzzzzz", "--json")
    assert done.returncode == 10
    check_error(result_json(done))


def test_without_a_state_database_exits_10(env: Env) -> None:
    done = env.run("stop", AGENT, "--json")
    assert done.returncode == 10
    check_error(result_json(done))


def test_a_torn_down_agent_exits_10_with_its_status(env: Env, fakebin: Fakebin) -> None:
    seed(env, fakebin, status="torn_down")
    done = env.run("stop", AGENT, "--json")
    assert done.returncode == 10
    body = result_json(done)
    check_error(body)
    assert body["error"]["details"]["status"] == "torn_down"


def test_a_tearing_down_agent_without_a_live_run_is_a_no_op(env: Env, fakebin: Fakebin) -> None:
    fakebin.seed_session(SESSION)  # its run is final
    seed(env, fakebin, status="tearing_down", teardown_step="started", csession_id=SESSION)
    done = env.run("stop", AGENT, "--json")
    body = result_json(done)
    assert done.returncode == 0
    check_stop(body)
    assert body["action"] == "none" and body["already_parked"] is True
    assert body["agent"]["status"] == "tearing_down"
    assert stop_calls(fakebin) == []


# --- session_stop ---------------------------------------------------------------------------


def test_a_live_run_is_stopped_through_claude_session_and_the_bookkeeping_awaited(
    env: Env, fakebin: Fakebin, proc_tracker: ProcTracker
) -> None:
    fakebin.seed_session(SESSION, adopted=True, final=False)
    script_live_run(fakebin)
    seed(env, fakebin, csession_id=SESSION)
    proc = start_stop(env, proc_tracker, "--grace", "5s")
    env.wait_for(lambda: bool(stop_calls(fakebin)), "claude-session stop")
    assert agent_row(env)["status"] == "live"  # stop records nothing itself
    park(env)  # the supervisor's bookkeeping
    code, body = finish(proc)
    assert code == 0
    check_stop(body)
    assert body["action"] == "session_stop" and body["already_parked"] is False
    assert body["session_stop"]["csession_id"] == SESSION and body["session_stop"]["ended"] is True
    assert body["agent"]["status"] == "parked" and body["agent"]["park_reason"] == "killed"
    call = stop_calls(fakebin)[0]
    assert call[:2] == ["stop", SESSION]
    assert "--grace=5s" in call
    assert "--actor=auto:tester" in call
    assert "--decision-id=d-1" in call
    record = audits(env)[-1]
    schema_registry.validate(record, "agent-spawn/audit.json")
    assert record["action"] == "stop" and record["target"] == AGENT
    assert record["actor"] == "auto:tester" and record["decision_id"] == ["d-1"]
    assert record["inputs"]["csession_id"] == SESSION and record["result"] == "ok"


def test_a_pending_run_of_a_launching_agent_is_canceled_through_claude_session(
    env: Env, fakebin: Fakebin, proc_tracker: ProcTracker
) -> None:
    fakebin.seed_session(SESSION, adopted=False, final=False)
    fakebin.script(
        "claude-session",
        "status",
        merge={
            "pending": True,
            "run": {
                "outcome": None,
                "end_reason": None,
                "exit_code": None,
                "phase": "launch",
                "pending": True,
                "state": "starting",
                "report": None,
            },
        },
    )
    seed(env, fakebin, status="launching", csession_id=SESSION)
    proc = start_stop(env, proc_tracker)
    env.wait_for(lambda: bool(stop_calls(fakebin)), "claude-session stop")
    park(env)
    code, body = finish(proc)
    assert code == 0
    check_stop(body)
    assert body["action"] == "session_stop" and body["session_stop"]["canceled"] is True


def test_a_run_still_live_after_claude_session_stop_exits_12(env: Env, fakebin: Fakebin) -> None:
    fakebin.seed_session(SESSION, adopted=True, final=False)
    script_live_run(fakebin)
    fakebin.script("claude-session", "stop", exit=12, merge={"ended": False})
    seed(env, fakebin, csession_id=SESSION)
    done = env.run("stop", AGENT, "--json", *AUDIT_ARGS)
    assert done.returncode == 12
    check_error(result_json(done))
    assert audits(env)[-1]["result"] == "error"
    assert agent_row(env)["status"] == "live"


def test_claude_sessions_refusal_passes_through_and_is_audited(env: Env, fakebin: Fakebin) -> None:
    fakebin.seed_session(SESSION, adopted=True, final=False)
    script_live_run(fakebin)
    fakebin.script("claude-session", "stop", error={"code": "refused", "message": "no"})
    seed(env, fakebin, csession_id=SESSION)
    done = env.run("stop", AGENT, "--json", *AUDIT_ARGS)
    assert done.returncode == 20
    body = result_json(done)
    check_error(body)
    assert body["error"]["tool"] == "claude-session" and body["error"]["agent_id"] == AGENT
    assert audits(env)[-1]["result"] == "refused"


def test_an_unknown_session_exits_10_through_claude_session(env: Env, fakebin: Fakebin) -> None:
    seed(env, fakebin, csession_id="cs-zzzzzzzz")
    done = env.run("stop", AGENT, "--json")
    assert done.returncode == 10
    check_error(result_json(done))


def test_dry_run_reports_the_action_and_changes_nothing(env: Env, fakebin: Fakebin) -> None:
    fakebin.seed_session(SESSION, adopted=True, final=False)
    script_live_run(fakebin)
    seed(env, fakebin, csession_id=SESSION)
    done = env.run("stop", AGENT, "--json", "--dry-run", *AUDIT_ARGS)
    body = result_json(done)
    assert done.returncode == 0
    check_stop(body)
    assert body["dry_run"] is True and body["action"] == "session_stop"
    assert "--dry-run" in stop_calls(fakebin)[0]
    assert body["agent"]["status"] == "live"
    assert audits(env) == []


def test_a_tearing_down_agent_with_a_live_run_is_stopped_until_the_run_is_final(
    env: Env, fakebin: Fakebin
) -> None:
    fakebin.seed_session(SESSION, adopted=True, final=False)
    script_live_run(fakebin, times=1)  # the first look: live; later looks: final
    seed(env, fakebin, status="tearing_down", teardown_step="started", csession_id=SESSION)
    done = env.run("stop", AGENT, "--json", *AUDIT_ARGS)
    body = result_json(done)
    assert done.returncode == 0
    check_stop(body)
    assert body["action"] == "session_stop" and body["already_parked"] is False
    assert body["agent"]["status"] == "tearing_down"


@pytest.mark.parametrize(
    ("outcome", "end_reason", "park_reason", "attention"),
    [
        ("killed", "stopped", "killed", None),
        # A run that had already reported keeps its report's outcome (claude-session's stop):
        # it parks `escalated` and is never resumed.
        ("needs-input", "stopped", "escalated", "escalated"),
    ],
)
def test_supervisor_lost_agent_is_stopped_and_parked_per_the_outcome_table(
    env: Env,
    fakebin: Fakebin,
    proc_tracker: ProcTracker,
    tmp_path: Path,
    outcome: str,
    end_reason: str,
    park_reason: str,
    attention: str | None,
) -> None:
    fakebin.seed_session(SESSION, adopted=True, final=False)
    script_live_run(fakebin, times=1)
    fakebin.script(
        "claude-session", "status", outcome=outcome, merge={"run": {"end_reason": end_reason}}
    )
    gone = dead_identity(env, proc_tracker, tmp_path)
    seed(env, fakebin, csession_id=SESSION, supervisor=gone, supervisor_lost=True)
    done = env.run("stop", AGENT, "--json", *AUDIT_ARGS)
    body = result_json(done)
    assert done.returncode == 0, done.stderr
    check_stop(body)
    assert body["action"] == "session_stop"
    assert body["agent"]["status"] == "parked"
    assert body["agent"]["park_reason"] == park_reason
    assert body["agent"]["needs_attention"] == attention
    assert body["agent"]["last_end_reason"] == end_reason
    assert [a for a in fakebin.argvs("claude-session") if a[0] == "start"] == []  # no resume


# --- signals to verified processes ----------------------------------------------------------


def test_a_supervisor_between_runs_gets_sigterm(
    env: Env, fakebin: Fakebin, proc_tracker: ProcTracker, tmp_path: Path
) -> None:
    fakebin.seed_session(SESSION)  # the current run is final: nothing for claude-session to stop
    supervisor, marker = stand_in(env, proc_tracker, tmp_path)
    seed(env, fakebin, csession_id=SESSION, supervisor=supervisor)
    proc = start_stop(env, proc_tracker)
    env.wait_for(marker.exists, "the supervisor to get SIGTERM")
    park(env)
    code, body = finish(proc)
    assert code == 0
    check_stop(body)
    assert body["action"] == "signal_supervisor" and body["session_stop"] is None
    assert body["agent"]["status"] == "parked"
    assert stop_calls(fakebin) == []
    assert audits(env)[-1]["result"] == "ok"


def test_a_process_with_another_start_time_is_never_signaled(
    env: Env, fakebin: Fakebin, proc_tracker: ProcTracker, tmp_path: Path
) -> None:
    fakebin.seed_session(SESSION)
    real, marker = stand_in(env, proc_tracker, tmp_path)
    forged = ProcessIdentity(real.pid, real.start_time + 1, real.boot_id)
    seed(env, fakebin, csession_id=SESSION, supervisor=forged)
    done = env.run("stop", AGENT, "--json", *AUDIT_ARGS)
    body = result_json(done)
    # The recorded supervisor is not that process: it counts as dead, the final run is parked
    # by the liveness rule and nothing is signaled.
    assert done.returncode == 0
    check_stop(body)
    assert body["action"] == "none" and body["agent"]["status"] == "parked"
    time.sleep(0.3)
    assert not marker.exists()
    assert procs.is_alive(real)


def test_a_launching_fork_deleted_by_its_launcher_is_removed(
    env: Env, fakebin: Fakebin, proc_tracker: ProcTracker, tmp_path: Path
) -> None:
    launcher, marker = stand_in(env, proc_tracker, tmp_path)
    seed(env, fakebin, status="launching", launcher=launcher, forked_from="ag-bbbbbbbb")
    proc = start_stop(env, proc_tracker)
    env.wait_for(marker.exists, "the launcher to get SIGTERM")
    delete_record(env)
    code, body = finish(proc)
    assert code == 0
    check_stop(body)
    assert body["removed"] is True and body["agent"] is None
    assert body["action"] == "signal_launcher" and body["already_parked"] is False
    assert audits(env) == [] or audits(env)[-1]["action"] == "stop"


def test_a_launching_agent_without_a_session_signals_its_launcher_and_waits_for_the_park(
    env: Env, fakebin: Fakebin, proc_tracker: ProcTracker, tmp_path: Path
) -> None:
    launcher, marker = stand_in(env, proc_tracker, tmp_path)
    seed(env, fakebin, status="launching", launcher=launcher)
    proc = start_stop(env, proc_tracker)
    env.wait_for(marker.exists, "the launcher to get SIGTERM")
    park(env)
    code, body = finish(proc)
    assert code == 0
    check_stop(body)
    assert body["action"] == "signal_launcher" and body["removed"] is False
    assert body["agent"]["status"] == "parked"


def test_the_wait_ends_with_exit_12_when_the_agent_stays_live(
    env: Env, fakebin: Fakebin, proc_tracker: ProcTracker, tmp_path: Path
) -> None:
    fakebin.seed_session(SESSION)
    supervisor, marker = stand_in(env, proc_tracker, tmp_path, exits=False)
    seed(env, fakebin, csession_id=SESSION, supervisor=supervisor)
    # The grace is what bounds the wait (plus a fixed margin); shrink it through the module.
    done = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import agent_spawn.stop as s; s.WAIT_MARGIN = 0.0; "
            "from agent_spawn.cli import main; sys.exit(main(sys.argv[1:]))",
            "stop",
            AGENT,
            "--json",
            "--grace",
            "1s",
        ],
        capture_output=True,
        text=True,
        env=env.env(),
        stdin=subprocess.DEVNULL,
        timeout=60,
        check=False,
    )
    assert done.returncode == 12
    check_error(json.loads(done.stdout))
    assert marker.exists()  # SIGTERM was sent; the agent never parked
    assert agent_row(env)["status"] == "live"


# --- races ----------------------------------------------------------------------------------


def make_stop(env: Env, fakebin: Fakebin) -> Stop:
    inv = SimpleNamespace(
        args=SimpleNamespace(dry_run=False, grace=None, agent_id=AGENT),
        state_dir=str(env.state_dir),
        warnings=[],
        actor="auto:tester",
        decision_ids=[],
        config=None,
    )
    stop = Stop(inv)
    stop.agent_id = AGENT
    stop.store = Store.open(env.state_dir, create=False)
    return stop


def test_a_signal_is_not_sent_when_the_row_changed_after_the_plan(
    env: Env, fakebin: Fakebin, proc_tracker: ProcTracker, tmp_path: Path
) -> None:
    supervisor, marker = stand_in(env, proc_tracker, tmp_path)
    seed(env, fakebin, supervisor=supervisor)
    stop = make_stop(env, fakebin)
    try:
        row = stop.store.get_agent(AGENT)  # type: ignore[union-attr]
        assert row is not None
        plan = Plan("signal_supervisor", row, target=supervisor, field="supervisor")
        park(env)  # a park (then a resume would follow) wins the race
        assert stop.send(plan) is False
        with Store.open(env.state_dir, create=False) as store:  # resumed: a different supervisor
            with store.lease(lifecycle.lease_key(AGENT)) as lease:
                store.update_agent(lease, AGENT, {"supervisor": None})
        assert stop.send(plan) is False
    finally:
        assert stop.store is not None
        stop.store.close()
    time.sleep(0.2)
    assert not marker.exists() and procs.is_alive(supervisor)


def test_the_signal_goes_out_when_the_row_still_names_the_process(
    env: Env, fakebin: Fakebin, proc_tracker: ProcTracker, tmp_path: Path
) -> None:
    supervisor, marker = stand_in(env, proc_tracker, tmp_path)
    seed(env, fakebin, supervisor=supervisor)
    stop = make_stop(env, fakebin)
    try:
        row = stop.store.get_agent(AGENT)  # type: ignore[union-attr]
        assert row is not None
        plan = Plan("signal_supervisor", row, target=supervisor, field="supervisor")
        assert stop.send(plan) is True
    finally:
        assert stop.store is not None
        stop.store.close()
    env.wait_for(marker.exists, "SIGTERM")


def test_stop_during_a_resume_acts_on_the_state_it_finds(
    env: Env, fakebin: Fakebin, proc_tracker: ProcTracker, tmp_path: Path
) -> None:
    """The row went `launching` (a resume won) with a new launcher before `stop` looked: the
    launcher is signaled, never the old supervisor recorded earlier."""
    fakebin.seed_session(SESSION)
    old, old_marker = stand_in(env, proc_tracker, tmp_path)  # an earlier, unrelated process
    launcher, launcher_marker = stand_in(env, proc_tracker, tmp_path)
    seed(env, fakebin, status="launching", csession_id=SESSION, launcher=launcher)
    proc = start_stop(env, proc_tracker)
    env.wait_for(launcher_marker.exists, "the launcher to get SIGTERM")
    park(env)
    code, body = finish(proc)
    assert code == 0 and body["action"] == "signal_launcher"
    assert not old_marker.exists() and procs.is_alive(old)


def test_an_ambiguous_prefix_exits_21_and_stops_nothing(env: Env, fakebin: Fakebin) -> None:
    seed(env, fakebin, status="parked")
    with Store.open(env.state_dir) as store:
        store.insert_agent("ag-abcdefgi", {"name": "other", "status": "parked"})
    done = env.run("stop", "ag-abcd", "--json")
    assert done.returncode == 21
    body = result_json(done)
    check_error(body)
    assert body["error"]["candidates"] == [AGENT, "ag-abcdefgi"]
    assert stop_calls(fakebin) == []


def test_a_unique_prefix_resolves(env: Env, fakebin: Fakebin) -> None:
    seed(env, fakebin, status="parked", park_reason="done")
    done = env.run("stop", "ag-abcdef", "--json")
    assert done.returncode == 0
    assert result_json(done)["agent"]["agent_id"] == AGENT
