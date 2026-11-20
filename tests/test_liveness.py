"""The inline liveness check (interface.md, Liveness) through ``show`` and ``list`` against seeded
rows and the fake claude-session: what each state of the processes and of the session makes of an
agent, that nothing is signaled or stopped, that read-only commands deliver no hook, and that
every recovery is a compare-and-set (concurrent checks apply it once).

``claude-session status`` defaults to a final ``done`` run in the fake; a test scripts a live,
pending or other state explicitly.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from agent_spawn import lifecycle, procs
from agent_spawn.liveness import Liveness
from agent_spawn.output import CliError
from agent_spawn.procs import ProcessIdentity
from agent_spawn.store import Store
from tests import schema_registry
from tests.conftest import Isolation
from tests.fakebin.harness import Fakebin
from tests.fixtures import sample_outcome
from tests.launch_support import Env, result_json, script_live_run
from tests.procguard import ProcTracker

AGENT = "ag-abcdefgh"
OTHER = "ag-bcdefghi"
SESSION = "cs-abcdefgh"


@pytest.fixture
def env(fakebin: Fakebin, isolation_env: Isolation) -> Env:
    fakebin.seed_session(SESSION)  # known to claude-session; a test seeds another state over it
    return Env(fakebin, isolation_env)


def dead(proc_tracker: ProcTracker) -> ProcessIdentity:
    """The identity of a process that has exited (and was reaped)."""
    child = proc_tracker.popen([sys.executable, "-c", "pass"], stdin=subprocess.DEVNULL)
    identity = procs.read_identity(child.pid)
    child.wait(timeout=30)
    assert identity is not None
    return identity


def me(env: Env) -> ProcessIdentity:
    with Store.open(env.state_dir) as store:
        return store.holder_identity()


def seed(env: Env, agent_id: str = AGENT, **fields: Any) -> str:
    row: dict[str, Any] = {
        "name": "seeded",
        "mode": "headless",
        "status": "live",
        "launch_kind": "implement",
        "csession_id": SESSION,
        "run": 1,
        "claude_session_path": str(env.fakebin.bin_dir / "claude-session"),
    }
    row.update(fields)
    with Store.open(env.state_dir) as store:
        store.insert_agent(agent_id, row)
        if isinstance(row["run"], int):
            with store.lease(lifecycle.lease_key(agent_id), wait=5.0) as lease:
                store.put_run(lease, agent_id, row["run"], {"run_kind": "implement"})
    return agent_id


def put_run(env: Env, agent_id: str, run: int, **fields: Any) -> None:
    with Store.open(env.state_dir) as store:
        with store.lease(lifecycle.lease_key(agent_id), wait=5.0) as lease:
            store.put_run(lease, agent_id, run, fields)


def events(env: Env, agent_id: str = AGENT) -> list[dict[str, Any]]:
    with Store.open(env.state_dir) as store:
        return store.pending_events(agent_id)


def show(env: Env, agent_id: str = AGENT) -> dict[str, Any]:
    done = env.run("show", agent_id, "--json")
    assert done.returncode == 0, (done.stdout, done.stderr)
    body = result_json(done)
    schema_registry.validate(body, "agent-spawn/cli/show.json")
    return body


def liveness_parks(env: Env, agent_id: str = AGENT) -> list[dict[str, Any]]:
    """The ``status_change`` log lines of the parks the liveness check made for the agent."""
    path = env.state_dir / "log.jsonl"
    lines = [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []
    return [
        x
        for x in lines
        if x["event"] == "status_change"
        and x["agent_id"] == agent_id
        and (x.get("details") or {}).get("cause") == "liveness"
    ]


def parked_first(env: Env, agent_id: str = AGENT) -> None:
    """The check itself parked the agent (a post-park event and a liveness status_change)."""
    assert "post-park" in [e["event"] for e in events(env, agent_id)]
    assert len(liveness_parks(env, agent_id)) == 1


def calls(fakebin: Fakebin, *names: str) -> list[list[str]]:
    return [a for a in fakebin.argvs("claude-session") if a[0] in names]


def script_pending(fakebin: Fakebin) -> None:
    fakebin.script("claude-session", "status", pending_unadopted=True)


def script_launch_phase(fakebin: Fakebin) -> None:
    """The run is adopted but still in its launch phase (live, not final, not pending)."""
    fakebin.script(
        "claude-session",
        "status",
        merge={
            "state": "starting",
            "live": True,
            "run": {
                "outcome": None,
                "end_reason": None,
                "exit_code": None,
                "phase": "launch",
                "state": "starting",
                "ended_at": None,
                "report": None,
            },
        },
    )


# -- the supervisor decides -----------------------------------------------------------------------


def test_a_live_supervisor_means_nothing_changes_and_claude_session_is_not_called(
    env: Env, fakebin: Fakebin
) -> None:
    seed(env, supervisor=me(env))
    done = env.run("list", "--json")  # (`show` reads the session's status itself)
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout)["agents"][0]["supervisor_lost"] is False
    assert env.agent(AGENT)["status"] == "live"
    assert fakebin.argvs("claude-session") == []  # nothing is read, let alone changed
    assert events(env) == []


def test_a_reused_pid_is_a_dead_supervisor(env: Env, fakebin: Fakebin) -> None:
    mine = me(env)
    fakebin.script("claude-session", "status", outcome="done")
    # the same PID with another start time is not the recorded process
    seed(env, supervisor=ProcessIdentity(mine.pid, mine.start_time + 1, mine.boot_id))
    assert show(env)["agent"]["status"] == "parked"
    parked_first(env)


def test_another_boot_id_is_a_dead_supervisor(env: Env, fakebin: Fakebin) -> None:
    mine = me(env)
    fakebin.script("claude-session", "status", outcome="done")
    seed(env, supervisor=ProcessIdentity(mine.pid, mine.start_time, "another-boot"))
    assert show(env)["agent"]["status"] == "parked"
    parked_first(env)


def test_a_zombie_supervisor_is_dead(env: Env, fakebin: Fakebin, proc_tracker: ProcTracker) -> None:
    fakebin.script("claude-session", "status", outcome="done")
    child = proc_tracker.popen([sys.executable, "-c", "pass"], stdin=subprocess.DEVNULL)
    identity = procs.read_identity(child.pid)
    assert identity is not None
    # not reaped yet: wait until it is a zombie (it has exited), without reaping it
    env.wait_for(lambda: not procs.is_alive(identity), "the child to exit")
    seed(env, supervisor=identity)
    assert show(env)["agent"]["status"] == "parked"
    parked_first(env)
    child.wait(timeout=30)


def test_an_unreadable_process_entry_is_an_error_never_dead(
    env: Env, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    store = Store.open(tmp_path / "state")
    store.insert_agent(
        AGENT,
        {"status": "live", "name": "x", "mode": "headless", "supervisor": store.holder_identity()},
    )

    def broken(*_a: object, **_k: object) -> bool:
        raise procs.ProcessIdentityError("unreadable")

    monkeypatch.setattr(procs, "is_alive", broken)
    with store:
        with pytest.raises(CliError) as caught:
            Liveness(store, state_dir=str(tmp_path / "state")).check(AGENT)
        assert caught.value.exit_code == 12
        assert (store.get_agent(AGENT) or {})["status"] == "live"  # untouched


# -- live with a dead supervisor ------------------------------------------------------------------


def test_a_final_run_is_parked_by_the_outcome_table_without_any_resume_or_stop(
    env: Env, fakebin: Fakebin, proc_tracker: ProcTracker
) -> None:
    fakebin.script("claude-session", "status", outcome="exited-unreported")
    seed(env, supervisor=dead(proc_tracker), exec_proc=dead(proc_tracker))
    body = show(env)["agent"]
    assert body["status"] == "parked" and body["park_reason"] == "unreported"
    assert body["needs_attention"] == "unreported"
    assert body["last_outcome"] == "exited-unreported"
    row = env.agent(AGENT)
    assert row["exec_proc"] is None and row["supervisor_lost"] is False
    # no automatic report resume or promotion (they belong to a supervisor), no stop
    assert calls(fakebin, "start", "stop", "resume", "exec") == []
    stored = env.agent(AGENT)["last_outcome"]
    assert stored["outcome"] == "exited-unreported" and stored["run"] == 1
    with Store.open(env.state_dir) as store:
        run = store.get_run(AGENT, 1)
    assert run is not None and run["outcome"]["outcome"] == "exited-unreported"
    assert run["end_reason"] == stored["end_reason"] and run["ended_at"] is not None


def test_read_only_commands_record_the_park_but_deliver_no_hook(
    env: Env, fakebin: Fakebin, proc_tracker: ProcTracker, tmp_path: Path
) -> None:
    fakebin.script("claude-session", "status", outcome="done")
    marker = tmp_path / "hook-ran"
    hook_config = (
        f"hooks:\n  post-park:\n    - cmd: 'touch {marker}'\n"
        f"  post-state-change:\n    - cmd: 'touch {marker}'\n"
    )
    env.config.parent.mkdir(parents=True, exist_ok=True)
    env.config.write_text(hook_config)
    seed(env, supervisor=dead(proc_tracker))
    assert show(env)["agent"]["status"] == "parked"
    kinds = [e["event"] for e in events(env)]
    assert "post-park" in kinds  # queued in the same transaction as the park
    assert not marker.exists()  # `show` never delivers
    seed(env, OTHER, supervisor=dead(proc_tracker), csession_id=SESSION)
    assert env.run("list", "--json").returncode == 0
    assert not marker.exists()  # `list` neither
    # (`wait` parking a fresh agent without a delivery: test_liveness_review.py)


def test_a_mutating_command_delivers_what_the_check_queued(
    env: Env, proc_tracker: ProcTracker, tmp_path: Path
) -> None:
    marker = tmp_path / "hook-ran"
    env.config.parent.mkdir(parents=True, exist_ok=True)
    env.config.write_text(f"hooks:\n  post-park:\n    - cmd: 'touch {marker}'\n")
    seed(env, supervisor=dead(proc_tracker))
    done = env.run("ack", AGENT, "--json")
    assert done.returncode == 0, done.stderr
    assert marker.exists()
    assert events(env) == []


def test_a_live_session_with_a_dead_supervisor_is_flagged_once(
    env: Env, fakebin: Fakebin, proc_tracker: ProcTracker
) -> None:
    script_live_run(fakebin)
    seed(env, supervisor=dead(proc_tracker))
    for _ in range(3):
        agent = show(env)["agent"]
        assert agent["status"] == "live" and agent["supervisor_lost"] is True
        assert agent["needs_attention"] == "lost"
    kinds = [e["event"] for e in events(env)]
    assert kinds.count("post-escalate") == 1  # the transition into supervisor_lost, once
    assert calls(fakebin, "stop", "start", "resume") == []


def test_ack_does_not_clear_supervisor_lost_and_it_clears_when_the_run_is_final(
    env: Env, fakebin: Fakebin, proc_tracker: ProcTracker
) -> None:
    script_live_run(fakebin, until_called="never")  # live until the test clears the rules
    seed(env, supervisor=dead(proc_tracker))
    assert show(env)["agent"]["supervisor_lost"] is True
    assert env.run("ack", AGENT, "--json").returncode == 0
    row = env.agent(AGENT)
    assert row["supervisor_lost"] is True and row["needs_attention"] is None
    fakebin.clear_script("claude-session", "status")  # the orphaned run ends
    agent = show(env)["agent"]
    assert agent["status"] == "parked" and agent["supervisor_lost"] is False


def test_a_pending_run_is_left_to_claude_sessions_expiry(
    env: Env, fakebin: Fakebin, proc_tracker: ProcTracker
) -> None:
    fakebin.seed_session(SESSION, adopted=False, final=False)
    script_pending(fakebin)
    seed(env, supervisor=dead(proc_tracker), promotion_pending=True)
    agent = show(env)["agent"]
    assert agent["status"] == "live" and agent["supervisor_lost"] is False  # never supervisor_lost
    assert agent["needs_attention"] is None
    assert calls(fakebin, "stop", "start", "resume") == []
    # once claude-session expired it (`never_adopted`), the next check settles it by the chain rule
    fakebin.script(
        "claude-session",
        "status",
        replace=True,
        outcome="not-started",
        merge={"run": {"end_reason": "never_adopted", "run": 2}, "current_run": 2},
    )
    put_run(
        env, AGENT, 1, outcome=_outcome("exited-unreported", "normal", run=1), end_reason="normal"
    )
    put_run(env, AGENT, 2, auto_resumed=True)
    with Store.open(env.state_dir) as store:
        with store.lease(lifecycle.lease_key(AGENT), wait=5) as lease:
            store.update_agent(lease, AGENT, {"run": 2})
    agent = show(env)["agent"]
    assert agent["status"] == "parked" and agent["promotion_pending"] is False
    # the run that triggered the chain run decides: exited-unreported
    assert agent["park_reason"] == "unreported" and agent["needs_attention"] == "unreported"
    assert agent["last_outcome"] == "exited-unreported"
    post_park = next(e for e in events(env) if e["event"] == "post-park")
    failure = post_park["payload"]["chain_failure"]
    assert failure["tool"] == "claude-session" and failure["run"] == 2
    assert failure["end_reason"] == "never_adopted"


def _outcome(outcome: str, end_reason: str, *, run: int, phase: str = "session") -> dict[str, Any]:
    return sample_outcome(
        agent_id=AGENT,
        run=run,
        phase=phase,
        outcome=outcome,
        end_reason=end_reason,
        exit_code=14,
        state="exited-unreported",
        report=None,
    )


def test_a_chain_run_canceled_by_stop_parks_per_its_own_row(
    env: Env, fakebin: Fakebin, proc_tracker: ProcTracker
) -> None:
    fakebin.script(
        "claude-session",
        "status",
        outcome="not-started",
        merge={"run": {"end_reason": "canceled", "run": 2}, "current_run": 2},
    )
    seed(env, supervisor=dead(proc_tracker))
    put_run(env, AGENT, 1, outcome=_outcome("exited-unreported", "normal", run=1))
    put_run(env, AGENT, 2, auto_resumed=True)
    agent = show(env)["agent"]
    assert agent["park_reason"] == "killed" and agent["needs_attention"] is None
    post_park = next(e for e in events(env) if e["event"] == "post-park")
    assert post_park["payload"]["chain_failure"] is None


def test_a_session_unknown_to_claude_session_parks_session_missing(
    env: Env, fakebin: Fakebin, proc_tracker: ProcTracker
) -> None:
    seed(env, supervisor=dead(proc_tracker), csession_id="cs-zzzzzzzz")
    agent = show(env)["agent"]
    assert (agent["status"], agent["park_reason"]) == ("parked", "session_missing")
    assert agent["needs_attention"] == "session_missing"


def test_a_failing_claude_session_changes_nothing(
    env: Env, fakebin: Fakebin, proc_tracker: ProcTracker
) -> None:
    fakebin.script("claude-session", "status", error={"code": "failed", "message": "x"}, exit=12)
    seed(env, supervisor=dead(proc_tracker))
    body = show(env)
    assert body["agent"]["status"] == "live" and body["agent"]["supervisor_lost"] is False
    assert any("status failed" in w for w in body["warnings"])


# -- launching ------------------------------------------------------------------------------------


def test_launching_with_a_dead_supervisor_and_a_session_that_started_is_live_then_lost(
    env: Env, fakebin: Fakebin, proc_tracker: ProcTracker
) -> None:
    script_live_run(fakebin)
    seed(env, status="launching", supervisor=dead(proc_tracker), launcher=dead(proc_tracker))
    agent = show(env)["agent"]
    assert agent["status"] == "live" and agent["supervisor_lost"] is True
    kinds = [e["event"] for e in events(env)]
    assert kinds.count("post-launch") == 1 and kinds.count("post-escalate") == 1


def test_launching_with_a_dead_supervisor_and_an_adopted_run_in_its_launch_phase_waits(
    env: Env, fakebin: Fakebin, proc_tracker: ProcTracker
) -> None:
    script_launch_phase(fakebin)
    seed(env, status="launching", supervisor=dead(proc_tracker))
    agent = show(env)["agent"]
    assert agent["status"] == "launching" and agent["needs_attention"] is None
    assert events(env) == []


def test_launching_with_a_dead_supervisor_and_a_final_run_is_parked_by_its_row(
    env: Env, fakebin: Fakebin, proc_tracker: ProcTracker
) -> None:
    fakebin.script("claude-session", "status", outcome="failed")
    seed(env, status="launching", supervisor=dead(proc_tracker))
    agent = show(env)["agent"]
    assert (agent["status"], agent["park_reason"], agent["needs_attention"]) == (
        "parked",
        "failed",
        "failed",
    )


def test_a_dead_supervisor_that_never_wrote_the_adoption_time_gets_it_from_the_check(
    env: Env, fakebin: Fakebin, proc_tracker: ProcTracker
) -> None:
    script_live_run(fakebin)
    seed(env, status="launching", supervisor=dead(proc_tracker), launcher=dead(proc_tracker))
    shown = show(env)
    assert shown["agent"]["status"] == "live"
    assert shown["runs"][0]["adopted_at"] is not None
    first = shown["runs"][0]["adopted_at"]
    assert show(env)["runs"][0]["adopted_at"] == first  # written once


def test_a_run_in_its_launch_phase_has_no_adoption_time_after_the_check(
    env: Env, fakebin: Fakebin, proc_tracker: ProcTracker
) -> None:
    script_launch_phase(fakebin)
    seed(env, status="launching", supervisor=dead(proc_tracker))
    assert show(env)["runs"][0]["adopted_at"] is None


def test_a_final_run_parked_by_the_check_in_its_session_phase_gets_an_adoption_time(
    env: Env, fakebin: Fakebin, proc_tracker: ProcTracker
) -> None:
    fakebin.script("claude-session", "status", outcome="failed")
    seed(env, status="launching", supervisor=dead(proc_tracker))
    assert show(env)["runs"][0]["adopted_at"] is not None


def test_a_final_run_parked_by_the_check_in_its_launch_phase_has_no_adoption_time(
    env: Env, fakebin: Fakebin, proc_tracker: ProcTracker
) -> None:
    fakebin.script("claude-session", "status", outcome="vetoed")
    seed(env, status="launching", supervisor=dead(proc_tracker))
    assert show(env)["runs"][0]["adopted_at"] is None


def test_launching_with_a_live_launcher_and_no_supervisor_is_left_alone(
    env: Env, fakebin: Fakebin
) -> None:
    seed(env, status="launching", launcher=me(env), csession_id=None, run=None)
    assert show(env)["agent"]["status"] == "launching"
    assert fakebin.argvs("claude-session") == []


def test_a_dead_launcher_with_a_pending_run_is_left_to_never_adopted(
    env: Env, fakebin: Fakebin, proc_tracker: ProcTracker
) -> None:
    fakebin.seed_session(SESSION, adopted=False, final=False)
    script_pending(fakebin)
    seed(env, status="launching", launcher=dead(proc_tracker))
    agent = show(env)["agent"]
    assert agent["status"] == "launching" and agent["needs_attention"] is None
    assert calls(fakebin, "stop") == []
    # claude-session's expiry ends the run `never_adopted`: a launch failure
    fakebin.script(
        "claude-session",
        "status",
        replace=True,
        outcome="not-started",
        merge={"run": {"end_reason": "never_adopted"}},
    )
    agent = show(env)["agent"]
    assert (agent["status"], agent["park_reason"]) == ("parked", "launch_failed")
    assert agent["needs_attention"] == "launch_failed"


def test_a_crash_between_start_and_the_session_write_is_recovered_by_its_label(
    env: Env, fakebin: Fakebin, proc_tracker: ProcTracker
) -> None:
    fakebin.seed_session(SESSION, adopted=False, final=False, labels=[f"agent={AGENT}"])
    script_pending(fakebin)
    seed(env, status="launching", launcher=dead(proc_tracker), csession_id=None, run=None)
    agent = show(env)["agent"]
    assert agent["csession_id"] == SESSION and agent["run"] == 1
    assert agent["status"] == "launching"  # pending: left to the expiry
    assert ["list", f"--label=agent={AGENT}", "--json"] in calls(fakebin, "list")
    with Store.open(env.state_dir) as store:
        assert store.get_run(AGENT, 1) is not None


def test_two_sessions_with_the_label_park_launch_failed_without_a_session(
    env: Env, fakebin: Fakebin, proc_tracker: ProcTracker
) -> None:
    fakebin.seed_session(SESSION, labels=[f"agent={AGENT}"])
    fakebin.seed_session("cs-bcdefghi", labels=[f"agent={AGENT}"])
    seed(env, status="launching", launcher=dead(proc_tracker), csession_id=None, run=None)
    agent = show(env)["agent"]
    assert (agent["status"], agent["park_reason"]) == ("parked", "launch_failed")
    assert agent["csession_id"] is None
    assert calls(fakebin, "stop") == []


def test_a_launch_that_died_before_start_parks_launch_failed(
    env: Env, fakebin: Fakebin, proc_tracker: ProcTracker
) -> None:
    seed(env, status="launching", launcher=dead(proc_tracker), csession_id=None, run=None)
    agent = show(env)["agent"]
    assert (agent["status"], agent["park_reason"]) == ("parked", "launch_failed")
    assert agent["needs_attention"] == "launch_failed"


PRE_RESUME = {
    "park_reason": "done",
    "needs_attention": None,
    "escalated_by": None,
    "last_outcome": None,
    "name": "seeded",
    "term_placement": None,
    "promote": None,
    "updated": 1.0,
}


def test_an_interrupted_resume_gets_its_pre_resume_row_back(
    env: Env, fakebin: Fakebin, proc_tracker: ProcTracker
) -> None:
    fakebin.seed_session(SESSION)  # current run 1: not newer than expect_run_after
    seed(
        env,
        status="launching",
        launcher=dead(proc_tracker),
        pre_resume=PRE_RESUME,
        expect_run_after=1,
    )
    agent = show(env)["agent"]
    assert (agent["status"], agent["park_reason"]) == ("parked", "done")
    row = env.agent(AGENT)
    assert row["launcher"] is None and row["pre_resume"] is None
    assert events(env) == []  # no hooks, no events
    log = (env.state_dir / "log.jsonl").read_text()
    assert '"event": "interrupted"' in log or '"event":"interrupted"' in log


def test_a_resume_whose_run_exists_is_not_restored(
    env: Env, fakebin: Fakebin, proc_tracker: ProcTracker
) -> None:
    fakebin.seed_session(SESSION, adopted=False, final=False, run=2)
    script_pending(fakebin)
    fakebin.script("claude-session", "status", merge={"current_run": 2, "run": {"run": 2}})
    seed(
        env,
        status="launching",
        launcher=dead(proc_tracker),
        pre_resume=PRE_RESUME,
        expect_run_after=1,
    )
    agent = show(env)["agent"]
    assert agent["status"] == "launching"  # the pending run is the resume's own


def test_an_abandoned_fork_is_removed_with_its_holds_and_prompt_copies(
    env: Env, fakebin: Fakebin, proc_tracker: ProcTracker
) -> None:
    seed(env, OTHER)  # the source
    seed(
        env,
        status="launching",
        launcher=dead(proc_tracker),
        csession_id=None,
        run=None,
        forked_from=OTHER,
    )
    prompts = env.state_dir / "prompts" / AGENT
    prompts.mkdir(parents=True)
    (prompts / "prompt.md").write_text("copy")
    with Store.open(env.state_dir) as store:
        with store.lease(lifecycle.lease_key(AGENT), wait=5) as lease:
            store.put_hold(lease, AGENT, "wt-aaaaaaaa", "/x/y", owned=False)
    done = env.run("show", AGENT, "--json")
    assert done.returncode == 10  # the record is gone
    with Store.open(env.state_dir) as store:
        assert store.get_agent(AGENT) is None
        assert store.is_reserved(AGENT)  # the id stays reserved
    assert not prompts.exists()
    log = (env.state_dir / "log.jsonl").read_text()
    assert "fork_abandoned" in log
    forgets = [a for a in fakebin.argvs("wtctl") if a[0] == "forget"]
    assert forgets and forgets[0][1] == "wt-aaaaaaaa" and f"--holder={AGENT}" in forgets[0]
    assert calls(fakebin, "stop") == []


# -- dry runs only report ------------------------------------------------------------------


def test_a_dry_run_reports_and_changes_nothing(
    env: Env, fakebin: Fakebin, proc_tracker: ProcTracker
) -> None:
    fakebin.script("claude-session", "status", outcome="done")
    seed(env, supervisor=dead(proc_tracker))
    done = env.run("teardown", AGENT, "--dry-run", "--json")
    assert done.returncode == 0, (done.stdout, done.stderr)  # the real run would park it first
    row = env.agent(AGENT)
    assert row["status"] == "live" and row["supervisor_lost"] is False
    assert events(env) == [] and liveness_parks(env) == []
    assert calls(fakebin, "stop") == []


def test_teardown_parks_a_dead_supervisors_agent_first_then_tears_it_down(
    env: Env, fakebin: Fakebin, proc_tracker: ProcTracker
) -> None:
    fakebin.script("claude-session", "status", outcome="done")
    seed(env, supervisor=dead(proc_tracker))
    done = env.run("teardown", AGENT, "--json")
    assert done.returncode == 0, (done.stdout, done.stderr)
    assert env.agent(AGENT)["status"] == "torn_down"
    assert len(liveness_parks(env)) == 1  # parked by the check first, then torn down


def test_teardown_still_refuses_a_live_session_with_a_dead_supervisor(
    env: Env, fakebin: Fakebin, proc_tracker: ProcTracker
) -> None:
    script_live_run(fakebin)
    seed(env, supervisor=dead(proc_tracker))
    done = env.run("teardown", AGENT, "--json")
    assert done.returncode == 20
    assert result_json(done)["error"]["reason"] == "live"
    assert env.agent(AGENT)["supervisor_lost"] is True
    assert calls(fakebin, "forget", "stop") == []


def test_resume_parks_a_dead_supervisors_agent_first(
    env: Env, fakebin: Fakebin, proc_tracker: ProcTracker
) -> None:
    fakebin.script("claude-session", "status", outcome="done")
    seed(env, supervisor=dead(proc_tracker))
    done = env.run("resume", AGENT, "--wait", "--json")
    assert done.returncode == 0, (done.stdout, done.stderr)
    assert result_json(done)["agent"]["agent_id"] == AGENT
    assert len(liveness_parks(env)) == 1  # the check parked it, then the resume entered


def test_resume_of_a_session_missing_agent_exits_10_with_the_status(
    env: Env, fakebin: Fakebin, proc_tracker: ProcTracker
) -> None:
    seed(env, supervisor=dead(proc_tracker), csession_id="cs-zzzzzzzz")
    done = env.run("resume", AGENT, "--json")
    assert done.returncode == 10
    assert result_json(done)["error"]["details"]["status"] == "session_missing"
    assert env.agent(AGENT)["park_reason"] == "session_missing"


# -- concurrent checks apply a recovery once -------------------------------------------------------

BARRIER = """
import os, sys, time
barrier = sys.argv[1]
while not os.path.exists(barrier):
    time.sleep(0.002)
from agent_spawn.cli import main
sys.exit(main(sys.argv[2:]))
"""


def start_together(
    env: Env, proc_tracker: ProcTracker, tmp_path: Path, commands: list[list[str]]
) -> list[subprocess.Popen[str]]:
    script = tmp_path / "barrier.py"
    script.write_text(BARRIER)
    barrier = tmp_path / "go"
    started = [
        proc_tracker.popen(
            [sys.executable, str(script), str(barrier), *command],
            env=env.env(),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for command in commands
    ]
    barrier.write_text("go")
    return started  # type: ignore[return-value]


# (concurrent checks apply a recovery once: test_liveness_review.py)


def test_list_runs_the_check_for_live_and_launching_rows(
    env: Env, fakebin: Fakebin, proc_tracker: ProcTracker
) -> None:
    fakebin.script("claude-session", "status", outcome="done")
    seed(env, supervisor=dead(proc_tracker))
    done = env.run("list", "--json")
    body = json.loads(done.stdout)
    assert [a["status"] for a in body["agents"]] == ["parked"]
    parked_first(env)
    assert os.path.exists(env.state_dir / "state.db")
