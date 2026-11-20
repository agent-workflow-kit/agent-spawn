"""Liveness against real processes: a launcher or a supervisor that is SIGKILLed (the exact
recorded process, by identity) and what the next command makes of the agent. The fakes stand in
for claude-session; the launcher, the detached supervisor and their ``exec`` stand-in are real.
"""

from __future__ import annotations

import signal
from pathlib import Path

import pytest

from agent_spawn import procs
from tests.conftest import Isolation
from tests.fakebin.harness import Fakebin
from tests.flow_support import Flow, schema
from tests.procguard import ProcTracker

LAUNCH = ("launch", "--mode", "headless", "--skill", "do-it", "--json")


@pytest.fixture
def flow(fakebin: Fakebin, isolation_env: Isolation, tmp_path: Path) -> Flow:
    return Flow(fakebin, isolation_env, tmp_path)


def kill_exactly(identity: procs.ProcessIdentity) -> None:
    handle = procs.open_verified(identity)
    assert handle is not None
    with handle:
        assert handle.send(signal.SIGKILL)
        assert handle.wait_exit(10)


def end_run_killed(flow: Flow, agent_id: str) -> None:
    """The run ends `killed`: `status` answers that (scripted before the signal, so no read can
    see the live script after `exec` ended), then `exec` gets its SIGTERM."""
    flow.fakebin.script("claude-session", "status", outcome="killed", replace=True)
    flow.fakebin.clear_script("claude-session", "exec")
    flow.end_live_run(agent_id)


def stop_calls(flow: Flow) -> list[list[str]]:
    return [a for a in flow.fakebin.argvs("claude-session") if a[0] == "stop"]


def test_a_launcher_killed_during_start_is_recovered_by_the_session_label(
    flow: Flow, proc_tracker: ProcTracker, tmp_path: Path
) -> None:
    gate = tmp_path / "gate"
    flow.fakebin.script("claude-session", "start", times=1)  # the classification dry run
    flow.fakebin.script("claude-session", "start", times=1, gate=str(gate))  # the real start
    flow.fakebin.script("claude-session", "status", pending_unadopted=True)
    job = flow.start(proc_tracker, *LAUNCH, "--wait")
    flow.fakebin.wait_for_gate(gate, timeout=60)
    (row,) = flow.env.agents()
    agent_id = row["agent_id"]
    assert row["status"] == "launching" and row["csession_id"] is None
    job.kill()  # the recorded launcher, a child this test started
    job.communicate(timeout=30)
    # `start` allocated the session (it carries the agent's label) but nothing recorded it: the
    # liveness check finds it by label; its run is pending, so it is left to the expiry
    shown = flow.cli(schema("show"), "show", agent_id, "--json")
    assert shown.code == 0, (shown.stdout, shown.stderr)
    agent = shown.body["agent"]
    assert agent["status"] == "launching" and agent["csession_id"] is not None
    assert agent["run"] == 1 and agent["needs_attention"] is None
    assert stop_calls(flow) == []
    gate.write_text("open")  # let the held fake finish (it is not this agent's any more)
    (held,) = [
        c
        for c in flow.fakebin.invocations("claude-session")
        if c.argv[0] == "start" and "--dry-run" not in c.argv
    ]
    flow.env.wait_for(lambda: not _alive(held.pid), "the held start to finish")


def _alive(pid: int) -> bool:
    identity = procs.read_identity(pid)
    return identity is not None and procs.is_alive(identity)


def test_a_launcher_killed_after_the_supervisor_took_over_changes_nothing(
    flow: Flow, proc_tracker: ProcTracker
) -> None:
    flow.script_live_status()
    flow.fakebin.script("claude-session", "exec", hang=True)
    job = flow.start(proc_tracker, *LAUNCH, "--wait")
    flow.env.wait_for(
        lambda: any(a.get("exec_proc") for a in flow.env.agents()), "the supervisor's exec"
    )
    (row,) = flow.env.agents()
    agent_id = row["agent_id"]
    flow.track(proc_tracker, agent_id)
    flow.fakebin.wait_for_hang(row["exec_proc"].pid)
    flow.env.wait_for(lambda: flow.agent(agent_id)["status"] == "live", "the agent to be live")
    job.kill()
    job.communicate(timeout=30)
    # the supervisor is alive: it alone decides, so the dead launcher changes nothing
    shown = flow.cli(schema("show"), "show", agent_id, "--json")
    assert shown.body["agent"]["status"] == "live"
    assert shown.body["agent"]["supervisor_lost"] is False
    assert shown.body["supervisor"]["alive"] is True
    # the run goes on and the supervisor finishes it: the final status is scripted first, so
    # whatever a supervisor reads once `exec` ends is the final one (never the live script)
    end_run_killed(flow, agent_id)
    waited = flow.cli(schema("wait"), "wait", agent_id, "--json")
    assert waited.body["agent"]["status"] == "parked"
    flow.wait_settled(agent_id)


def test_a_sigkilled_supervisor_leaves_supervisor_lost_and_stop_is_the_remedy(
    flow: Flow, proc_tracker: ProcTracker
) -> None:
    agent_id = flow.live_launch(proc_tracker)
    row = flow.agent(agent_id)
    kill_exactly(row["supervisor"])
    assert flow.agent(agent_id)["status"] == "live"  # nothing parked it
    shown = flow.cli(schema("show"), "show", agent_id, "--json")
    assert shown.body["agent"]["supervisor_lost"] is True
    assert shown.body["agent"]["needs_attention"] == "lost"
    assert shown.body["supervisor"]["alive"] is False
    assert stop_calls(flow) == []  # liveness never stops the run
    assert procs.is_alive(row["exec_proc"])  # and never signals: `exec` is untouched
    # the remedy: `stop` ends the orphaned run through claude-session, then the next look parks
    stopper = flow.start(proc_tracker, "stop", agent_id, "--json")
    flow.env.wait_for(lambda: bool(stop_calls(flow)), "claude-session stop")
    end_run_killed(flow, agent_id)
    stopped = flow.finish(schema("stop"), stopper)
    assert stopped.code == 0, (stopped.stdout, stopped.stderr)
    agent = flow.agent(agent_id)
    assert agent["status"] == "parked" and agent["park_reason"] == "killed"
    assert agent["supervisor_lost"] is False and agent["needs_attention"] is None


def test_a_sigkilled_exec_under_a_surviving_child_leaves_supervisor_lost_and_stop_is_the_remedy(
    flow: Flow, proc_tracker: ProcTracker
) -> None:
    """The wrapper-dead-child-alive row (issue #97): only ``exec`` is SIGKILLed and the session is
    still live (claude-session reports no outcome, the child ignoring SIGTERM survives). The
    supervisor finds no outcome and ends without parking; the next look flags ``supervisor_lost``
    once and never stops or signals anything; ``stop`` is the remedy."""
    agent_id = flow.live_launch(proc_tracker)
    row = flow.agent(agent_id)
    kill_exactly(row["exec_proc"])  # the status script stays live: no outcome, no outcome file
    flow.wait_settled(agent_id)  # the supervisor read no outcome and ended
    assert flow.agent(agent_id)["status"] == "live"  # nothing parked it
    shown = flow.cli(schema("show"), "show", agent_id, "--json")
    assert shown.body["agent"]["supervisor_lost"] is True
    assert shown.body["agent"]["needs_attention"] == "lost"
    assert shown.body["supervisor"]["alive"] is False
    again = flow.cli(schema("show"), "show", agent_id, "--json")
    assert again.body["agent"]["supervisor_lost"] is True
    flagged = [
        r
        for r in flow.log()
        if r.get("event") == "reconcile_change"
        and r.get("agent_id") == agent_id
        and r["details"]["kind"] == "supervisor_lost"
    ]
    assert len(flagged) == 1  # set once, on the transition
    assert stop_calls(flow) == []  # liveness never stops the run
    stopper = flow.start(proc_tracker, "stop", agent_id, "--json")
    flow.env.wait_for(lambda: bool(stop_calls(flow)), "claude-session stop")
    flow.fakebin.script("claude-session", "status", outcome="killed", replace=True)
    stopped = flow.finish(schema("stop"), stopper)
    assert stopped.code == 0, (stopped.stdout, stopped.stderr)
    agent = flow.agent(agent_id)
    assert agent["status"] == "parked" and agent["park_reason"] == "killed"
    assert agent["supervisor_lost"] is False and agent["needs_attention"] is None


def test_a_supervisor_killed_between_runs_is_parked_without_the_automatic_resume(
    flow: Flow, proc_tracker: ProcTracker
) -> None:
    agent_id = flow.live_launch(proc_tracker)
    row = flow.agent(agent_id)
    kill_exactly(row["supervisor"])
    # the run ended unreported while nobody supervised: no automatic report resume happens
    flow.fakebin.script("claude-session", "status", outcome="exited-unreported", replace=True)
    flow.fakebin.clear_script("claude-session", "exec")
    kill_exactly(row["exec_proc"])  # no outcome file: `status` is the only record of the end
    waited = flow.cli(schema("wait"), "wait", agent_id, "--json")
    assert waited.code == 14, (waited.stdout, waited.stderr)
    agent = flow.agent(agent_id)
    assert (agent["park_reason"], agent["needs_attention"]) == ("unreported", "unreported")
    starts = [
        a for a in flow.fakebin.argvs("claude-session") if a[0] == "start" and "--dry-run" not in a
    ]
    assert len(starts) == 1  # only the launch's own start
