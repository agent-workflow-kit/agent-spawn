"""Stage 1 end-to-end flows (issue #19): launch, wait, resume, fork, stop, list, label, teardown
and export, composed the way callers use them, against the fake lower tools.

Every ``--json`` output goes through :meth:`Flow.cli` (validated against its schema), and each
scenario ends with :meth:`Flow.validate_everything` (hook payloads, log, audit). Intermediate
states are scripted (``live`` status, hung ``exec``, gated ``forget``); nothing depends on timing.
"""

from __future__ import annotations

import signal
from pathlib import Path

import pytest

from agent_spawn import procs
from tests.conftest import Isolation
from tests.fakebin.harness import Fakebin
from tests.flow_support import HOOKS, LIVE_LAUNCH_PHASE, Flow, schema
from tests.launch_support import script_hung_exec
from tests.procguard import ProcTracker

LAUNCH_HEADLESS = ("launch", "--mode", "headless", "--skill", "do-it", "--json")
REVIEW = ("launch", "--mode", "headless", "--kind", "review", "--skill", "review-it", "--json")


@pytest.fixture
def flow(fakebin: Fakebin, isolation_env: Isolation, tmp_path: Path) -> Flow:
    return Flow(fakebin, isolation_env, tmp_path)


# --- the headless review run --------------------------------------------------------------


def test_headless_review_run_to_report_result_file_and_teardown(flow: Flow, tmp_path: Path) -> None:
    result_file = tmp_path / "out" / "review.json"
    result_file.parent.mkdir()
    done = flow.cli(
        schema("launch"),
        *REVIEW,
        "--wait",
        "--label",
        "ticket=T-1",
        "--result-file",
        str(result_file),
    )
    assert done.code == 0, (done.stdout, done.stderr)
    agent_id = done.body["agent"]["agent_id"]
    assert done.body["outcome"] == "done" and done.body["run"]["report"]["status"] == "done"
    assert done.body["agent"]["status"] == "parked" and done.body["agent"]["park_reason"] == "done"
    # the result file holds exactly what --json printed, mode 0600
    assert result_file.read_text() == done.stdout
    assert result_file.stat().st_mode & 0o777 == 0o600
    assert flow.settled(agent_id)  # --wait returned only after the supervisor exited

    shown = flow.cli(schema("show"), "show", agent_id, "--json")
    assert shown.code == 0 and shown.body["runs"][0]["outcome"]["outcome"] == "done"
    waited = flow.cli(schema("wait"), "wait", agent_id, "--json")
    assert waited.code == 0 and waited.body["outcome"] == "done"

    gone = flow.cli(schema("teardown"), "teardown", agent_id, "--json")
    assert gone.code == 0, (gone.stdout, gone.stderr)
    assert gone.body["agent"]["status"] == "torn_down" and gone.body["session_forgotten"] is True
    assert not (flow.state_dir / "prompts" / agent_id).exists()
    assert not (flow.state_dir / "outcomes" / agent_id).exists()
    tomb = flow.cli(schema("wait"), "wait", agent_id, "--json")
    assert tomb.code == 0 and tomb.body["agent"]["status"] == "torn_down"
    # a torn-down agent is not found by a prefix either, whatever the command
    by_prefix = flow.cli(schema("resume"), "resume", agent_id[:8], "--json")
    assert by_prefix.code == 10, (by_prefix.stdout, by_prefix.stderr)

    exported = flow.cli(schema("export"), "export", "--json")
    assert exported.code == 0 and exported.body == []  # no worktree, no --force
    park = flow.hooks_named("post-park", agent_id)
    assert [p["reason"] for p in park] == ["done"]
    flow.validate_everything(
        expect_hooks=[
            "pre-launch",
            "post-launch",
            "post-state-change",
            "post-park",
            "post-teardown",
        ]
    )


def test_blocked_review_run_is_acknowledged_resumed_and_finished(flow: Flow) -> None:
    flow.fakebin.script("claude-session", "exec", times=1, outcome="blocked")
    first = flow.cli(schema("launch"), *REVIEW, "--wait")
    assert first.code == 13, (first.stdout, first.stderr)
    agent_id = first.body["agent"]["agent_id"]
    assert first.body["outcome"] == "blocked"
    assert first.body["agent"]["needs_attention"] == "blocked"
    assert flow.settled(agent_id)
    listed = flow.cli(schema("list"), "list", "--needs-attention", "--json")
    assert [a["agent_id"] for a in listed.body["agents"]] == [agent_id]
    acked = flow.cli(schema("ack"), "ack", agent_id, "--json")
    assert acked.code == 0 and acked.body["changed"] is True
    assert acked.body["agent"]["needs_attention"] is None
    resumed = flow.cli(schema("resume"), "resume", agent_id, "--wait", "--json")
    assert resumed.code == 0, (resumed.stdout, resumed.stderr)
    assert resumed.body["outcome"] == "done" and resumed.body["agent"]["status"] == "parked"
    assert flow.agent(agent_id)["run"] == 2
    assert flow.settled(agent_id)
    assert flow.cli(schema("list"), "list", "--needs-attention", "--json").body["agents"] == []
    flow.validate_everything(expect_hooks=["post-launch", "post-escalate", "post-park"])
    escalations = flow.hooks_named("post-escalate", agent_id)
    assert [p["needs_attention"] for p in escalations] == ["blocked"]
    launches = flow.hooks_named("post-launch", agent_id)
    assert sorted(p["run"] for p in launches) == [1, 2]
    assert [p["resumed"] for p in sorted(launches, key=lambda p: p["run"])] == [False, True]


# --- launch without --wait, wait, resume, fork, stop, list, labels -----------------------------


def stop_calls(flow: Flow) -> list[list[str]]:
    return [a for a in flow.fakebin.argvs("claude-session") if a[0] == "stop"]


def test_live_agent_wait_label_stop_resume_fork_and_export(
    flow: Flow, proc_tracker: ProcTracker, isolation_env: Isolation
) -> None:
    csd = isolation_env.state / "claude-session" / "sessions" / "cs-flowflow"
    csd.mkdir(parents=True)
    secret = csd / "secret"
    secret.write_text("flow-secret\n")
    secret.chmod(0o600)
    flow.fakebin.script("claude-session", "verify-secret", secret="flow-secret")
    agent_id = flow.live_launch(
        proc_tracker, "--label", "ticket=T-1", "--label", "team=blue", csession_dir=str(csd)
    )
    session = flow.agent(agent_id)["csession_id"]

    # a wait that times out leaves the agent running; a resume of a live agent is refused
    timed_out = flow.cli(schema("wait"), "wait", agent_id, "--timeout", "200ms", "--json")
    assert timed_out.code == 22 and timed_out.body["timed_out"] is True
    assert timed_out.body["run"] is None
    refused = flow.cli(schema("resume"), "resume", agent_id, "--json")
    assert refused.code == 20 and refused.body["error"]["reason"] == "live"
    assert flow.agent(agent_id)["status"] == "live"

    # list by label; labels by hand and by the agent itself (self-labeling)
    by_ticket = flow.cli(schema("list"), "list", "--label", "ticket=T-1", "--json")
    assert [a["agent_id"] for a in by_ticket.body["agents"]] == [agent_id]
    assert by_ticket.body["agents"][0]["status"] == "live"
    added = flow.cli(schema("label-add"), "label", "add", agent_id, "stage=review", "--json")
    assert added.code == 0 and added.body["added"] == ["stage=review"]
    removed = flow.cli(schema("label-remove"), "label", "remove", agent_id, "team=blue", "--json")
    assert removed.code == 0 and removed.body["removed"] == ["team=blue"]
    own = flow.cli(
        schema("label-add"),
        "label", "add", "--session", session, "pr=h/o/r#7", "--json",
        extra_env={"CSESSION_ID": session, "CSESSION_DIR": str(csd)},
    )  # fmt: skip
    assert own.code == 0, (own.stdout, own.stderr)
    assert own.body["agent_id"] == agent_id and own.body["added"] == ["pr=h/o/r#7"]
    assert flow.cli(schema("list"), "list", "--label", "team=blue", "--json").body["agents"] == []
    both = flow.cli(
        schema("list"), "list", "--label", "ticket=T-1", "--label", "pr=h/o/r#7", "--json"
    )
    assert [a["agent_id"] for a in both.body["agents"]] == [agent_id]
    assert sorted(both.body["agents"][0]["labels"])[:2] == ["pr=h/o/r#7", "stage=review"]

    # stop: a waiter and the stopper both see the run end `killed`
    waiter = flow.start(proc_tracker, "wait", agent_id, "--json")
    stopper = flow.start(
        proc_tracker, "stop", agent_id, "--json", "--actor", "auto:flow", "--decision-id", "d-1"
    )
    flow.env.wait_for(lambda: bool(stop_calls(flow)), "claude-session stop")
    flow.end_live_run(agent_id)  # claude-session stop signals the run's exec
    stopped = flow.finish(schema("stop"), stopper)
    waited = flow.finish(schema("wait"), waiter)
    assert flow.settled(agent_id)  # stop and wait returned only after the supervisor exited
    assert stopped.code == 0, (stopped.stdout, stopped.stderr)
    assert stopped.body["action"] == "session_stop" and stopped.body["already_parked"] is False
    assert stopped.body["agent"]["status"] == "parked"
    assert stopped.body["agent"]["park_reason"] == "killed"
    assert stopped.body["agent"]["needs_attention"] is None
    assert waited.code == 15 and waited.body["outcome"] == "killed"
    flow.back_to_final_status()
    noop = flow.cli(schema("stop"), "stop", agent_id, "--json")
    assert noop.code == 0 and noop.body["already_parked"] is True and noop.body["action"] == "none"

    # resume the stopped agent, then fork it
    resumed = flow.cli(schema("resume"), "resume", agent_id, "--wait", "--json")
    assert resumed.code == 0, (resumed.stdout, resumed.stderr)
    assert resumed.body["outcome"] == "done" and flow.agent(agent_id)["run"] == 2
    assert flow.settled(agent_id)
    forked = flow.cli(
        schema("resume"),
        "resume", agent_id, "--fork", "--cwd", "session-dir", "--wait",
        "--label", "branch=x", "--json",
    )  # fmt: skip
    assert forked.code == 0, (forked.stdout, forked.stderr)
    fork_id = forked.body["agent"]["agent_id"]
    assert fork_id != agent_id and forked.body["agent"]["forked_from"] == agent_id
    children = flow.cli(schema("list"), "list", "--forked-from", agent_id, "--json")
    assert [a["agent_id"] for a in children.body["agents"]] == [fork_id]
    by_prefix = flow.cli(schema("list"), "list", "--forked-from", agent_id[:8], "--json")
    assert [a["agent_id"] for a in by_prefix.body["agents"]] == [fork_id]
    assert flow.agent(agent_id)["status"] == "parked" and flow.agent(agent_id)["run"] == 2
    assert flow.settled(agent_id) and flow.settled(fork_id)

    # the audit trail: the stop is there, with its actor and decision id
    exported = flow.cli(schema("export"), "export", "--action", "stop", "--json")
    assert len(exported.body) == 2  # the stop and the no-op stop of the parked agent (both audited)
    record = exported.body[0]
    assert exported.body[1]["target"] == agent_id and exported.body[1]["decision_id"] == []
    assert record["action"] == "stop" and record["target"] == agent_id
    assert record["actor"] == "auto:flow" and record["decision_id"] == ["d-1"]
    assert record["result"] == "ok" and record["inputs"]["csession_id"] == session
    assert flow.cli(schema("export"), "export", "--action", "teardown", "--json").body == []

    # both agents end torn down
    for target in (fork_id, agent_id):
        assert flow.cli(schema("teardown"), "teardown", target, "--json").code == 0
    assert flow.cli(schema("list"), "list", "--json").body["agents"] == []
    gone = flow.cli(schema("list"), "list", "--status", "torn_down", "--json")
    assert {a["agent_id"] for a in gone.body["agents"]} == {agent_id, fork_id}
    flow.validate_everything(
        expect_hooks=["post-launch", "post-state-change", "post-park", "post-teardown"],
        expect_audit=2,
    )
    assert {p["run"] for p in flow.hooks_named("post-launch", agent_id)} == {1, 2}
    assert [p["fork"] for p in flow.hooks_named("post-launch", fork_id)] == [True]


# --- a job-style caller canceled from outside -------------------------------------------------


def test_launch_wait_under_sigterm_during_the_prelaunch_hooks_parks_killed_with_a_launch_result(
    flow: Flow, proc_tracker: ProcTracker
) -> None:
    # claude-session's pre-launch hooks are still running: the run is in its launch phase
    flow.fakebin.script("claude-session", "status", merge={"run": LIVE_LAUNCH_PHASE})
    # `exec` ends with exit 12 on the forwarded SIGTERM, as it does while claude-session is
    # still in its launch phase (the scripted `status` below then reports the not-started run)
    flow.fakebin.script("claude-session", "exec", hang=True, on_signal={"SIGTERM": {"exit": 12}})
    job = flow.start(proc_tracker, *LAUNCH_HEADLESS, "--wait")
    flow.env.wait_for(
        lambda: any(a.get("exec_proc") for a in flow.env.agents()), "the supervisor's exec"
    )
    (row,) = flow.env.agents()
    agent_id = row["agent_id"]
    flow.track(proc_tracker, agent_id)
    flow.fakebin.wait_for_hang(row["exec_proc"].pid)
    assert flow.agent(agent_id)["status"] == "launching"
    # the cancellation arrives: the run ends in its launch phase, never started
    flow.fakebin.script("claude-session", "status", replace=True, outcome="not-started")
    job.send_signal(signal.SIGTERM)  # the jobs runner's SIGTERM to the launcher
    done = flow.finish(schema("launch"), job)
    assert done.code == 12, (done.stdout, done.stderr)
    assert done.body["phase"] == "launch" and done.body["outcome"] == "not-started"
    parked = flow.agent(agent_id)
    assert parked["status"] == "parked" and parked["park_reason"] == "killed"
    assert parked["needs_attention"] is None
    assert flow.settled(agent_id)
    assert flow.cli(schema("wait"), "wait", agent_id, "--json").code == 12
    flow.validate_everything(expect_hooks=["post-park"])
    assert [p["reason"] for p in flow.hooks_named("post-park", agent_id)] == ["killed"]


# --- teardown combined with wait, resume and stop (issue #18, criterion 3) ---------------------

EPHEMERAL = (*LAUNCH_HEADLESS, "--ephemeral")


def test_wait_and_resume_during_an_ephemeral_teardown(
    flow: Flow, proc_tracker: ProcTracker, tmp_path: Path
) -> None:
    gate = tmp_path / "forget-gate"
    flow.fakebin.script("claude-session", "forget", gate=str(gate))  # holds teardown step 6
    started = flow.cli(schema("launch"), *EPHEMERAL)
    assert started.code == 0, (started.stdout, started.stderr)
    agent_id = started.body["agent"]["agent_id"]
    # tracked before anything can fail, and the gate opens whatever happens, so a failing
    # assertion never leaves the supervisor or the gated fake unrecorded or running
    flow.track(proc_tracker, agent_id)
    try:
        flow.fakebin.wait_for_gate(gate)
        row = flow.agent(agent_id)
        assert row["status"] == "tearing_down" and row["teardown_step"] == "hooks_done"
        supervisor = row["supervisor"]
        assert supervisor is not None and procs.is_alive(supervisor)

        # a resume is refused for as long as the teardown runs
        refused = flow.cli(schema("resume"), "resume", agent_id, "--json")
        assert refused.code == 20 and refused.body["error"]["reason"] == "tearing_down"
        assert flow.agent(agent_id)["status"] == "tearing_down"
        # a wait does not return while the supervisor is alive
        early = flow.cli(schema("wait"), "wait", agent_id, "--timeout", "300ms", "--json")
        assert early.code == 22 and early.body["timed_out"] is True
        waiter = flow.start(proc_tracker, "wait", agent_id, "--json")
    finally:
        gate.write_text("")  # claude-session forgets the session; the teardown finishes
    done = flow.finish(schema("wait"), waiter)
    assert done.code == 0, (done.stdout, done.stderr)
    assert done.body["agent"]["status"] == "torn_down" and done.body["outcome"] == "done"
    assert not procs.is_alive(supervisor)  # it returned only after the supervisor exited

    # the tombstone: resume is refused, wait still answers, no second teardown
    gone = flow.cli(schema("resume"), "resume", agent_id, "--json")
    assert gone.code == 10 and gone.body["error"]["details"]["status"] == "torn_down"
    assert flow.cli(schema("wait"), "wait", agent_id, "--json").code == 0
    again = flow.cli(schema("teardown"), "teardown", agent_id, "--json")
    assert again.code == 10 and again.body["error"]["details"]["status"] == "torn_down"
    assert flow.cli(schema("export"), "export", "--json").body == []
    flow.validate_everything(expect_hooks=["post-launch", "post-teardown"])
    assert len(flow.hooks_named("post-teardown", agent_id)) == 1
    assert flow.hooks_named("post-park", agent_id) == []  # never parked: it went to tombstone


def test_stop_is_the_remedy_for_an_ephemeral_agent_left_tearing_down(
    flow: Flow, proc_tracker: ProcTracker
) -> None:
    # the session still has an orphaned child, so claude-session refuses to forget it
    flow.fakebin.script(
        "claude-session",
        "forget",
        times=1,
        error={"code": "refused", "message": "a run is still live", "reason": "live"},
    )
    started = flow.cli(schema("launch"), *EPHEMERAL, "--wait")
    agent_id = started.body["agent"]["agent_id"]
    assert flow.settled(agent_id)
    row = flow.agent(agent_id)
    assert row["status"] == "tearing_down" and row["teardown_step"] == "hooks_done"
    assert row["park_reason"] is None  # a tearing_down row carries no park reason

    # no supervisor is left: wait answers at once with the last outcome
    waited = flow.cli(schema("wait"), "wait", agent_id, "--json")
    assert waited.code == 0 and waited.body["agent"]["status"] == "tearing_down"
    # the orphan is live: stop acts on it, and a resume stays refused
    # the run reads live until a `stop` was logged, whatever number of looks come before
    flow.script_live_status(until_called="stop")
    # the fake session recorded the run as done; the orphan is a live process of it, so
    # claude-session's stop kills it (scripted: the fake's own bookkeeping says already final)
    flow.fakebin.script(
        "claude-session",
        "stop",
        merge={
            "already_final": False,
            "canceled": False,
            "run_exit_code": 15,
            "state": "killed",
            "outcome": "killed",
            "end_reason": "stopped",
        },
    )
    stopped = flow.cli(schema("stop"), "stop", agent_id, "--json", "--grace", "1s")
    assert stopped.code == 0, (stopped.stdout, stopped.stderr)
    assert stopped.body["action"] == "session_stop" and stopped.body["already_parked"] is False
    assert stopped.body["agent"]["status"] == "tearing_down"
    assert [a for a in flow.fakebin.argvs("claude-session") if a[0] == "stop"]
    refused = flow.cli(schema("resume"), "resume", agent_id, "--json")
    assert refused.code == 20 and refused.body["error"]["reason"] == "tearing_down"

    # the retry now completes where it stopped (the session step)
    retried = flow.cli(schema("teardown"), "teardown", agent_id, "--json")
    assert retried.code == 0, (retried.stdout, retried.stderr)
    assert retried.body["agent"]["status"] == "torn_down" and retried.body["session_forgotten"]
    stops = flow.cli(schema("export"), "export", "--action", "stop", "--json").body
    assert [r["target"] for r in stops] == [agent_id]
    flow.validate_everything(expect_hooks=["post-teardown"], expect_audit=1)


def test_ephemeral_agent_runs_its_pre_teardown_hook_before_the_teardown(
    fakebin: Fakebin, isolation_env: Isolation, tmp_path: Path
) -> None:
    flow = Flow(fakebin, isolation_env, tmp_path)
    started = flow.cli(schema("launch"), *EPHEMERAL, "--wait")
    assert started.code == 0, (started.stdout, started.stderr)
    agent_id = started.body["agent"]["agent_id"]
    assert flow.settled(agent_id)
    assert flow.agent(agent_id)["status"] == "torn_down"
    (pre,) = flow.hooks_named("pre-teardown", agent_id)
    assert pre["override_escalation"] is False and pre["actor"] == "auto:agent-spawn"
    assert len(flow.hooks_named("post-teardown", agent_id)) == 1
    flow.validate_everything(expect_hooks=["pre-teardown", "post-teardown"])


def test_job_launch_wait_canceled_after_the_run_started_parks_killed(
    flow: Flow, proc_tracker: ProcTracker
) -> None:
    flow.script_live_status()
    script_hung_exec(flow.fakebin)
    job = flow.start(
        proc_tracker, *LAUNCH_HEADLESS, "--wait", extra_env={"JOBSCTL_JOB_ID": "job-42"}
    )
    flow.env.wait_for(
        lambda: any(a["status"] == "live" for a in flow.env.agents()), "the run to start"
    )
    (row,) = flow.env.agents()
    agent_id = row["agent_id"]
    flow.track(proc_tracker, agent_id)
    flow.fakebin.wait_for_hang(row["exec_proc"].pid)
    # the cancellation chain: launcher -> supervisor -> exec, each forwarding to the next
    job.send_signal(signal.SIGTERM)
    done = flow.finish(schema("launch"), job)
    assert done.code == 15, (done.stdout, done.stderr)
    assert done.body["phase"] == "session" and done.body["outcome"] == "killed"
    assert done.body["agent"]["park_reason"] == "killed"
    assert done.body["agent"]["needs_attention"] is None
    assert flow.settled(agent_id)
    flow.back_to_final_status()
    # the job id travelled as a label and finds the agent
    found = flow.cli(schema("list"), "list", "--label", "job=job-42", "--json")
    assert [a["agent_id"] for a in found.body["agents"]] == [agent_id]
    # the label cap is 64 pairs, the job= pair included: 63 more are accepted, a 65th is not
    fill = [f"k{i}=v" for i in range(63)]
    filled = flow.cli(schema("label-add"), "label", "add", agent_id, *fill, "--json")
    assert filled.code == 0, (filled.stdout, filled.stderr)
    assert len(flow.cli(schema("show"), "show", agent_id, "--json").body["agent"]["labels"]) >= 64
    over = flow.cli(schema("label-add"), "label", "add", agent_id, "one=more", "--json")
    assert over.code == 2, (over.stdout, over.stderr)
    # a stopped run is never resumed automatically; the caller may resume it deliberately
    assert flow.agent(agent_id)["run"] == 1
    flow.validate_everything(expect_hooks=["post-launch", "post-park"])
    assert [p["reason"] for p in flow.hooks_named("post-park", agent_id)] == ["killed"]
