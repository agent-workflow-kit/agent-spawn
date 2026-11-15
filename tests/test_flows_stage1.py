"""Stage 1 end-to-end flows (issue #19): launch, wait, resume, fork, stop, list, label, teardown
and export, composed the way callers use them, against the fake lower tools.

Every ``--json`` output goes through :meth:`Flow.cli` (validated against its schema), and each
scenario ends with :meth:`Flow.validate_everything` (hook payloads, log, audit). Intermediate
states are scripted (``live`` status, hung ``exec``, gated ``forget``); nothing depends on timing.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.conftest import Isolation
from tests.fakebin.harness import Fakebin
from tests.flow_support import Flow, schema
from tests.procguard import ProcTracker

REVIEW = ("launch", "--mode", "headless", "--kind", "review", "--skill", "review-it", "--json")


@pytest.fixture
def flow(fakebin: Fakebin, isolation_env: Isolation, tmp_path: Path) -> Flow:
    return Flow(fakebin, isolation_env, tmp_path)


# --- the headless review run --------------------------------------------------------------


def test_headless_review_run_to_report_result_file_and_teardown(flow: Flow, tmp_path: Path) -> None:
    result_file = tmp_path / "out" / "review.json"
    result_file.parent.mkdir()
    done = flow.cli(
        schema("launch"), *REVIEW, "--wait", "--label", "ticket=T-1", "--result-file", str(result_file)
    )
    assert done.code == 0, (done.stdout, done.stderr)
    agent_id = done.body["agent"]["agent_id"]
    assert done.body["outcome"] == "done" and done.body["run"]["report"]["status"] == "done"
    assert done.body["agent"]["status"] == "parked" and done.body["agent"]["park_reason"] == "done"
    # the result file holds exactly what --json printed, mode 0600
    assert result_file.read_text() == done.stdout
    assert result_file.stat().st_mode & 0o777 == 0o600
    flow.wait_settled(agent_id)

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

    exported = flow.cli(schema("export"), "export", "--json")
    assert exported.code == 0 and exported.body == []  # no worktree, no --force
    park = flow.hooks_named("post-park", agent_id)
    assert [p["reason"] for p in park] == ["done"]
    flow.validate_everything(
        expect_hooks=["pre-launch", "post-launch", "post-state-change", "post-park", "post-teardown"]
    )


def test_blocked_review_run_is_acknowledged_resumed_and_finished(flow: Flow) -> None:
    flow.fakebin.script("claude-session", "exec", times=1, outcome="blocked")
    first = flow.cli(schema("launch"), *REVIEW, "--wait")
    assert first.code == 13, (first.stdout, first.stderr)
    agent_id = first.body["agent"]["agent_id"]
    assert first.body["outcome"] == "blocked"
    assert first.body["agent"]["needs_attention"] == "blocked"
    flow.wait_settled(agent_id)
    listed = flow.cli(schema("list"), "list", "--needs-attention", "--json")
    assert [a["agent_id"] for a in listed.body["agents"]] == [agent_id]
    acked = flow.cli(schema("ack"), "ack", agent_id, "--json")
    assert acked.code == 0 and acked.body["changed"] is True
    assert acked.body["agent"]["needs_attention"] is None
    resumed = flow.cli(schema("resume"), "resume", agent_id, "--wait", "--json")
    assert resumed.code == 0, (resumed.stdout, resumed.stderr)
    assert resumed.body["outcome"] == "done" and resumed.body["agent"]["status"] == "parked"
    assert flow.agent(agent_id)["run"] == 2
    flow.wait_settled(agent_id)
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
    forked = flow.cli(
        schema("resume"),
        "resume", agent_id, "--fork", "--cwd", "session-dir", "--wait", "--label", "branch=x", "--json",
    )  # fmt: skip
    assert forked.code == 0, (forked.stdout, forked.stderr)
    fork_id = forked.body["agent"]["agent_id"]
    assert fork_id != agent_id and forked.body["agent"]["forked_from"] == agent_id
    children = flow.cli(schema("list"), "list", "--forked-from", agent_id, "--json")
    assert [a["agent_id"] for a in children.body["agents"]] == [fork_id]
    assert flow.agent(agent_id)["status"] == "parked" and flow.agent(agent_id)["run"] == 2
    flow.wait_settled(agent_id)
    flow.wait_settled(fork_id)

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
        expect_hooks=["post-launch", "post-state-change", "post-park", "post-teardown"]
    )
    assert {p["run"] for p in flow.hooks_named("post-launch", agent_id)} == {1, 2}
    assert [p["fork"] for p in flow.hooks_named("post-launch", fork_id)] == [True]
