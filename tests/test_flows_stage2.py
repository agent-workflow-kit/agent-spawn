"""Stage 2 end-to-end flows (issue #25): the ticket-work and review-respond flows with worktrees,
throwaways, teardown and prune, composed the way callers use them, against the fake lower tools
(claude-session, wtctl, termctl, tmux; no real process of any of them).

Every ``--json`` output goes through :meth:`Flow.cli` (validated against its schema), and each
scenario ends with :meth:`Flow.validate_everything` (hook payloads, log, audit). Intermediate
states are scripted; nothing depends on timing. Every run here is headless with ``--wait`` or
parks before the command returns, so no scenario leaves a process behind.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Any

import pytest

from agent_spawn.lifecycle import LEASE_WAIT, lease_key
from tests.conftest import Isolation
from tests.fakebin.harness import Fakebin
from tests.flow_support import Flow, schema
from tests.test_worktrees_launch import REPO, fake_throwaway, records, wtctl_calls

LAUNCH = ("launch", "--mode", "headless", "--skill", "do-it", "--json", "--wait")
REVIEW = ("launch", "--mode", "headless", "--kind", "review", "--skill", "review-it")
SHA = "b" * 40
OLD = time.time() - 40 * 86400


@pytest.fixture
def flow(fakebin: Fakebin, isolation_env: Isolation, tmp_path: Path) -> Flow:
    return Flow(fakebin, isolation_env, tmp_path)


def backdate(flow: Flow, agent_id: str, created: float = OLD) -> None:
    """``created`` is immutable through the store: age an agent in the database directly."""
    db = sqlite3.connect(flow.state_dir / "state.db")
    try:
        with db:
            db.execute(
                "UPDATE agents SET created = ?, last_live_at = NULL WHERE agent_id = ?",
                (created, agent_id),
            )
    finally:
        db.close()


def worktree_launch(flow: Flow, *extra: str) -> dict[str, Any]:
    done = flow.cli(
        schema("launch"),
        *LAUNCH,
        "--repo",
        REPO,
        "--worktree",
        "new",
        "--branch",
        "feat/ticket-1",
        "--base",
        "main",
        *extra,
    )
    assert done.code == 0, (done.stdout, done.stderr)
    return done.body  # type: ignore[no-any-return]


# --- ticket work: worktree launch, report, wait, resume, fork in place, teardown ---------------


def test_ticket_work_in_a_new_worktree_to_teardown_with_merged_head(
    flow: Flow, tmp_path: Path
) -> None:
    result_file = tmp_path / "result.json"
    body = worktree_launch(flow, "--label", "ticket=T-1", "--result-file", str(result_file))
    agent_id = body["agent"]["agent_id"]
    assert body["outcome"] == "done" and body["run"]["report"]["status"] == "done"
    assert body["agent"]["status"] == "parked" and body["agent"]["park_reason"] == "done"
    assert flow.settled(agent_id)
    (record,) = records(flow.fakebin).values()
    worktree = body["agent"]["worktree"]
    assert worktree == {"record_id": record["id"], "path": record["path"], "owned": True}
    assert record["holders"] == [agent_id] and os.path.isdir(record["path"])
    assert json.loads(result_file.read_text())["agent"]["agent_id"] == agent_id
    assert result_file.stat().st_mode & 0o777 == 0o600

    # the agent's run started in the worktree
    starts = [a for a in flow.fakebin.argvs("claude-session") if a[0] == "start"]
    assert f"--cwd={os.path.realpath(record['path'])}" in starts[-1]

    shown = flow.cli(schema("show"), "show", agent_id, "--json")
    assert shown.code == 0 and shown.body["agent"]["worktree"] == worktree
    assert shown.body["third_party_run"] is False
    assert shown.body["repo_config_changed"]["changed"] is False
    waited = flow.cli(schema("wait"), "wait", agent_id, "--json")
    assert waited.code == 0 and waited.body["outcome"] == "done"

    # a second run (the ticket is reworked) takes no new hold
    resumed = flow.cli(schema("resume"), "resume", agent_id, "--wait", "--json")
    assert resumed.code == 0 and flow.agent(agent_id)["run"] == 2
    assert flow.settled(agent_id) and records(flow.fakebin)[record["id"]]["holders"] == [agent_id]

    # a fork in place holds the source's record under its own id
    forked = flow.cli(schema("resume"), "resume", agent_id, "--fork", "--wait", "--json")
    assert forked.code == 0, (forked.stdout, forked.stderr)
    fork_id = forked.body["agent"]["agent_id"]
    assert sorted(records(flow.fakebin)[record["id"]]["holders"]) == sorted([agent_id, fork_id])
    assert flow.settled(fork_id)

    # the source cannot remove the worktree while the fork holds it
    held = flow.cli(schema("teardown"), "teardown", agent_id, "--json")
    assert held.code == 20 and held.body["error"]["reason"] == "held"
    assert flow.agent(agent_id)["status"] == "parked"
    assert os.path.isdir(record["path"])

    # the fork releases its hold (forget: wtctl removes no files) ...
    gone_fork = flow.cli(schema("teardown"), "teardown", fork_id, "--json")
    assert gone_fork.code == 0, (gone_fork.stdout, gone_fork.stderr)
    assert [w["action"] for w in gone_fork.body["worktrees"]] == ["released"]
    assert os.path.isdir(record["path"])
    assert records(flow.fakebin)[record["id"]]["holders"] == [agent_id]

    # ... and then the source removes it, with the merged head
    dry = flow.cli(
        schema("teardown-dry-run"), "teardown", agent_id, "--merged-head", SHA, "--dry-run",
        "--json",
    )  # fmt: skip
    assert dry.code == 0 and dry.body["would_refuse"] is None
    assert flow.agent(agent_id)["status"] == "parked"
    gone = flow.cli(
        schema("teardown"), "teardown", agent_id, "--merged-head", SHA, "--decision-id", "d-1",
        "--actor", "auto:flow", "--json",
    )  # fmt: skip
    assert gone.code == 0, (gone.stdout, gone.stderr)
    (entry,) = gone.body["worktrees"]
    assert entry["action"] == "removed" and entry["owned"] is True
    assert gone.body["session_forgotten"] is True
    assert not os.path.exists(record["path"])
    assert records(flow.fakebin)[record["id"]]["state"] == "removed"
    remove = wtctl_calls(flow.fakebin, "remove")[-1]
    assert f"--merged-head={SHA}" in remove and f"--holder={agent_id}" in remove

    assert flow.cli(schema("list"), "list", "--json").body["agents"] == []
    audit = flow.cli(schema("export"), "export", "--action", "teardown", "--json")
    assert [r["target"] for r in audit.body] == [agent_id]
    assert audit.body[0]["decision_id"] == ["d-1"]
    flow.validate_everything(
        expect_hooks=["post-launch", "post-park", "pre-teardown", "post-teardown"],
        expect_audit=1,
    )
    (post,) = flow.hooks_named("post-teardown", agent_id)
    assert [(w["record_id"], w["action"]) for w in post["worktrees"]] == [(record["id"], "removed")]


def test_ephemeral_ticket_agent_with_a_worktree_is_torn_down_by_its_supervisor(
    flow: Flow,
) -> None:
    # #23 made this work: before it the worktree step refused and the agent parked done with
    # teardown_refused
    body = worktree_launch(flow, "--ephemeral")
    agent_id = body["agent"]["agent_id"]
    assert flow.settled(agent_id)
    assert flow.agent(agent_id)["status"] == "torn_down"
    (record,) = records(flow.fakebin).values()
    assert record["state"] == "removed" and not os.path.exists(record["path"])
    assert flow.hooks_named("post-park", agent_id) == []
    gone = flow.cli(schema("wait"), "wait", agent_id, "--json")
    assert gone.code == 0 and gone.body["agent"]["status"] == "torn_down"
    flow.validate_everything(expect_hooks=["post-launch", "post-teardown"], expect_audit=1)


# --- crash and retry of a worktree launch ---------------------------------------------------


def test_crashed_worktree_launch_is_retried_by_resume_without_a_second_worktree(
    flow: Flow,
) -> None:
    busy = {"error": {"code": "busy", "message": "locked"}, "exit": 12}
    flow.fakebin.script("wtctl", "create", times=1, **busy)
    failed = flow.cli(
        schema("launch"), *LAUNCH, "--repo", REPO, "--worktree", "new", "--prompt", "P"
    )
    assert failed.code == 12, (failed.stdout, failed.stderr)
    (row,) = flow.env.agents()
    agent_id = row["agent_id"]
    assert row["status"] == "parked" and row["park_reason"] == "launch_failed"
    assert row["csession_id"] is None

    # the launcher also died after wtctl created the record, before it stored it
    done = flow.fakebin.run(
        "wtctl", "create", "--repo", REPO, "--holder", agent_id, "--name", agent_id, "--json"
    )
    assert done.returncode == 0, done.stderr
    left = json.loads(done.stdout)["record"]
    creates = len(wtctl_calls(flow.fakebin, "create"))

    resumed = flow.cli(schema("resume"), "resume", agent_id, "--wait", "--json")
    assert resumed.code == 0, (resumed.stdout, resumed.stderr)
    assert len(wtctl_calls(flow.fakebin, "create")) == creates  # found by name, not recreated
    assert len(records(flow.fakebin)) == 1
    assert resumed.body["agent"]["worktree"]["record_id"] == left["id"]
    assert flow.settled(agent_id)
    assert flow.agent(agent_id)["status"] == "parked"
    assert flow.agent(agent_id)["park_reason"] == "done"

    final = flow.cli(schema("teardown"), "teardown", agent_id, "--json")
    assert final.code == 0, (final.stdout, final.stderr)
    assert final.body["agent"]["status"] == "torn_down"
    assert records(flow.fakebin)[left["id"]]["state"] == "removed"
    flow.validate_everything(expect_hooks=["post-launch", "post-teardown"], expect_audit=1)


# --- prune of old agents ---------------------------------------------------------------------


def test_prune_tears_down_an_old_parked_worktree_agent_and_leaves_the_rest(flow: Flow) -> None:
    old = worktree_launch(flow)["agent"]["agent_id"]
    recent = flow.cli(schema("launch"), *LAUNCH).body["agent"]["agent_id"]
    owned = flow.cli(schema("launch"), *LAUNCH, "--label", "owner=me").body["agent"]["agent_id"]
    escalated = flow.cli(schema("launch"), *LAUNCH).body["agent"]["agent_id"]
    with flow.env.store() as store:  # escalated: needs_attention set, as the supervisor would
        with store.lease(lease_key(escalated), wait=LEASE_WAIT) as lease:
            store.update_agent(lease, escalated, {"needs_attention": "failed"})
    for agent_id in (old, owned, escalated):
        backdate(flow, agent_id)
    (record,) = records(flow.fakebin).values()

    # a second pre-teardown policy (after the recording one) vetoes the first attempt only
    once = flow.state_dir.parent / "vetoed-once"
    veto = f"[ -e {once} ] || {{ touch {once}; exit 1; }}"
    text = flow.env.config.read_text()
    head, _, tail = text.partition("  pre-teardown:\n")
    recording, _, rest = tail.partition("\n")
    flow.env.config.write_text(f'{head}  pre-teardown:\n{recording}\n    - cmd: "{veto}"\n{rest}')

    dry = flow.cli(schema("prune"), "prune", "--older-than", "7d", "--dry-run", "--json")
    assert dry.code == 0 and dry.body["dry_run"] is True
    assert os.path.isdir(record["path"])
    vetoed = flow.cli(schema("prune"), "prune", "--older-than", "7d", "--json")
    assert vetoed.code == 0, (vetoed.stdout, vetoed.stderr)
    refused = {r["agent_id"]: r for r in vetoed.body["results"]}[old]
    assert (refused["result"], refused["reason"], refused["tool"]) == (
        "refused",
        "vetoed",
        "agent-spawn",
    )
    assert refused["details"] == {"hook": "pre-teardown"}
    assert flow.agent(old)["status"] == "parked" and os.path.isdir(record["path"])
    pruned = flow.cli(schema("prune"), "prune", "--older-than", "7d", "--json")
    assert pruned.code == 0, (pruned.stdout, pruned.stderr)
    results = {r["agent_id"]: r for r in pruned.body["results"]}
    assert results[old]["result"] == "torn_down"
    assert (results[owned]["result"], results[owned]["reason"]) == ("skipped", "protected")
    assert recent not in results  # too young: not a candidate
    assert (results[escalated]["result"], results[escalated]["reason"]) == (
        "skipped",
        "needs_attention",
    )
    assert flow.agent(old)["status"] == "torn_down"
    for kept in (owned, recent, escalated):
        assert flow.agent(kept)["status"] == "parked"
    assert not os.path.exists(record["path"])
    assert records(flow.fakebin)[record["id"]]["state"] == "removed"
    # prune acts as the automation actor and its removal is audited
    audit = flow.cli(schema("export"), "export", "--action", "teardown", "--json").body
    assert [(r["target"], r["actor"]) for r in audit] == [(old, "auto:agent-spawn")]
    flow.validate_everything(expect_hooks=["pre-teardown", "post-teardown"], expect_audit=1)


def test_prune_refuses_the_source_of_a_young_fork_until_the_fork_is_old_too(flow: Flow) -> None:
    source = worktree_launch(flow)["agent"]["agent_id"]
    fork = flow.cli(schema("resume"), "resume", source, "--fork", "--wait", "--json")
    assert fork.code == 0
    fork_id = fork.body["agent"]["agent_id"]
    (record,) = records(flow.fakebin).values()
    backdate(flow, source)  # only the source is old: the fork still holds the worktree
    first = flow.cli(schema("prune"), "prune", "--older-than", "7d", "--json")
    assert first.code == 0, (first.stdout, first.stderr)
    (refused,) = first.body["results"]
    assert (refused["agent_id"], refused["result"]) == (source, "refused")
    assert (refused["reason"], refused["tool"]) == ("held", "wtctl")
    assert flow.agent(source)["status"] == "parked" and os.path.isdir(record["path"])

    # prune goes oldest first: the source is tried while the fork still holds, so it is refused
    # again in the pass that tears the fork down; the next pass finishes it
    backdate(flow, fork_id, OLD + 60)
    second = flow.cli(schema("prune"), "prune", "--older-than", "7d", "--json")
    assert {r["agent_id"]: r["result"] for r in second.body["results"]} == {
        source: "refused",
        fork_id: "torn_down",
    }
    third = flow.cli(schema("prune"), "prune", "--older-than", "7d", "--json")
    assert {r["agent_id"]: r["result"] for r in third.body["results"]} == {source: "torn_down"}
    assert not os.path.exists(record["path"])
    assert records(flow.fakebin)[record["id"]]["state"] == "removed"
    audit = flow.cli(schema("export"), "export", "--action", "teardown", "--json").body
    # only a teardown that removed a worktree is audited: the fork only released its hold
    assert flow.agent(fork_id)["status"] == "torn_down"
    assert flow.agent(source)["status"] == "torn_down"
    assert [(r["target"], r["actor"]) for r in audit] == [(source, "auto:agent-spawn")]
    flow.validate_everything(expect_hooks=["post-teardown"], expect_audit=1)


# --- review-respond: third-party reading in a throwaway, trusted in the worktree ---------------


def test_review_respond_reads_third_party_content_in_a_throwaway_and_acts_in_the_worktree(
    flow: Flow, tmp_path: Path
) -> None:
    # the trusted work: a ticket agent in its worktree
    ticket = worktree_launch(flow)["agent"]["agent_id"]
    (wt,) = records(flow.fakebin).values()
    throwaway = fake_throwaway(flow.fakebin)
    outside = tmp_path / "out"
    outside.mkdir()

    # the cwd rule: a third-party-reading run never runs in a worktree ...
    refused = flow.cli(
        schema("launch"), *LAUNCH, "--repo", REPO, "--worktree", "new", "--reads-third-party"
    )
    assert refused.code == 20 and refused.body["error"]["reason"] == "third_party_cwd"
    assert refused.body["error"]["tool"] == "agent-spawn"
    in_worktree = flow.cli(
        schema("resume"), "resume", ticket, "--fork", "--reads-third-party", "--wait", "--json"
    )
    assert in_worktree.code == 20 and in_worktree.body["error"]["reason"] == "third_party_cwd"
    # ... nor in a subdirectory of a throwaway ...
    sub = flow.cli(
        schema("launch"), *LAUNCH, "--cwd", throwaway["path"] + "/pkg", "--reads-third-party"
    )
    assert sub.code == 20 and sub.body["error"]["reason"] == "third_party_cwd"
    # ... and its result file may not be inside the throwaway
    inside = Path(throwaway["path"]) / "r.json"
    bad = flow.cli(
        schema("launch"), *LAUNCH, "--cwd", throwaway["path"], "--reads-third-party",
        "--result-file", str(inside),
    )  # fmt: skip
    assert bad.code == 2 and not inside.exists()
    # and a trusted run never runs in a throwaway
    trusted = flow.cli(schema("launch"), *LAUNCH, "--cwd", throwaway["path"])
    assert trusted.code == 20 and trusted.body["error"]["reason"] == "third_party_run"
    assert [a["agent_id"] for a in flow.env.agents()] == [ticket]  # refusals left no trace

    # the review itself: at the throwaway's root, result file outside it
    result_file = outside / "review.json"
    review = flow.cli(
        schema("launch"), "launch", "--mode", "headless", "--kind", "review", "--skill",
        "review-it", "--json", "--wait", "--cwd", throwaway["path"], "--reads-third-party",
        "--label", "ticket=T-1", "--result-file", str(result_file),
    )  # fmt: skip
    assert review.code == 0, (review.stdout, review.stderr)
    review_id = review.body["agent"]["agent_id"]
    assert review.body["outcome"] == "done" and review.body["agent"]["holds"] == []
    assert json.loads(result_file.read_text())["agent"]["agent_id"] == review_id
    assert flow.settled(review_id)
    shown = flow.cli(schema("show"), "show", review_id, "--json")
    assert shown.body["third_party_run"] is True

    # the response: the ticket agent resumes trusted in its worktree
    again = flow.cli(schema("resume"), "resume", ticket, "--wait", "--json")
    assert again.code == 0 and flow.agent(ticket)["run"] == 2
    assert flow.settled(ticket)
    # a trusted resume of the review agent is refused: its cwd is the throwaway
    no_resume = flow.cli(schema("resume"), "resume", review_id, "--json")
    assert no_resume.code == 20 and no_resume.body["error"]["reason"] == "third_party_run"

    # a third-party-reading fork of the review run into the same throwaway root, held-free
    fork = flow.cli(
        schema("resume"), "resume", review_id, "--fork", "--cwd", throwaway["path"],
        "--reads-third-party", "--wait", "--json",
    )  # fmt: skip
    assert fork.code == 0, (fork.stdout, fork.stderr)
    assert fork.body["agent"]["holds"] == [] and flow.settled(fork.body["agent"]["agent_id"])

    # clean-up: the review agents hold nothing (the throwaway is wtctl's), the ticket's
    # worktree is removed with its owner
    for agent_id in (fork.body["agent"]["agent_id"], review_id):
        gone = flow.cli(schema("teardown"), "teardown", agent_id, "--json")
        assert gone.code == 0 and gone.body["worktrees"] == []
    assert records(flow.fakebin)[throwaway["id"]]["state"] != "removed"
    removed = flow.cli(schema("teardown"), "teardown", ticket, "--json")
    assert removed.code == 0 and removed.body["worktrees"][0]["action"] == "removed"
    assert not os.path.exists(wt["path"])
    flow.validate_everything(
        expect_hooks=["post-launch", "post-park", "post-teardown"], expect_audit=1
    )


# --- ticket work in a window: the window step comes after the worktree, before the session ------


def test_ticket_work_in_a_window_closes_its_dead_window_in_the_order_of_the_teardown(
    flow: Flow, fakebin: Fakebin, isolation_env: Isolation, private_tmux: Any
) -> None:
    from tests.launch_support import script_done_run
    from tests.window_support import WindowEnv

    script_done_run(fakebin)  # the run ends `done`, scripted rather than the fake's default
    private_tmux.start()
    flow.env = WindowEnv(fakebin, isolation_env, private_tmux)  # same state dir and hook config
    done = flow.cli(
        schema("launch"), "launch", "--skill", "do-it", "--json", "--wait",
        "--repo", REPO, "--worktree", "new", "--branch", "feat/ticket-1", "--base", "main",
    )  # fmt: skip
    assert done.code == 0, (done.stdout, done.stderr)
    agent_id = done.body["agent"]["agent_id"]
    assert done.body["agent"]["mode"] == "interactive" and done.body["agent"]["window"]
    (record,) = records(flow.fakebin).values()
    cwd = [a for a in fakebin.argvs("termctl") if "open" in a and "--dry-run" not in a][-1]
    assert f"--cwd={os.path.realpath(record['path'])}" in cwd  # the window opens in the worktree
    flow.wait_settled(agent_id)
    # a window process of an earlier crash: its dead pane is still there, labeled with the agent
    leftover = fakebin.run(
        "termctl", "open", f"--label=agent={agent_id}", "--keep-on-exit=failed", "--json",
        "--", "sh", "-c", "exit 3", env=flow.env.env(),
    )  # fmt: skip
    assert leftover.returncode == 0, leftover.stderr
    flow.env.wait_for(
        lambda: any(p["state"] == "dead" for p in flow.env.panes()),  # type: ignore[attr-defined]
        "the leftover pane to die",
    )

    gone = flow.cli(schema("teardown"), "teardown", agent_id, "--merged-head", SHA, "--json")
    assert gone.code == 0, (gone.stdout, gone.stderr)
    (closed,) = gone.body["windows_closed"]
    assert closed.startswith("tc-")
    steps = [
        (i.tool, i.argv[0])
        for i in fakebin.invocations()
        if (i.tool, i.argv[0])
        in {("wtctl", "remove"), ("termctl", "close"), ("claude-session", "forget")}
    ]
    assert steps == [("wtctl", "remove"), ("termctl", "close"), ("claude-session", "forget")]
    assert flow.env.panes() == []  # type: ignore[attr-defined]
    flow.validate_everything(
        expect_hooks=["post-launch", "post-park", "post-teardown"], expect_audit=1
    )
