"""agent-spawn over the real wtctl when ``wtctl create`` is vetoed, escalates or fails its bootstrap,
and the ``repo_config_changed`` view of a changed worktree (wave 2 of #100). A stub ``claude``, no
Claude session."""

# ruff: noqa: E501  (assertion messages are prose: assumption, then the document line)

from __future__ import annotations

import json
from pathlib import Path

import pytest

from .harness import World, check
from .test_agent_spawn_hooks import HUMAN, write_script
from .test_agent_spawn_stub import show, stub_setup
from .test_wtctl import Wt
from .test_wtctl_hooks import wt_config

SC = pytest.mark.live_scenario


def new_worktree_args(wt: Wt, *extra: str) -> list[str]:
    return [
        "launch", "--mode", "headless", "--prompt", "reply ok", "--model", "haiku",
        "--repo", wt.repo.repo_id, "--worktree", "new", *extra,
    ]  # fmt: skip


def active_records(wt: Wt) -> list[dict]:
    return wt.ok("list")["records"]


@SC("as-wtctl-create-failures")
def test_launch_over_a_vetoing_escalating_and_bootstrap_failing_wtctl(world: World) -> None:
    """interface.md Launch order step 4 (``wtctl create`` exit 11 -> park ``vetoed`` without
    attention; 13 ``escalated`` -> ``escalated`` with ``escalated_by: wtctl``; 12
    ``bootstrap_failed`` -> ``launch_failed`` with the record stored), Resume order step 5 and
    Escalation overrides (``resume --override-escalation`` goes to ``wtctl create`` when
    ``escalated_by`` is ``wtctl``; the resume of a ``bootstrap_failed`` launch continues at step 5
    with the stored record), the exit codes 11, 13 and 12 of ``launch``. #100 (wtctl ``create``
    escalation (13) and hook veto (11), ``bootstrap_failed``), #36 #8 wtctl, #78."""
    wt = stub_setup(world, "exit 0")
    veto = write_script(world, "veto.sh", "cat > /dev/null; echo 'wt says no' >&2; exit 1\n")
    escalate = write_script(world, "escalate.sh", "cat > /dev/null; exit 13\n")

    # 11: parked vetoed, no attention, nothing started, no record
    wt_config(world, wt, hooks={"pre-create": [(veto, None)]})
    vetoed = world.spawn_json(*new_worktree_args(wt, "--wait"), ok=(11,))
    err = vetoed["error"]
    shown = show(world, err["agent_id"])["agent"]
    world.observe("agent-spawn.wtctl-veto", err)
    check(
        err["code"] == "vetoed" and err["tool"] == "wtctl",
        "a wtctl pre-create veto passes through: launch exits 11, code vetoed, tool wtctl",
        "interface.md Launch order step 4 (11 -> park vetoed); Error envelope",
        err,
    )
    check(
        shown["status"] == "parked"
        and shown["park_reason"] == "vetoed"
        and shown["needs_attention"] is None
        and shown["worktree"] is None,
        "the agent parks vetoed without attention and records no worktree",
        "interface.md Launch order step 4",
        {k: shown[k] for k in ("status", "park_reason", "needs_attention", "worktree")},
    )
    check(
        active_records(wt) == [] and world.json_tool("claude-session", "list")["sessions"] == [],
        "no wtctl record and no claude-session session exist",
        "interface.md Launch order step 4 (the session starts at step 6)",
    )

    # 13: parked escalated by wtctl; the human override goes to wtctl create
    wt_config(world, wt, hooks={"pre-create": [(escalate, None)]})
    escalated = world.spawn_json(*new_worktree_args(wt, "--wait"), ok=(13,))
    eid = escalated["error"]["agent_id"]
    shown = show(world, eid)["agent"]
    check(
        escalated["error"]["code"] == "escalated"
        and shown["park_reason"] == "escalated"
        and shown["escalated_by"] == "wtctl"
        and shown["needs_attention"] == "escalated",
        "a wtctl pre-create escalation parks the agent escalated with escalated_by wtctl (exit 13)",
        "interface.md Launch order step 4 (13 escalated -> escalated, escalated_by wtctl)",
        {k: shown[k] for k in ("park_reason", "escalated_by", "needs_attention")},
    )
    refused = world.spawn_json("resume", eid, "--override-escalation", "--wait", ok=(2,))
    check(
        refused["error"]["code"] == "usage" and active_records(wt) == [],
        "resume --override-escalation without a human actor exits 2 and creates nothing",
        "interface.md Global flags (Human-only flags)",
        refused["error"],
    )
    resumed = world.spawn_json("resume", eid, "--override-escalation", *HUMAN, "--wait", ok=(14,))
    records = active_records(wt)
    after = show(world, eid)
    world.observe(
        "agent-spawn.wtctl-escalation.resumed",
        {k: resumed.get(k) for k in ("outcome", "exit_code")},
    )
    check(
        len(records) == 1
        and eid in records[0]["holders"]
        and after["agent"]["worktree"] is not None
        and after["agent"]["worktree"]["record_id"] == records[0]["id"]
        and resumed["outcome"] == "exited-unreported",
        "the override reached wtctl create: the worktree exists, held by the agent, and the launch ran on (the stub exited 0: 14)",
        "interface.md Escalation overrides (resume --override-escalation: wtctl -> wtctl create --override-escalation); Resume order step 5",
        (records, after["agent"]["worktree"]),
    )
    check(
        all(r["adopted_at"] is not None for r in after["runs"]) and len(after["runs"]) == 1,
        "the run that started after the override has adopted_at (the earlier escalation was before any run)",
        "interface.md show (adopted_at); #91",
        [r["adopted_at"] for r in after["runs"]],
    )

    # the claude class of the baseline: only CLAUDE.md changed in the agent's worktree
    path = Path(records[0]["path"])
    check(
        after["repo_config_changed"] == {"changed": False, "changed_classes": [], "items": []}
        or after["repo_config_changed"]["changed"] is False,
        "an untouched worktree shows repo_config_changed.changed false",
        "interface.md show (repo_config_changed from wtctl baseline check)",
        after["repo_config_changed"],
    )
    (path / "CLAUDE.md").write_text("project instructions the agent changed\n")
    changed = show(world, eid)
    world.observe("agent-spawn.show.repo_config_changed.claude", changed["repo_config_changed"])
    check(
        changed["repo_config_changed"]["changed"] is True
        and changed["repo_config_changed"]["changed_classes"] == ["claude"],
        "adding CLAUDE.md changes only the claude class: changed true, changed_classes [claude]",
        "interface.md show (repo_config_changed: {changed, changed_classes, items}); #36 #25",
        changed["repo_config_changed"],
    )
    # teardown: the claude class alone is a warning for wtctl remove, not a refusal
    torn = world.agent_spawn("teardown", eid, "--json")
    world.observe(
        "agent-spawn.teardown.claude-class",
        (torn.returncode, json.loads(torn.stdout).get("error", {}).get("reason")),
    )
    check(
        torn.returncode in (0, 20),
        "teardown with only the claude class changed is decided by wtctl remove (warning, or the dirty check for the untracked file)",
        "wtctl interface.md Baseline (remove refuses only on a git-class change)",
        (torn.returncode, torn.stdout[-400:]),
    )
    if torn.returncode == 20:
        reason = json.loads(torn.stdout)["error"]["reason"]
        check(
            reason != "repo_config_changed",
            "a refused teardown names the untracked file (dirty), never repo_config_changed, for a claude-only change",
            "wtctl interface.md Baseline (remove refuses repo_config_changed only for the git class)",
            reason,
        )

    # 12 bootstrap_failed: the record is stored; a human fixes the step and the resume continues
    marker = world.root / "bs.log"
    failing = write_script(world, "bs-fail.sh", f"echo ran >> {marker}; exit 3\n")
    wt_config(world, wt, steps=[failing])
    failed = world.spawn_json(*new_worktree_args(wt, "--wait"), ok=(12,))
    fid = failed["error"]["agent_id"]
    fshown = show(world, fid)["agent"]
    world.observe("agent-spawn.bootstrap_failed", failed["error"])
    check(
        failed["error"]["code"] in ("launch_failed", "failed")
        and failed["error"]["tool"] == "wtctl"
        and failed["error"]["lower_code"] == "bootstrap_failed"
        and fshown["park_reason"] == "launch_failed"
        and fshown["worktree"] is not None,
        "bootstrap_failed passes wtctl's code through (exit 12), parks launch_failed and stores the worktree record",
        "interface.md Launch order step 4 (12 bootstrap_failed -> launch_failed, the record is stored)",
        (failed["error"], fshown["worktree"]),
    )
    check(
        any(r["id"] == fshown["worktree"]["record_id"] for r in active_records(wt)),
        "the stored record is a live wtctl record",
        "interface.md Launch order step 4 (wt_record_id stored from bootstrap_failed details.record)",
    )
    wt_config(world, wt)
    marker.write_text("")
    continued = world.spawn_json("resume", fid, "--wait", ok=(14,))
    check(
        continued["agent"]["worktree"]["record_id"] == fshown["worktree"]["record_id"]
        and marker.read_text() == ""
        and len([r for r in active_records(wt) if fid in r["holders"]]) == 1,
        "the resume continues at step 5 with the stored record: no second worktree, bootstrap not run again, the run started",
        "interface.md Resume order step 5 (wt_record_id set, active: skip create)",
        (continued["agent"]["worktree"], marker.read_text()),
    )
