"""wtctl behaviors agent-spawn relies on that need a configured hook, a failing bootstrap step,
a human actor or a squash-merged branch (wave 2 of #100), against the real wtctl and real git.

Human-only flags and commands are driven with an explicit ``--actor human:tester`` (the children
have no TTY on stdin). Hook scripts and bootstrap steps are small shell scripts under the world's
private root.
"""

# ruff: noqa: E501  (assertion messages are prose: assumption, then the document line)

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from .billed import pid_gone
from .harness import World, check, wait_for
from .test_wtctl import AUTO, Wt

SC = pytest.mark.live_scenario
HUMAN = "human:tester"


@pytest.fixture
def wt(world: World) -> Wt:
    return Wt(world)


def write_script(world: World, name: str, body: str) -> Path:
    path = world.root / "hooks" / name
    path.parent.mkdir(exist_ok=True)
    path.write_text("#!/bin/sh\n" + body)
    path.chmod(0o700)
    return path


def wt_config(
    world: World,
    wt: Wt,
    *,
    hooks: dict[str, list[tuple[Path, int | None]]] | None = None,
    steps: list[Path] | None = None,
) -> None:
    """Rewrite wtctl's config: the harness repo, hooks (``name -> [(script, timeout)]``) and
    bootstrap steps (scripts run with ``argv: [script]``)."""
    repo = wt.repo
    lines = [
        f"root: {world.root / 'wtctl'}\n",
        "repos:\n",
        f"  {repo.repo_id}:\n",
        f"    clone: {repo.clone}\n",
        f"    clone_url: {repo.origin}\n",
    ]
    if steps:
        lines.append("    bootstrap:\n      steps:\n")
        for step in steps:
            lines.append(f'        - argv: ["{step}"]\n')
    if hooks:
        lines.append("hooks:\n")
        for name, entries in hooks.items():
            lines.append(f"  {name}:\n")
            for script, timeout in entries:
                lines.append(f'    - cmd: "{script}"\n')
                if timeout is not None:
                    lines.append(f"      timeout: {timeout}\n")
    world.write_config("wtctl", "".join(lines))


def err_of(done: Any) -> dict[str, Any]:
    return json.loads(done.stdout)["error"]


def create_args(wt: Wt, name: str, *extra: str) -> list[str]:
    return ["create", "--repo", wt.repo.repo_id, "--name", name, "--holder", "ag-hook", *extra]


@SC("wtctl-create-hooks")
def test_pre_create_hook_veto_timeout_escalation_and_human_override(wt: Wt, world: World) -> None:
    """wtctl interface.md Hooks (``pre-create``: any non-zero exit or a timeout vetoes with exit 11,
    exit 13 escalates and the command exits 13 having changed nothing, the remaining hooks still
    run, ``--override-escalation`` counts a 13 as allow and is human-only, a hook's descendants die
    with a timed-out hook) and interface.md Launch order step 4 (11 and 13 from ``wtctl create``).
    #36 #8 wtctl (not covered: a ``create`` hook veto (11) and escalation (13) with
    ``--override-escalation``, a ``create`` killed on timeout, a human override), #100."""
    ran = world.root / "ran.log"
    payload = world.root / "payload.json"
    veto = write_script(
        world,
        "veto.sh",
        f"cat > {payload}; echo veto >> {ran}; echo 'no creates today' >&2; exit 1\n",
    )
    escalate = write_script(
        world,
        "escalate.sh",
        f"cat > /dev/null; echo escalate >> {ran}; echo 'ask a human' >&2; exit 13\n",
    )
    slow = write_script(
        world, "slow.sh", f"cat > /dev/null; echo $$ > {world.root / 'slow.pid'}; sleep 120\n"
    )

    wt_config(world, wt, hooks={"pre-create": [(veto, None)]})
    vetoed = wt.run(*create_args(wt, "v1"), f"--actor={AUTO}")
    check(
        vetoed.returncode == 11 and err_of(vetoed)["code"] == "vetoed",
        "a pre-create veto exits 11 with error.code vetoed",
        "wtctl interface.md Hooks Outcomes / Exit codes (11)",
        (vetoed.returncode, vetoed.stdout[-300:]),
    )
    world.observe("wtctl.pre-create.veto", err_of(vetoed))
    body = json.loads(payload.read_text())
    check(
        body["hook"] == "pre-create"
        and body["object_type"] == "worktree"
        and body["repo"] == wt.repo.repo_id,
        "the pre-create payload names the hook, the object type and the repo",
        "wtctl interface.md Hooks (payload)",
        {k: body.get(k) for k in ("hook", "object_type", "repo")},
    )
    check(
        wt.ok("list")["records"] == []
        and (
            not (world.root / "wtctl" / "worktrees").exists()
            or not any((world.root / "wtctl" / "worktrees").rglob("v1"))
        ),
        "a vetoed create leaves no record and no worktree directory",
        "wtctl interface.md Create order (nothing is written to the clone before pre-create)",
        wt.ok("list"),
    )
    check(
        [
            b.lstrip("* ").strip()
            for b in world.git("branch", "--list", cwd=wt.repo.clone).splitlines()
        ]
        == ["main"],
        "a vetoed create leaves no branch in the clone (only main)",
        "wtctl interface.md Create order step 13 (the branch is created after pre-create)",
    )

    wt_config(world, wt, hooks={"pre-create": [(escalate, None)]})
    escalated = wt.run(*create_args(wt, "e1"), f"--actor={AUTO}")
    check(
        escalated.returncode == 13 and err_of(escalated)["code"] == "escalated",
        "a pre-create escalation exits 13 with error.code escalated",
        "wtctl interface.md Hooks Outcomes (exit 13 escalated)",
        (escalated.returncode, escalated.stdout[-300:]),
    )
    check(
        wt.ok("list")["records"] == [],
        "an escalated create changed nothing",
        "wtctl interface.md Hooks Outcomes (having changed nothing)",
    )
    non_human = wt.run(*create_args(wt, "e1"), "--override-escalation", f"--actor={AUTO}")
    check(
        non_human.returncode == 2,
        "--override-escalation with a non-human actor exits 2",
        "wtctl interface.md Hooks Outcomes (--override-escalation is a human-only flag)",
        (non_human.returncode, non_human.stdout[-300:]),
    )
    overridden = wt.run(*create_args(wt, "e1"), "--override-escalation", f"--actor={HUMAN}")
    check(
        overridden.returncode == 0 and json.loads(overridden.stdout)["record"]["name"] == "e1",
        "with a human actor and --override-escalation the exit 13 counts as allow and the worktree is created",
        "wtctl interface.md Hooks Outcomes",
        (overridden.returncode, overridden.stdout[-300:]),
    )

    # the remaining hooks still run after an escalation; a later veto wins and the override does not lift it
    ran.write_text("")
    wt_config(world, wt, hooks={"pre-create": [(escalate, None), (veto, None)]})
    both = wt.run(*create_args(wt, "b1"), "--override-escalation", f"--actor={HUMAN}")
    check(
        both.returncode == 11 and ran.read_text().split() == ["escalate", "veto"],
        "a veto after an escalation still vetoes under --override-escalation, and both hooks ran",
        "wtctl interface.md Hooks Outcomes (the remaining hooks still run; other outcomes are unchanged)",
        (both.returncode, ran.read_text()),
    )

    # a per-invocation hook runs after the configured ones
    wt_config(world, wt)
    ran.write_text("")
    inv = wt.run(
        *create_args(wt, "i1"),
        f"--hook=pre-create=cat > /dev/null; echo hook-cmd >> {ran}; exit 1",
        f"--actor={AUTO}",
    )
    check(
        inv.returncode == 11 and ran.read_text().split() == ["hook-cmd"],
        "a --hook pre-create=CMD of the invocation can veto",
        "wtctl interface.md Hooks (--hook NAME=CMD adds a hook for this invocation)",
        (inv.returncode, inv.stdout[-200:]),
    )

    # a timeout vetoes (11) and SIGKILLs the hook's process group
    wt_config(world, wt, hooks={"pre-create": [(slow, 1)]})
    timed = wt.run(*create_args(wt, "t1"), f"--actor={AUTO}", timeout=90)
    check(
        timed.returncode == 11,
        "a pre-create hook that times out vetoes (exit 11)",
        "wtctl interface.md Hooks Outcomes (a timeout vetoes)",
        (timed.returncode, timed.stdout[-300:]),
    )
    world.observe("wtctl.pre-create.timeout", err_of(timed))
    pid = int((world.root / "slow.pid").read_text())
    wait_for(lambda: pid_gone(pid), 20, "the timed-out hook to be killed")
    check(
        [r["name"] for r in wt.ok("list")["records"]] == ["e1"],
        "only the overridden create made a record",
        "wtctl interface.md Create order",
        wt.ok("list"),
    )


@SC("wtctl-bootstrap-failed")
def test_a_failing_bootstrap_step_keeps_the_record_and_a_human_rerun_succeeds(
    wt: Wt, world: World
) -> None:
    """wtctl interface.md Bootstrap and Exit codes (``bootstrap_failed``: exit 12, the record is
    kept and ``details.record`` carries it; the first failing step stops the rest; ``bootstrap``
    re-run is human-only (13 without a human actor); a re-run that succeeds records the baseline),
    interface.md Launch order step 4 (``bootstrap_failed`` -> ``details.record``). #36 #8 wtctl
    (not covered: ``bootstrap_failed`` with ``details.record``), #100."""
    marker = world.root / "bootstrap.marker"
    fail = write_script(world, "fail.sh", f"echo step-output; echo seen >> {marker}; exit 3\n")
    second = write_script(world, "second.sh", f"echo second >> {marker}\n")
    wt_config(world, wt, steps=[fail, second])
    done = wt.run(*create_args(wt, "bs1"), f"--actor={AUTO}")
    err = err_of(done)
    world.observe("wtctl.bootstrap_failed", err)
    check(
        done.returncode == 12 and err["code"] == "bootstrap_failed",
        "a failing bootstrap step exits 12 with error.code bootstrap_failed",
        "wtctl interface.md Exit codes (12 bootstrap_failed)",
        (done.returncode, done.stdout[-400:]),
    )
    record = (err.get("details") or {}).get("record")
    check(
        isinstance(record, dict)
        and record.get("name") == "bs1"
        and "ag-hook" in record.get("holders", []),
        "the error carries the whole record in details.record (name and holder)",
        "interface.md Launch order step 4 (bootstrap_failed: details.record, wtctl's whole record object)",
        err.get("details"),
    )
    assert isinstance(record, dict)  # the check above failed otherwise
    step = (err.get("details") or {}).get("step")
    check(
        isinstance(step, dict) and step.get("argv") == [str(fail)] and step.get("exit") == 3,
        "details.step names the failing step: its argv and exit code",
        "interface.md Launch order step 4 (bootstrap_failed: details.step); wtctl interface.md Bootstrap",
        err.get("details", {}).get("step"),
    )
    check(
        marker.read_text().split() == ["seen"],
        "the first failing step stopped the remaining steps",
        "wtctl interface.md Bootstrap step 3 (the first failing step stops the remaining steps)",
        marker.read_text(),
    )
    listed = wt.ok("list")["records"]
    check(
        len(listed) == 1
        and listed[0]["id"] == record["id"]
        and listed[0]["state"] == "active"
        and Path(listed[0]["path"]).is_dir(),
        "the record and worktree are kept (active) after a bootstrap failure",
        "wtctl interface.md Create order (bootstrap runs after the record exists; post-create fires after bootstrap_failed)",
        listed,
    )
    refused = wt.run("bootstrap", record["id"], f"--actor={AUTO}")
    check(
        refused.returncode == 13,
        "bootstrap re-run without a human actor exits 13 (human-only)",
        "wtctl interface.md Bootstrap (Re-run: exit 13 without a human actor)",
        (refused.returncode, refused.stdout[-300:]),
    )
    # the human fixes the step and re-runs
    wt_config(world, wt, steps=[second])
    marker.write_text("")
    rerun = wt.run("bootstrap", record["id"], f"--actor={HUMAN}")
    check(
        rerun.returncode == 0,
        "a human re-run of bootstrap succeeds once the step is fixed",
        "wtctl interface.md Bootstrap (Re-run)",
        (rerun.returncode, rerun.stdout[-300:], rerun.stderr[-300:]),
    )
    check(
        wt.run("baseline", "check", record["id"]).returncode == 0,
        "the successful re-run re-recorded the baseline",
        "wtctl interface.md Bootstrap (re-records the baseline after a successful run)",
    )


@SC("wtctl-remove-merged-head")
def test_remove_merged_head_lets_a_squash_merged_branch_go(wt: Wt, world: World) -> None:
    """wtctl interface.md Content checks (``unpushed_head``: HEAD not pushed and not equal to a
    verified ``--merged-head``) and Remove order step 10 (the branch is deleted when its tip is
    pushed or equals ``--merged-head``): a worktree commit pushed to origin and then deleted there
    (a squash-merge) is ``unpushed`` without ``--merged-head`` and removable with it; a sha that is
    not the HEAD (the pushed base commit) is refused; the branch is deleted. Depends on wtctl's
    mirror keeping the pushed commit after the remote branch is deleted (the wtctl version is
    observed). #36 #8 wtctl (not covered: ``remove --merged-head``), #100."""
    world.observe("wtctl.version", wt.run("--version").stdout.strip())
    rec = wt.create("mh1", "ag-mh")["record"]
    path = Path(rec["path"])
    (path / "feature.txt").write_text("feature\n")
    world.git("add", "feature.txt", cwd=path)
    world.git("commit", "-q", "-m", "feature", cwd=path)
    head = world.git("rev-parse", "HEAD", cwd=path)
    branch = rec["branch"]
    world.git("push", "-q", "origin", f"HEAD:refs/heads/{branch}", cwd=path)
    # let wtctl's mirror see the branch, then delete it on the remote (the squash merge)
    seen = wt.run("status", rec["id"], "--holder=ag-mh")
    check(
        seen.returncode == 0,
        "status refreshes the mirror",
        "wtctl interface.md status",
        seen.stdout[-300:],
    )
    world.git("--git-dir", str(wt.repo.origin), "update-ref", "-d", f"refs/heads/{branch}")
    status = wt.run("status", rec["id"], "--holder=ag-mh")
    st = json.loads(status.stdout)
    world.observe("wtctl.status.unmerged", {k: st.get(k) for k in ("would_remove", "reason")})
    refused = wt.run("remove", rec["id"], "--yes", "--holder=ag-mh", f"--actor={AUTO}")
    check(
        refused.returncode == 20 and err_of(refused)["reason"] == "unpushed",
        "HEAD that is on no remote branch any more is refused unpushed",
        "wtctl interface.md Content checks (unpushed_head)",
        (refused.returncode, refused.stdout[-300:]),
    )
    # a real commit the mirror has seen (the pushed base, on origin/main) that is not HEAD: only
    # the "equals HEAD" guard can refuse it, since the commit itself is pushed
    base = world.git("rev-parse", "HEAD~1", cwd=path)
    wrong = wt.run(
        "remove", rec["id"], "--yes", "--holder=ag-mh", f"--actor={AUTO}", "--merged-head", base
    )
    check(
        wrong.returncode == 20 and err_of(wrong)["reason"] == "unpushed",
        "a --merged-head that is a pushed commit but not HEAD does not lift the check",
        "wtctl interface.md Content checks (accepted only when it equals HEAD)",
        (wrong.returncode, wrong.stdout[-300:]),
    )
    dry = wt.run("remove", rec["id"], "--dry-run", "--holder=ag-mh", "--merged-head", head)
    world.observe("wtctl.remove.merged-head.dry-run", json.loads(dry.stdout).get("would_remove"))
    removed = wt.run(
        "remove", rec["id"], "--yes", "--holder=ag-mh", f"--actor={AUTO}", "--merged-head", head
    )
    out = json.loads(removed.stdout)
    world.observe(
        "wtctl.remove.merged-head",
        {k: out.get(k) for k in ("branch_deleted", "branch_kept_reason", "warnings")},
    )
    check(
        removed.returncode == 0 and not path.exists(),
        "with the verified --merged-head the worktree is removed",
        "wtctl interface.md Content checks / Remove order",
        (removed.returncode, removed.stdout[-400:]),
    )
    check(
        out["branch_deleted"] is True
        and branch not in world.git("branch", "--list", cwd=wt.repo.clone),
        "the branch whose tip equals --merged-head is deleted",
        "wtctl interface.md Remove order step 10",
        {k: out.get(k) for k in ("branch_deleted", "branch_kept_reason")},
    )


@SC("wtctl-baseline-check")
def test_baseline_check_when_only_the_claude_class_changed(wt: Wt, world: World) -> None:
    """#36 #25 (not covered before: a change of only the ``claude`` class): a new ``CLAUDE.md`` in
    the worktree makes ``baseline check`` refuse (exit 20, ``repo_config_changed``) with
    ``details.changed_classes == ["claude"]``, while ``wtctl status`` reports the baseline as a
    warning, not a failure (wtctl interface.md Baseline: ``remove`` refuses only a ``git``-class
    change; ``baseline check`` refuses either); a git-class change adds ``git``."""
    made = wt.create("cl1", "ag-cl")["record"]
    path = Path(made["path"])
    (path / "CLAUDE.md").write_text("changed instructions\n")
    done = wt.run("baseline", "check", made["id"])
    err = err_of(done)
    check(
        done.returncode == 20
        and err["reason"] == "repo_config_changed"
        and err["details"]["changed_classes"] == ["claude"],
        "a changed CLAUDE.md alone is a claude-class change: baseline check exits 20 with changed_classes [claude]",
        "wtctl interface.md Baseline (baseline check refuses on a change in either class); #36 #25",
        (done.returncode, err),
    )
    status = json.loads(wt.run("status", made["id"], "--holder=ag-cl").stdout)
    world.observe("wtctl.status.claude-class", status.get("checks", {}).get("baseline"))
    check(
        status["checks"]["baseline"]["status"] == "warn",
        "wtctl status reports a claude-class-only change as a baseline warning, not a failure",
        "wtctl interface.md Baseline (remove reports claude-class changes as a warning)",
        status["checks"]["baseline"],
    )
    config = wt.repo.clone / ".git" / "config"
    config.write_text(config.read_text() + "\n[alias]\n\tlive = status\n")
    both = err_of(wt.run("baseline", "check", made["id"]))
    check(
        sorted(both["details"]["changed_classes"]) == ["claude", "git"],
        "adding a git config change reports both classes",
        "wtctl interface.md Baseline; #36 #25",
        both["details"],
    )
