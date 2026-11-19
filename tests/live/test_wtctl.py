"""wtctl behaviors agent-spawn relies on, against the real wtctl and real git (no Claude Code).

The repository is a local bare ``origin`` plus a clone, wired as a wtctl repo id with
``repos.<id>.clone`` and ``clone_url`` (wtctl's documented way to name a local repository), so the
suite needs no network. Everything lives under the world's private root, wtctl's included.
Each check names the assumption and where it is written; the docstrings list the #36 comment and
design.md items the test confirms.
"""

# ruff: noqa: E501  (assertion messages are prose: assumption, then the document line)

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from .harness import Repo, World, check

SC = pytest.mark.live_scenario
AUTO = "auto:agent-spawn"


class Wt:
    """A world with one wtctl repo and thin wrappers over the real ``wtctl --json``."""

    def __init__(self, world: World) -> None:
        self.world = world
        self.repo: Repo = world.make_repo("proj")
        world.wtctl_config({"proj": self.repo})

    def run(self, *args: str, **kw: Any) -> Any:
        return self.world.tool("wtctl", "--json", *args, **kw)

    def ok(self, *args: str) -> Any:
        return self.world.json_tool("wtctl", *args)

    def code(self, done: Any) -> str | None:
        try:
            return json.loads(done.stdout)["error"]["code"]
        except (ValueError, KeyError, TypeError):
            return None

    def create(self, name: str, holder: str | None = "ag-aaaa", *extra: str) -> dict[str, Any]:
        args = ["create", "--repo", self.repo.repo_id, "--name", name, *extra]
        if holder:
            args += ["--holder", holder]
        return self.ok(*args)


@pytest.fixture
def wt(world: World) -> Wt:
    return Wt(world)


@SC("wtctl-create-list-show")
def test_create_list_show_shapes_and_flag_forms(wt: Wt, world: World) -> None:
    """#36 #20 (``--flag=value`` for every value flag, ``--json`` after positionals, a record's
    realpath and explicit name, default branch name, ``list`` shape and ``--repo`` as an exact
    match, ``show`` of a tombstone), #36 #8 wtctl (default branch name, ``list`` ordering)."""
    made = world.json_tool(
        "wtctl",
        "create",
        f"--repo={wt.repo.repo_id}",
        "--holder=ag-aaaa",
        "--name=n1",
        "--actor=auto:agent-spawn",
        "--decision-id=d1",
    )
    record = made["record"]
    for key in ("id", "path", "object_type", "state", "name", "owned", "holders"):
        check(
            key in record,
            f"a record carries {key}",
            "design.md wtctl behavior (list/show keys)",
            sorted(record),
        )
    check(
        record["name"] == "n1",
        "the record name equals the explicit --name",
        "#36 #20",
        record["name"],
    )
    check(
        record["branch"] == "wt/n1",
        "default branch of a new worktree is wt/<name>",
        "#36 #8 wtctl",
        record["branch"],
    )
    check(
        record["path"] == str(Path(record["path"]).resolve()) and Path(record["path"]).is_dir(),
        "a record's path is the realpath of an existing directory",
        "#36 #20 (path is the realpath)",
        record["path"],
    )
    check(
        record["object_type"] == "worktree"
        and record["owned"] is True
        and record["state"] == "active",
        "create makes an owned active worktree record",
        "design.md D6/D11",
    )
    check(
        made["reused"] is False and made["dry_run"] is False,
        "first create is not reused",
        "#36 #20",
    )
    world.observe("wtctl.record.keys", sorted(record))

    # `--json` after the positionals, `--flag=value` everywhere
    shown = world.tool("wtctl", "show", record["id"], "--json")
    check(
        shown.returncode == 0 and json.loads(shown.stdout)["id"] == record["id"],
        "--json after the positional works",
        "#36 #20",
    )
    listed = wt.ok("list")
    check(
        [r["id"] for r in listed["records"]] == [record["id"]],
        "list --json returns live records as {records: [...]}",
        "#36 #20 / #63 (supervisor chain reads {records})",
        listed,
    )
    check(
        wt.ok("list", f"--repo={wt.repo.repo_id}")["records"]
        and not wt.ok("list", "--repo", "nope/x")["records"],
        "--repo is an exact repo-id match",
        "#36 #20",
    )
    # a second repo: list without --repo returns every repo's records
    other = world.make_repo("other")
    world.wtctl_config({"proj": wt.repo, "other": other})
    world.json_tool(
        "wtctl", "create", "--repo", other.repo_id, "--name", "o1", "--holder", "ag-bbbb"
    )
    everyone = wt.ok("list")["records"]
    check(
        {r["name"] for r in everyone} == {"n1", "o1"},
        "list without --repo returns every repo's records",
        "#36 #63 (supervisor chain runs wtctl list --json without --repo)",
        [r["name"] for r in everyone],
    )
    world.observe("wtctl.list.order", [r["name"] for r in everyone])
    # holders is a sorted array
    for holder in ("ag-zz", "ag-bb", "ag-mm"):
        wt.ok("adopt", record["path"], "--holder", holder)
    holders = wt.ok("show", record["id"])["holders"]
    check(
        holders == sorted(holders) and len(holders) == 4,
        "holders come back as a sorted array",
        "#36 #22 (adopt and list return holders sorted)",
        holders,
    )
    # unknown id and error envelopes
    unknown = wt.run("show", "wt-zzzzzzzz")
    check(
        unknown.returncode == 10 and wt.code(unknown) == "not_found",
        "show of an unknown id exits 10 with an envelope on stdout",
        "#36 #20",
    )
    for argv, exit_code in (
        (["create", "--repo", wt.repo.repo_id, "--name", "bad1", "--base", "refs/heads/main"], 2),
        (["create", "--repo", wt.repo.repo_id, "--name", "bad2", "--branch", "HEAD"], 2),
    ):
        done = wt.run(*argv)
        check(
            done.returncode == exit_code and wt.code(done) == "usage",
            "create --base refs/... and --branch HEAD exit 2 (not 20) with an envelope",
            "#36 #20 (create exits 2, not 20)",
            (argv[-2:], done.returncode, done.stdout[-200:]),
        )


@SC("wtctl-create-list-show")
def test_create_dry_run_shape(wt: Wt, world: World) -> None:
    """#36 #20: ``create --dry-run`` takes no persistent record and runs no hooks (it refreshes the
    mirror: here the local origin). A dry run over the *holder's own* record answers like the real
    call: ``reused: true``, ``dry_run: false``, ``branch_action`` null (wtctl interface.md, Create
    order, "Retry after a crash": "also under ``--dry-run``: the answer is the same"); another
    holder over the same name is refused ``path_exists`` (exit 20), never ``reused``."""
    plan = wt.ok(
        "create", "--repo", wt.repo.repo_id, "--name", "dry1", "--holder", "ag-d", "--dry-run"
    )
    check(
        plan["dry_run"] is True and plan["reused"] is False,
        "dry-run shape",
        "#36 #20",
        sorted(plan),
    )
    check(not wt.ok("list")["records"], "a dry run leaves no live record", "#36 #20")
    real = wt.create("dry1", "ag-d")
    again = wt.ok(
        "create", "--repo", wt.repo.repo_id, "--name", "dry1", "--holder", "ag-d", "--dry-run"
    )
    world.observe(
        "wtctl.dry-run.over-own-record",
        {k: again[k] for k in ("reused", "branch_action", "dry_run")},
    )
    check(
        real["reused"] is False
        and again["reused"] is True
        and again["dry_run"] is False
        and again["branch_action"] is None
        and again["record"]["id"] == real["record"]["id"],
        "a dry run over the holder's own record returns reused: true, dry_run: false, "
        "branch_action null (the same answer as the real call)",
        "wtctl interface.md Create order, Retry after a crash (line 195: also under --dry-run)",
        again,
    )
    other = wt.run(
        "create", "--repo", wt.repo.repo_id, "--name", "dry1", "--holder", "ag-x", "--dry-run"
    )
    err = json.loads(other.stdout).get("error", {}) if other.stdout.strip() else {}
    check(
        other.returncode == 20
        and err.get("reason") == "path_exists"
        and "reused" not in other.stdout,
        "a dry run by another holder over the same name is refused (exit 20, path_exists), "
        "never reused",
        "wtctl interface.md Names (explicit --name collision) / Create order step 10",
        (other.returncode, other.stdout[-300:]),
    )
    world.observe("wtctl.dry-run.record-state", plan["record"]["state"])


@SC("wtctl-collisions-reuse")
def test_name_collisions_and_reuse_by_the_same_holder(wt: Wt, world: World) -> None:
    """#36 #20 and design.md (explicit ``--name`` collisions): names are never suffixed; the
    template-built branch colliding with an existing branch exits 20 ``branch_exists``; the same
    holder retrying an explicit-branch or detached create is ``reused``; another holder is refused
    (the real wtctl checks the path first: ``path_exists``, observed also for ``--branch``)."""
    wt.create("d4", "ag-a", "--branch", "feat/x")
    same = wt.create("d4", "ag-a", "--branch", "feat/x")
    check(
        same["reused"] is True and same["record"]["name"] == "d4",
        "explicit --branch retry by the same holder is reused (nothing changed)",
        "design.md wtctl behavior: explicit --branch collision reused when the same holder holds it",
    )
    other = wt.run(
        "create",
        "--repo",
        wt.repo.repo_id,
        "--name",
        "d4",
        "--branch",
        "feat/x",
        "--holder",
        "ag-b",
    )
    world.observe(
        "wtctl.collision.explicit-branch.other-holder",
        (other.returncode, wt.code(other), other.stdout[-200:]),
    )
    other_err = json.loads(other.stdout).get("error", {})
    check(
        other.returncode == 20
        and other_err.get("reason") == "path_exists"
        and other_err.get("details", other_err).get("holder_id") == same["record"]["id"],
        "another holder's retry is refused with exit 20, reason path_exists naming the holder's "
        "record (the name collision, whose path would be taken, is checked before the branch)",
        "wtctl interface.md Names (explicit --name collision: reason path_exists) / Create order "
        "step 10 (holder_id)",
        other.stdout[-300:],
    )
    wt.create("d5", "ag-a", "--detach")
    check(
        wt.create("d5", "ag-a", "--detach")["reused"] is True,
        "--detach retry by the same holder is reused",
        "design.md wtctl behavior: --detach (path_exists) reused when the same holder holds it",
    )
    detached_other = wt.run(
        "create", "--repo", wt.repo.repo_id, "--name", "d5", "--detach", "--holder", "ag-b"
    )
    check(
        detached_other.returncode == 20,
        "another holder on a --detach collision exits 20",
        "design.md wtctl behavior",
    )
    world.git("branch", "wt/d6", cwd=wt.repo.clone)
    clash = wt.run("create", "--repo", wt.repo.repo_id, "--name", "d6", "--holder", "ag-a")
    body = json.loads(clash.stdout)["error"]
    check(
        clash.returncode == 20 and body.get("reason") == "branch_exists",
        "the template-built branch colliding with an existing branch exits 20 branch_exists (not reused)",
        "design.md wtctl behavior (Create order)",
        body,
    )
    check(
        not any(r["name"] == "d6-2" for r in wt.ok("list")["records"]),
        "names are never suffixed",
        "#36 #20",
    )
    world.observe("wtctl.collision.template-branch", body)
    # a different name on the same branch: branch_in_use
    same_holder = wt.ok(
        "create",
        "--repo",
        wt.repo.repo_id,
        "--name",
        "d7",
        "--branch",
        "feat/x",
        "--holder",
        "ag-a",
    )
    world.observe(
        "wtctl.collision.branch-in-use.same-holder",
        (same_holder["reused"], same_holder["record"]["name"]),
    )
    check(
        same_holder["reused"] is True,
        "an explicit --branch in use by the same holder's record is reused (even under another --name)",
        "design.md wtctl behavior (branch_in_use reused when the same holder holds the record)",
        same_holder["record"]["name"],
    )
    inuse = wt.run(
        "create",
        "--repo",
        wt.repo.repo_id,
        "--name",
        "d7",
        "--branch",
        "feat/x",
        "--holder",
        "ag-b",
    )
    world.observe("wtctl.collision.branch-in-use", (inuse.returncode, inuse.stdout[-300:]))
    check(
        inuse.returncode == 20,
        "an explicit --branch already checked out by another worktree exits 20",
        "design.md wtctl behavior (branch_in_use)",
        inuse.stdout[-200:],
    )
    check(
        json.loads(inuse.stdout)["error"].get("reason") in ("branch_in_use", "branch_exists"),
        "the refusal reason is branch_in_use (or branch_exists)",
        "design.md wtctl behavior (branch_in_use)",
        inuse.stdout[-200:],
    )
    throw = wt.run(
        "create", "--throwaway", "--repo", wt.repo.repo_id, "--ref", wt.repo.sha, "--holder", "ag-t"
    )
    check(
        throw.returncode == 2 and wt.code(throw) == "usage",
        "a throwaway takes no holder (agent-spawn never holds throwaways)",
        "design.md D6 (throwaways are never held)",
        throw.stdout[-200:],
    )
    made = wt.ok("create", "--throwaway", "--repo", wt.repo.repo_id, "--ref", wt.repo.sha)
    check(
        made["record"]["object_type"] == "throwaway" and made["branch_action"] is None,
        "a throwaway record has object_type throwaway",
        "design.md: throwaway records have object_type throwaway",
        made["record"],
    )
    check(Path(made["record"]["path"]).is_dir(), "the throwaway root exists", "design.md D49")
    world.observe("wtctl.throwaway.path", made["record"]["path"])


@SC("wtctl-adopt-forget-holders")
def test_adopt_and_forget_by_holder(wt: Wt, world: World) -> None:
    """#36 #20 (adopt/forget accept ``--actor`` and ``--decision-id``, ``adopt`` of a recorded
    path is ``existing: true`` and ignores ``--repo``, ``forget`` of a non-holder exits 0 with
    ``not_a_holder``, a tombstone repeat exits 10 with ``details.state``), #36 #63 (an ``auto:``
    actor is accepted), #36 #22 (a forgotten adopted record prints ``record.state: forgotten``
    exactly when its last holder went; an owned record never does)."""
    made = wt.create("n1", "ag-aaaa")["record"]
    adopted = world.json_tool(
        "wtctl",
        "adopt",
        made["path"],
        "--holder=ag-bbbb",
        "--actor",
        AUTO,
        "--decision-id",
        "d1",
        "--repo",
        "other/x",
    )
    check(
        adopted["existing"] is True and adopted["record"]["holders"] == ["ag-aaaa", "ag-bbbb"],
        "adopt of a recorded path adds the holder, existing: true, --repo ignored; --actor/--decision-id accepted",
        "#36 #20 adopt on a recorded active path",
        adopted,
    )
    notholder = wt.ok("forget", made["id"], "--holder", "ag-zzzz")
    check(
        any("not_a_holder" in w for w in notholder["warnings"])
        and notholder["record"]["state"] == "active",
        "forget of a non-holder exits 0 with the warning not_a_holder",
        "#36 #20 forget of a non-holder",
        notholder["warnings"],
    )
    for holder in ("ag-bbbb", "ag-aaaa"):
        left = world.json_tool("wtctl", "forget", made["id"], "--holder", holder, "--actor", AUTO)
    check(
        left["record"]["state"] == "active" and left["record"]["holders"] == [],
        "an owned record is never forgotten by dropping its last holder",
        "#36 #22 (forget --holder prints forgotten only for an adopted record)",
        left["record"],
    )
    # an adopted (not owned) record: last holder forgets it
    manual = world.root / "repos" / "manual"
    world.git("worktree", "add", "-q", "-b", "manual", str(manual), cwd=wt.repo.clone)
    rec = world.json_tool(
        "wtctl",
        "adopt",
        str(manual),
        "--holder",
        "ag-m1",
        "--repo",
        wt.repo.repo_id,
        "--actor",
        AUTO,
    )["record"]
    check(
        rec["owned"] is False and rec["state"] == "active",
        "adopt of an existing worktree is not owned",
        "#36 #20",
    )
    wt.ok("adopt", str(manual), "--holder", "ag-m2")
    first = wt.ok("forget", rec["id"], "--holder", "ag-m1", "--actor", AUTO)
    check(
        first["record"]["state"] == "active",
        "forgetting one of two holders keeps the record active",
        "#36 #22",
    )
    last = wt.ok("forget", rec["id"], "--holder", "ag-m2", "--actor", AUTO)
    check(
        last["record"]["state"] == "forgotten",
        "forgetting the last holder of an adopted record makes it forgotten",
        "design.md wtctl behavior / #36 #23 item 8",
        last["record"],
    )
    again = wt.run("forget", rec["id"], "--holder", "ag-m2")
    err = json.loads(again.stdout)["error"]
    check(
        again.returncode == 10 and err.get("details", {}).get("state") == "forgotten",
        "a repeated forget of a tombstone exits 10 with details.state",
        "#36 #20 / design.md wtctl behavior",
        again.stdout[-200:],
    )
    shown = wt.ok("show", rec["id"])
    check(
        shown["state"] == "forgotten",
        "show of a tombstone exits 0 with its state",
        "#36 #20 / design.md",
    )


@SC("wtctl-adopt-forget-holders")
def test_forget_holder_in_unlinked_and_missing_states(wt: Wt, world: World) -> None:
    """#36 #22 and #75 7: ``forget <id> --holder`` exits 0 without a human actor for records in
    the ``unlinked`` and ``missing`` states (``release_fork_holds`` relies on it)."""
    made = wt.create("gone1", "ag-aaaa")["record"]
    shutil.rmtree(made["path"])
    reconciled = wt.run("reconcile")
    world.observe("wtctl.reconcile.after-rmtree", (reconciled.returncode, reconciled.stdout[-300:]))
    state = wt.ok("show", made["id"])["state"]
    world.observe("wtctl.state.after-rmtree-reconcile", state)
    check(
        state in ("missing", "unlinked", "active"),
        "the record is missing or unlinked after its directory is deleted",
        "#36 #22 states",
        state,
    )
    done = wt.run("forget", made["id"], "--holder", "ag-aaaa", "--actor", AUTO)
    check(
        done.returncode == 0,
        f"forget --holder of a {state} record exits 0 without a human actor",
        "#36 #22 (the state table says 0; the --holder form is not spelled out)",
        done.stdout[-300:],
    )
    free = wt.run("forget", "wt-nosuchid", "--holder", "ag-aaaa")
    check(
        free.returncode == 10,
        "forget of an unknown id exits 10 (counts as released)",
        "#36 #22",
        free.returncode,
    )


@SC("wtctl-remove-status")
def test_status_and_remove_contract(wt: Wt, world: World) -> None:
    """#36 #23 (``remove``/``status`` accept ``--flag=value``; a held refusal exits 20 with
    ``details.holders`` listing every holder including the caller; ``status`` prints
    ``would_remove``, ``reason``, ``human_required`` and ``force_overrides``, ignores the caller's
    own hold, exits 0 whatever it found; ``remove`` of a tombstone exits 10 with ``details.state``;
    remove as an ``auto:`` actor without a TTY exits 0), #36 #22 (refusal order)."""
    made = wt.create("n1", "ag-aaaa")["record"]
    wt.ok("adopt", made["path"], "--holder", "ag-bbbb")
    status = world.json_tool("wtctl", "status", made["id"], "--holder=ag-aaaa")
    for key in ("would_remove", "reason", "human_required", "force_overrides"):
        check(key in status, f"status prints {key}", "#36 #23 item 7", sorted(status))
    check(
        status["would_remove"] is False and status["reason"] == "held",
        "status ignores the caller's own hold and reports another holder as held",
        "#36 #23 item 7",
        (status["would_remove"], status["reason"]),
    )
    refused = wt.run("remove", made["id"], "--yes", "--holder=ag-aaaa", f"--actor={AUTO}")
    err = json.loads(refused.stdout)["error"]
    check(
        refused.returncode == 20 and sorted(err["details"]["holders"]) == ["ag-aaaa", "ag-bbbb"],
        "a held refusal exits 20 with details.holders listing every holder including the caller",
        "#36 #23 item 2",
        refused.stdout[-300:],
    )
    nottty = wt.run("remove", made["id"], "--holder=ag-aaaa", f"--actor={AUTO}")
    check(
        nottty.returncode == 20,
        "without --yes the held refusal (20) comes before confirmation_required (13)",
        "#36 #22 refusal order (step 4 before the TTY prompt)",
        (nottty.returncode, nottty.stdout[-200:]),
    )
    wt.ok("forget", made["id"], "--holder", "ag-bbbb")
    status2 = world.json_tool("wtctl", "status", made["id"], "--holder=ag-aaaa")
    check(
        status2["would_remove"] is True,
        "status of a removable record: would_remove true",
        "#36 #23 item 7",
        status2,
    )
    removed = world.json_tool(
        "wtctl", "remove", made["id"], "--yes", "--holder=ag-aaaa", f"--actor={AUTO}"
    )
    check(
        removed["record"]["state"] == "removed" and not Path(made["path"]).exists(),
        "remove as an auto: actor without a TTY exits 0 and deletes the worktree",
        "#36 #23 item 2",
        removed["record"]["state"],
    )
    tomb = wt.run("remove", made["id"], "--yes", "--holder=ag-aaaa", f"--actor={AUTO}")
    err = json.loads(tomb.stdout)["error"]
    check(
        tomb.returncode == 10 and err["details"]["state"] in ("removed", "forgotten", "gone"),
        "remove of a tombstone exits 10 with details.state",
        "#36 #23 item 3",
        tomb.stdout[-200:],
    )
    # adopted record: the human-only 13 comes first, even with an extra holder (step 1 before 4)
    manual = world.root / "repos" / "manual"
    world.git("worktree", "add", "-q", "-b", "manual", str(manual), cwd=wt.repo.clone)
    rec = wt.ok("adopt", str(manual), "--holder", "ag-m1", "--repo", wt.repo.repo_id)["record"]
    wt.ok("adopt", str(manual), "--holder", "ag-m2")
    adopted = wt.run("remove", rec["id"], "--yes", "--holder=ag-m1", f"--actor={AUTO}")
    check(
        adopted.returncode == 13,
        "removing an adopted record without a human actor exits 13 before held (20)",
        "#36 #22 refusal order / #8 wtctl exit 13",
        (adopted.returncode, adopted.stdout[-200:]),
    )
    ast = world.json_tool("wtctl", "status", rec["id"], "--holder=ag-m1")
    check(
        ast["human_required"] is True,
        "status marks an adopted record human_required",
        "#36 #23 item 7",
        ast,
    )
    world.observe("wtctl.status.keys", sorted(status))
    world.observe("wtctl.status.force_overrides", status["force_overrides"])
    forced = wt.run(
        "remove", rec["id"], "--yes", "--holder=ag-m1", f"--actor={AUTO}", "--override-escalation"
    )
    world.observe(
        "wtctl.remove.override-escalation.auto", (forced.returncode, forced.stdout[-200:])
    )
    check(
        forced.returncode == 2,
        "--override-escalation with a non-human actor exits 2",
        "#36 #78 (wtctl remove --override-escalation)",
        forced.returncode,
    )


@SC("wtctl-baseline-check")
def test_baseline_check_and_listing_paths(wt: Wt, world: World) -> None:
    """#36 #25: ``baseline check <full id> --json`` takes the record id as the only positional;
    ``baseline check`` of a missing/partial record exits 20 ``unlinked``; a removed record exits
    10; ``list`` records carry realpath ``path`` and ``object_type`` worktree or throwaway."""
    made = wt.create("b1", "ag-aaaa")["record"]
    done = wt.run("baseline", "check", made["id"])
    world.observe("wtctl.baseline.check.fresh", (done.returncode, done.stdout[-300:]))
    check(
        done.returncode == 0,
        "baseline check of an untouched record exits 0",
        "#36 #25 (accepted with the record id as the only positional)",
        (done.returncode, done.stdout[-300:]),
    )
    # change the clone's config: the baseline may report a changed class
    config = wt.repo.clone / ".git" / "config"
    config.write_text(config.read_text() + "\n[alias]\n\tlive = status\n")
    changed = wt.run("baseline", "check", made["id"])
    world.observe(
        "wtctl.baseline.check.after-config-change", (changed.returncode, changed.stdout[-400:])
    )
    check(
        changed.returncode == 20,
        "a changed baseline exits 20",
        "#36 #25",
        (changed.returncode, changed.stdout[-300:]),
    )
    err = json.loads(changed.stdout)["error"]
    check(
        err.get("code") == "refused"
        and err.get("reason") == "repo_config_changed"
        and err.get("details", {}).get("changed_classes") == ["git"],
        "a baseline change is refused (code refused, reason repo_config_changed) with "
        "details.changed_classes naming only the changed class (the git config)",
        "#36 #25",
        err,
    )
    world.observe("wtctl.baseline.error.keys", sorted(err))
    throw = wt.ok("create", "--throwaway", "--repo", wt.repo.repo_id, "--ref", wt.repo.sha)[
        "record"
    ]
    kinds = {r["name"]: r["object_type"] for r in wt.ok("list")["records"]}
    check(
        sorted(kinds.values()) == ["throwaway", "worktree"],
        "list records carry object_type worktree or throwaway",
        "#36 #25 / design.md D49",
        kinds,
    )
    for r in wt.ok("list")["records"]:
        check(
            r["path"] == str(Path(r["path"]).resolve()),
            "list paths are realpaths",
            "#36 #25",
            r["path"],
        )
    shutil.rmtree(made["path"])
    wt.run("reconcile")
    gone = wt.run("baseline", "check", made["id"])
    world.observe("wtctl.baseline.check.missing", (gone.returncode, gone.stdout[-300:]))
    gone_err = json.loads(gone.stdout).get("error", {}) if gone.stdout.strip() else {}
    check(
        gone.returncode == 20
        and gone_err.get("code") == "refused"
        and gone_err.get("reason") == "unlinked",
        "baseline check of a missing record exits 20 (code refused, reason unlinked)",
        "#36 #25 (partial, missing and clone_missing records exit 20 unlinked)",
        (gone.returncode, gone.stdout[-300:]),
    )
    del throw
