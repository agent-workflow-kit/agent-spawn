"""The wtctl backend against the fake wtctl: exact argv, the exit mapping and the pure rules of
the records of a cwd."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from agent_spawn import worktrees as wt
from agent_spawn.gateway import SystemGateway
from agent_spawn.output import CliError
from tests import sibling_schemas
from tests.fakebin.harness import Fakebin
from tests.session_backend_support import FakebinGateway

AGENT = "ag-abcdefgh"
REPO = "example.org/team/project"


def backend(fakebin: Fakebin, **kwargs: Any) -> wt.WtctlBackend:
    return wt.WtctlBackend(
        [str(fakebin.bin_dir / "wtctl")], gateway=FakebinGateway(fakebin), **kwargs
    )


def plan(**changes: Any) -> wt.CreatePlan:
    base: dict[str, Any] = {"agent_id": AGENT, "repo": REPO, "name": AGENT}
    return wt.CreatePlan(**{**base, **changes})


# --- argv ---------------------------------------------------------------------------------------


def test_create_argv_is_exact_with_every_flag(fakebin: Fakebin) -> None:
    api = backend(fakebin, actor="auto:spawn", decision_ids=["d1", "d2"])
    record, reused = api.create(
        plan(
            name="my-tree",
            branch="feat/x",
            base="main",
            ref="a" * 40,
            no_bootstrap=True,
            copy_env=True,
        )
    )
    assert reused is False and record.name == "my-tree" and record.holders == (AGENT,)
    fakebin.assert_argv(
        "wtctl",
        [
            "create",
            f"--repo={REPO}",
            f"--holder={AGENT}",
            "--name=my-tree",
            "--branch=feat/x",
            "--base=main",
            f"--ref={'a' * 40}",
            "--no-bootstrap",
            "--copy-env",
            "--actor=auto:spawn",
            "--decision-id=d1",
            "--decision-id=d2",
            "--json",
        ],
    )


def test_create_detach_and_the_override_flag(fakebin: Fakebin) -> None:
    api = backend(fakebin)
    # --override-escalation is human-only: the fake wants a human actor for it.
    human = backend(fakebin, actor="human:me")
    human.create(plan(detach=True), override_escalation=True)
    fakebin.assert_argv(
        "wtctl",
        [
            "create",
            f"--repo={REPO}",
            f"--holder={AGENT}",
            f"--name={AGENT}",
            "--detach",
            "--override-escalation",
            "--actor=human:me",
            "--json",
        ],
    )
    api.create(plan(name="plain", detach=True))
    assert "--override-escalation" not in fakebin.argvs("wtctl")[-1]
    assert not any(a.startswith("--actor") for a in fakebin.argvs("wtctl")[-1])


def test_adopt_list_show_forget_argv(fakebin: Fakebin, tmp_path: Path) -> None:
    api = backend(fakebin, actor="auto:spawn", decision_ids=["d1"])
    target = tmp_path / "tree"
    target.mkdir()
    record, existing = api.adopt(str(target), AGENT, repo=REPO)
    assert existing is False and record.owned is False
    fakebin.assert_argv(
        "wtctl",
        [
            "adopt",
            str(target),
            f"--holder={AGENT}",
            f"--repo={REPO}",
            "--actor=auto:spawn",
            "--decision-id=d1",
            "--json",
        ],
    )
    api.list_records()
    fakebin.assert_argv("wtctl", ["list", "--json"])  # a read carries no actor
    api.list_records(repo=REPO)
    fakebin.assert_argv("wtctl", ["list", f"--repo={REPO}", "--json"])
    found = api.show(record.id)
    assert found is not None and found.id == record.id
    fakebin.assert_argv("wtctl", ["show", record.id, "--json"])
    assert api.forget(record.id, AGENT) == []
    fakebin.assert_argv(
        "wtctl",
        [
            "forget",
            record.id,
            f"--holder={AGENT}",
            "--actor=auto:spawn",
            "--decision-id=d1",
            "--json",
        ],
    )


def test_adopt_without_repo_passes_none_and_refuses_a_relative_path(fakebin: Fakebin) -> None:
    api = backend(fakebin)
    with pytest.raises(CliError) as caught:
        api.adopt("relative/dir", AGENT)
    assert caught.value.exit_code == 2
    assert fakebin.invocations("wtctl") == []


def test_ids_that_could_be_flags_never_reach_argv(fakebin: Fakebin) -> None:
    api = backend(fakebin)
    for bad in ("--evil", "wt-short", "wt-aaaaaaaa --json", ""):
        with pytest.raises(CliError):
            api.show(bad)
        with pytest.raises(CliError):
            api.forget(bad, AGENT)
    assert fakebin.invocations("wtctl") == []


def test_values_after_equals_stay_values(fakebin: Fakebin) -> None:
    # check_branch refuses a leading '-' before argv is built; the = form is the second guard.
    argv = backend(fakebin).create_argv(plan(base="--x"))
    assert "--base=--x" in argv and "--x" not in argv
    with pytest.raises(CliError):
        wt.check_branch("--x", "--base")
    with pytest.raises(CliError):
        backend(fakebin).create_argv(plan(base="a\0b"))


@pytest.mark.parametrize(
    "value",
    ["", "-x", "a..b", "a b", "a~b", "a^b", "a:b", "a?b", "a*b", "a[b", "a\\b", "x.lock", "a/",
     "/a", "a//b", "@", "a@{b", ".hidden", "a/.b", "trailing.", "tab\tx", "nul\0", "z" * 256],
)  # fmt: skip
def test_check_branch_refuses_what_cannot_be_a_branch(value: str) -> None:
    with pytest.raises(CliError):
        wt.check_branch(value, "--branch")


@pytest.mark.parametrize("value", ["main", "feat/x-1", "release/1.2", "ünï"])
def test_check_branch_accepts_ordinary_names(value: str) -> None:
    assert wt.check_branch(value, "--branch") == value


def test_other_grammars() -> None:
    assert wt.check_repo(REPO) == REPO
    for bad in ("", "no-slash", "Host/UP", "a/b c", "-x/y"):
        with pytest.raises(CliError):
            wt.check_repo(bad)
    assert wt.check_sha("a" * 40)
    for bad in ("a" * 39, "A" * 40, "g" * 40, ""):
        with pytest.raises(CliError):
            wt.check_sha(bad)
    for bad in ("", "-x", ".x", "a b", "a" * 65, "é"):
        with pytest.raises(CliError):
            wt.check_worktree_name(bad)
    assert wt.check_worktree_name("a" * 64)
    assert wt.planned_name(AGENT, None) == AGENT and wt.planned_name(AGENT, "n") == "n"


# --- the exit mapping -----------------------------------------------------------------------------


def scripted(fakebin: Fakebin, command: str, **respond: Any) -> None:
    fakebin.script("wtctl", command, **respond)


@pytest.mark.parametrize(
    ("respond", "kind", "park"),
    [
        (
            {"error": {"code": "vetoed", "message": "no"}, "exit": 11},
            "vetoed",
            {"park_reason": "vetoed", "needs_attention": None, "escalated_by": None},
        ),
        (
            {"error": {"code": "escalated", "message": "ask"}, "exit": 13},
            "escalated",
            {"park_reason": "escalated", "needs_attention": "escalated", "escalated_by": "wtctl"},
        ),
        (
            {"error": {"code": "busy", "message": "later"}, "exit": 12},
            "busy",
            {"park_reason": "launch_failed", "needs_attention": None, "escalated_by": None},
        ),
        (
            {"error": {"code": "fetch_failed", "message": "x"}, "exit": 12},
            "other",
            {
                "park_reason": "launch_failed",
                "needs_attention": "launch_failed",
                "escalated_by": None,
            },
        ),
        (
            {"error": {"code": "confirmation_required", "message": "x"}, "exit": 13},
            "other",
            {
                "park_reason": "launch_failed",
                "needs_attention": "launch_failed",
                "escalated_by": None,
            },
        ),
        (
            {"error": {"code": "refused", "message": "x", "reason": "path_exists"}, "exit": 20},
            "other",
            {
                "park_reason": "launch_failed",
                "needs_attention": "launch_failed",
                "escalated_by": None,
            },
        ),
        (
            {"error": {"code": "internal", "message": "x"}, "exit": 1},
            "other",
            {
                "park_reason": "launch_failed",
                "needs_attention": "launch_failed",
                "escalated_by": None,
            },
        ),
    ],
)
def test_create_failures_map_per_the_spec(
    fakebin: Fakebin, respond: dict[str, Any], kind: str, park: dict[str, Any]
) -> None:
    scripted(fakebin, "create", **respond)
    with pytest.raises(wt.WtctlError) as caught:
        backend(fakebin).create(plan())
    error = caught.value
    assert wt.failure_kind(error) == kind and wt.park_fields(wt.failure_kind(error)) == park
    assert error.tool == "wtctl" and error.exit_code == respond["exit"]
    assert error.lower_error is not None and error.lower_error["code"] == respond["error"]["code"]
    if kind == "busy":
        assert error.details == {"retryable": True}
    else:
        assert error.details is None


def test_bootstrap_failed_names_the_record(fakebin: Fakebin) -> None:
    scripted(fakebin, "create", bootstrap_failed="install")
    with pytest.raises(wt.WtctlError) as caught:
        backend(fakebin).create(plan())
    error = caught.value
    assert wt.failure_kind(error) == "bootstrap_failed" and error.exit_code == 12
    shown = backend(fakebin).show(error.record_id or "")
    assert shown is not None and shown.state == "active" and shown.holders == (AGENT,)
    assert wt.park_fields("bootstrap_failed")["needs_attention"] == "launch_failed"


def test_a_record_id_that_is_not_an_id_is_not_taken_from_an_error(fakebin: Fakebin) -> None:
    scripted(
        fakebin,
        "create",
        error={"code": "bootstrap_failed", "message": "x", "details": {"record": "--evil"}},
        exit=12,
    )
    with pytest.raises(wt.WtctlError) as caught:
        backend(fakebin).create(plan())
    assert caught.value.record_id is None


def test_adopt_busy_is_retryable_and_never_vetoed(fakebin: Fakebin, tmp_path: Path) -> None:
    scripted(fakebin, "adopt", error={"code": "busy", "message": "later"}, exit=12)
    with pytest.raises(wt.WtctlError) as caught:
        backend(fakebin).adopt(str(tmp_path), AGENT)
    assert wt.failure_kind(caught.value) == "busy" and caught.value.details == {"retryable": True}


def test_missing_wtctl_is_tool_missing(tmp_path: Path) -> None:
    api = wt.WtctlBackend([str(tmp_path / "nowhere")], gateway=SystemGateway())
    with pytest.raises(CliError) as caught:
        api.list_records()
    assert caught.value.code == "tool_missing" and caught.value.exit_code == 12


# --- list, show, forget -----------------------------------------------------------------------


def test_list_drops_tombstones_and_rejects_unusable_records(fakebin: Fakebin) -> None:
    api = backend(fakebin)
    live, _ = api.create(plan())
    gone, _ = api.create(plan(name="other", detach=True))
    scripted(fakebin, "list", times=1, stdout={"records": [
        {"id": live.id, "path": live.path, "object_type": "worktree", "state": "active",
         "name": "n", "owned": True, "holders": [AGENT]},
        {"id": gone.id, "path": gone.path, "object_type": "worktree", "state": "removed",
         "name": "n2", "owned": True, "holders": []},
    ], "foreign": []})  # fmt: skip
    assert [r.id for r in api.list_records()] == [live.id]
    for bad in (
        {"records": "x"},
        {"records": [{"id": "x"}]},
        {"records": [{"id": live.id, "path": "rel", "object_type": "worktree", "state": "active",
                      "holders": []}]},
        {"records": [{"id": live.id, "path": live.path, "object_type": "worktree",
                      "state": "weird", "holders": []}]},
        {"records": [{"id": live.id, "path": live.path, "object_type": "worktree",
                      "state": "active", "holders": [1]}]},
    ):  # fmt: skip
        scripted(fakebin, "list", times=1, stdout=bad)
        with pytest.raises(CliError) as caught:
            api.list_records()
        assert caught.value.code == "internal" and caught.value.tool == "wtctl"


def test_list_failure_passes_wtctls_code_through(fakebin: Fakebin) -> None:
    scripted(fakebin, "list", error={"code": "busy", "message": "x"}, exit=12)
    with pytest.raises(CliError) as caught:
        backend(fakebin).list_records()
    assert caught.value.exit_code == 12 and caught.value.tool == "wtctl"


def test_show_tombstone_unknown_and_failure(fakebin: Fakebin) -> None:
    api = backend(fakebin)
    record, _ = api.create(plan())
    assert api.forget(record.id, AGENT) == []  # an owned record is kept (never forgotten by holder)
    owned = api.show(record.id)
    assert owned is not None and owned.holders == ()
    assert api.show("wt-aaaaaaaa") is None
    scripted(fakebin, "show", error={"code": "internal", "message": "x"}, exit=1)
    with pytest.raises(CliError):
        api.show(record.id)


def test_forget_of_a_non_holder_returns_the_warning_and_a_tombstone_counts_as_done(
    fakebin: Fakebin, tmp_path: Path
) -> None:
    api = backend(fakebin)
    target = tmp_path / "t"
    target.mkdir()
    record, _ = api.adopt(str(target), AGENT)
    assert api.forget(record.id, "ag-bbbbbbbb") == ["not_a_holder"]
    assert api.forget(record.id, AGENT) == []  # the last holder: the adopted record is forgotten
    assert api.forget(record.id, AGENT) == []  # exit 10 with a tombstone state: done


def test_create_dry_run_returns_the_plan_and_creates_nothing(fakebin: Fakebin) -> None:
    api = backend(fakebin)
    dry = api.create_dry_run(plan())
    sibling_schemas.validate(dry, "wtctl", "cli/create.json")
    assert dry["dry_run"] is True
    assert api.list_records() == []
    assert "--dry-run" in fakebin.argvs("wtctl")[0]


# --- the records of a cwd -------------------------------------------------------------------------


def record(
    path: Path | str, kind: str = "worktree", state: str = "active", rid: str = "wt-aaaaaaaa"
):
    return wt.WtRecord(rid, str(path), kind, state, "n", None, ())


def test_containment_uses_realpaths_and_components(tmp_path: Path) -> None:
    root = tmp_path / "wt"
    (root / "sub").mkdir(parents=True)
    (tmp_path / "wt-sibling").mkdir()
    link = tmp_path / "link"
    link.symlink_to(root)
    records = [record(root)]
    assert wt.records_containing(records, str(root))
    assert wt.records_containing(records, str(root / "sub"))
    assert wt.records_containing(records, str(link / "sub"))  # a symlinked cwd resolves first
    assert not wt.records_containing(records, str(tmp_path / "wt-sibling"))
    assert not wt.records_containing(records, str(tmp_path))
    assert wt.records_containing([record(link)], str(root))  # a symlinked record path too
    assert not wt.records_containing([record(root, state="removed")], str(root))


def test_throwaway_root_needs_equality_and_no_worktree_around_it(tmp_path: Path) -> None:
    tw = tmp_path / "tw"
    (tw / "sub").mkdir(parents=True)
    throwaway = record(tw, "throwaway", rid="wt-bbbbbbbb")
    assert wt.is_throwaway_root([throwaway], str(tw))
    assert not wt.is_throwaway_root([throwaway], str(tw / "sub"))
    assert wt.inside_throwaway([throwaway], str(tw / "sub"))
    nested = [throwaway, record(tmp_path)]  # an adopted worktree record around the throwaway
    assert not wt.is_throwaway_root(nested, str(tw))
    assert not wt.is_throwaway_root([record(tw, "throwaway", "removed")], str(tw))
    assert [r.id for r in wt.worktrees_containing(nested, str(tw))] == ["wt-aaaaaaaa"]


def test_record_paths_with_dots_are_normalized(tmp_path: Path) -> None:
    tw = tmp_path / "tw"
    tw.mkdir()
    sneaky = os.path.join(str(tw), "..", "tw")
    assert wt.is_throwaway_root([record(sneaky, "throwaway")], str(tw))
