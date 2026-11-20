"""The wtctl backend against the fake wtctl: exact argv, the exit mapping and the pure rules of
the records of a cwd."""

from __future__ import annotations

import json
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
     "/a", "a//b", "@", "a@{b", "HEAD", "refs/heads/x", "refs/x", ".hidden", "a/.b", "trailing.",
     "tab\tx", "nul\0", "z" * 256],
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
    assert error.record is not None and error.record.id == error.record_id
    assert error.record.holders == (AGENT,) and error.record.state == "active"
    shown = backend(fakebin).show(error.record_id or "")
    assert shown is not None and shown.state == "active" and shown.holders == (AGENT,)
    assert wt.park_fields("bootstrap_failed")["needs_attention"] == "launch_failed"


def test_an_id_string_in_details_record_is_still_accepted(fakebin: Fakebin) -> None:
    scripted(
        fakebin,
        "create",
        error={"code": "bootstrap_failed", "message": "x", "details": {"record": "wt-aaaaaaab"}},
        exit=12,
    )
    with pytest.raises(wt.WtctlError) as caught:
        backend(fakebin).create(plan())
    assert caught.value.record_id == "wt-aaaaaaab" and caught.value.record is None


def test_only_bootstrap_failed_names_a_record(fakebin: Fakebin) -> None:
    api = backend(fakebin)
    other, _ = api.create(plan(name="other", detach=True))
    for respond in (
        {"error": {"code": "refused", "message": "x", "reason": "path_exists",
                   "details": {"record": other.id}}, "exit": 20},
        {"error": {"code": "fetch_failed", "message": "x", "details": {"record": other.id}},
         "exit": 12},
    ):  # fmt: skip
        scripted(fakebin, "create", times=1, **respond)
        with pytest.raises(wt.WtctlError) as caught:
            api.create(plan(name="again", detach=True))
        assert caught.value.record_id is None and caught.value.record is None
    # the adopt of a failed bootstrap code does not name one either
    scripted(
        fakebin,
        "adopt",
        error={"code": "bootstrap_failed", "message": "x", "details": {"record": other.id}},
        exit=12,
    )
    with pytest.raises(wt.WtctlError) as caught:
        api.adopt(other.path, AGENT)
    assert caught.value.record_id is None


class _TimingOutGateway(FakebinGateway):
    def run(self, argv: Any, **kwargs: Any) -> Any:
        import subprocess

        raise subprocess.TimeoutExpired(list(argv), 1.0)


@pytest.mark.parametrize("command", ["adopt", "forget", "show", "list"])
def test_a_timeout_of_a_repeatable_call_is_busy_and_retryable(
    fakebin: Fakebin, tmp_path: Path, command: str
) -> None:
    api = wt.WtctlBackend([str(fakebin.bin_dir / "wtctl")], gateway=_TimingOutGateway(fakebin))
    calls = {
        "adopt": lambda: api.adopt(str(tmp_path), AGENT),
        "forget": lambda: api.forget("wt-aaaaaaab", AGENT),
        "show": lambda: api.show("wt-aaaaaaab"),
        "list": lambda: api.list_records(),
    }
    with pytest.raises(wt.WtctlError) as caught:
        calls[command]()
    assert wt.failure_kind(caught.value) == "busy" and caught.value.details == {"retryable": True}
    assert wt.park_fields("busy")["needs_attention"] is None


def test_a_timeout_of_create_stays_a_plain_failure(fakebin: Fakebin) -> None:
    api = wt.WtctlBackend([str(fakebin.bin_dir / "wtctl")], gateway=_TimingOutGateway(fakebin))
    with pytest.raises(CliError) as caught:
        api.create(plan())
    assert not isinstance(caught.value, wt.WtctlError) and caught.value.code == "failed"


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


@pytest.mark.parametrize("exit_code", [11, 13])
def test_only_create_can_be_vetoed_or_escalated(
    fakebin: Fakebin, tmp_path: Path, exit_code: int
) -> None:
    code = "vetoed" if exit_code == 11 else "escalated"
    api = backend(fakebin)
    scripted(fakebin, "adopt", error={"code": code, "message": "x"}, exit=exit_code)
    with pytest.raises(wt.WtctlError) as caught:
        api.adopt(str(tmp_path), AGENT)
    assert wt.failure_kind(caught.value) == "other"
    scripted(fakebin, "list", error={"code": code, "message": "x"}, exit=exit_code)
    with pytest.raises(wt.WtctlError) as listed:
        api.list_records()
    assert wt.failure_kind(listed.value) == "other"
    scripted(fakebin, "create", error={"code": code, "message": "x"}, exit=exit_code)
    with pytest.raises(wt.WtctlError) as created:
        api.create(plan())
    assert wt.failure_kind(created.value) == ("vetoed" if exit_code == 11 else "escalated")


# --- a reused record must be the planned worktree (#99) ----------------------------------------


def saved(fakebin: Fakebin) -> dict[str, Any]:
    path = Path(str(fakebin.log) + ".records.json")
    return json.loads(path.read_text()) if path.exists() else {}  # type: ignore[no-any-return]


def test_a_matching_reuse_is_accepted_and_changes_nothing(fakebin: Fakebin) -> None:
    api = backend(fakebin)
    first, first_reused = api.create(plan(name="n1", branch="feat/x"))
    again, reused = api.create(plan(name="n1", branch="feat/x"))  # the retry after a crash
    assert (first_reused, reused) == (False, True)
    assert again == first and len(saved(fakebin)) == 1


def test_the_same_holder_with_a_new_name_and_an_explicit_branch_gets_the_old_record_refused(
    fakebin: Fakebin,
) -> None:
    api = backend(fakebin)
    old, _ = api.create(plan(name="old", branch="feat/x"))
    before = saved(fakebin)
    with pytest.raises(CliError) as caught:
        # wtctl answers with the OLD record (reused: true, name "old"): the fake models it
        api.create(plan(name="new", branch="feat/x"))
    error = caught.value
    assert (error.code, error.exit_code, error.tool) == ("launch_failed", 12, "agent-spawn")
    assert (error.details or {}).get("cause") == "worktree_mismatch"
    assert (error.details or {}).get("retryable") is False
    assert old.id in str((error.details or {}).get("hint"))
    for text in ("'new'", "'old'", "feat/x", old.id):
        assert text in str(error)
    assert saved(fakebin) == before  # nothing created, changed or removed
    assert [c[0] for c in fakebin.argvs("wtctl")] == ["create", "create"]


def reuse_with(fakebin: Fakebin, record: dict[str, Any], **plan_changes: Any) -> wt.WtRecord:
    """A create that wtctl answers with ``reused: true`` and the planned record changed by
    ``record`` (a reply the fake would not produce itself)."""
    fakebin.script("wtctl", "create", merge={"reused": True, "record": record})
    return backend(fakebin).create(plan(**plan_changes))[0]


@pytest.mark.parametrize(
    ("record", "plan_changes"),
    [
        pytest.param({"name": AGENT.upper()}, {}, id="name-case"),
        pytest.param({"name": f"{AGENT} "}, {}, id="name-trailing-space"),
        pytest.param({"name": f" {AGENT}"}, {}, id="name-leading-space"),
        pytest.param({"name": None}, {}, id="name-null"),
        pytest.param({"name": ""}, {}, id="name-empty"),
        pytest.param({"branch": "Feat/x"}, {"branch": "feat/x"}, id="branch-case"),
        pytest.param({"branch": "feat/x "}, {"branch": "feat/x"}, id="branch-space"),
        pytest.param({"branch": "feat/y"}, {"branch": "feat/x"}, id="branch-other"),
        pytest.param({"branch": "refs/tags/feat/x"}, {"branch": "feat/x"}, id="branch-other-ref"),
        pytest.param({"branch": None}, {"branch": "feat/x"}, id="branch-null-but-planned"),
        pytest.param({"branch": "feat/x"}, {"detach": True}, id="branch-on-a-detach-plan"),
        pytest.param({"holders": ["ag-zzzzzzzz"]}, {}, id="another-holder-only"),
        pytest.param({"holders": []}, {}, id="no-holders"),
        pytest.param({"object_type": "throwaway"}, {}, id="throwaway"),
    ],
)
def test_a_reuse_that_is_not_the_planned_worktree_is_refused(
    fakebin: Fakebin, record: dict[str, Any], plan_changes: dict[str, Any]
) -> None:
    with pytest.raises(CliError) as caught:
        reuse_with(fakebin, record, **plan_changes)
    assert (caught.value.code, caught.value.exit_code) == ("launch_failed", 12)
    assert (caught.value.details or {}).get("cause") == "worktree_mismatch"


@pytest.mark.parametrize(
    ("record", "plan_changes"),
    [
        pytest.param({"branch": "refs/heads/feat/x"}, {"branch": "feat/x"}, id="record-qualified"),
        pytest.param({"branch": "feat/x"}, {"branch": "refs/heads/feat/x"}, id="plan-qualified"),
        pytest.param({"branch": None}, {}, id="template-built-branch-not-compared"),
        pytest.param({"branch": "wt/whatever"}, {}, id="template-branch-any-value"),
        pytest.param({"base": "other", "ref": "b" * 40}, {"base": "main"}, id="base-not-compared"),
        pytest.param({"holders": ["ag-zzzzzzzz", AGENT]}, {}, id="another-holder-too"),
    ],
)
def test_a_reuse_that_is_the_planned_worktree_is_accepted(
    fakebin: Fakebin, record: dict[str, Any], plan_changes: dict[str, Any]
) -> None:
    got = reuse_with(fakebin, record, **plan_changes)
    assert got.name == AGENT


def test_a_create_that_is_not_reused_is_not_compared(fakebin: Fakebin) -> None:
    # the check is only for `reused: true`: a fresh record is the one the plan asked for
    fakebin.script("wtctl", "create", merge={"reused": False, "record": {"name": "other"}})
    record, reused = backend(fakebin).create(plan())
    assert reused is False and record.name == "other"


def test_a_reused_flag_that_is_not_the_boolean_true_is_not_a_reuse(fakebin: Fakebin) -> None:
    fakebin.script("wtctl", "create", merge={"reused": "true", "record": {"name": "other"}})
    _, reused = backend(fakebin).create(plan())
    assert reused is False


def test_the_record_carries_its_branch_and_tolerates_a_null_or_absent_one() -> None:
    base: dict[str, Any] = {
        "id": "wt-aaaaaaaa",
        "path": "/x/y",
        "object_type": "worktree",
        "state": "active",
        "name": "n",
        "owned": True,
        "holders": [],
    }
    parsed = wt.parse_record({**base, "branch": "feat/x"})
    assert parsed is not None and parsed.branch == "feat/x"
    for extra in ({}, {"branch": None}):
        parsed = wt.parse_record({**base, **extra})
        assert parsed is not None and parsed.branch is None
    assert wt.parse_record({**base, "branch": 5}) is None
    assert wt.parse_record({**base, "branch": ["a"]}) is None


def test_a_bare_refs_heads_prefix_never_matches() -> None:
    record = wt.WtRecord(
        "wt-aaaaaaaa", "/x/y", "worktree", "active", AGENT, True, (AGENT,), "refs/heads/"
    )
    with pytest.raises(CliError):
        wt.check_reused(plan(branch="refs/heads/"), record)
