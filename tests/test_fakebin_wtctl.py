"""The fake wtctl: canned outputs validate against the pinned schemas, argv is checked."""

from __future__ import annotations

import json
import os

import pytest

from tests import sibling_schemas
from tests.fakebin.harness import Fakebin
from tests.fakebin_support import error_of, run_json

REPO = "example.org/team/project"
SHA = "1" * 40


def create(fakebin: Fakebin, *extra: str) -> dict[str, object]:
    code, body = run_json(
        fakebin, "wtctl", "create", "--repo", REPO, *extra, schema="cli/create.json"
    )
    assert code == 0
    return body


def test_create_worktree_validates_and_makes_the_directory(fakebin: Fakebin) -> None:
    body = create(
        fakebin, "--branch", "feat/x y", "--holder", "ag-abcdefgh", "--label", "agent=ag-abcdefgh"
    )
    record = body["record"]
    assert isinstance(record, dict)
    assert record["name"] == "feat-x-y" and record["branch"] == "feat/x y"
    assert record["holders"] == ["ag-abcdefgh"] and record["labels"] == ["agent=ag-abcdefgh"]
    assert body["branch_action"] == "new" and body["reused"] is False
    assert os.path.isdir(record["path"])


def test_create_throwaway_validates(fakebin: Fakebin) -> None:
    body = create(fakebin, "--throwaway", "--ref", SHA, "--depth", "1", "--source", "local")
    record = body["record"]
    assert isinstance(record, dict)
    assert record["object_type"] == "throwaway" and record["ref"] == SHA and record["name"] is None
    assert body["bootstrap"] is None and body["branch_action"] is None


def test_names_are_made_unique_per_repo(fakebin: Fakebin) -> None:
    first = create(fakebin, "--name", "same")["record"]
    second = create(fakebin, "--name", "same")["record"]
    assert isinstance(first, dict) and isinstance(second, dict)
    assert (first["name"], second["name"]) == ("same", "same-2")


def test_retry_after_a_crash_returns_the_record_unchanged_for_a_holder(fakebin: Fakebin) -> None:
    first = create(fakebin, "--branch", "b", "--holder", "ag-1")
    again = create(fakebin, "--branch", "b", "--holder", "ag-1")
    assert again["reused"] is True and again["record"] == first["record"]
    assert again["branch_action"] is None and again["bootstrap"] is None
    code, error = error_of(
        fakebin, "wtctl", "create", "--repo", REPO, "--branch", "b", "--holder", "ag-2"
    )
    assert (code, error["code"], error["reason"]) == (20, "refused", "branch_in_use")
    assert error["holder_id"] == first["record"]["id"]  # pyright: ignore[reportIndexIssue]


@pytest.mark.parametrize(
    "argv",
    [
        pytest.param(["create"], id="no-repo"),
        pytest.param(["create", "--repo", "Bad Repo"], id="bad-repo"),
        pytest.param(
            ["create", "--repo", REPO, "--branch", "b", "--detach"], id="branch-and-detach"
        ),
        pytest.param(["create", "--repo", REPO, "--throwaway"], id="throwaway-without-ref"),
        pytest.param(["create", "--repo", REPO, "--throwaway", "--ref", "abc"], id="short-ref"),
        pytest.param(
            ["create", "--repo", REPO, "--throwaway", "--ref", SHA, "--branch", "b"],
            id="throwaway-branch",
        ),
        pytest.param(
            ["create", "--repo", REPO, "--source", "local"], id="source-without-throwaway"
        ),
        pytest.param(["create", "--repo", REPO, "--name", "-bad"], id="bad-name"),
        pytest.param(["create", "--repo", REPO, "--holder", "a b"], id="bad-holder"),
        pytest.param(["create", "--repo", REPO, "--hook", "pre-remove=true"], id="wrong-hook"),
        pytest.param(["create", "--repo", REPO, "--override-escalation"], id="human-only-flag"),
        pytest.param(["create", "--repo", REPO, "--actor", "x"], id="bad-actor"),
        pytest.param(["remove", "relative/path"], id="relative-path"),
        pytest.param(["show", "wt-ab"], id="short-id"),
        pytest.param(["remove"], id="remove-no-target"),
        pytest.param(["label", "add", "wt-abcdefgh"], id="label-without-pair"),
        pytest.param(["prune"], id="prune-without-throwaways"),
        pytest.param(["prune", "--throwaways"], id="prune-without-age"),
        pytest.param(["list", "--throwaways", "--worktrees"], id="list-both-kinds"),
        pytest.param(["frobnicate"], id="unknown-command"),
    ],
)
def test_usage_errors_exit_2_with_the_error_envelope(fakebin: Fakebin, argv: list[str]) -> None:
    done = fakebin.run("wtctl", *argv, "--json")
    assert done.returncode == 2, done.stdout
    body = json.loads(done.stdout)
    sibling_schemas.validate(body, "wtctl", "cli/error.json")
    assert body["error"]["code"] == "usage"


def test_list_show_and_addressing_by_prefix_and_path(fakebin: Fakebin) -> None:
    a = create(fakebin, "--branch", "a")["record"]
    b = create(fakebin, "--throwaway", "--ref", SHA)["record"]
    assert isinstance(a, dict) and isinstance(b, dict)
    _, listed = run_json(fakebin, "wtctl", "list", schema="cli/list.json")
    assert [r["id"] for r in listed["records"]] == [b["id"], a["id"]]  # newest first, by id on ties
    _, only = run_json(fakebin, "wtctl", "list", "--throwaways", schema="cli/list.json")
    assert [r["id"] for r in only["records"]] == [b["id"]]
    _, shown = run_json(fakebin, "wtctl", "show", str(a["path"]), schema="cli/show.json")
    assert shown["id"] == a["id"]
    code, error = error_of(fakebin, "wtctl", "show", str(a["id"])[3:7])
    assert (code, error["code"]) == (21, "ambiguous") and len(error["candidates"]) == 2
    assert error_of(fakebin, "wtctl", "show", "/no/such/path")[0] == 10


def test_label_add_remove_validate(fakebin: Fakebin) -> None:
    record = create(fakebin, "--label", "a=b")["record"]
    assert isinstance(record, dict)
    rid = str(record["id"])
    _, added = run_json(fakebin, "wtctl", "label", "add", rid, "c=d", schema="cli/label-add.json")
    assert added["labels"] == ["a=b", "c=d"]
    _, removed = run_json(
        fakebin, "wtctl", "label", "remove", rid, "a=b", schema="cli/label-remove.json"
    )
    assert removed["labels"] == ["c=d"]


def test_remove_rules_holders_confirmation_and_tombstones(fakebin: Fakebin) -> None:
    record = create(fakebin, "--branch", "r", "--holder", "ag-1")["record"]
    assert isinstance(record, dict)
    rid, path = str(record["id"]), str(record["path"])
    _, status = run_json(fakebin, "wtctl", "status", rid, schema="cli/status.json")
    assert status["would_remove"] is False and status["reason"] == "held"
    _, own = run_json(fakebin, "wtctl", "status", rid, "--holder", "ag-1", schema="cli/status.json")
    assert own["would_remove"] is True
    code, error = error_of(fakebin, "wtctl", "remove", rid, "--yes")
    assert (code, error["code"], error["reason"]) == (20, "refused", "held")
    assert error_of(fakebin, "wtctl", "remove", rid, "--holder", "ag-1", "--force")[0] == 2
    code, error = error_of(fakebin, "wtctl", "remove", rid, "--holder", "ag-1")  # no TTY, no --yes
    assert (code, error["code"]) == (13, "confirmation_required")
    assert os.path.isdir(path)
    _, dry = run_json(
        fakebin, "wtctl", "remove", rid, "--holder", "ag-1", "--dry-run", schema="cli/remove.json"
    )
    assert dry["dry_run"] is True and os.path.isdir(path)
    _, done = run_json(
        fakebin,
        "wtctl",
        "remove",
        rid,
        "--holder",
        "ag-1",
        "--yes",
        "--keep-branch",
        schema="cli/remove.json",
    )
    assert done["record"]["state"] == "removed" and done["branch_kept_reason"] == "keep_branch"
    assert not os.path.exists(path)
    # a full id finds the tombstone for show; every other command exits 10 with its details
    _, tomb = run_json(fakebin, "wtctl", "show", rid, schema="cli/show.json")
    assert tomb["state"] == "removed"
    code, error = error_of(fakebin, "wtctl", "remove", rid, "--yes")
    assert (code, error["code"], error["details"]["state"]) == (10, "not_found", "removed")
    # prefixes never match tombstones
    assert error_of(fakebin, "wtctl", "show", rid[3:7])[0] == 10


def test_forget_with_holder_is_idempotent_and_plain_forget_is_human_only(fakebin: Fakebin) -> None:
    record = create(fakebin, "--branch", "f", "--holder", "ag-1")["record"]
    assert isinstance(record, dict)
    rid = str(record["id"])
    _, gone = run_json(
        fakebin, "wtctl", "forget", rid, "--holder", "ag-1", schema="cli/forget.json"
    )
    assert gone["record"]["holders"] == [] and gone["warnings"] == []
    _, retry = run_json(
        fakebin, "wtctl", "forget", rid, "--holder", "ag-1", schema="cli/forget.json"
    )
    assert retry["warnings"] == ["not_a_holder"]
    code, error = error_of(fakebin, "wtctl", "forget", rid)
    assert (code, error["code"]) == (13, "confirmation_required")
    _, forgotten = run_json(
        fakebin, "wtctl", "forget", rid, "--actor", "human:me", schema="cli/forget.json"
    )
    assert forgotten["record"]["state"] == "forgotten"
    throwaway = create(fakebin, "--throwaway", "--ref", SHA)["record"]
    assert isinstance(throwaway, dict)
    assert error_of(fakebin, "wtctl", "forget", str(throwaway["id"]), "--holder", "x")[0] == 2


def test_locate_baseline_check_reconcile_prune_validate(fakebin: Fakebin) -> None:
    run_json(fakebin, "wtctl", "locate", REPO, schema="cli/locate.json")
    record = create(fakebin, "--branch", "bc")["record"]
    assert isinstance(record, dict)
    _, check = run_json(
        fakebin, "wtctl", "baseline", "check", str(record["id"]), schema="cli/baseline-check.json"
    )
    assert check["changed"] is False
    run_json(fakebin, "wtctl", "reconcile", schema="cli/reconcile.json")
    run_json(
        fakebin, "wtctl", "prune", "--throwaways", "--older-than", "7d", schema="cli/prune.json"
    )
    run_json(fakebin, "wtctl", "prune", "--throwaways", "--if-configured", schema="cli/prune.json")


def test_scripted_failure_changes_nothing(fakebin: Fakebin) -> None:
    fakebin.script(
        "wtctl", "create", times=1, error={"code": "busy", "message": "lease busy"}, exit=12
    )
    code, error = error_of(fakebin, "wtctl", "create", "--repo", REPO, "--branch", "x")
    assert (code, error["code"]) == (12, "busy")
    _, listed = run_json(fakebin, "wtctl", "list", schema="cli/list.json")
    assert listed["records"] == []
    create(fakebin, "--branch", "x")  # the rule had one use


def test_merge_overrides_canned_fields_and_stays_schema_valid(fakebin: Fakebin) -> None:
    fakebin.script(
        "wtctl", "create", merge={"record": {"path": "/elsewhere/wt"}, "warnings": ["w"]}
    )
    body = create(fakebin, "--branch", "m")
    assert body["record"]["path"] == "/elsewhere/wt" and body["warnings"] == ["w"]  # pyright: ignore[reportIndexIssue]


def test_version_and_unsupported(fakebin: Fakebin) -> None:
    assert fakebin.run("wtctl", "--version").stdout == "wtctl 0.1.0\n"
    done = fakebin.run("wtctl", "adopt", "/x", "--json")
    assert done.returncode == 1 and json.loads(done.stdout)["error"]["code"] == "fake_unsupported"
