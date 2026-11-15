"""Classification, the cwd rule and the post-start decision (pure; no subprocess)."""

from __future__ import annotations

from typing import Any

import pytest

from agent_spawn.output import CliError
from agent_spawn.third_party import (
    SESSION_DIR,
    Classification,
    PlanError,
    check_error,
    cwd_rule_allows,
    decide_check,
    parse_plan,
    refusal,
    require_cwd_rule,
)


def plan(**changes: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "dry_run": True,
        "run_kind": "implement",
        "taints": False,
        "repo": None,
        "cwd": "/somewhere",
        "profile": {"reads_third_party": False, "cwd": "repo"},
        "would_refuse": None,
        "would_fail": None,
    }
    return {**base, **changes}


def test_a_trusted_plan() -> None:
    result = parse_plan(plan())
    assert (result.reads_third_party, result.run_kind, result.classification) == (
        False,
        "implement",
        "trusted",
    )


def test_profile_flag_or_taint_makes_a_run_third_party_reading() -> None:
    assert parse_plan(plan(profile={"reads_third_party": True})).reads_third_party
    assert parse_plan(plan(taints=True)).reads_third_party


@pytest.mark.parametrize(
    "bad",
    [
        None,
        [],
        "plan",
        plan(dry_run=False),
        plan(dry_run="true"),
        plan(profile=None),
        plan(profile={}),
        plan(profile={"reads_third_party": "false"}),
        plan(profile={"reads_third_party": 0}),
        plan(taints=None),
        plan(taints=1),
        plan(taints="no"),
        plan(run_kind=None),
        plan(run_kind="Bad Kind"),
        plan(run_kind="x" * 65),
        plan(run_kind="../x"),
        plan(would_refuse=5),
        plan(would_fail=["x"]),
        plan(repo=5),
        plan(repo="r" * 5000),
        plan(cwd="a\0b"),
        plan(cwd=["x"]),
    ],
)
def test_an_unusable_plan_fails_closed(bad: object) -> None:
    with pytest.raises(PlanError):
        parse_plan(bad)


def test_missing_taints_is_unusable_not_false() -> None:
    value = plan()
    del value["taints"]
    with pytest.raises(PlanError):
        parse_plan(value)


def test_cwd_is_the_requested_one_else_the_kinds_default() -> None:
    assert parse_plan(plan(), requested_cwd="/x").cwd == "/x"
    assert parse_plan(plan(profile={"reads_third_party": True, "cwd": "session-dir"})).cwd == (
        SESSION_DIR
    )
    assert parse_plan(plan()).cwd == "/somewhere"
    assert parse_plan(plan(cwd=None)).cwd is None


def test_repo_comes_from_the_plan() -> None:
    assert parse_plan(plan(repo="host/o/r")).repo == "host/o/r"


def test_would_refuse_is_an_exit_20_refusal_naming_claude_session() -> None:
    error = refusal(parse_plan(plan(would_refuse="tainted")))
    assert error is not None
    assert (error.exit_code, error.code, error.reason, error.tool) == (
        20,
        "refused",
        "tainted",
        "claude-session",
    )
    assert refusal(parse_plan(plan())) is None


def test_a_would_refuse_that_is_not_a_reason_becomes_unknown() -> None:
    error = refusal(parse_plan(plan(would_refuse="Not A Reason!\n")))
    assert error is not None and error.reason == "unknown"


def test_would_fail_alone_does_not_refuse() -> None:
    assert refusal(parse_plan(plan(would_fail="claude_not_found"))) is None


# --- the cwd rule -------------------------------------------------------------------------------


def test_a_third_party_run_runs_only_in_its_session_dir(tmp_path: Any) -> None:
    work = tmp_path / "work"
    work.mkdir()
    assert cwd_rule_allows(True, SESSION_DIR)
    assert cwd_rule_allows(True, str(work), session_work=str(work))
    assert cwd_rule_allows(True, str(work / ".." / "work"), session_work=str(work))
    assert not cwd_rule_allows(True, str(tmp_path))
    assert not cwd_rule_allows(True, str(tmp_path), session_work=str(work))
    assert not cwd_rule_allows(True, None)
    assert not cwd_rule_allows(True, "")


def test_a_symlink_to_the_session_work_dir_counts_by_realpath(tmp_path: Any) -> None:
    work = tmp_path / "work"
    work.mkdir()
    link = tmp_path / "link"
    link.symlink_to(work)
    assert cwd_rule_allows(True, str(link), session_work=str(work))


def test_a_trusted_run_has_no_throwaway_to_be_refused_in_without_wtctl() -> None:
    assert cwd_rule_allows(False, "/any/where")
    assert cwd_rule_allows(False, None)


def test_require_cwd_rule_refuses_with_third_party_cwd() -> None:
    with pytest.raises(CliError) as caught:
        require_cwd_rule(True, "/repo")
    error = caught.value
    assert (error.exit_code, error.code, error.reason, error.tool) == (
        20,
        "refused",
        "third_party_cwd",
        "agent-spawn",
    )
    require_cwd_rule(True, SESSION_DIR)
    require_cwd_rule(False, "/repo")


# --- the post-start decision --------------------------------------------------------------------


def allows(value: bool):
    return lambda _actual: value


def test_agreement_is_accepted_whatever_the_rule_says() -> None:
    for expected in (True, False):
        decision = decide_check(expected, expected, rule_allows=allows(False))
        assert (decision.verdict, decision.cause) == ("accept", None)


def test_a_mismatch_the_rule_allows_continues_with_a_warning() -> None:
    decision = decide_check(False, True, rule_allows=allows(True))
    assert (decision.verdict, decision.cause) == ("warn", None)
    decision = decide_check(True, False, rule_allows=allows(True))
    assert decision.verdict == "warn"


def test_a_mismatch_the_rule_forbids_is_cancelled() -> None:
    decision = decide_check(False, True, rule_allows=allows(False))
    assert (decision.verdict, decision.cause) == ("cancel", "classification_mismatch")


def test_the_rule_is_asked_about_the_actual_class() -> None:
    asked: list[bool] = []

    def rule(actual: bool) -> bool:
        asked.append(actual)
        return True

    decide_check(False, True, rule_allows=rule)
    decide_check(True, False, rule_allows=rule)
    assert asked == [True, False]


def test_an_actual_third_party_run_also_needs_its_result_file_check() -> None:
    calls: list[str] = []

    def result_file() -> bool:
        calls.append("checked")
        return False

    decision = decide_check(False, True, rule_allows=allows(True), result_file_allows=result_file)
    assert (decision.verdict, decision.cause) == ("cancel", "classification_mismatch")
    assert calls == ["checked"]


def test_the_result_file_check_is_not_run_for_an_actual_trusted_run() -> None:
    def never() -> bool:
        raise AssertionError("not for a trusted run")

    assert decide_check(
        True, False, rule_allows=allows(True), result_file_allows=never
    ).verdict == ("warn")


def test_an_unreadable_status_is_cancelled_as_unknown() -> None:
    decision = decide_check(False, None, rule_allows=allows(True))
    assert (decision.verdict, decision.cause) == ("cancel", "classification_unknown")


def test_the_error_of_a_cancelled_check_is_launch_failed_with_its_cause() -> None:
    for cause, actual in (("classification_mismatch", True), ("classification_unknown", None)):
        error = check_error(
            decide_check(False, actual, rule_allows=allows(False)), agent_id="ag-aaaaaaaa"
        )
        assert (error.exit_code, error.code, error.tool, error.agent_id) == (
            12,
            "launch_failed",
            "agent-spawn",
            "ag-aaaaaaaa",
        )
        assert error.details == {"cause": cause}


def test_only_a_cancelled_check_has_an_error() -> None:
    with pytest.raises(ValueError):
        check_error(decide_check(False, False, rule_allows=allows(True)), agent_id=None)


def test_classification_is_a_value_object() -> None:
    value = Classification(True, "review", None, None, None, None)
    assert value.classification == "third_party"
