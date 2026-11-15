"""Third-party runs: the classification of a run and the rules that follow from it.

claude-session is the one source of truth (design D39): before every ``start`` agent-spawn
runs the same command with ``--dry-run --json`` and reads the plan; after ``start`` it compares
the plan with ``claude-session status``. This module is pure (no subprocess): it parses a plan,
applies the cwd rule and decides the post-start check. ``session_backend`` does the calls.

Everything parsed here is claude-session's output and therefore untrusted: a field of the wrong
type is an unusable plan (fail closed), never a guess.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

from .output import CliError
from .validate import KIND
from .worktrees import WtRecord, inside_throwaway, is_throwaway_root

SESSION_DIR = "session-dir"

#: The reasons of the pinned dry-run schema (``would_refuse``); anything else is "unknown".
WOULD_REFUSE_REASONS = frozenset({"live", "tainted", "already_adopted", "expired"})
_REPO_MAX = 512
_CWD_MAX = 4096


class PlanError(ValueError):
    """The dry-run output is not a usable plan."""


@dataclass(frozen=True)
class Classification:
    """What a dry run says about a run (interface.md, Third-party runs, Classification)."""

    reads_third_party: bool
    run_kind: str
    repo: str | None
    cwd: str | None  # "session-dir", or the plan's cwd; None when neither is known
    would_refuse: str | None
    would_fail: str | None

    @property
    def classification(self) -> Literal["third_party", "trusted"]:
        return "third_party" if self.reads_third_party else "trusted"


def _text(value: object, name: str, limit: int) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or len(value) > limit or "\0" in value:
        raise PlanError(f"plan field {name} is not a short string")
    return value


PLACEHOLDER_SESSION = "cs-aaaaaaaa"


def _is_new_session_placeholder(
    plan: dict[str, object], profile: dict[str, object], path: str
) -> bool:
    """Whether ``path`` is the ``.../sessions/cs-aaaaaaaa/work`` a new-session dry run shows.

    Exact, not a pattern: the plan has no ``csession_id`` (a new session or fork), the kind's
    default cwd is ``session-dir``, and the normalized path ends ``sessions/cs-aaaaaaaa/work``.
    A repo path that merely looks like a session work directory is not this.
    """
    if plan.get("csession_id") is not None or profile.get("cwd") != SESSION_DIR:
        return False
    if not os.path.isabs(path):
        return False
    parts = os.path.normpath(path).split(os.sep)
    return parts[-3:] == ["sessions", PLACEHOLDER_SESSION, "work"]


def parse_plan(plan: object, *, requested_cwd: str | None = None) -> Classification:
    """Read the classification out of a dry-run plan.

    The run is third-party-reading when ``profile.reads_third_party`` or ``taints`` is true.
    ``requested_cwd`` is the ``--cwd`` the dry run was given; without one the run's cwd is the
    plan's cwd, mapped to ``session-dir`` only when that is exactly the new-session placeholder
    the dry run emits (see ``_is_new_session_placeholder``). The plan carries no
    ``csession_dir`` (only ``csession_id``), so the work directory of a resumed session cannot
    be recognized here: it keeps its real path, and the cwd rule recognizes it through the
    caller's ``session_work`` (or fails closed). Any other path, including a repo path that looks
    like ``.../cs-xxxxxxxx/work``, stays the real path.
    """
    if not isinstance(plan, dict):
        raise PlanError("the plan is not an object")
    if plan.get("dry_run") is not True:
        raise PlanError("the output is not a dry-run plan")
    profile = plan.get("profile")
    if not isinstance(profile, dict):
        raise PlanError("the plan has no profile")
    reads = profile.get("reads_third_party")
    taints = plan.get("taints")
    if not isinstance(reads, bool) or not isinstance(taints, bool):
        raise PlanError("the plan does not say whether the run reads third-party content")
    run_kind = plan.get("run_kind")
    if not isinstance(run_kind, str) or KIND.fullmatch(run_kind) is None:
        raise PlanError("the plan has no valid run kind")
    refuse = _text(plan.get("would_refuse"), "would_refuse", 64)
    fail = _text(plan.get("would_fail"), "would_fail", 64)
    repo = _text(plan.get("repo"), "repo", _REPO_MAX)
    plan_cwd = _text(plan.get("cwd"), "cwd", _CWD_MAX)
    cwd: str | None
    if requested_cwd is not None:
        cwd = requested_cwd
    elif plan_cwd is not None and _is_new_session_placeholder(plan, profile, plan_cwd):
        cwd = SESSION_DIR
    else:
        cwd = plan_cwd
    return Classification(
        reads_third_party=reads or taints,
        run_kind=run_kind,
        repo=repo,
        cwd=cwd,
        would_refuse=refuse,
        would_fail=fail,
    )


def refusal(classification: Classification) -> CliError | None:
    """The exit-20 refusal for a plan with ``would_refuse`` set (``tool: claude-session``)."""
    reason = classification.would_refuse
    if reason is None:
        return None
    if not isinstance(reason, str) or reason not in WOULD_REFUSE_REASONS:
        reason = "unknown"
    return CliError(
        "refused",
        f"claude-session would refuse the run: {reason}",
        reason=reason,
        tool="claude-session",
        lower_code="refused",
        lower_error={"code": "refused", "reason": reason},
    )


# --- the cwd rule -------------------------------------------------------------------------------


def _real(path: str) -> str:
    return os.path.realpath(path)


def cwd_rule_allows(
    reads_third_party: bool,
    cwd: str | None,
    *,
    session_work: str | None = None,
    records: Sequence[WtRecord] | None = None,
) -> bool:
    """Whether ``cwd`` satisfies the third-party rule for a run of this class.

    ``records`` are the live wtctl records (one ``wtctl list``; ``[]`` when config
    ``wtctl_cmd`` is null). A third-party-reading run must run in its own session dir
    (``session-dir``, or the realpath of its ``<csession_dir>/work``) or at the root of a live
    throwaway record that lies inside no worktree record; a trusted run must not run inside any
    throwaway. Without ``records`` (a caller that has not looked them up) a trusted run is not
    judged and a third-party-reading run outside its session dir fails closed.
    """
    if not reads_third_party:
        if records is None or cwd is None or cwd == SESSION_DIR:
            return True
        try:
            return not inside_throwaway(records, cwd)
        except (ValueError, OSError):  # a NUL or another path the OS refuses
            return False
    if cwd is None:
        return False
    if cwd == SESSION_DIR:
        return True
    try:
        if session_work is not None and _real(cwd) == _real(session_work):
            return True
        return records is not None and is_throwaway_root(records, cwd)
    except (ValueError, OSError):
        return False


def require_cwd_rule(
    reads_third_party: bool,
    cwd: str | None,
    *,
    session_work: str | None = None,
    records: Sequence[WtRecord] | None = None,
) -> None:
    """Raise the exit-20 refusal when the rule forbids the cwd: ``third_party_cwd`` for a
    third-party-reading run, ``third_party_run`` for a trusted run inside a throwaway."""
    if cwd_rule_allows(reads_third_party, cwd, session_work=session_work, records=records):
        return
    if not reads_third_party:
        raise CliError(
            "refused",
            "throwaways are for third-party-reading runs only",
            reason="third_party_run",
        )
    raise CliError(
        "refused",
        "third-party-reading runs run only in their session dir or at a throwaway's root",
        reason="third_party_cwd",
    )


# --- the post-start classification check --------------------------------------------------------

Verdict = Literal["accept", "warn", "cancel"]
CheckCause = Literal["classification_mismatch", "classification_unknown"]


@dataclass(frozen=True)
class CheckDecision:
    """The outcome of comparing the dry-run classification with ``claude-session status``."""

    verdict: Verdict
    cause: CheckCause | None
    expected: bool
    actual: bool | None


def decide_check(
    expected: bool,
    actual: bool | None,
    *,
    cwd: str | None,
    rule_allows: Callable[[bool, str | None], bool],
    result_file_allows: Callable[[], bool] = lambda: True,
) -> CheckDecision:
    """Apply interface.md, Classification check.

    ``actual`` is None when ``status`` failed or could not be read: cancel as unknown.
    ``cwd`` is the run's actual cwd (``status`` ``run.cwd``), null while the run is pending
    (claude-session schemas/cli/status.json, docs/interface.md status). When the classes agree
    no cwd is needed (the cwd rule was enforced before start); on a mismatch a third-party
    actual run with no string cwd is cancelled as unknown, and a string cwd is always checked
    with ``rule_allows``. ``rule_allows(c, cwd)`` says whether ``cwd``
    satisfies the cwd rule for class ``c`` (supplied by the caller: #11, #13, #14, which own
    the records of the cwd); ``result_file_allows()`` runs the result-file check for an actual
    third-party-reading run. A callback that raises cancels as unknown (fail closed).
    """
    if actual is None:
        return CheckDecision("cancel", "classification_unknown", expected, None)
    if actual == expected:
        if isinstance(cwd, str):
            try:
                ok = rule_allows(actual, cwd)
            except Exception:  # noqa: BLE001 - any failure of a caller's check fails closed
                return CheckDecision("cancel", "classification_unknown", expected, actual)
            if not ok:
                return CheckDecision("cancel", "classification_mismatch", expected, actual)
        return CheckDecision("accept", None, expected, actual)
    if actual and not isinstance(cwd, str):
        return CheckDecision("cancel", "classification_unknown", expected, actual)
    try:
        allowed = rule_allows(actual, cwd) and (not actual or result_file_allows())
    except Exception:  # noqa: BLE001 - any failure of a caller's check fails closed
        return CheckDecision("cancel", "classification_unknown", expected, actual)
    if allowed:
        return CheckDecision("warn", None, expected, actual)
    return CheckDecision("cancel", "classification_mismatch", expected, actual)


def check_error(decision: CheckDecision, *, agent_id: str | None) -> CliError:
    """The exit-12 ``launch_failed`` error for a cancelled run (``details.cause`` set)."""
    cause = decision.cause
    if cause is None:
        raise ValueError("only a cancelled check has an error")
    message = (
        "the run's third-party classification changed after start and its cwd is not allowed"
        if cause == "classification_mismatch"
        else "the run's third-party classification could not be confirmed after start"
    )
    return CliError("launch_failed", message, agent_id=agent_id, details={"cause": cause})
