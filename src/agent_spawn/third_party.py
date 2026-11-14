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
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from .output import CliError

SESSION_DIR = "session-dir"

_KIND = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}")
_REASON = re.compile(r"[a-z][a-z0-9_]{0,63}")
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


def parse_plan(plan: object, *, requested_cwd: str | None = None) -> Classification:
    """Read the classification out of a dry-run plan.

    The run is third-party-reading when ``profile.reads_third_party`` or ``taints`` is true.
    ``requested_cwd`` is the ``--cwd`` the dry run was given; without one the run's cwd is the
    kind's default, which is ``session-dir`` when ``profile.cwd`` says so, else the plan's cwd.
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
    if not isinstance(run_kind, str) or _KIND.fullmatch(run_kind) is None:
        raise PlanError("the plan has no valid run kind")
    refuse = _text(plan.get("would_refuse"), "would_refuse", 64)
    fail = _text(plan.get("would_fail"), "would_fail", 64)
    repo = _text(plan.get("repo"), "repo", _REPO_MAX)
    plan_cwd = _text(plan.get("cwd"), "cwd", _CWD_MAX)
    cwd: str | None
    if requested_cwd is not None:
        cwd = requested_cwd
    elif profile.get("cwd") == SESSION_DIR:
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
    if _REASON.fullmatch(reason) is None:
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
) -> bool:
    """Whether ``cwd`` satisfies the third-party rule for a run of this class.

    wtctl's records are not consulted here (no throwaways or worktrees exist yet): a
    third-party-reading run must run in its own session dir (``session-dir``, or the realpath
    of its ``<csession_dir>/work``), and a trusted run is refused only inside a throwaway, of
    which there is none without wtctl.
    TODO(#20): the records of the cwd (throwaway roots, worktree containment) join this rule.
    """
    if not reads_third_party:
        return True
    if cwd is None:
        return False
    if cwd == SESSION_DIR:
        return True
    return session_work is not None and _real(cwd) == _real(session_work)


def require_cwd_rule(
    reads_third_party: bool,
    cwd: str | None,
    *,
    session_work: str | None = None,
) -> None:
    """Raise the exit-20 ``third_party_cwd`` refusal when the rule forbids the cwd."""
    if cwd_rule_allows(reads_third_party, cwd, session_work=session_work):
        return
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
    rule_allows: Callable[[bool], bool],
    result_file_allows: Callable[[], bool] = lambda: True,
) -> CheckDecision:
    """Apply interface.md, Classification check.

    ``actual`` is None when ``status`` failed or could not be read: cancel as unknown.
    ``rule_allows(c)`` says whether the run's cwd satisfies the cwd rule for class ``c``;
    ``result_file_allows()`` runs the result-file check for an actual third-party-reading run.
    """
    if actual is None:
        return CheckDecision("cancel", "classification_unknown", expected, None)
    if actual == expected:
        return CheckDecision("accept", None, expected, actual)
    if rule_allows(actual) and (not actual or result_file_allows()):
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
