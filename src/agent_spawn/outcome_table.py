"""The outcome table (interface.md, Outcome table): how a final run parks its agent."""

from __future__ import annotations

from dataclasses import dataclass

#: End reasons of a deliberate stop or a forwarded signal: the park carries no attention.
DELIBERATE_END_REASONS = frozenset({"stopped", "forwarded_term", "forwarded_hup", "forwarded_int"})
#: ``not-started`` end reasons of a deliberate cancel.
CANCEL_END_REASONS = frozenset({"canceled", "launch_interrupted"})
#: Outcomes a run can end with in its launch phase (a supervisor-initiated run that does so
#: stops the chain early).
LAUNCH_PHASE_OUTCOMES = frozenset({"not-started", "vetoed", "escalated"})


@dataclass(frozen=True)
class ParkRow:
    """One row: the agent's ``park_reason``, ``needs_attention`` and ``escalated_by``."""

    park_reason: str
    needs_attention: str | None
    escalated_by: str | None = None


def park_row(outcome: str, end_reason: str | None) -> ParkRow:
    """The park fields for a run that ended with ``outcome`` and ``end_reason``.

    An outcome the table does not know (a newer claude-session) parks ``failed`` with
    attention rather than guessing success.
    """
    if outcome == "done":
        return ParkRow("done", None)
    if outcome == "blocked":
        return ParkRow("blocked", "blocked")
    if outcome == "needs-input":
        return ParkRow("escalated", "escalated")
    if outcome == "failed":
        return ParkRow("failed", "failed")
    if outcome == "exited-unreported":
        # A headless run got one automatic report resume before this row applied (supervise.py).
        return ParkRow("unreported", "unreported")
    if outcome == "killed":
        if end_reason in DELIBERATE_END_REASONS:
            return ParkRow("killed", None)
        return ParkRow("killed", "failed")
    if outcome == "timeout":
        return ParkRow("timeout", "timeout")
    if outcome == "lost":
        return ParkRow("lost", "lost")
    if outcome == "not-started":
        if end_reason in CANCEL_END_REASONS:
            return ParkRow("killed", None)
        return ParkRow("launch_failed", "launch_failed")
    if outcome == "vetoed":
        return ParkRow("vetoed", None)
    if outcome == "escalated":
        return ParkRow("escalated", "escalated", "claude-session")
    return ParkRow("failed", "failed")
