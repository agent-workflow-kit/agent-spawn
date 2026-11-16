"""The agent and outcome objects, and the outbox events of a status change.

``agent_object`` renders a stored agent row as ``schemas/defs.json#/$defs/agent``;
``outcome_object`` renders a final run as ``$defs/outcome``. ``lifecycle_events`` builds the
outbox events (``post-park``, ``post-escalate``, ``post-launch``, ``post-state-change``) that are
written in the same transaction as the change they describe (interface.md, Hook delivery).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

from .hooks import agent_event
from .ids import is_agent_id
from .output import CliError
from .records import format_ts
from .session_backend import RunOutcome
from .store import OutboxEvent, Store

#: What rendering a corrupt stored row can raise (an out-of-range or non-numeric time, ...).
CORRUPT_ROW = (OverflowError, TypeError, ValueError, OSError)


def unusable_row(agent_id: str | None = None) -> CliError:
    """The expected failure for a row that cannot be rendered (exit 12, never a traceback)."""
    known = agent_id if agent_id is not None and is_agent_id(agent_id) else None
    return CliError("failed", "state.db row unusable", agent_id=known, exit_code=12)


def ts(moment: float | None) -> str | None:
    """A stored epoch time as the schemas' ``date-time``; ``None`` stays ``None``."""
    if moment is None:
        return None
    return format_ts(datetime.fromtimestamp(moment, UTC))


def _outcome_value(last: object) -> str | None:
    if isinstance(last, dict):
        value = last.get("outcome")  # pyright: ignore[reportUnknownMemberType]
        return value if isinstance(value, str) else None
    return None


def _ref(record_id: object, path: object, owned: object) -> dict[str, Any] | None:
    if not isinstance(record_id, str) or not isinstance(path, str):
        return None
    return {"record_id": record_id, "path": path, "owned": bool(owned)}


def agent_object(
    store: Store, row: Mapping[str, Any], *, warnings: Sequence[str] = ()
) -> dict[str, Any]:
    """The agent object of a stored row (labels, holds and the current run come from ``store``)."""
    agent_id = row["agent_id"]
    run_number = row.get("run")
    run_row = store.get_run(agent_id, run_number) if isinstance(run_number, int) else None
    window = None
    if run_row is not None and run_row.get("term_id") and run_row.get("socket_path"):
        window = {"term_id": run_row["term_id"], "socket_path": run_row["socket_path"]}
    holds = [
        ref
        for hold in store.get_holds(agent_id)
        if (ref := _ref(hold["record_id"], hold["path"], hold["owned"])) is not None
    ]
    mode = row.get("mode")
    placement = row.get("placement") if mode == "interactive" else None
    return {
        "agent_id": agent_id,
        "status": row["status"],
        "needs_attention": row.get("needs_attention"),
        "park_reason": row.get("park_reason"),
        "kind": row.get("launch_kind"),
        "run_kind": row.get("run_kind"),
        "escalated_by": row.get("escalated_by"),
        "mode": mode,
        "placement": placement,
        "term_placement": row.get("term_placement"),
        "name": row["name"],
        "repo": row.get("repo"),
        "cwd": row.get("cwd"),
        "labels": store.get_labels(agent_id).texts(),
        "csession_id": row.get("csession_id"),
        "run": run_number,
        "session_state": row.get("session_state"),
        "activity": row.get("activity"),
        "needs_input_reason": row.get("needs_input_reason"),
        "forked_from": row.get("forked_from"),
        "worktree": _ref(row.get("wt_record_id"), row.get("wt_path"), row.get("wt_owned")),
        "holds": holds,
        "window": window,
        "promote": row.get("promote"),
        "promotion_pending": bool(row.get("promotion_pending")),
        "ephemeral": bool(row.get("ephemeral")),
        "last_outcome": _outcome_value(row.get("last_outcome")),
        "last_end_reason": row.get("last_end_reason"),
        "tainted": row.get("tainted"),
        "supervisor_lost": bool(row.get("supervisor_lost")),
        "teardown_step": row.get("teardown_step"),
        "created": ts(row["created"]),
        "last_live_at": ts(row.get("last_live_at")),
        "updated": ts(row["updated"]),
        "torn_down_at": ts(row.get("torn_down_at")),
        "warnings": list(warnings),
    }


def chain_first_working_at(
    store: Store, row: Mapping[str, Any], run: int | None = None
) -> str | None:
    """``first_working_at`` of the run chain's first run (launch, resume and wait results).

    An automatic report resume belongs to the run before it.
    """
    number = run if run is not None else row.get("run")
    if not isinstance(number, int) or number < 1:
        return None
    run_row = store.get_run(row["agent_id"], number)
    while run_row is not None and run_row.get("auto_resumed") and number > 1:
        earlier = store.get_run(row["agent_id"], number - 1)
        if earlier is None:
            break
        number, run_row = number - 1, earlier
    return ts(run_row.get("first_working_at")) if run_row else None


def _text_or_none(value: object) -> str | None:
    return value if isinstance(value, str) else None


_OWN_STATE = ("killed", "timeout", "lost", "exited-unreported", "not-started")


def _state_of(outcome: str) -> str:
    """The final state of an outcome (claude-session interface, Outcome mapping)."""
    if outcome in ("vetoed", "escalated"):
        return "not-started"
    return outcome if outcome in _OWN_STATE else "exited"


def outcome_object(
    agent_id: str,
    outcome: RunOutcome,
    extra: Mapping[str, Any] | None = None,
    *,
    promoted: bool = False,
    auto_resumed: bool = False,
    first_working_at: str | None = None,
) -> dict[str, Any]:
    """The outcome object of a final run: claude-session's fields plus agent-spawn's.

    ``extra`` is the raw outcome file, or the run object of ``status``; the fields
    ``outcome`` does not carry (``claude_session_id``, ``cost_usd``, ``message``,
    ``session_started_at``, ``warnings``, ``forked_from``) are taken from it when well-formed.
    """
    if outcome.outcome is None or outcome.exit_code is None:
        raise ValueError("only a final run has an outcome object")
    raw: Mapping[str, Any] = extra or {}
    cost = raw.get("cost_usd")
    warnings = raw.get("warnings")
    return {
        "agent_id": agent_id,
        "csession_id": outcome.csession_id,
        "run": outcome.run,
        "forked_from": _text_or_none(raw.get("forked_from")),
        "claude_session_id": _text_or_none(raw.get("claude_session_id")),
        "phase": outcome.phase,
        "outcome": outcome.outcome,
        "end_reason": outcome.end_reason,
        "exit_code": outcome.exit_code,
        "state": _state_of(outcome.outcome),
        "report": outcome.report,
        "session_started_at": _text_or_none(raw.get("session_started_at")),
        "cost_usd": cost if isinstance(cost, int | float) and not isinstance(cost, bool) else None,
        "message": _text_or_none(raw.get("message")),
        "warnings": [w for w in warnings if isinstance(w, str)]
        if isinstance(warnings, list)
        else [],
        "promoted": promoted,
        "auto_resumed": auto_resumed,
        "first_working_at": first_working_at or outcome.first_working_at,
    }


def lifecycle_events(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    *,
    reads_third_party: bool | None,
    outcome: Mapping[str, Any] | None = None,
    chain_failure: Mapping[str, Any] | None = None,
    started_run: int | None = None,
    resumed: bool = False,
    fork: bool = False,
    auto_resumed: bool = False,
    promoted: bool = False,
    teardown_refused: bool = False,
) -> list[OutboxEvent]:
    """The outbox events of a status or attention change from ``before`` to ``after`` (agent
    objects): ``post-launch`` when ``started_run`` is given, ``post-park`` when the agent became
    parked, ``post-escalate`` when ``needs_attention`` changed to a non-null value, and
    ``post-state-change`` (``change: agent``) when the status or attention changed."""
    events: list[OutboxEvent] = []
    if started_run is not None:
        events.append(
            agent_event(
                "post-launch",
                after,
                reads_third_party=reads_third_party,
                extra={
                    "run": started_run,
                    "resumed": resumed,
                    "fork": fork,
                    "promoted": promoted,
                    "auto_resumed": auto_resumed,
                },
            )
        )
    if after["status"] == "parked" and before["status"] != "parked":
        events.append(
            agent_event(
                "post-park",
                after,
                reads_third_party=reads_third_party,
                extra={
                    "reason": after["park_reason"],
                    "outcome": None if outcome is None else dict(outcome),
                    "teardown_refused": teardown_refused,
                    "chain_failure": None if chain_failure is None else dict(chain_failure),
                },
            )
        )
    attention = after["needs_attention"]
    if attention is not None and attention != before["needs_attention"]:
        events.append(
            agent_event(
                "post-escalate",
                after,
                reads_third_party=reads_third_party,
                extra={
                    "outcome": None if outcome is None else dict(outcome),
                    "chain_failure": None if chain_failure is None else dict(chain_failure),
                },
            )
        )
    if (after["status"], attention) != (before["status"], before["needs_attention"]):
        events.append(
            agent_event(
                "post-state-change",
                after,
                reads_third_party=reads_third_party,
                extra={
                    "change": "agent",
                    "previous_status": before["status"],
                    "previous_needs_attention": before["needs_attention"],
                },
            )
        )
    return events
