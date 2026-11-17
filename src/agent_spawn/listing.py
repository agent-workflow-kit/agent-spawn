"""``list``: agents filtered and ordered from the state database alone.

The session fields come from the relay's records; claude-session is never called and no hook is
delivered. A missing database is an empty list (nothing is created).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from .agent_lookup import (
    liveness_for,
    open_existing,
    resolve_existing,
    state_dir_of,
    store_failed,
)
from .agent_view import CORRUPT_ROW, agent_object, unusable_row
from .commands.base import Invocation
from .ids import ID_CHARS, check_agent_id_text
from .labels import Label, parse_user_label
from .output import CliError, Outcome, PayloadKind, safe_text
from .records import RecordsError, append_log
from .session_backend import STATES
from .store import Store, StoreError
from .store_schema import STATUSES
from .validate import GrammarError, parse_since

__all__ = ["list_agents"]


def _log_unreadable(inv: Invocation, agent_id: str, error: CliError) -> None:
    """A row whose process entry cannot be read stays as it is in the listing; the log says so."""
    try:
        append_log(
            state_dir_of(inv),
            "warning",
            level="warn",
            agent_id=agent_id,
            message="liveness_unreadable",
            details={"cause": "liveness_unreadable", "reason": safe_text(error.message)},
        )
    except (RecordsError, ValueError, OSError):
        pass  # a log that cannot be written never fails a listing


def _check_choices(values: list[str], allowed: tuple[str, ...] | frozenset[str], what: str) -> None:
    for value in values:
        if value not in allowed:
            raise CliError.usage(
                f"unknown {what} {safe_text(value)!r}: expected one of "
                + ", ".join(sorted(allowed))
            )


def human_line(agent: dict[str, Any]) -> str:
    attention = agent["needs_attention"] or "-"
    state = agent["session_state"] or "-"
    return "  ".join(
        [
            agent["agent_id"],
            agent["status"],
            state,
            attention,
            safe_text(str(agent["kind"] or "-")),
            safe_text(agent["name"]),
        ]
    )


def _matches(row: dict[str, Any], ids: set[str] | None, wanted: dict[str, Any]) -> bool:
    if ids is not None and row["agent_id"] not in ids:
        return False
    statuses: list[str] = wanted["statuses"]
    if statuses:
        if row["status"] not in statuses:
            return False
    elif row["status"] == "torn_down":
        return False
    if wanted["needs_attention"] and row.get("needs_attention") is None:
        return False
    states: list[str] = wanted["session_states"]
    if states and row.get("session_state") not in states:
        return False
    kind = wanted["kind"]
    if kind is not None and kind not in (row.get("launch_kind"), row.get("run_kind")):
        return False
    if wanted["repo"] is not None and row.get("repo") != wanted["repo"]:
        return False
    if wanted["mode"] is not None and row.get("mode") != wanted["mode"]:
        return False
    if wanted["forked_from"] is not None and row.get("forked_from") != wanted["forked_from"]:
        return False
    since: datetime | None = wanted["since"]
    return since is None or datetime.fromtimestamp(row["updated"], UTC) >= since


def list_agents(inv: Invocation) -> Outcome:
    args = inv.args
    statuses: list[str] = list(args.status or [])
    session_states: list[str] = list(args.session_state or [])
    _check_choices(statuses, STATUSES, "status")
    _check_choices(session_states, tuple(STATES), "session state")
    labels: list[Label] = []
    for text in args.label or []:
        try:
            labels.append(parse_user_label(text))
        except GrammarError as exc:
            raise CliError.usage(str(exc)) from None
    since: datetime | None = None
    if args.since is not None:
        try:
            since = parse_since(args.since, datetime.now(UTC))
        except GrammarError as exc:
            raise CliError.usage(str(exc)) from None

    store = open_existing(inv)
    agents: list[dict[str, Any]] = []
    if store is not None:
        try:
            with store:
                agents = _select(inv, store, args, statuses, session_states, labels, since)
        except StoreError as error:
            raise store_failed(error) from None
    human = "\n".join(human_line(agent) for agent in agents) if agents else "no agents"
    return Outcome(0, {"agents": agents}, PayloadKind.COMMAND, human)


def _select(
    inv: Invocation,
    store: Store,
    args: Any,
    statuses: list[str],
    session_states: list[str],
    labels: list[Label],
    since: datetime | None,
) -> list[dict[str, Any]]:
    forked_from = None
    if args.forked_from is not None:
        text: str = args.forked_from
        try:
            body = check_agent_id_text(text)
        except GrammarError as exc:
            raise CliError.usage(str(exc)) from None
        if len(body) == ID_CHARS:
            # a full id is compared as it is: the parent's record may be gone, its children not
            forked_from = "ag-" + body
        else:
            forked_from = resolve_existing(store, text).agent_id
    ids = set(store.find_agents(labels, include_torn_down=True)) if labels else None
    wanted = {
        "statuses": statuses,
        "needs_attention": args.needs_attention,
        "session_states": session_states,
        "kind": args.kind,
        "repo": args.repo,
        "mode": args.mode,
        "forked_from": forked_from,
        "since": since,
    }
    try:
        # The inline liveness check of the live and launching rows runs first (it may park an
        # agent); it records its changes and delivers no hook. A row whose process entry cannot
        # be read is listed as it is: unreadable is an error, never "dead".
        live = liveness_for(inv, store)
        for row in store.list_agents(statuses=("launching", "live")):
            try:
                live.check(row["agent_id"])
            except CliError as error:
                if error.exit_code == 12:
                    # unreadable process entry (documented): listed unchanged, and logged
                    _log_unreadable(inv, row["agent_id"], error)
                continue
        rows = [row for row in store.list_agents() if _matches(row, ids, wanted)]
        rows.sort(key=lambda row: row["agent_id"])
        rows.sort(key=lambda row: row["created"], reverse=True)
    except CORRUPT_ROW:
        raise unusable_row() from None
    agents: list[dict[str, Any]] = []
    for row in rows:
        try:
            agents.append(agent_object(store, row))
        except CORRUPT_ROW:
            raise unusable_row(row.get("agent_id")) from None
    return agents
