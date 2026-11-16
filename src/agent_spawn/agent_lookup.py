"""What list, show, ack and label share: opening an existing store, resolving ids, the backend.

None of these commands may create a database: a missing ``state.db`` is "no agents" for ``list``
and exit 10 for the commands that name an agent. Store failures are reported as exit 12.
"""

from __future__ import annotations

import shlex
from collections.abc import Mapping
from typing import Any

from . import gateway as gw
from .commands.base import Invocation
from .ids import ResolvedId, check_agent_id_text, resolve_agent_id
from .launch import resolve_tool
from .output import CliError
from .session_backend import SessionBackend
from .store import SchemaOlder, Store, StoreBusy, StoreError, StoreMissing, default_state_dir
from .validate import GrammarError

__all__ = [
    "backend_for",
    "open_existing",
    "resolve_existing",
    "state_dir_of",
    "store_failed",
]


def state_dir_of(inv: Invocation) -> str:
    return inv.state_dir or str(default_state_dir())


def store_failed(error: StoreError) -> CliError:
    """The expected failure for a store that cannot be used (exit 12, never a traceback)."""
    if isinstance(error, StoreBusy):
        return CliError(
            "failed", "state.db is busy: try again", details={"retryable": True, "hint": "retry"}
        )
    if isinstance(error, SchemaOlder):
        return CliError(
            "failed",
            "state.db has an older schema: run a command that migrates it (such as `launch`)",
        )
    return CliError("failed", f"the state database could not be used ({type(error).__name__})")


def open_existing(inv: Invocation, *, migrate: bool = True) -> Store | None:
    """The store, or ``None`` when no ``state.db`` exists (nothing is created either way)."""
    try:
        return Store.open(
            state_dir_of(inv), create=False, migrate=migrate, gateway=gw.get_gateway()
        )
    except StoreMissing:
        return None
    except StoreError as error:
        raise store_failed(error) from None


def resolve_existing(store: Store | None, text: str) -> ResolvedId:
    """The stored agent ``text`` names (``ids.resolve_agent_id``); exit 10 without a store."""
    if store is None:
        try:
            check_agent_id_text(text)
        except GrammarError as exc:
            raise CliError.usage(str(exc)) from None
        raise CliError.of("not_found", "no such agent")
    return resolve_agent_id(text, store)


def torn_down_error(agent_id: str) -> CliError:
    """Every command but ``show`` treats a tombstone as gone (exit 10, ``details.status``)."""
    return CliError.of(
        "not_found",
        f"agent {agent_id} is torn down",
        agent_id=agent_id,
        details={"status": "torn_down"},
    )


def backend_for(inv: Invocation, row: Mapping[str, Any] | None = None) -> SessionBackend:
    """claude-session as the agent recorded it, else as the config names it."""
    stored = None if row is None else row.get("claude_session_path")
    if isinstance(stored, str) and stored:
        command = shlex.split(stored)
    else:
        assert inv.config is not None
        command = resolve_tool(inv.config.claude_session_cmd, "claude-session")
    return SessionBackend(
        command, actor=inv.actor, decision_ids=inv.decision_ids, gateway=gw.get_gateway()
    )
