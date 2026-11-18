"""Dead terminal panes of an agent (interface.md, Teardown order step 5 and ``reconcile`` step 6).

Ownership rule, shared by teardown and reconcile: a pane is the agent's when it carries the label
``agent=<agent_id>`` (agent-spawn labels every pane it opens, and the id is allocated by it) on
a server the agent's runs record, or on termctl's default server when a run was placed in a
window. The default server covers a window whose ``open`` was killed after the pane existed and
whose recovery (``Launch.recover_window``) failed: no run records its socket path.

Only panes termctl itself lists as dead are candidates, and ``termctl close`` is never forced.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from . import gateway as gw
from .config import Config
from .output import CliError
from .term import TermBackend

__all__ = ["had_window", "sockets_to_search", "termctl_for"]


def had_window(runs: Sequence[Mapping[str, Any]]) -> bool:
    """Whether any run was placed in a window (its pane may exist even when it was never
    recorded): a recorded pane or socket, or the stored placement ``window``."""
    for run in runs:
        if run.get("placement") == "window":
            return True
        if isinstance(run.get("term_id"), str) or isinstance(run.get("socket_path"), str):
            return True
    return False


def sockets_to_search(runs: Sequence[Mapping[str, Any]]) -> list[str | None]:
    """The servers to look on, in run order: every distinct recorded socket path, then the
    default server (``None``) for the panes no run recorded. Empty without a window run."""
    if not had_window(runs):
        return []
    found: list[str | None] = []
    for run in runs:
        socket_path = run.get("socket_path")
        if isinstance(socket_path, str) and socket_path and socket_path not in found:
            found.append(socket_path)
    found.append(None)
    return found


def termctl_for(
    config: Config,
    row: Mapping[str, Any],
    *,
    actor: str | None,
    decision_ids: Sequence[str],
    gateway: gw.Gateway,
) -> TermBackend | None:
    """termctl as launch resolved it for this agent (stored path first); None when it cannot be
    found."""
    from .launch import resolve_stored_tool  # the tool resolution launch uses

    stored = row.get("termctl_path")
    try:
        argv = resolve_stored_tool(
            config.termctl_cmd, "termctl", stored if isinstance(stored, str) else None
        )
    except CliError:
        return None
    return TermBackend(argv, actor=actor, decision_ids=decision_ids, gateway=gateway)
