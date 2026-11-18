"""Dead terminal panes of an agent (interface.md, Teardown order step 5 and ``reconcile`` step 6).

Ownership rule, shared by teardown and reconcile: a pane is the agent's when it carries the label
``agent=<agent_id>`` (agent-spawn labels every pane it opens, and the id is allocated by it) on
a server the agent's runs record, or on termctl's default server when a run was placed in a
window. The default server covers a window whose ``open`` was killed after the pane existed and
whose recovery (``Launch.recover_window``) failed: no run records its socket path.

The default server is a best-effort sweep: which server termctl calls "default" depends on the
caller's environment (``TERMCTL_SOCKET``, ``TMUX``, ``TMUX_TMPDIR``), so its coverage does too.
The recorded sockets are authoritative. Labels are untrusted pane options (anything that can
reach the server can set one); the risk is bounded by closing only dead panes and never forcing.

Only panes termctl itself lists as dead are candidates, and ``termctl close`` is never forced.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from . import gateway as gw
from .config import Config
from .output import CliError
from .term import TermBackend

__all__ = ["DeadScan", "had_window", "scan_dead", "sockets_to_search", "termctl_for"]


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


@dataclass
class DeadScan:
    """The dead panes of an agent over several servers. ``complete`` is False when a recorded
    server could not be listed or a pane had no socket path (a retry must look again); a failure
    of the default-server sweep alone only warns."""

    panes: list[tuple[str, str]] = field(default_factory=lambda: [])
    complete: bool = True
    warnings: list[str] = field(default_factory=lambda: [])


def _canonical(socket_path: str) -> str:
    return os.path.realpath(socket_path)


def scan_dead(term: TermBackend, agent_id: str, targets: Sequence[str | None]) -> DeadScan:
    """List the dead panes of ``agent_id`` on each target (``None``: the default server). A pane
    reachable through two targets (a recorded socket that is also the default server) is returned
    once, by ``(id, canonical socket path)``. Exit 10 (no server) is nothing to close."""
    scan = DeadScan()
    seen: set[tuple[str, str]] = set()
    for target in targets:
        try:
            dead = term.find_dead(agent_id, target)
        except CliError as error:
            if error.exit_code != 10:
                scan.warnings.append(f"termctl list failed (exit {error.exit_code})")
                if target is not None:  # the default server is a best-effort sweep
                    scan.complete = False
            continue
        if dead.unresolved:
            scan.warnings.append(
                f"{dead.unresolved} dead pane(s) listed without a socket path: not closed"
            )
            scan.complete = False
        for term_id, socket_path in dead.panes:
            key = (term_id, _canonical(socket_path))
            if key not in seen:
                seen.add(key)
                scan.panes.append((term_id, socket_path))
    return scan


def termctl_for(
    config: Config,
    row: Mapping[str, Any],
    *,
    actor: str | None,
    decision_ids: Sequence[str],
    gateway: gw.Gateway,
    cache: MutableMapping[str | None, list[str] | None] | None = None,
) -> TermBackend | None:
    """termctl as launch resolved it for this agent (stored path first); None when it cannot be
    found. ``cache`` (a dict kept by the caller for one pass) remembers the resolution per stored
    path, so a pass over many agents resolves termctl once."""
    from .launch import resolve_stored_tool  # the tool resolution launch uses

    stored = row.get("termctl_path")
    key = stored if isinstance(stored, str) else None
    if cache is not None and key in cache:
        argv = cache[key]
    else:
        try:
            argv = resolve_stored_tool(config.termctl_cmd, "termctl", key)
        except CliError:
            argv = None
        if cache is not None:
            cache[key] = argv
    if argv is None:
        return None
    return TermBackend(argv, actor=actor, decision_ids=decision_ids, gateway=gateway)
