"""``attach``: bring the window of a live agent's current run into view.

Only the current run's window is ever used (an older run's pane may be gone or reused), the
agent is resolved like every other command (a prefix that matches several agents is exit 21 and
nothing is called), and what termctl answers is validated by ``TermBackend.focus`` before this
process replaces itself with it.
"""

from __future__ import annotations

from typing import Any

from . import gateway as gw
from .agent_lookup import open_existing, resolve_existing, store_failed
from .annotate import termctl_for
from .commands.base import Invocation
from .output import CliError, Outcome, PayloadKind, safe_text
from .store import StoreError

__all__ = ["attach"]

NO_WINDOW_HINT = (
    "headless agents may be promoted; read the session with `claude-session transcript "
    "<csession_id>`"
)


def attach(inv: Invocation) -> Outcome:
    args = inv.args
    store = open_existing(inv)
    try:
        try:
            resolved = resolve_existing(store, args.agent_id)
        except CliError as error:
            if error.code == "not_found" and error.details is None:
                error.details = {"status": None}  # an unknown id
            raise
        agent_id = resolved.agent_id
        if resolved.torn_down:
            raise CliError.of(
                "not_found",
                f"agent {agent_id} is torn down",
                agent_id=agent_id,
                details={"status": "torn_down"},
            )
        assert store is not None
        try:
            row = store.get_agent(agent_id)
            if row is None:
                raise CliError.of("not_found", "no such agent", details={"status": None})
            window = _window(store, row) if row["status"] == "live" else None
        except StoreError as error:
            raise store_failed(error) from None
    finally:
        if store is not None:
            store.close()
    status = row["status"]
    if status != "live":
        details: dict[str, Any] = {"status": status}
        if status == "parked":
            details["hint"] = f"agent-spawn resume {agent_id}"
        raise CliError.of(
            "not_found",
            f"agent {agent_id} is not live ({status})",
            agent_id=agent_id,
            details=details,
        )
    if window is None:
        raise CliError.of(
            "no_window",
            f"agent {agent_id}'s current run has no window",
            agent_id=agent_id,
            details={"hint": NO_WINDOW_HINT},
        )
    term_id, socket_path = window
    result = termctl_for(inv, row).focus(
        term_id, socket_path, client=args.client, no_attach=bool(args.no_attach)
    )
    payload = {
        "agent_id": agent_id,
        "action": result["action"],
        "client": result["client"],
        "argv": result["argv"],
        "term_id": term_id,
        "socket_path": socket_path,
    }
    if result["action"] == "attach" and not inv.as_json:
        try:
            gw.get_gateway().exec_replace(result["argv"])
        except OSError as error:
            raise CliError(
                "failed", f"could not attach ({error.strerror})", agent_id=agent_id
            ) from None
    human = (
        f"{agent_id}: switched {safe_text(result['client'])}"
        if result["action"] == "switched"
        else f"{agent_id}: attach with {safe_text(' '.join(result['argv']))}"
    )
    return Outcome(0, payload, PayloadKind.COMMAND, human)


def _window(store: Any, row: dict[str, Any]) -> tuple[str, str] | None:
    run = row.get("run")
    if not isinstance(run, int) or isinstance(run, bool):
        return None
    run_row = store.get_run(row["agent_id"], run) or {}
    term_id, socket_path = run_row.get("term_id"), run_row.get("socket_path")
    if isinstance(term_id, str) and term_id and isinstance(socket_path, str) and socket_path:
        return term_id, socket_path
    return None
