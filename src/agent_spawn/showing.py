"""``show``: one agent with its runs, session summary, supervisor and cwd classification.

Read-only: it never delivers hooks and never creates a database. Everything a lower tool prints
is untrusted: it is validated field by field and a malformed answer becomes ``null`` with a
warning, never a guess.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from . import gateway as gw
from . import procs
from .agent_lookup import backend_for, open_existing, resolve_existing, store_failed
from .agent_view import agent_object, ts
from .commands.base import Invocation
from .output import CliError, Outcome, PayloadKind, safe_text
from .session_backend import STATES, SessionStatus
from .store import Store, StoreError

__all__ = ["show"]

_REPORT_STATUSES = frozenset({"done", "blocked", "needs-input", "failed"})
_ACTIVITIES = frozenset({"working", "idle"})
_NEEDS_INPUT = frozenset({"permission", "question", "elicitation"})
_MODES = frozenset({"interactive", "headless"})


def session_summary(status: SessionStatus) -> dict[str, Any] | None:
    """The ``session`` object of ``claude-session status``; ``None`` when a field is unusable."""
    raw = status.raw
    run = status.run
    state, activity, reason = raw.get("state"), raw.get("activity"), raw.get("needs_input_reason")
    run_raw = raw.get("run")
    sandbox = run_raw.get("sandbox_active") if isinstance(run_raw, dict) else None
    report = run.report
    report_status = report.get("status") if isinstance(report, dict) else None
    if state not in STATES or (activity is not None and activity not in _ACTIVITIES):
        return None
    if reason is not None and reason not in _NEEDS_INPUT:
        return None
    return {
        "state": state,
        "needs_input_reason": reason,
        "activity": activity,
        "run": run.run,
        "report_status": report_status if report_status in _REPORT_STATUSES else None,
        "sandbox_active": sandbox if isinstance(sandbox, bool) else None,
        "tainted": status.tainted,
    }


def run_objects(store: Store, row: Mapping[str, Any]) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for stored in store.get_runs(row["agent_id"]):
        window = None
        if stored.get("term_id") and stored.get("socket_path"):
            window = {"term_id": stored["term_id"], "socket_path": stored["socket_path"]}
        outcome = stored.get("outcome")
        mode = stored.get("mode") or row.get("mode")
        runs.append(
            {
                "run": stored["run"],
                "csession_id": row.get("csession_id"),
                "run_kind": stored.get("run_kind"),
                "mode": mode if mode in _MODES else row.get("mode"),
                "promoted": bool(stored.get("promoted")),
                "auto_resumed": bool(stored.get("auto_resumed")),
                "adopted_at": ts(stored.get("started_at")),
                "ended_at": ts(stored.get("ended_at")),
                "first_working_at": ts(stored.get("first_working_at")),
                "window": window,
                "outcome": outcome if isinstance(outcome, dict) else None,
            }
        )
    return runs


def supervisor_object(row: Mapping[str, Any], warnings: list[str]) -> dict[str, Any] | None:
    sup, exec_proc = row.get("supervisor"), row.get("exec_proc")
    if sup is None:
        return None
    try:
        alive = procs.is_alive(sup, gateway=gw.get_gateway())
    except procs.ProcessIdentityError:
        warnings.append("the supervisor's liveness could not be determined")
        alive = False
    return {
        "pid": sup.pid,
        "alive": alive,
        "exec_pid": None if exec_proc is None else exec_proc.pid,
    }


def _session(inv: Invocation, row: Mapping[str, Any], warnings: list[str]) -> dict[str, Any] | None:
    session_id = row.get("csession_id")
    if not isinstance(session_id, str):
        return None
    try:
        status = backend_for(inv, row).status(session_id)
    except CliError as error:
        if error.exit_code == 10:
            # TODO(#31): park the agent `session_missing` (the liveness check owns the park).
            warnings.append("claude-session reports the session not found")
        else:
            warnings.append(f"claude-session status failed ({safe_text(error.code)})")
        return None
    summary = session_summary(status)
    if summary is None:
        warnings.append("claude-session printed an unusable status")
    return summary


def third_party_run(inv: Invocation, row: Mapping[str, Any], warnings: list[str]) -> bool | None:
    """Whether the cwd lies inside a live throwaway record (None: unknown, with a warning)."""
    cwd = row.get("cwd")
    if not isinstance(cwd, str):
        return None
    assert inv.config is not None
    if inv.config.wtctl_cmd is None:
        return False  # no wtctl on this machine: the records of every cwd are empty
    # TODO(#21): the records of a cwd come from `wtctl list --json` (interface.md, Records of a
    # cwd); the lookup, its fail-closed rules and the holds belong to that issue.
    warnings.append("the throwaway records of the cwd were not looked up")
    return None


def repo_config_changed(row: Mapping[str, Any], warnings: list[str]) -> dict[str, Any] | None:
    if row.get("wt_record_id") is None:
        return None
    # TODO(#20): `wtctl baseline check <record>` of the agent's own worktree record.
    warnings.append("the repo config baseline was not checked")
    return None


def _human(payload: Mapping[str, Any]) -> str:
    agent = payload["agent"]
    lines = [
        f"{agent['agent_id']}  {agent['status']}  {safe_text(agent['name'])}",
        f"  needs_attention: {agent['needs_attention'] or '-'}",
        f"  session: {agent['csession_id'] or '-'}  state: {agent['session_state'] or '-'}",
        f"  labels: {safe_text(' '.join(agent['labels'])) or '-'}",
    ]
    for run in payload["runs"]:
        outcome = run["outcome"]["outcome"] if run["outcome"] else "-"
        lines.append(f"  run {run['run']}: {run['mode']}  outcome: {safe_text(str(outcome))}")
    for warning in payload["warnings"]:
        lines.append(f"  warning: {safe_text(warning)}")
    return "\n".join(lines)


def show(inv: Invocation) -> Outcome:
    store = open_existing(inv)
    resolved = resolve_existing(store, inv.args.agent_id)
    assert store is not None
    warnings: list[str] = []
    try:
        with store:
            row = store.get_agent(resolved.agent_id)
            if row is None:
                raise CliError.of("not_found", "no such agent")
            agent = agent_object(store, row)
            runs = run_objects(store, row)
    except StoreError as error:
        raise store_failed(error) from None
    if row["status"] == "torn_down":
        session = supervisor = config_changed = None
        third = None
    else:
        # TODO(#31): the inline liveness check of a live or launching agent runs first.
        supervisor = supervisor_object(row, warnings)
        session = _session(inv, row, warnings)
        third = third_party_run(inv, row, warnings)
        config_changed = repo_config_changed(row, warnings)
    payload = {
        "agent": agent,
        "runs": runs,
        "session": session,
        "supervisor": supervisor,
        "third_party_run": third,
        "repo_config_changed": config_changed,
        "warnings": warnings,
    }
    return Outcome(0, payload, PayloadKind.COMMAND, _human(payload))
