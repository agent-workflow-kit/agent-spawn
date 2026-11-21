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
from .agent_lookup import (
    backend_for,
    liveness_for,
    open_existing,
    resolve_existing,
    store_failed,
)
from .agent_view import CORRUPT_ROW, agent_object, ts, unusable_row
from .commands.base import Invocation
from .launch import resolve_wtctl
from .output import CliError, Outcome, PayloadKind, safe_text
from .session_backend import STATES, SessionStatus
from .store import Store, StoreError
from .worktrees import WtctlBackend, inside_throwaway

__all__ = ["show"]

_REPORT_STATUSES = frozenset({"done", "blocked", "needs-input", "failed"})
_ACTIVITIES = frozenset({"working", "idle"})
_NEEDS_INPUT = frozenset({"permission", "question", "elicitation"})
_MODES = frozenset({"interactive", "headless"})
SHOW_TIMEOUT = 15.0  # seconds per read-only wtctl call: show must stay quick


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
                # `started_at` is written once when the supervisor adopts the run (null before).
                "adopted_at": ts(stored.get("started_at")),
                "ended_at": ts(stored.get("ended_at")),
                "first_working_at": ts(stored.get("first_working_at")),
                "window": window,
                "outcome": outcome if isinstance(outcome, dict) else None,
            }
        )
    return runs


def supervisor_object(row: Mapping[str, Any]) -> dict[str, Any] | None:
    sup, exec_proc = row.get("supervisor"), row.get("exec_proc")
    if sup is None:
        return None
    try:
        alive = procs.is_alive(sup, gateway=gw.get_gateway())
    except procs.ProcessIdentityError:
        # An unreadable process entry is an error, never "dead" (interface.md, Liveness).
        raise CliError(
            "failed",
            "the supervisor's process entry could not be read",
            agent_id=row["agent_id"],
            exit_code=12,
        ) from None
    return {
        "pid": sup.pid,
        "alive": alive,
        "exec_pid": None if exec_proc is None else exec_proc.pid,
    }


class _SessionMissing(Exception):
    """claude-session reports the agent's session not found (exit 10)."""


def _session(inv: Invocation, row: Mapping[str, Any], warnings: list[str]) -> dict[str, Any] | None:
    session_id = row.get("csession_id")
    if not isinstance(session_id, str):
        return None
    try:
        status = backend_for(inv, row).status(session_id)
    except CliError as error:
        if error.exit_code == 10:
            warnings.append("claude-session reports the session not found")
            raise _SessionMissing from None
        warnings.append(f"claude-session status failed ({safe_text(error.code)})")
        return None
    if status.csession_id != session_id:
        warnings.append("claude-session answered for another session")
        return None
    summary = session_summary(status)
    if summary is None:
        warnings.append("claude-session printed an unusable status")
    return summary


def _wtctl(inv: Invocation, row: Mapping[str, Any]) -> WtctlBackend | None:
    """wtctl as the agent was launched with it (config's command, else the absolute path stored
    with the agent); ``None`` when there is no wtctl on this machine (``wtctl_cmd: null``)."""
    assert inv.config is not None
    command = inv.config.wtctl_cmd
    if command is None:
        return None
    stored = row.get("wtctl_path")
    argv = resolve_wtctl(command, stored if isinstance(stored, str) else None)
    return WtctlBackend(argv)


def third_party_run(inv: Invocation, row: Mapping[str, Any], warnings: list[str]) -> bool | None:
    """Whether the cwd lies inside a live throwaway record (None: unknown, with a warning)."""
    cwd = row.get("cwd")
    if not isinstance(cwd, str):
        return None
    try:
        backend = _wtctl(inv, row)
        if backend is None:
            return False  # no wtctl on this machine: the records of every cwd are empty
        return inside_throwaway(backend.list_records(timeout=SHOW_TIMEOUT), cwd)
    except CliError as error:
        warnings.append(
            f"the throwaway records of the cwd were not looked up ({safe_text(error.code)})"
        )
        return None


def repo_config_changed(
    inv: Invocation, row: Mapping[str, Any], warnings: list[str]
) -> dict[str, Any] | None:
    """``wtctl baseline check`` of the agent's own worktree record (informational)."""
    record_id = row.get("wt_record_id")
    if not isinstance(record_id, str):
        return None
    try:
        backend = _wtctl(inv, row)
        if backend is None:
            warnings.append("the repo config baseline was not checked (no wtctl)")
            return None
        return backend.baseline_check(record_id, timeout=SHOW_TIMEOUT)
    except CliError as error:
        warnings.append(f"the repo config baseline was not checked ({safe_text(error.code)})")
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


def _show_unchecked(inv: Invocation) -> Outcome:
    """``show --no-liveness``: the stored row and runs only. No liveness check, no process
    probe, no claude-session or wtctl call, nothing written or migrated; what needs one of
    those (``session``, ``supervisor.alive``, ``third_party_run``, ``repo_config_changed``) is
    null."""
    store = open_existing(inv, migrate=False, readonly=True)
    try:
        resolved = resolve_existing(store, inv.args.agent_id)
    except BaseException:
        if store is not None:
            store.close()
        raise
    assert store is not None
    try:
        with store:
            row = store.get_agent(resolved.agent_id)
            if row is None:
                raise CliError.of("not_found", "no such agent")
            supervisor = None
            sup, exec_proc = row.get("supervisor"), row.get("exec_proc")
            if sup is not None and row["status"] != "torn_down":
                supervisor = {
                    "pid": sup.pid,
                    "alive": None,
                    "exec_pid": None if exec_proc is None else exec_proc.pid,
                }
            try:
                agent = agent_object(store, row)
                runs = run_objects(store, row)
            except CORRUPT_ROW:
                raise unusable_row(resolved.agent_id) from None
    except StoreError as error:
        raise store_failed(error) from None
    payload = {
        "agent": agent,
        "runs": runs,
        "session": None,
        "supervisor": supervisor,
        "third_party_run": None,
        "repo_config_changed": None,
        "warnings": [],
        "liveness_checked": False,
    }
    return Outcome(0, payload, PayloadKind.COMMAND, _human(payload) + "\n  (liveness not checked)")


def show(inv: Invocation) -> Outcome:
    if inv.args.no_liveness:
        return _show_unchecked(inv)
    store = open_existing(inv)
    try:
        resolved = resolve_existing(store, inv.args.agent_id)
    except BaseException:
        if store is not None:
            store.close()
        raise
    assert store is not None
    warnings: list[str] = []
    session = supervisor = config_changed = third = None
    try:
        with store:
            row = store.get_agent(resolved.agent_id)
            if row is None:
                raise CliError.of("not_found", "no such agent")
            live = liveness_for(inv, store)
            if row["status"] != "torn_down":
                # The inline liveness check runs first: it may park the agent, and the reads below
                # use the row it leaves. It records its changes and delivers no hook.
                checked = live.check(resolved.agent_id)
                warnings.extend(checked.warnings)
                if checked.row is None:
                    raise CliError.of("not_found", "no such agent")
                row = checked.row
            if row["status"] not in ("torn_down",):
                supervisor = supervisor_object(row)
                try:
                    session = _session(inv, row, warnings)
                except _SessionMissing:
                    parked = live.session_missing(resolved.agent_id, row["csession_id"])
                    if parked.row is not None:
                        row = parked.row
                    supervisor = supervisor_object(row)
            try:
                agent = agent_object(store, row)
                runs = run_objects(store, row)
            except CORRUPT_ROW:
                raise unusable_row(resolved.agent_id) from None
    except StoreError as error:
        raise store_failed(error) from None
    if row["status"] != "torn_down":
        third = third_party_run(inv, row, warnings)
        config_changed = repo_config_changed(inv, row, warnings)
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
