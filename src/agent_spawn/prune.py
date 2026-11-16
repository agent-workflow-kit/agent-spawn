"""``prune`` (interface.md, ``prune``): tear down old parked agents under automation rules.

A bulk-destructive command, so every decision is conservative:

* the teardown engine (``teardown.Teardown``) does the work; prune never deletes a row, a file or
  a worktree itself, never passes ``--force`` or ``--merged-head``, and always acts as
  ``auto:agent-spawn`` with the decision ids stored with the agent (the caller's ``--actor`` and
  ``--decision-id`` are never used);
* the age and the protections are judged twice: once unlocked to choose the candidates and report,
  and again, authoritatively, on the agent's fresh row under its lease in the same critical
  section as the compare-and-set that enters ``tearing_down`` (``Teardown.entry_guard``), so a
  label, an attention flag or a run that arrived in between is never overridden;
* only a ``parked`` agent, or a ``tearing_down`` one whose recorded teardown process is dead, is
  ever entered; the engine's own live checks (the session's run, a pending promotion) still run;
* over ``worktree_partial`` only a tombstone continues (``retry_partial_remove=False``);
* one agent's failure, whatever it is, is reported for that agent and never stops the others.

TODO(#31): the liveness check (``lost``, ``session_missing``) joins the engine's live checks in
``Teardown.live_reason``; prune inherits it.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from . import gateway as gw
from . import labels as labellib
from .config import Config
from .output import CliError, Outcome, PayloadKind
from .store import CasLost, Store, StoreError, StoreMissing, default_state_dir
from .teardown import (
    SUPERVISOR_ACTOR,
    UNSUPPORTED,
    GuardSkip,
    Teardown,
    _dry_run,  # pyright: ignore[reportPrivateUsage]
)
from .validate import parse_duration

PARTIAL = "worktree_partial"
CANDIDATE_STATUSES = ("parked", "tearing_down")


def protected_by(label_set: labellib.LabelSet, patterns: list[str]) -> bool:
    """Whether a label matches ``prune.protect_labels``: a bare key matches any value of that
    key, a ``key=value`` pair matches exactly that pair."""
    for pattern in patterns:
        if labellib.KEY.fullmatch(pattern) is not None:
            if any(label.key == pattern for label in label_set.pairs):
                return True
        elif labellib.parse_label(pattern) in label_set:
            return True
    return False


class Judge:
    """The age and the protections, applied to a row (fresh when called under the lease)."""

    def __init__(
        self,
        store: Store,
        config: Config,
        *,
        older_than: float,
        include_owned: bool,
        include_attention: bool,
    ) -> None:
        self.store = store
        self.patterns = list(config.prune.protect_labels)
        self.older_than = older_than
        self.include_owned = include_owned
        self.include_attention = include_attention

    def age(self, row: Mapping[str, Any]) -> float:
        since = row.get("last_live_at") or row.get("created")
        if not isinstance(since, (int, float)):
            return 0.0  # no usable time: never old enough
        return self.store.gateway.wall_time() - float(since)

    def old_enough(self, row: Mapping[str, Any]) -> bool:
        return self.age(row) > self.older_than

    def skip_reason(self, row: Mapping[str, Any]) -> str | None:
        """Why this agent must not be pruned, or ``None``. Every condition fails closed."""
        if row["status"] not in CANDIDATE_STATUSES:
            return "not_parked"
        if not self.old_enough(row):
            return "too_new"
        if not self.include_owned and protected_by(
            self.store.get_labels(row["agent_id"]), self.patterns
        ):
            return "protected"
        if row.get("needs_attention") and not self.include_attention:
            return "needs_attention"
        return None


def _result(
    agent_id: str,
    result: str,
    reason: str | None = None,
    tool: str | None = None,
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "agent_id": agent_id,
        "result": result,
        "reason": reason,
        "tool": tool,
        "details": None if details is None else dict(details),
    }


def _from_error(error: CliError, agent_id: str, *, was_partial: bool) -> dict[str, Any]:
    """Map one agent's engine error to its result entry."""
    lower = error.lower_error
    if error.code == "refused":
        if error.tool == "agent-spawn" and error.reason == "tearing_down":
            return _result(agent_id, "skipped", "tearing_down", "agent-spawn")
        return _result(agent_id, "refused", error.reason, error.tool, lower or error.details)
    if error.code == "partial" and was_partial:
        # The marker was already set: skipped, left for reconcile and a human (Teardown order 4).
        return _result(agent_id, "skipped", PARTIAL, error.tool, lower)
    if error.code == "not_found":
        return _result(agent_id, "skipped", "gone", "agent-spawn")
    return _result(agent_id, "failed", error.code, error.tool, lower or error.details)


def _open(state_dir: str) -> Store | None:
    try:
        return Store.open(state_dir, create=False, gateway=gw.get_gateway())
    except StoreMissing:
        return None
    except StoreError as error:
        raise CliError.of("failed", f"cannot open the state: {type(error).__name__}") from None


def prune_command(inv: Any) -> Outcome:
    """The handler of ``prune`` (``commands.ending``)."""
    args = inv.args
    config: Config = inv.config
    warnings: list[str] = list(inv.warnings)
    if inv.actor_explicit or inv.decision_ids:
        warnings.append(
            "prune always acts as auto:agent-spawn: --actor and --decision-id are ignored"
        )
    older = args.older_than if args.older_than is not None else config.prune.older_than
    dry_run = bool(args.dry_run)
    if older is None:
        warnings.append("nothing pruned: give --older-than or set prune.older_than in the config")
        return _outcome([], warnings, dry_run)
    threshold = parse_duration(older) if isinstance(older, str) else float(older)
    state_dir = inv.state_dir or str(default_state_dir())
    store = _open(state_dir)
    if store is None:
        return _outcome([], warnings, dry_run)
    results: list[dict[str, Any]] = []
    try:
        judge = Judge(
            store,
            config,
            older_than=threshold,
            include_owned=bool(args.include_owned),
            include_attention=bool(args.include_attention),
        )
        candidates = sorted(
            store.list_agents(statuses=CANDIDATE_STATUSES),
            key=lambda r: (r.get("created") or 0, r["agent_id"]),
        )
        for row in candidates:
            if not judge.old_enough(row):
                continue  # not reported: younger agents are not candidates
            results.append(_one(store, config, judge, row, state_dir, dry_run, warnings))
    finally:
        store.close()
    unsupported = sum(1 for r in results if r["reason"] == UNSUPPORTED)
    if unsupported:
        warnings.append(
            f"{unsupported} agent(s) skipped: teardown of agents with windows or pre-teardown "
            "hooks is not implemented yet (tracked in issues #27 and #32)"
        )
    return _outcome(results, warnings, dry_run)


def _one(
    store: Store,
    config: Config,
    judge: Judge,
    row: dict[str, Any],
    state_dir: str,
    dry_run: bool,
    warnings: list[str],
) -> dict[str, Any]:
    """One agent, isolated: nothing it raises reaches the other agents."""
    agent_id = row["agent_id"]
    was_partial = row.get("teardown_step") == PARTIAL
    td: Teardown | None = None
    try:
        td = Teardown(
            store,
            config,
            state_dir=state_dir,
            actor=SUPERVISOR_ACTOR,  # never the caller's
            decision_ids=row.get("decision_ids") or [],
            gateway=gw.get_gateway(),
            entry_guard=None if dry_run else judge.skip_reason,
            retry_partial_remove=False,
        )
        skip = judge.skip_reason(row)
        if skip is not None:
            entry = _result(agent_id, "skipped", skip, "agent-spawn")
        elif td.unsupported(row) is not None:
            entry = _result(agent_id, "skipped", UNSUPPORTED, "agent-spawn")
        elif dry_run:
            entry = _plan(td, row, agent_id, was_partial)
        else:
            entry = _tear_down(td, agent_id)
    except GuardSkip as skipped:  # fresh state under the lease disagreed with the listing
        entry = _result(agent_id, "skipped", skipped.reason, "agent-spawn")
    except CliError as error:
        entry = _from_error(error, agent_id, was_partial=was_partial)
    except CasLost:
        entry = _result(agent_id, "skipped", "changed", "agent-spawn")
    except Exception as error:  # noqa: BLE001 - bulk command: one agent never stops the others
        name = type(error).__name__
        entry = _result(agent_id, "failed", "internal", "agent-spawn", {"error": name})
    if td is not None:
        for problem in td.warnings:
            warnings.append(f"{agent_id}: {problem}")
        if not dry_run:
            try:
                td.log("prune", agent_id=agent_id, message=_message(entry))
            except Exception:  # noqa: BLE001 - a log failure never changes a result
                pass
            td.deliver(agent_id)
    return entry


def _message(entry: Mapping[str, Any]) -> str:
    return f"{entry['result']}" + (f": {entry['reason']}" if entry["reason"] else "")


def _tear_down(td: Teardown, agent_id: str) -> dict[str, Any]:
    row, retry = td.enter(agent_id)  # runs the entry guard under the lease
    td.run_from_step_3(row, retry=retry, abort_on_refusal=True)
    return _result(agent_id, "torn_down")


def _plan(td: Teardown, row: dict[str, Any], agent_id: str, was_partial: bool) -> dict[str, Any]:
    """The dry run: the teardown dry run (read-only), never a mutation."""
    if was_partial:
        record = row.get("teardown_partial_record")
        shown = td.wtctl(row).show(record) if isinstance(record, str) else None
        if shown is None or not shown.tombstone:
            return _result(agent_id, "skipped", PARTIAL, "wtctl")
    outcome = _dry_run(td, agent_id)
    reason = outcome.payload.get("would_refuse") if isinstance(outcome.payload, dict) else None
    if reason is None:
        return _result(agent_id, "torn_down")
    if reason == "tearing_down":
        return _result(agent_id, "skipped", "tearing_down", "agent-spawn")
    tool = "agent-spawn" if reason in ("live", "not_held") else "wtctl"
    return _result(agent_id, "refused", str(reason), tool)


def _outcome(results: list[dict[str, Any]], warnings: list[str], dry_run: bool) -> Outcome:
    payload = {"results": results, "warnings": warnings, "dry_run": dry_run}
    torn = sum(1 for r in results if r["result"] == "torn_down")
    verb = "would tear down" if dry_run else "torn down"
    lines = [f"{verb} {torn} of {len(results)} agent(s)"]
    for entry in results:
        detail = f" ({entry['reason']})" if entry["reason"] else ""
        lines.append(f"  {entry['agent_id']}: {entry['result']}{detail}")
    lines.extend(f"warning: {w}" for w in warnings)
    return Outcome(0, payload, PayloadKind.COMMAND, "\n".join(lines))


__all__ = ["Judge", "prune_command", "protected_by", "StoreError"]
