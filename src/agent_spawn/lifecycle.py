"""Status transitions that carry outbox events: park and the started (``live``) transition.

Both are compare-and-sets under the agent's lease (interface.md, Lifecycle transitions). The
events are built from the agent objects before and after and written in the same transaction.
No lower tool and no hook runs while the lease is held.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .agent_view import agent_object, lifecycle_events
from .store import CasLost, Store

LEASE_WAIT = 30.0  # seconds a lifecycle write waits for another live holder of the lease


def lease_key(agent_id: str) -> str:
    return f"agent:{agent_id}"


def reads_third_party_of(store: Store, row: Mapping[str, Any]) -> bool | None:
    """The classification of the agent's current run; ``None`` before a run."""
    run = row.get("run")
    if not isinstance(run, int):
        return None
    stored = store.get_run(row["agent_id"], run)
    if stored is None:
        return None
    value = stored.get("reads_third_party")
    return value if isinstance(value, bool) else None


def _prospective(row: Mapping[str, Any], changes: Mapping[str, object]) -> dict[str, Any]:
    after = {**row, **changes}
    if after.get("status") == "parked":
        after["promotion_pending"] = False
    return after


def park(
    store: Store,
    agent_id: str,
    *,
    park_reason: str,
    needs_attention: str | None,
    escalated_by: str | None = None,
    from_status: str | Sequence[str] = ("launching", "live"),
    changes: Mapping[str, object] | None = None,
    outcome: Mapping[str, Any] | None = None,
    chain_failure: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Park the agent; ``None`` when another process already did (the compare-and-set lost)."""
    with store.lease(lease_key(agent_id), wait=LEASE_WAIT) as lease:
        row = store.get_agent(agent_id)
        if row is None:
            return None
        wanted = (from_status,) if isinstance(from_status, str) else tuple(from_status)
        if row["status"] not in wanted:
            return None
        all_changes: dict[str, object] = {
            **(changes or {}),
            "status": "parked",
            "park_reason": park_reason,
            "needs_attention": needs_attention,
            "escalated_by": escalated_by,
        }
        before = agent_object(store, row)
        after = agent_object(store, _prospective(row, all_changes))
        events = lifecycle_events(
            before,
            after,
            reads_third_party=reads_third_party_of(store, row),
            outcome=outcome,
            chain_failure=chain_failure,
        )
        try:
            return store.park(
                lease,
                agent_id,
                park_reason=park_reason,
                needs_attention=needs_attention,
                escalated_by=escalated_by,
                from_status=wanted,
                changes=changes,
                events=events,
            )
        except CasLost:
            return None


def go_live(
    store: Store, agent_id: str, *, run: int, changes: Mapping[str, object] | None = None
) -> dict[str, Any] | None:
    """``launching`` → ``live`` when the run is started: clears ``needs_attention``, records
    ``post-launch`` (and the status change). ``None`` when the agent is no longer launching."""
    with store.lease(lease_key(agent_id), wait=LEASE_WAIT) as lease:
        row = store.get_agent(agent_id)
        if row is None or row["status"] != "launching":
            return None
        all_changes: dict[str, object] = {
            **(changes or {}),
            "status": "live",
            "needs_attention": None,
            "park_reason": None,
            "escalated_by": None,
        }
        before = agent_object(store, row)
        after = agent_object(store, {**row, **all_changes})
        events = lifecycle_events(
            before, after, reads_third_party=reads_third_party_of(store, row), started_run=run
        )
        try:
            return store.compare_and_set(
                lease,
                agent_id,
                expect_status="launching",
                changes=all_changes,
                events=events,
            )
        except CasLost:
            return None
