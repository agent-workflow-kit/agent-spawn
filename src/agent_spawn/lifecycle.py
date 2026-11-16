"""Status transitions that carry outbox events: park and the started (``live``) transition.

Both are compare-and-sets under the agent's lease (interface.md, Lifecycle transitions). The
events are built from the agent objects before and after and written in the same transaction.
No lower tool and no hook runs while the lease is held.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .agent_view import agent_object, lifecycle_events
from .procs import ProcessIdentity
from .store import CasLost, Store

LEASE_WAIT = 30.0  # seconds a lifecycle write waits for another live holder of the lease


def lease_key(agent_id: str) -> str:
    return f"agent:{agent_id}"


def run_settled(row: Mapping[str, Any], *, supervisor_alive: bool) -> bool:
    """The end condition of a waiting command: the agent is not `launching` or `live`, no
    promotion is pending and its supervisor has exited (interface.md, `wait`)."""
    return (
        row["status"] not in ("launching", "live")
        and not row.get("promotion_pending")
        and not supervisor_alive
    )


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
        now = store.gateway.wall_time()
        all_changes: dict[str, object] = {
            **(changes or {}),
            "updated": now,  # the stored value is this one: the events' `after` matches it
            "pre_resume": None,  # the run exists (or the launch failed): never restored again
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
                changes={**(changes or {}), "updated": now, "pre_resume": None},
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
            "updated": store.gateway.wall_time(),
            "status": "live",
            "needs_attention": None,
            "park_reason": None,
            "escalated_by": None,
            "pre_resume": None,
        }
        before = agent_object(store, row)
        after = agent_object(store, {**row, **all_changes})
        # A plain resume still carries its pre-resume row here; a fork's first run is the one
        # run of a record with `forked_from` and no pre-resume row (a fork is also a resume).
        resumed = row.get("pre_resume") is not None
        fork = row.get("forked_from") is not None and not resumed
        events = lifecycle_events(
            before,
            after,
            reads_third_party=reads_third_party_of(store, row),
            started_run=run,
            resumed=resumed or fork,
            fork=fork,
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


def run_started(
    store: Store,
    agent_id: str,
    *,
    run: int,
    auto_resumed: bool = False,
    changes: Mapping[str, object] | None = None,
) -> dict[str, Any] | None:
    """A later run of the chain started (status stays ``live``): clears ``needs_attention`` and
    records ``post-launch`` for that run. ``None`` when the agent is no longer live."""
    with store.lease(lease_key(agent_id), wait=LEASE_WAIT) as lease:
        row = store.get_agent(agent_id)
        if row is None or row["status"] != "live":
            return None
        all_changes: dict[str, object] = {
            **(changes or {}),
            "updated": store.gateway.wall_time(),
            "needs_attention": None,
        }
        before = agent_object(store, row)
        after = agent_object(store, {**row, **all_changes})
        events = lifecycle_events(
            before,
            after,
            reads_third_party=reads_third_party_of(store, {**row, "run": run}),
            started_run=run,
            auto_resumed=auto_resumed,
        )
        try:
            return store.compare_and_set(
                lease, agent_id, expect_status="live", changes=all_changes, events=events
            )
        except CasLost:
            return None


def enter_ephemeral_teardown(
    store: Store,
    agent_id: str,
    *,
    supervisor: ProcessIdentity,
    changes: Mapping[str, object] | None = None,
) -> dict[str, Any] | None:
    """The supervisor of an ephemeral agent whose run is ``done``: ``live`` → ``tearing_down``
    with the supervisor as the teardown process. ``None`` when the compare-and-set lost."""
    with store.lease(lease_key(agent_id), wait=LEASE_WAIT) as lease:
        row = store.get_agent(agent_id)
        if row is None or row["status"] != "live":
            return None
        now = store.gateway.wall_time()
        all_changes: dict[str, object] = {
            **(changes or {}),
            "updated": now,
            "status": "tearing_down",
            "teardown_step": "started",
            "teardown_proc": supervisor,
            "teardown_partial_record": None,
        }
        before = agent_object(store, row)
        after = agent_object(store, {**row, **all_changes})
        events = lifecycle_events(before, after, reads_third_party=reads_third_party_of(store, row))
        try:
            return store.compare_and_set(
                lease,
                agent_id,
                expect_status="live",
                expect={"ephemeral": True, "supervisor": supervisor},
                changes=all_changes,
                events=events,
            )
        except CasLost:
            return None


def refuse_ephemeral_teardown(
    store: Store, agent_id: str, *, outcome: Mapping[str, Any] | None
) -> dict[str, Any] | None:
    """An ephemeral teardown that was refused at a step that changed nothing: ``tearing_down``
    → ``parked`` (``park_reason: done``) with ``post-park`` carrying ``teardown_refused``."""
    with store.lease(lease_key(agent_id), wait=LEASE_WAIT) as lease:
        row = store.get_agent(agent_id)
        if row is None or row["status"] != "tearing_down":
            return None
        # Only the supervisor that entered the teardown may refuse it, and only before the
        # worktree step began: marker ``started``, or ``hooks_done`` (the worktree step runs
        # later). ``worktree_partial`` and beyond are never refused.
        step = row.get("teardown_step")
        if row.get("teardown_proc") != store.holder_identity() or step not in (
            "started",
            "hooks_done",
        ):
            return None
        all_changes: dict[str, object] = {
            "updated": store.gateway.wall_time(),
            "status": "parked",
            "park_reason": "done",
            "needs_attention": None,
            "escalated_by": None,
            "promotion_pending": False,
            "teardown_step": None,
            "teardown_proc": None,
            "teardown_partial_record": None,
        }
        before = agent_object(store, row)
        after = agent_object(store, {**row, **all_changes})
        events = lifecycle_events(
            before,
            after,
            reads_third_party=reads_third_party_of(store, row),
            outcome=outcome,
            teardown_refused=True,
        )
        try:
            return store.compare_and_set(
                lease,
                agent_id,
                expect_status="tearing_down",
                expect={"teardown_proc": store.holder_identity(), "teardown_step": step},
                changes=all_changes,
                events=events,
            )
        except CasLost:
            return None
