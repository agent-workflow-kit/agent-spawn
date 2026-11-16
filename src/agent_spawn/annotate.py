"""``ack``, ``label add`` and ``label remove``: edits of one agent under its lease.

Every write is a read-modify-write under ``agent:<id>``; no lower-tool call and no hook runs
while the lease is held. Label changes are mirrored to the session afterwards; a failed mirror is
a warning and the change stays in the pending-mirror queue for ``reconcile``.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any, Literal

from . import gateway as gw
from . import hooks as hooklib
from .agent_lookup import (
    backend_for,
    open_existing,
    resolve_existing,
    state_dir_of,
    store_failed,
    torn_down_error,
)
from .agent_view import agent_object, lifecycle_events, ts
from .commands.base import Invocation
from .labels import Label, LabelChange, LabelSet, parse_user_label
from .lifecycle import LEASE_WAIT, lease_key, reads_third_party_of
from .output import CliError, Outcome, PayloadKind, safe_text
from .records import RecordsError, append_log
from .store import AgentMissing, Lease, LeaseHeld, LeaseLost, Store, StoreError
from .validate import GrammarError

__all__ = ["ack", "label_edit", "mirror_session", "print_warnings"]


@contextmanager
def agent_lease(store: Store, agent_id: str) -> Iterator[Lease]:
    try:
        with store.lease(lease_key(agent_id), wait=LEASE_WAIT) as lease:
            yield lease
    except LeaseHeld:
        raise CliError(
            "failed",
            f"agent {agent_id} is busy: another process holds its lease",
            agent_id=agent_id,
            details={"retryable": True, "hint": "retry"},
        ) from None
    except (LeaseLost, AgentMissing):
        raise CliError.of("not_found", "no such agent", agent_id=agent_id) from None


def log_quietly(inv: Invocation, warnings: list[str], event: str, **fields: Any) -> None:
    try:
        append_log(
            state_dir_of(inv),
            event,
            actor=inv.actor,
            decision_ids=inv.decision_ids,
            **fields,
        )
    except (RecordsError, ValueError, OSError) as error:
        warnings.append(f"could not write the log ({type(error).__name__})")


def print_warnings(inv: Invocation, warnings: Sequence[str]) -> None:
    """In human mode each warning is one ``warning: ...`` line on stderr (``--json`` carries them
    in the result instead)."""
    if inv.as_json:
        return
    for warning in warnings:
        try:
            sys.stderr.write(safe_text(f"warning: {warning}") + "\n")
            sys.stderr.flush()
        except (BrokenPipeError, OSError):
            return


def _open_and_resolve(inv: Invocation, text: str) -> tuple[Store, str]:
    """The store and the agent id ``text`` names; the store is closed if resolving fails."""
    store = open_existing(inv)
    try:
        resolved = resolve_existing(store, text)
        if resolved.torn_down:
            raise torn_down_error(resolved.agent_id)
    except BaseException:
        if store is not None:
            store.close()
        raise
    assert store is not None
    return store, resolved.agent_id


def _live_row(store: Store, agent_id: str) -> dict[str, Any]:
    row = store.get_agent(agent_id)
    if row is None:
        raise CliError.of("not_found", "no such agent")
    if row["status"] == "torn_down":
        raise torn_down_error(agent_id)
    return row


# --- ack ---------------------------------------------------------------------------------------


def ack(inv: Invocation) -> Outcome:
    store, agent_id = _open_and_resolve(inv, inv.args.agent_id)
    warnings: list[str] = []
    try:
        with store:
            # TODO(#31): the inline liveness check of the agent runs before the write.
            with agent_lease(store, agent_id) as lease:
                row = _live_row(store, agent_id)
                before = agent_object(store, row)
                if row["needs_attention"] is None:
                    changed, after = False, before
                else:
                    reads = reads_third_party_of(store, row)
                    now = store.gateway.wall_time()  # the stored value: the event matches it
                    prospective = {**before, "needs_attention": None, "updated": ts(now)}
                    events = lifecycle_events(before, prospective, reads_third_party=reads)
                    updated = store.update_agent(
                        lease,
                        agent_id,
                        {"needs_attention": None, "updated": now},
                        events=events,
                    )
                    changed, after = True, agent_object(store, updated)
            if changed:
                log_quietly(
                    inv,
                    warnings,
                    "ack",
                    agent_id=agent_id,
                    details={"previous_needs_attention": before["needs_attention"]},
                )
                _deliver(inv, store, agent_id)
            after = {**after, "warnings": [*after["warnings"], *warnings]} if warnings else after
    except StoreError as error:
        raise store_failed(error) from None
    print_warnings(inv, warnings)
    human = (
        f"{agent_id}: needs_attention cleared" if changed else f"{agent_id}: nothing to acknowledge"
    )
    return Outcome(0, {"agent": after, "changed": changed}, PayloadKind.COMMAND, human)


def _deliver(inv: Invocation, store: Store, agent_id: str) -> None:
    """Mutating commands deliver the hook events they queued (best effort, never fatal)."""
    assert inv.config is not None
    try:
        hooklib.deliver_outbox(
            store,
            agent_id,
            inv.config.hooks,
            state_dir=state_dir_of(inv),
            gateway=gw.get_gateway(),
        )
    except (StoreError, OSError):
        pass  # hook failures never change the command's result


# --- labels ------------------------------------------------------------------------------------

Direction = Literal["add", "remove"]


def mirror_session(
    inv: Invocation,
    store: Store,
    row: dict[str, Any],
    direction: Direction,
    pairs: Sequence[Label],
    warnings: list[str],
) -> None:
    """Push the pairs to the agent's session; a failure is a warning and stays queued."""
    session = row.get("csession_id")
    if not pairs or not isinstance(session, str):
        return  # nothing to push, or no session yet: the queue keeps the change
    texts = [str(pair) for pair in pairs]
    try:
        backend = backend_for(inv, row)
        if direction == "add":
            backend.label_add(session, texts)
        else:
            backend.label_remove(session, texts)
    except CliError as error:
        warnings.append(
            f"claude-session label {direction} failed (exit {error.exit_code}); "
            "queued for reconcile"
        )
        return
    try:
        with agent_lease(store, row["agent_id"]) as lease:
            store.mirror_done(lease, row["agent_id"], [(pair, direction) for pair in pairs])
    except (CliError, StoreError):
        warnings.append("the label mirror could not be recorded; reconcile re-syncs")


def labels_after(store: Store, agent_id: str) -> list[str]:
    return store.get_labels(agent_id).texts()


def label_edit(inv: Invocation, direction: Direction) -> Outcome:
    args = inv.args
    try:
        pairs = list(dict.fromkeys(parse_user_label(text) for text in args.pairs))
    except GrammarError as exc:
        raise CliError.usage(str(exc)) from None
    store, agent_id = _open_and_resolve(inv, args.agent_id)
    warnings: list[str] = []
    key = "added" if direction == "add" else "removed"
    try:
        with store:
            # TODO(#31): the inline liveness check of the agent runs before the write.
            if args.dry_run:
                row = _live_row(store, agent_id)
                change = _plan(store.get_labels(agent_id), direction, pairs)
                final = change.labels.texts()
                applied: tuple[Label, ...] = change.applied
            else:
                with agent_lease(store, agent_id) as lease:
                    row = _live_row(store, agent_id)
                    try:
                        change = store.edit_labels(lease, agent_id, direction, pairs)
                    except GrammarError as exc:
                        raise CliError.usage(str(exc)) from None
                    final = labels_after(store, agent_id)
                    applied = change.applied
                if change.changed:
                    log_quietly(
                        inv,
                        warnings,
                        "label_change",
                        agent_id=agent_id,
                        details={"op": direction, "labels": [str(p) for p in applied]},
                    )
                    mirror_session(inv, store, row, direction, applied, warnings)
                    # TODO(#27): mirror to the live pane (termctl label add|remove) for the first
                    # 30 stored labels; a change beyond them is a warning, never queued.
                # a mutating command delivers what earlier commands queued (label queues nothing)
                _deliver(inv, store, agent_id)
    except StoreError as error:
        raise store_failed(error) from None
    print_warnings(inv, warnings)
    payload = {
        "agent_id": agent_id,
        "labels": final,
        key: [str(pair) for pair in applied],
        "changed": bool(applied),
        "warnings": warnings,
        "dry_run": bool(args.dry_run),
    }
    verb = "would be " if args.dry_run else ""
    human = f"{agent_id}: {verb}{key} {safe_text(' '.join(payload[key])) or 'nothing'}"
    return Outcome(0, payload, PayloadKind.COMMAND, human)


def _plan(current: LabelSet, direction: Direction, pairs: Sequence[Label]) -> LabelChange:
    try:
        return current.add(pairs) if direction == "add" else current.remove(pairs)
    except GrammarError as exc:
        raise CliError.usage(str(exc)) from None
