"""``reconcile`` (interface.md, ``reconcile``): the idempotent maintenance pass cron runs.

One caller at a time (lease ``reconcile``: a second live caller exits 0 with
``already_running: true`` and does nothing). In order:

1. ``claude-session reconcile`` (a failure is a warning);
2. the liveness check of every ``launching`` and ``live`` agent (``Liveness``; it reports
   ``parked``, ``supervisor_lost``, ``launch_failed``, ``session_missing``, ``pending_run``,
   ``ambiguous_session`` and ``fork_abandoned``, never signals and never stops a session);
3. delivery of the hook events nobody delivered (the outbox rules of ``hooks.deliver_outbox``);
4. ``teardown_incomplete`` reports for agents with ``teardown_step`` set (reported, never
   continued: ``prune`` and ``teardown`` continue them);
5. the pending label mirror changes, replayed to sessions and live panes
   (``annotate.mirror_labels``: the queue holds the latest op per pair, so a replay converges);
6. housekeeping: leftover files of torn-down agents and orphaned prompt copies, tombstones past
   ``tombstone_days`` (the id stays reserved), handoff directories of dead listeners.

Safety, because steps 6 deletes:

* every file removal goes through descriptors opened without following symlinks, inside the
  state directory, on real directories owned by this user (``teardown.remove_tree_at``, the
  discipline of ``remove_agent_files``); names come from ``readdir`` and are checked against the
  id grammar, never from stored values;
* prompt copies go only when the agent has no record and the directory is older than an hour,
  or the agent is a tombstone: a record that is not torn down keeps its copies (a failed launch
  is resumed from them);
* a handoff directory goes only when its ``owner`` is verifiably dead (``handoff.sweep_dead``);
* a tombstone is deleted under the agent's lease and its delivery lease, judged again on the
  fresh row;
* one agent's failure, whatever it is, is a warning and never stops the others; a step that
  raises is a warning and never stops the later steps;
* ``--dry-run`` takes no lease and changes nothing: it reports what the pass would change.
  Files (not agents) that the pass would remove are not part of the reported changes (the schema
  has no kind for them); they are logged as ``reconcile_change`` when really removed.

Lower tools are called as ``auto:agent-spawn`` with the decision ids stored with the agent.
"""

from __future__ import annotations

import dataclasses
import os
import stat
from collections.abc import Mapping, Sequence
from typing import Any

from . import gateway as gw
from . import handoff, procs
from . import hooks as hooklib
from .agent_lookup import (
    backend_for,
    liveness_for,
    open_existing,
    state_dir_of,
    store_failed,
)
from .agent_view import ts
from .annotate import mirror_labels, print_warnings
from .commands.base import Invocation
from .ids import is_agent_id
from .labels import Label
from .lifecycle import LEASE_WAIT, lease_key
from .output import CliError, Outcome, PayloadKind, safe_text
from .records import RecordsError, append_log
from .store import LeaseHeld, Store, StoreError
from .teardown import SUPERVISOR_ACTOR, remove_agent_files, remove_tree_at

__all__ = ["LEASE", "reconcile_command"]

LEASE = "reconcile"
#: Prompt copies without an agent record younger than this are a command still classifying.
ORPHAN_PROMPT_AGE = 3600.0
TEARDOWN_INCOMPLETE = "teardown_incomplete"
LABELS_RESYNCED = "labels_resynced"
EVENTS_DELIVERED = "events_delivered"
TOMBSTONE_DELETED = "tombstone_deleted"


def _internal(inv: Invocation, row: Mapping[str, Any] | None = None) -> Invocation:
    """The invocation as the internal actor sees it (never the caller's TTY-derived actor)."""
    ids = row.get("decision_ids") if row is not None else None
    return dataclasses.replace(
        inv,
        actor=SUPERVISOR_ACTOR,
        actor_explicit=False,
        decision_ids=[i for i in ids if isinstance(i, str)] if isinstance(ids, list) else [],  # pyright: ignore[reportUnknownVariableType]
    )


class _Pass:
    """One reconcile pass over one open store."""

    def __init__(self, inv: Invocation, store: Store, state_dir: str, dry_run: bool) -> None:
        assert inv.config is not None
        self.inv = inv
        self.config = inv.config
        self.store = store
        self.state_dir = state_dir
        self.dry_run = dry_run
        self.gateway = gw.get_gateway()
        self.changes: list[dict[str, Any]] = []
        self.warnings: list[str] = []

    # -- plumbing ----------------------------------------------------------------------------

    def run(self) -> None:
        steps = (
            ("claude-session reconcile", "session_reconcile"),
            ("liveness", "liveness"),
            ("hook delivery", "deliver"),
            ("teardown report", "teardown_incomplete"),
            ("label re-sync", "labels"),
            ("leftover files", "leftover_files"),
            ("tombstones", "tombstones"),
            ("handoff directories", "handoffs"),
        )
        for label, method in steps:
            try:
                getattr(self, method)()
            except Exception as error:  # noqa: BLE001 - a step never stops the later ones
                self.warnings.append(f"{label} failed ({type(error).__name__})")

    def change(self, agent_id: str, kind: str, **details: Any) -> None:
        self.changes.append({"agent_id": agent_id, "kind": kind, "details": details})

    def log(self, agent_id: str | None, kind: str, **details: Any) -> None:
        """Record a change this pass made (never in a dry run; never raises)."""
        if self.dry_run:
            return
        try:
            append_log(
                self.state_dir,
                "reconcile_change",
                agent_id=agent_id,
                actor=SUPERVISOR_ACTOR,
                details={"kind": kind, **details},
            )
        except (RecordsError, ValueError, OSError):
            pass

    def warn(self, agent_id: str | None, text: str) -> None:
        self.warnings.append(text if agent_id is None else f"{agent_id}: {text}")

    # -- 1: claude-session --------------------------------------------------------------------

    def session_reconcile(self) -> None:
        try:
            backend = backend_for(_internal(self.inv))
            backend.reconcile(dry_run=self.dry_run)
        except CliError as error:
            self.warn(None, f"claude-session reconcile failed ({error.code})")

    # -- 2: liveness --------------------------------------------------------------------------

    def liveness(self) -> None:
        check = liveness_for(self.inv, self.store, dry_run=self.dry_run)
        for row in self.store.list_agents(statuses=("launching", "live")):
            agent_id = row["agent_id"]
            try:
                checked = check.check(agent_id)
            except CliError as error:  # an unreadable process entry: never "dead"
                self.warn(agent_id, safe_text(error.message))
                continue
            except Exception as error:  # noqa: BLE001 - one agent never stops the others
                self.warn(agent_id, f"liveness failed ({type(error).__name__})")
                continue
            for item in checked.changes:
                self.change(item.agent_id, item.kind, **item.details)
            for text in checked.warnings:
                self.warn(agent_id, text)

    # -- 3: the outbox ------------------------------------------------------------------------

    def deliver(self) -> None:
        for row in self.store.list_agents():
            agent_id = row["agent_id"]
            try:
                if self.dry_run:
                    count = len(self.store.pending_events(agent_id))
                    if count:
                        self.change(agent_id, EVENTS_DELIVERED, count=count)
                    continue
                if not self.store.pending_events(agent_id, limit=1):
                    continue
                result = hooklib.deliver_outbox(
                    self.store,
                    agent_id,
                    self.config.hooks,
                    state_dir=self.state_dir,
                    gateway=self.gateway,
                )
            except Exception as error:  # noqa: BLE001 - delivery never stops the pass
                self.warn(agent_id, f"hook delivery failed ({type(error).__name__})")
                continue
            if result.error is not None:
                self.warn(agent_id, "hook delivery stopped")
            if result.delivered:
                self.change(
                    agent_id,
                    EVENTS_DELIVERED,
                    count=result.delivered,
                    hooks_failed=result.hooks_failed,
                )
                self.log(agent_id, EVENTS_DELIVERED, count=result.delivered)

    # -- 4: teardown reports ------------------------------------------------------------------

    def teardown_incomplete(self) -> None:
        for row in self.store.list_agents(statuses=("tearing_down",)):
            step = row.get("teardown_step")
            if not step:
                continue
            agent_id = row["agent_id"]
            alive = self._alive(row.get("teardown_proc"))
            details: dict[str, Any] = {"teardown_step": step, "teardown_alive": alive}
            partial = row.get("teardown_partial_record")
            if step == "worktree_partial" and isinstance(partial, str):
                details["partial_record"] = partial
            if step == "started" and alive is False:
                # The pre-teardown hook ran (or was about to) in a process that is gone. A
                # SIGTERM/SIGHUP/SIGINT stops the hook with its teardown; only a SIGKILL can
                # leave it running, and its identity is not recorded: the retry runs it again.
                details["hook_may_be_running"] = True
            self.change(agent_id, TEARDOWN_INCOMPLETE, **details)

    def _alive(self, identity: object) -> bool | None:
        if not isinstance(identity, procs.ProcessIdentity):
            return False
        try:
            if identity.boot_id != procs.read_boot_id(gateway=self.gateway):
                return False
            return procs.is_alive(identity, gateway=self.gateway)
        except procs.ProcessIdentityError:
            return None  # unreadable: neither alive nor dead

    # -- 5: label mirror ----------------------------------------------------------------------

    def labels(self) -> None:
        for row in self.store.list_agents(statuses=("launching", "live", "parked")):
            agent_id = row["agent_id"]
            try:
                self._labels_of(agent_id)
            except Exception as error:  # noqa: BLE001 - one agent never stops the others
                self.warn(agent_id, f"label re-sync failed ({type(error).__name__})")

    def _labels_of(self, agent_id: str) -> None:
        pending = self.store.mirror_pending(agent_id)
        if not pending:
            return
        row = self.store.get_agent(agent_id)  # liveness may have changed it
        if row is None or row["status"] not in ("launching", "live", "parked"):
            return
        if self.dry_run:
            self._report_labels(agent_id, pending, set(), row)
            return
        warnings: list[str] = []
        inv = _internal(self.inv, row)
        for direction in ("add", "remove"):
            pairs = [pair for pair, op in pending if op == direction]
            if pairs:
                mirror_labels(inv, self.store, row, direction, pairs, warnings)
        for text in warnings:
            self.warn(agent_id, text)
        left = set(self.store.mirror_pending(agent_id))
        self._report_labels(agent_id, pending, left, row)

    def _report_labels(
        self,
        agent_id: str,
        before: Sequence[tuple[Label, str]],
        left: set[tuple[Label, str]],
        row: Mapping[str, Any],
    ) -> None:
        done = [item for item in before if item not in left]
        if self.dry_run and not isinstance(row.get("csession_id"), str):
            return  # nothing can be pushed without a session
        if not done:
            return
        added = [str(pair) for pair, op in done if op == "add"]
        removed = [str(pair) for pair, op in done if op == "remove"]
        self.change(agent_id, LABELS_RESYNCED, added=added, removed=removed)
        self.log(agent_id, LABELS_RESYNCED, added=len(added), removed=len(removed))

    # -- 6: housekeeping ----------------------------------------------------------------------

    def leftover_files(self) -> None:
        """Prompt copies and outcome files: a tombstone's, whatever their age, and the prompt
        copies of an id with no record once they are older than an hour."""
        try:
            base = os.open(self.state_dir, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        except OSError as error:
            self.warn(None, f"cannot open the state directory ({error.strerror})")
            return
        try:
            for sub in ("prompts", "outcomes"):
                self._sweep_files(base, sub)
        finally:
            os.close(base)

    def _sweep_files(self, base: int, sub: str) -> None:
        try:
            parent = os.open(
                sub, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=base
            )
        except FileNotFoundError:
            return
        except OSError as error:
            self.warn(None, f"cannot open {sub}/ ({error.strerror})")
            return
        try:
            if os.fstat(parent).st_uid != self.gateway.getuid():
                self.warn(None, f"{sub}/ belongs to another user: not swept")
                return
            now = self.gateway.wall_time()
            for name in sorted(os.listdir(parent)):
                if not is_agent_id(name):
                    continue
                try:
                    self._sweep_entry(parent, sub, name, now)
                except OSError as error:
                    self.warn(name, f"cannot delete {sub}/{name} ({error.strerror})")
                except StoreError as error:
                    self.warn(name, f"{sub}/{name} not judged ({type(error).__name__})")
        finally:
            os.close(parent)

    def _sweep_entry(self, parent: int, sub: str, name: str, now: float) -> None:
        info = os.lstat(name, dir_fd=parent)
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != self.gateway.getuid():
            return  # a symlink or a stranger's file is none of ours to delete
        row = self.store.get_agent(name)
        if row is None:
            if sub != "prompts" or now - info.st_mtime <= ORPHAN_PROMPT_AGE:
                return
            reason = "no_record"
        elif row["status"] == "torn_down":
            reason = "torn_down"
        else:
            return  # a live, launching, tearing-down or parked agent keeps its files
        if self.dry_run:
            return
        remove_tree_at(parent, name)
        self.log(name if row is not None else None, "files_removed", directory=sub, reason=reason)

    def tombstones(self) -> None:
        limit = self.config.tombstone_days * 86400.0
        now = self.gateway.wall_time()
        for row in self.store.list_agents(statuses=("torn_down",)):
            agent_id = row["agent_id"]
            if not self._expired(row, now, limit):
                continue
            if self.dry_run:
                self.change(agent_id, TOMBSTONE_DELETED, torn_down_at=ts(row["torn_down_at"]))
                continue
            try:
                if self._delete_tombstone(agent_id, now, limit):
                    self.change(agent_id, TOMBSTONE_DELETED, torn_down_at=ts(row["torn_down_at"]))
                    self.log(agent_id, TOMBSTONE_DELETED)
            except Exception as error:  # noqa: BLE001 - one agent never stops the others
                self.warn(agent_id, f"tombstone not deleted ({type(error).__name__})")

    @staticmethod
    def _expired(row: Mapping[str, Any], now: float, limit: float) -> bool:
        at = row.get("torn_down_at")
        if isinstance(at, bool) or not isinstance(at, (int, float)):
            return False  # no usable time: never old enough
        return now - float(at) > limit

    def _delete_tombstone(self, agent_id: str, now: float, limit: float) -> bool:
        """Delete the record under the agent's lease and its delivery lease, on the fresh row.
        A delivery in progress (its lease held) postpones the deletion to the next pass."""
        try:
            delivery = self.store.acquire_lease(f"delivery:{agent_id}")
        except LeaseHeld:
            return False
        try:
            with self.store.lease(lease_key(agent_id), wait=LEASE_WAIT) as lease:
                fresh = self.store.get_agent(agent_id)
                if fresh is None or fresh["status"] != "torn_down":
                    return False
                if not self._expired(fresh, now, limit):
                    return False
                problems = remove_agent_files(self.state_dir, agent_id)
                if problems:
                    self.warn(agent_id, "its leftover files could not all be deleted")
                return self.store.delete_agent(lease, agent_id)
        except LeaseHeld:
            return False
        finally:
            self.store.release_lease(delivery)

    def handoffs(self) -> None:
        sweep = handoff.sweep_dead(self.state_dir, dry_run=self.dry_run, gateway=self.gateway)
        for text in sweep.warnings:
            self.warn(None, text)
        for name in sweep.removed:
            self.log(None, "handoff_removed", directory=name)


def _outcome(
    inv: Invocation,
    changes: list[dict[str, Any]],
    warnings: list[str],
    already_running: bool,
    dry_run: bool,
) -> Outcome:
    """The result. In human mode the summary goes to stdout (not with ``--quiet``, which is what
    cron passes) and every warning to stderr, so a problem still reaches cron's mail."""
    payload = {
        "changes": changes,
        "warnings": warnings,
        "already_running": already_running,
        "dry_run": dry_run,
    }
    print_warnings(inv, warnings)
    if already_running:
        human = "another reconcile is running: nothing done"
    else:
        verb = "would change" if dry_run else "changed"
        lines = [f"{verb} {len(changes)} thing(s)"]
        lines.extend(f"  {c['agent_id']}: {c['kind']}" for c in changes)
        human = "\n".join(lines)
    quiet = inv.quiet and not inv.as_json
    return Outcome(0, payload, PayloadKind.COMMAND, human, quiet=quiet)


def reconcile_command(inv: Invocation) -> Outcome:
    """The handler of ``reconcile`` (``commands.maintenance``)."""
    dry_run = bool(inv.args.dry_run)
    state_dir = state_dir_of(inv)
    store = open_existing(inv, migrate=not dry_run)  # a dry run never migrates
    warnings = list(inv.warnings)
    if store is None:  # no state.db: no agents, nothing to reconcile, nothing is created
        return _outcome(inv, [], warnings, False, dry_run)
    try:
        run = _Pass(inv, store, state_dir, dry_run)
        run.warnings = warnings
        if dry_run:
            run.run()
        else:
            try:
                lease = store.acquire_lease(LEASE)
            except LeaseHeld:
                return _outcome(inv, [], warnings, True, dry_run)
            except StoreError as error:  # e.g. a lease row whose holder cannot be parsed
                raise store_failed(error) from None
            try:
                run.run()
            finally:
                try:
                    store.release_lease(lease)
                except StoreError:
                    run.warn(None, "the reconcile lease could not be released")
        return _outcome(inv, run.changes, run.warnings, False, dry_run)
    finally:
        store.close()
