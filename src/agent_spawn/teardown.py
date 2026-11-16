"""``teardown`` (interface.md, Teardown order).

The steps this module owns: 1 (resolve and usage checks), 2 (the live checks and the lifecycle
entry), 4 (worktrees), 5 (windows), 6 (forget the session) and 7 (the tombstone). The named markers
(``teardown_step``) make a retry continue where a dead teardown process stopped. The supervisor of
an ephemeral agent runs the same steps from step 3 (``ephemeral``).

TODO(#32): the ``pre-teardown`` hooks (step 3); until then an agent that has any configured or
given ``pre-teardown`` hook is refused before anything changes (a veto must never be skipped).
Step 5 closes the dead panes labeled ``agent=<agent_id>`` of every server the agent's runs used;
a failure there is a warning, never a refusal.

Every lifecycle write is a compare-and-set under the agent's lease; no lower tool and no hook
runs while the lease is held.
"""

from __future__ import annotations

import os
import re
import shlex
import stat
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from . import gateway as gw
from . import hooks as hooklib
from . import procs
from .agent_view import agent_object, lifecycle_events
from .config import Config
from .hooks import agent_event
from .ids import is_agent_id, resolve_agent_id
from .lifecycle import LEASE_WAIT, lease_key, reads_third_party_of
from .output import CliError, Outcome, PayloadKind
from .records import RecordsError, append_audit, append_log
from .session_backend import SessionBackend, SessionStatus
from .store import CasLost, Store, StoreError, StoreMissing, default_state_dir
from .term import TermBackend
from .worktrees import WtctlBackend, WtctlError

TOOL = "claude-session"
WTCTL = "wtctl"
# The order of the named markers; ``worktree_partial`` is a branch of the worktree step.
STEP_ORDER = (
    "started",
    "hooks_done",
    "worktree_partial",
    "worktree_done",
    "windows_done",
    "session_done",
)
PRE_TEARDOWN = "pre-teardown"
POST_TEARDOWN = "post-teardown"
TEARDOWN_HOOKS = (PRE_TEARDOWN, POST_TEARDOWN)
SUPERVISOR_ACTOR = "auto:agent-spawn"
MAX_TREE_DEPTH = 32
# Stable refusal reasons the supervisor records in its log when an ephemeral teardown is refused.
UNSUPPORTED = "teardown_unsupported"


class GuardSkip(Exception):
    """An entry guard (``prune``'s protections) declined the agent, judged on its fresh row
    under the lease: nothing was changed. ``reason`` is the stable skip reason."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


EntryGuard = Callable[[Mapping[str, Any]], "str | None"]


@dataclass
class WtItem:
    """One worktree record of the agent that step 4 processes."""

    record_id: str
    path: str
    owned: bool
    released: bool  # the hold is released already (an earlier run of this teardown, or none)


def reached(row: Mapping[str, Any], marker: str) -> bool:
    """Whether the recorded ``teardown_step`` is ``marker`` or later."""
    step = row.get("teardown_step")
    return step in STEP_ORDER and STEP_ORDER.index(step) >= STEP_ORDER.index(marker)


# --- deleting the agent's own files ---------------------------------------------------------------


def _remove_at(dir_fd: int, name: str, depth: int = 0) -> None:
    """Delete ``name`` below ``dir_fd`` without ever following a symlink: a symlink (to anything)
    is unlinked itself, a directory is emptied through descriptors opened with ``O_NOFOLLOW``."""
    if depth > MAX_TREE_DEPTH:
        raise OSError("the directory nests too deeply")
    info = os.lstat(name, dir_fd=dir_fd)
    if not stat.S_ISDIR(info.st_mode):
        os.unlink(name, dir_fd=dir_fd)
        return
    fd = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=dir_fd)
    try:
        opened = os.fstat(fd)
        if (opened.st_dev, opened.st_ino) != (info.st_dev, info.st_ino):
            raise OSError("the directory changed while it was being deleted")
        for entry in os.listdir(fd):
            _remove_at(fd, entry, depth + 1)
    finally:
        os.close(fd)
    os.rmdir(name, dir_fd=dir_fd)


def remove_agent_files(state_dir: str, agent_id: str) -> list[str]:
    """Delete ``<state>/prompts/<agent_id>`` and ``<state>/outcomes/<agent_id>``.

    The paths come from the state directory and a checked agent id, never from stored values.
    The two parent directories must be real directories (not symlinks). Returns the problems
    met (the tombstone does not wait for them: ``reconcile`` removes leftovers of torn-down
    agents).
    """
    if not is_agent_id(agent_id):
        raise ValueError(f"not an agent id: {agent_id!r}")
    problems: list[str] = []
    try:
        base = os.open(state_dir, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    except OSError as error:
        return [f"cannot open the state directory ({error.strerror})"]
    try:
        for sub in ("prompts", "outcomes"):
            try:
                parent = os.open(
                    sub, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=base
                )
            except FileNotFoundError:
                continue
            except OSError as error:
                problems.append(f"cannot open {sub}/ ({error.strerror})")
                continue
            try:
                _remove_at(parent, agent_id)
            except FileNotFoundError:
                pass
            except OSError as error:
                problems.append(f"cannot delete {sub}/{agent_id} ({error.strerror})")
            finally:
                os.close(parent)
    finally:
        os.close(base)
    return problems


# --- the engine -----------------------------------------------------------------------------------


def _store_failure(error: StoreError, agent_id: str | None = None) -> CliError:
    return CliError(
        "failed",
        f"the state database could not be used ({type(error).__name__})",
        agent_id=agent_id,
    )


class Teardown:
    """One teardown of one agent, by the command or by the supervisor of an ephemeral agent."""

    def __init__(
        self,
        store: Store,
        config: Config,
        *,
        state_dir: str,
        actor: str,
        decision_ids: Sequence[str] = (),
        merged_head: str | None = None,
        force: bool = False,
        override_escalation: bool = False,
        hooks: Sequence[Mapping[str, str]] = (),
        backend: SessionBackend | None = None,
        gateway: gw.Gateway | None = None,
        log: Callable[..., None] | None = None,
        entry_guard: EntryGuard | None = None,
        retry_partial_remove: bool = True,
    ) -> None:
        self.store = store
        self.config = config
        self.state_dir = state_dir
        self.actor = actor
        self.decision_ids = list(decision_ids)
        self.merged_head = merged_head
        self.force = force
        self.override_escalation = override_escalation
        self.hooks = [dict(item) for item in hooks]
        self.backend = backend
        self.gateway = gateway or gw.get_gateway()
        self.warnings: list[str] = []
        self.session_forgotten = False
        self.session_missing = False
        self.past_refusals = False  # step 6 started: a failure there is not a refusal
        self.hooks_before: object = None  # agents.hooks before this call's entry wrote it
        self.hooks_written = False
        self.wt_backend: WtctlBackend | None = None
        self.worktrees: list[dict[str, Any]] = []  # step 4's output entries, in processing order
        self.windows_closed: list[str] = []  # step 5: the dead panes it closed
        self.removed: WtItem | None = None  # the owned record this run removed
        self.audited = False
        self.sticky = False  # a remove with a possibly partial outcome: never back to parked
        self.step4_refused: str | None = None  # step 4 stopped with nothing changed for a record
        self._log = log
        # ``prune``: judged on the fresh row under the lease, in the same critical section as
        # the compare-and-set that enters ``tearing_down`` (no gap between check and entry).
        self.entry_guard = entry_guard
        # False (``prune``): over ``worktree_partial`` only a tombstone continues; any other
        # state leaves the agent ``tearing_down`` instead of calling ``remove`` again.
        self.retry_partial_remove = retry_partial_remove

    # -- records ----------------------------------------------------------------------------

    def log(self, event: str, *, level: str = "info", agent_id: str | None, **fields: Any) -> None:
        if self._log is not None:
            self._log(event, level=level, **fields)
            return
        try:
            append_log(
                self.state_dir,
                event,
                level=level,
                agent_id=agent_id,
                actor=self.actor,
                decision_ids=self.decision_ids or None,
                **fields,
            )
        except (RecordsError, ValueError, OSError) as error:
            self.warnings.append(f"could not write the log ({type(error).__name__})")

    def audit(
        self,
        row: Mapping[str, Any],
        *,
        action: str = "teardown",
        result: str = "ok",
        item: WtItem | None = None,
    ) -> None:
        """An audit record: ``teardown`` (a teardown that removed a worktree or was forced) or
        ``teardown_escalation_override`` (``--override-escalation`` reached ``wtctl remove``)."""
        removed = item or self.removed
        worktree = (
            {"record_id": removed.record_id, "path": removed.path, "owned": removed.owned}
            if removed is not None
            else None
        )
        try:
            append_audit(
                self.state_dir,
                actor=self.actor,
                decision_ids=self.decision_ids,
                target=row["agent_id"],
                action=action,
                inputs={
                    "labels": self.store.get_labels(row["agent_id"]).texts(),
                    "repo": row.get("repo"),
                    "worktree": worktree,
                    "csession_id": row.get("csession_id"),
                    "merged_head": self.merged_head,
                    "force": self.force,
                },
                result=result,
            )
        except (RecordsError, ValueError, OSError) as error:
            self.warnings.append(f"could not write the audit log ({type(error).__name__})")

    def audit_removal(self, row: Mapping[str, Any], item: WtItem) -> None:
        """The removal's audit record, once per teardown that finished a removal."""
        self.removed = item
        if not self.audited:
            self.audit(row)
            self.audited = True

    # -- lower tools ------------------------------------------------------------------------

    def session_backend(self, row: Mapping[str, Any]) -> SessionBackend:
        if self.backend is None:
            from .launch import resolve_tool  # the tool resolution launch uses

            stored = row.get("claude_session_path")
            argv = (
                shlex.split(stored)
                if isinstance(stored, str) and stored
                else resolve_tool(self.config.claude_session_cmd, TOOL)
            )
            self.backend = SessionBackend(
                argv, actor=self.actor, decision_ids=self.decision_ids, gateway=self.gateway
            )
        return self.backend

    def session_status(self, row: Mapping[str, Any]) -> SessionStatus | None:
        """The session's ``status``; ``None`` for no session or one claude-session reports
        missing (exit 10). Any other failure passes through."""
        session = row.get("csession_id")
        if not isinstance(session, str) or not session:
            return None
        backend = self.session_backend(row)
        try:
            return backend.status(session)
        except CliError as error:
            if error.tool == TOOL and error.code == "not_found":
                self.session_missing = True
                return None
            raise
        except ValueError:
            raise CliError("internal", "the stored session id is malformed") from None

    @staticmethod
    def run_is_live(status: SessionStatus | None) -> bool:
        """The session's current run is pending or live (not final)."""
        return status is not None and (
            status.pending or status.run.pending or not status.outcome().final
        )

    # -- step 2: the live checks ---------------------------------------------------------------

    def refuse(
        self, reason: str, message: str, agent_id: str, *, status: str | None = None
    ) -> CliError:
        details = {"status": status} if status in ("launching", "tearing_down") else None
        return CliError("refused", message, reason=reason, agent_id=agent_id, details=details)

    def live_reason(self, row: Mapping[str, Any]) -> str | None:
        """``live`` when nothing may tear the agent down (the checks no flag overrides).

        TODO(#31): the liveness check of step 2 (``lost``, ``session_missing``) runs here once
        issue #31 lands; until then only the status, the promotion and the session's run count."""
        if row["status"] in ("launching", "live") or row.get("promotion_pending"):
            return "live"
        if self.run_is_live(self.session_status(row)):
            return "live"
        return None

    def unsupported(self, row: Mapping[str, Any]) -> CliError | None:
        """The pieces of teardown not implemented yet, refused before any change."""
        agent_id = row["agent_id"]
        what: str | None = None
        if not reached(row, "hooks_done") and self.pre_hook_entries():
            what = "pre-teardown hooks"  # TODO(#32)
        if what is None:
            return None
        return CliError(
            "internal",
            f"teardown with {what} is not implemented yet (tracked in issue #32)",
            agent_id=agent_id,
        )

    def pre_hook_entries(self) -> list[hooklib.HookEntry]:
        return [
            *hooklib.config_entries(self.config.hooks, PRE_TEARDOWN),
            *hooklib.invocation_entries(self.hooks, PRE_TEARDOWN),
        ]

    # -- the lifecycle ----------------------------------------------------------------------

    def entry_changes(self, row: Mapping[str, Any], **extra: object) -> dict[str, object]:
        """The entry's changes. ``--hook`` values replace the stored hooks; without any, the
        stored value (the first invocation's) is left alone."""
        # Note: ``--hook`` entries given on a retry replace (not merge with) those stored by
        # the first invocation; the spec leaves this open.
        changes: dict[str, object] = {"updated": self.store.gateway.wall_time(), **extra}
        if self.hooks:
            self.hooks_before = row.get("hooks")
            self.hooks_written = True
            changes["hooks"] = self.hooks
        return changes

    def dead_teardown_process(self, row: Mapping[str, Any]) -> procs.ProcessIdentity:
        """The recorded teardown process of a ``tearing_down`` agent, which must be dead for a
        retry (checked outside the lease). An unknown or unreadable one is never "dead"."""
        recorded = row.get("teardown_proc")
        agent_id = row["agent_id"]
        if not isinstance(recorded, procs.ProcessIdentity):
            raise self.refuse(
                "tearing_down", "a teardown is in progress", agent_id, status="tearing_down"
            )
        try:
            alive = procs.is_alive(recorded, gateway=self.gateway)
        except procs.ProcessIdentityError as error:
            raise CliError(
                "failed",
                f"cannot tell whether the teardown process is alive: {error}",
                agent_id=agent_id,
            ) from None
        if alive:
            raise self.refuse(
                "tearing_down",
                "a teardown of this agent is in progress",
                agent_id,
                status="tearing_down",
            )
        return recorded

    def enter(self, agent_id: str) -> tuple[dict[str, Any], bool]:
        """Step 2: the live checks, then the compare-and-set into ``tearing_down`` (or, over a
        dead teardown process, the retry). Returns the row and whether it is a retry."""
        for _ in range(3):
            row = self.store.get_agent(agent_id)
            if row is None or row["status"] == "torn_down":
                raise self.torn_down(agent_id, row is None)
            retry_from: procs.ProcessIdentity | None = None
            if row["status"] == "tearing_down":
                retry_from = self.dead_teardown_process(row)
            elif row["status"] in ("launching", "live") or row.get("promotion_pending"):
                raise self.refuse("live", "the agent is live", agent_id, status=row["status"])
            unsupported = self.unsupported(row)
            if unsupported is not None:
                raise unsupported
            if self.live_reason(row) is not None:
                raise self.refuse("live", "the agent's session has a pending or live run", agent_id)
            try:
                return self.compare_and_set_entry(agent_id, retry_from)
            except CasLost as lost:
                if lost.status == "parked" or (
                    retry_from is not None and lost.status == "tearing_down"
                ):
                    continue  # the row moved under us: judge it again
                if lost.status == "tearing_down":
                    raise self.refuse(
                        "tearing_down",
                        "a teardown of this agent is in progress",
                        agent_id,
                        status="tearing_down",
                    ) from None
                if lost.status == "torn_down" or lost.current is None:
                    raise self.torn_down(agent_id, lost.current is None) from None
                raise self.refuse(
                    "live", "the agent is live", agent_id, status=lost.status
                ) from None
        raise self.refuse("tearing_down", "the agent keeps changing", agent_id)

    def compare_and_set_entry(
        self, agent_id: str, retry_from: procs.ProcessIdentity | None
    ) -> tuple[dict[str, Any], bool]:
        store = self.store
        me = store.holder_identity()
        with store.lease(lease_key(agent_id), wait=LEASE_WAIT) as lease:
            row = store.get_agent(agent_id)
            if row is None:
                raise CasLost(agent_id, ("parked",), None)
            if self.entry_guard is not None:
                skip = self.entry_guard(row)
                if skip is not None:
                    raise GuardSkip(skip)
            if retry_from is not None:
                done = store.compare_and_set(
                    lease,
                    agent_id,
                    expect_status="tearing_down",
                    expect={"teardown_proc": retry_from},
                    changes=self.entry_changes(row, teardown_proc=me),
                )
                return done, True
            changes = self.entry_changes(
                row,
                status="tearing_down",
                teardown_step="started",
                teardown_proc=me,
                teardown_partial_record=None,
            )
            before = agent_object(store, row)
            after = agent_object(store, {**row, **changes})
            events = lifecycle_events(
                before, after, reads_third_party=reads_third_party_of(store, row)
            )
            done = store.compare_and_set(
                lease,
                agent_id,
                expect_status="parked",
                expect={"promotion_pending": False},
                changes=changes,
                events=events,
            )
            return done, False

    def torn_down(self, agent_id: str, missing: bool) -> CliError:
        if missing:
            return CliError("not_found", f"no agent {agent_id}", agent_id=agent_id)
        return CliError(
            "not_found",
            f"agent {agent_id} is already torn down",
            agent_id=agent_id,
            details={"status": "torn_down"},
        )

    def mine(self, lease_row: Mapping[str, Any] | None) -> bool:
        return (
            lease_row is not None
            and lease_row["status"] == "tearing_down"
            and lease_row.get("teardown_proc") == self.store.holder_identity()
        )

    def set_step(self, agent_id: str, marker: str, **extra: object) -> None:
        """Record a finished step; only the recorded teardown process may."""
        with self.store.lease(lease_key(agent_id), wait=LEASE_WAIT) as lease:
            if not self.mine(self.store.get_agent(agent_id)):
                raise CliError(
                    "failed", "this teardown no longer owns the agent", agent_id=agent_id
                )
            self.store.compare_and_set(
                lease,
                agent_id,
                expect_status="tearing_down",
                expect={"teardown_proc": self.store.holder_identity()},
                changes={
                    "teardown_step": marker,
                    "updated": self.store.gateway.wall_time(),
                    **extra,
                },
            )

    def abort(self, agent_id: str) -> None:
        """Back to ``parked``, unchanged (a refusal at a step that changed nothing)."""
        store = self.store
        with store.lease(lease_key(agent_id), wait=LEASE_WAIT) as lease:
            row = store.get_agent(agent_id)
            if not self.mine(row) or row is None or row.get("teardown_step") == "worktree_partial":
                return
            now = store.gateway.wall_time()
            changes = {
                "status": "parked",
                "teardown_step": None,
                "teardown_proc": None,
                "teardown_partial_record": None,
                "promotion_pending": False,
                "updated": now,
            }
            if self.hooks_written:
                changes["hooks"] = self.hooks_before  # this call's --hook values go with it
            before = agent_object(store, row)
            after = agent_object(store, {**row, **changes})
            events = lifecycle_events(
                before, after, reads_third_party=reads_third_party_of(store, row)
            )
            store.compare_and_set(
                lease,
                agent_id,
                expect_status="tearing_down",
                expect={"teardown_proc": store.holder_identity()},
                changes=changes,
                events=events,
            )

    # -- the steps --------------------------------------------------------------------------

    def run_from_step_3(
        self, row: dict[str, Any], *, retry: bool, abort_on_refusal: bool
    ) -> dict[str, Any]:
        """Steps 3, 6 and 7 for an agent that is ``tearing_down`` with this process as the
        teardown process. Returns the tombstoned row."""
        agent_id = row["agent_id"]
        if not reached(row, "hooks_done"):
            self.step_hooks(row, abort_on_refusal=abort_on_refusal)
            row = self.store.get_agent(agent_id) or row
        elif retry:  # a retry past the hooks step skips it
            self.log(
                "teardown_retry", agent_id=agent_id, message=f"retry after {row['teardown_step']}"
            )
        if not reached(row, "worktree_done"):
            self.step_worktrees(row, abort_on_refusal=abort_on_refusal)
            row = self.store.get_agent(agent_id) or row
        else:
            self.worktrees = [self.entry(i, "none", None) for i in self.finished_items(row)]
        if not reached(row, "windows_done"):
            if self.step_windows(row):
                self.set_step(agent_id, "windows_done")
        if not reached(row, "session_done"):
            self.step_session(row)
            self.set_step(agent_id, "session_done")
        return self.step_tombstone(agent_id)

    def step_hooks(self, row: Mapping[str, Any], *, abort_on_refusal: bool) -> None:
        """Step 3. TODO(#32): run the ``pre-teardown`` hooks (veto 11, escalation 13) here;
        ``unsupported`` refuses any agent that has one until then. The live checks run again."""
        agent_id = row["agent_id"]
        try:
            if self.live_reason(row) is not None:
                raise self.refuse("live", "the agent's session has a pending or live run", agent_id)
        except CliError:
            if abort_on_refusal:
                self.abort(agent_id)
            raise
        self.set_step(agent_id, "hooks_done")

    # -- step 4: worktrees ------------------------------------------------------------------

    @staticmethod
    def entry(item: WtItem, action: str, result: Mapping[str, Any] | None) -> dict[str, Any]:
        return {
            "record_id": item.record_id,
            "path": item.path,
            "owned": item.owned,
            "action": action,
            "wtctl": None if result is None else dict(result),
        }

    def finished_items(self, row: Mapping[str, Any]) -> list[WtItem]:
        """The hold rows of an agent whose worktree step is done, in processing order."""
        rows = self.store.get_holds(row["agent_id"], include_released=True)
        ordered = [r for r in rows if not r["owned"]] + [r for r in rows if r["owned"]]
        return [WtItem(r["record_id"], r["path"], r["owned"], True) for r in ordered]

    def wtctl(self, row: Mapping[str, Any]) -> WtctlBackend:
        """wtctl as the agent was launched with it: config's command, else the absolute path
        stored with the agent. Teardown with a record and no wtctl is refused (nothing changed)."""
        if self.wt_backend is not None:
            return self.wt_backend
        from .launch import resolve_wtctl  # the tool resolution launch uses

        command = self.config.wtctl_cmd
        if command is None:
            raise CliError(
                "tool_missing",
                "this agent holds worktree records, but wtctl_cmd is null in the config",
                tool=WTCTL,
            )
        stored = row.get("wtctl_path")
        argv = resolve_wtctl(command, stored if isinstance(stored, str) else None)
        self.wt_backend = WtctlBackend(
            argv, actor=self.actor, decision_ids=self.decision_ids, gateway=self.gateway
        )
        return self.wt_backend

    def worktree_plan(self, row: Mapping[str, Any]) -> list[WtItem]:
        """The records to process: every held record in stored order, then the owned one.

        A launch whose ``create`` failed ``bootstrap_failed`` can leave ``wt_record_id`` without
        a hold row; that record is looked up with ``wtctl show`` and must be held by the agent."""
        agent_id = row["agent_id"]
        items = self.finished_items(row)
        rows = {
            r["record_id"]: r["released"]
            for r in self.store.get_holds(agent_id, include_released=True)
        }
        for item in items:
            item.released = rows[item.record_id]
        stored = row.get("wt_record_id")
        if isinstance(stored, str) and row.get("wt_owned") and stored not in rows:
            record = self.wtctl(row).show(stored)
            if record is None:
                self.warnings.append("wtctl does not know the agent's worktree record")
            elif record.tombstone:
                items.append(WtItem(record.id, record.path, True, True))
            elif record.object_type != "worktree" or agent_id not in record.holders:
                raise CliError(
                    "refused",
                    "the agent's stored worktree record is not a worktree it holds",
                    reason="not_held",
                    agent_id=agent_id,
                )
            else:
                items.append(WtItem(record.id, record.path, True, False))
        return items

    def mark_released(self, agent_id: str, item: WtItem) -> None:
        with self.store.lease(lease_key(agent_id), wait=LEASE_WAIT) as lease:
            if not self.mine(self.store.get_agent(agent_id)):
                raise CliError(
                    "failed", "this teardown no longer owns the agent", agent_id=agent_id
                )
            self.store.release_hold(lease, agent_id, item.record_id)
        item.released = True

    @staticmethod
    def left_unknown(error: CliError) -> bool:
        """Whether a failed worktree call may have changed something (then the agent stays
        ``tearing_down``): a partial removal, a call whose outcome is unknown (a timeout,
        wtctl's exit 1, or an exit the mapping does not know: 3 and up, a signal) or one of
        agent-spawn's own failures. wtctl's exit 2 (usage) ran nothing, and every other mapped
        result (refusals, exit 12 and 13) changed nothing."""
        if error.code == "partial":
            return True
        if isinstance(error, WtctlError):
            if error.lower_exit == 2:
                return False
            return error.exit_code == 1
        return error.code == "failed" or error.exit_code == 1

    def step_worktrees(self, row: dict[str, Any], *, abort_on_refusal: bool) -> None:
        agent_id = row["agent_id"]
        try:
            if not self.process_records(row):
                return  # no record: the step is skipped and the marker stays
        except CliError as error:
            error.agent_id = agent_id
            if self.sticky or self.left_unknown(error):
                self.past_refusals = True
                raise
            self.step4_refused = error.reason or error.code
            if abort_on_refusal:
                self.abort(agent_id)
            raise
        self.set_step(agent_id, "worktree_done")

    def process_records(self, row: dict[str, Any]) -> bool:
        """Release and remove the agent's records in order; false when it had none."""
        agent_id = row["agent_id"]
        items = self.worktree_plan(row)
        partial = row.get("teardown_partial_record")
        if row.get("teardown_step") != "worktree_partial":
            partial = None
        if partial is not None and all(i.record_id != partial for i in items):
            raise CliError(
                "failed",
                f"the stored partial record {partial} is not one of this agent's records: "
                "a human must inspect it",
                agent_id=agent_id,
            )
        if not items:
            return False
        for item in items:
            if item.released:
                self.worktrees.append(self.entry(item, "none", None))
            elif item.record_id == partial:
                self.finish_marked(row, item)
            else:
                self.release_or_remove(row, item)
        return True

    def finish_marked(self, row: dict[str, Any], item: WtItem) -> None:
        """A retry over ``worktree_partial`` (a partial removal, or one whose outcome is
        unknown): ``wtctl show`` decides. A tombstone completes the record; ``partial`` (or a
        record wtctl does not know) leaves the agent ``tearing_down``; any other state calls
        ``remove`` again, and nothing but success lets the agent leave ``tearing_down``."""
        agent_id = row["agent_id"]
        record = self.wtctl(row).show(item.record_id)
        if record is None or record.state == "partial":
            raise self.unfinished(agent_id, item)
        if not record.tombstone:
            if not self.retry_partial_remove:
                raise self.unfinished(agent_id, item)
            if agent_id not in record.holders or record.owned is not True:
                raise self.unfinished(agent_id, item)
            self.remove_owned(row, item, marked=True)
            return
        self.mark_released(agent_id, item)
        self.set_step(agent_id, "hooks_done", teardown_partial_record=None)
        self.worktrees.append(self.entry(item, "removed", None))
        if record.state == "removed":  # a forgotten or gone record was not removed by this agent
            self.audit_removal(row, item)

    @staticmethod
    def unfinished(agent_id: str, item: WtItem) -> CliError:
        return CliError(
            "partial",
            f"the worktree record {item.record_id} may be partly removed: "
            "finish it with `wtctl remove`",
            tool=WTCTL,
            agent_id=agent_id,
        )

    def release_or_remove(self, row: dict[str, Any], item: WtItem) -> None:
        agent_id = row["agent_id"]
        if item.owned:
            # The local row alone never decides a removal: wtctl must say the agent holds a
            # record that it created. A record wtctl did not create is released instead.
            record = self.wtctl(row).show(item.record_id)
            if record is not None and not record.tombstone:
                if agent_id not in record.holders:
                    raise CliError(
                        "refused",
                        f"the agent does not hold worktree record {item.record_id}",
                        reason="not_held",
                        agent_id=agent_id,
                    )
                if record.owned is not True:
                    item.owned = False
            if item.owned:
                self.remove_owned(row, item, marked=False)
                return
        result = self.wtctl(row).release(item.record_id, agent_id)
        if result.already:
            action = "none"
        else:
            action = "forgotten" if result.record_state == "forgotten" else "released"
        self.mark_released(agent_id, item)
        self.worktrees.append(self.entry(item, action, result.data))

    def remove_owned(self, row: dict[str, Any], item: WtItem, *, marked: bool) -> None:
        """``wtctl remove`` of the owned record. A failure that is partial or of unknown
        outcome is sticky: the marker and the record are stored, and the agent never returns
        to ``parked`` (``marked``: an earlier call had such an outcome, so any failure is)."""
        agent_id = row["agent_id"]
        try:
            result = self.wtctl(row).remove(
                item.record_id,
                agent_id,
                merged_head=self.merged_head,
                force=self.force,
                override_escalation=self.override_escalation,
            )
        except CliError as error:
            if self.override_escalation:
                outcome = "refused" if error.exit_code == 20 else "error"
                self.audit(row, action="teardown_escalation_override", result=outcome, item=item)
            if marked or self.left_unknown(error):
                self.sticky = True
                if not marked:
                    self.set_step(
                        agent_id, "worktree_partial", teardown_partial_record=item.record_id
                    )
            raise
        if self.override_escalation:
            self.audit(row, action="teardown_escalation_override", result="ok", item=item)
        self.mark_released(agent_id, item)
        if marked:
            self.set_step(agent_id, "hooks_done", teardown_partial_record=None)
        self.worktrees.append(
            self.entry(item, "none" if result.already else "removed", result.data)
        )
        if not result.already or result.state == "removed":  # forgotten/gone: not a removal
            self.audit_removal(row, item)

    # -- step 5: windows --------------------------------------------------------------------

    def window_sockets(self, agent_id: str) -> list[str]:
        """The distinct socket paths of the agent's runs' windows, in run order."""
        found: list[str] = []
        for run in self.store.get_runs(agent_id):
            socket_path = run.get("socket_path")
            if isinstance(socket_path, str) and socket_path and socket_path not in found:
                found.append(socket_path)
        return found

    def termctl(self, row: Mapping[str, Any]) -> TermBackend | None:
        from .launch import resolve_stored_tool  # the tool resolution launch uses

        stored = row.get("termctl_path")
        try:
            argv = resolve_stored_tool(
                self.config.termctl_cmd, "termctl", stored if isinstance(stored, str) else None
            )
        except CliError:
            return None
        return TermBackend(
            argv, actor=self.actor, decision_ids=self.decision_ids, gateway=self.gateway
        )

    def step_windows(self, row: Mapping[str, Any]) -> bool:
        """Step 5 (skipped, False, when no run of the agent had a window): for each distinct
        socket path ``termctl list --label agent=<id> --dead``, then ``termctl close`` for each
        dead pane listed. Exit 10 and 20 ``running`` are no-ops; any other failure is a warning
        (a window left behind never blocks a teardown). Returns whether the step is done: it is
        not when termctl could not be found or a ``list`` failed, so a retry looks again."""
        agent_id = row["agent_id"]
        sockets = self.window_sockets(agent_id)
        if not sockets:
            return False
        term = self.termctl(row)
        if term is None:
            self.warnings.append("termctl could not be found: the agent's dead windows stay open")
            return False
        complete = True
        for socket_path in sockets:
            try:
                dead = term.list_dead(agent_id, socket_path)
            except CliError as error:
                if error.exit_code != 10:
                    self.warnings.append(f"termctl list failed (exit {error.exit_code})")
                    complete = False
                continue
            for term_id in dead:
                try:
                    term.close(term_id, socket_path)
                except CliError as error:
                    if error.exit_code == 10 or (
                        error.exit_code == 20 and error.reason == "running"
                    ):
                        continue
                    self.warnings.append(f"termctl close failed (exit {error.exit_code})")
                    continue
                self.windows_closed.append(term_id)
        return complete

    def step_session(self, row: Mapping[str, Any]) -> None:
        """Step 6: ``claude-session forget``; exit 10 counts as done, any other failure stops
        the teardown with the marker unchanged."""
        agent_id = row["agent_id"]
        self.past_refusals = True
        session = row.get("csession_id")
        if not isinstance(session, str) or not session or self.session_missing:
            return
        try:
            self.session_backend(row).forget(session)
        except CliError as error:
            if error.tool == TOOL and error.code == "not_found":
                return
            error.agent_id = agent_id
            if error.code == "refused" and error.reason == "live":
                error.details = {**(error.details or {}), "hint": f"agent-spawn stop {agent_id}"}
            raise
        except ValueError:
            raise CliError(
                "internal", "the stored session id is malformed", agent_id=agent_id
            ) from None
        self.session_forgotten = True

    def step_tombstone(self, agent_id: str) -> dict[str, Any]:
        """Step 7: delete the agent's prompt copies and outcome files, then ``torn_down`` with
        ``post-teardown`` in the same transaction, then the log (and audit) records."""
        store = self.store
        with store.lease(lease_key(agent_id), wait=LEASE_WAIT) as lease:
            row = store.get_agent(agent_id)
            if not self.mine(row) or row is None:
                raise CliError(
                    "failed", "this teardown no longer owns the agent", agent_id=agent_id
                )
            # Ownership first, then the files, then the compare-and-set: a crash in between
            # leaves ``tearing_down`` and a retry deletes what is left (idempotent).
            problems = remove_agent_files(self.state_dir, agent_id)
            changes: dict[str, object] = {
                "status": "torn_down",
                "needs_attention": None,
                "teardown_step": None,
                "teardown_proc": None,
                "promotion_pending": False,
                "torn_down_at": store.gateway.wall_time(),
                "updated": store.gateway.wall_time(),
            }
            before = agent_object(store, row)
            after = agent_object(store, {**row, **changes})
            third_party = reads_third_party_of(store, row)
            events = lifecycle_events(before, after, reads_third_party=third_party)
            events.append(
                agent_event(
                    POST_TEARDOWN,
                    after,
                    reads_third_party=third_party,
                    extra={
                        "worktrees": [
                            {k: v for k, v in entry.items() if k != "wtctl"}
                            for entry in self.worktrees
                        ],
                        "merged_head": self.merged_head,
                        "force": self.force,
                    },
                )
            )
            done = store.compare_and_set(
                lease,
                agent_id,
                expect_status="tearing_down",
                expect={"teardown_proc": store.holder_identity()},
                changes=changes,
                events=events,
            )
        for problem in problems:
            self.warnings.append(problem)
            self.log("warning", level="warn", agent_id=agent_id, message=problem)
        self.log("teardown", agent_id=agent_id, message="torn down")
        if self.force and not self.audited:
            self.audit(done)
        return done

    # -- delivery -----------------------------------------------------------------------------

    def deliver(self, agent_id: str) -> None:
        try:
            hooklib.deliver_outbox(
                self.store,
                agent_id,
                self.config.hooks,
                state_dir=self.state_dir,
                gateway=self.gateway,
            )
        except (StoreError, OSError):
            pass  # hook failures never change the command's result


# --- the command ----------------------------------------------------------------------------------


def dry_run_outcome(td: Teardown, agent_id: str) -> Outcome:
    """Steps 1 and 2 without the compare-and-set; the liveness check only reports.

    TODO(#31): the liveness check of step 2 is reported here once issue #31 lands."""
    row = td.store.get_agent(agent_id)
    if row is None or row["status"] == "torn_down":
        raise td.torn_down(agent_id, row is None)
    would_refuse: str | None = None
    if row["status"] == "tearing_down":
        try:
            td.dead_teardown_process(row)
        except CliError as error:
            if error.code != "refused":
                raise
            would_refuse = error.reason
    if would_refuse is None:
        would_refuse = td.live_reason(row)  # TODO(#31): includes the liveness check
    if would_refuse is None:  # a refusal the live checks report wins over "not implemented"
        unsupported = td.unsupported(row)
        if unsupported is not None:
            raise unsupported
    worktrees, predicted = _dry_run_worktrees(td, row, ask_wtctl=would_refuse is None)
    would_refuse = would_refuse or predicted
    payload: dict[str, Any] = {
        "dry_run": True,
        "agent_id": agent_id,
        "would_refuse": would_refuse,
        "worktrees": worktrees,
    }
    return Outcome(
        20 if would_refuse is not None else 0,
        payload,
        PayloadKind.DRY_RUN,
        f"would refuse: {would_refuse}" if would_refuse else f"would tear down {agent_id}",
    )


_REASON = re.compile(r"[a-z][a-z0-9_]*")


# What wtctl's human ``--force`` overrides (wtctl interface, Remove order): the refusal reasons
# ``wtctl status`` reports for these (its ``force_overrides`` lists check names, not reasons).
_FORCE_OVERRIDES = ("held", "dirty", "unpushed")


def _predicted_refusal(td: Teardown, status: Mapping[str, Any]) -> str | None:
    """The refusal a ``wtctl status`` object predicts for ``remove`` with this call's flags.

    Anything but ``would_remove`` true or false (null: wtctl could not tell, a real remove
    would exit 12; missing or malformed) is ``unknown``: the dry run never fails open."""
    human = td.actor.startswith("human:")
    if status.get("human_required") is True and not human:
        return "confirmation_required"
    would_remove = status.get("would_remove")
    if would_remove is True:
        return None
    if would_remove is not False:
        return "unknown"
    reason = status.get("reason")
    if not isinstance(reason, str) or _REASON.fullmatch(reason) is None:
        return "unknown"
    if td.force and reason in _FORCE_OVERRIDES:
        return None
    return reason


def _dry_run_worktrees(
    td: Teardown, row: Mapping[str, Any], *, ask_wtctl: bool
) -> tuple[list[dict[str, Any]], str | None]:
    """The dry run's plan: one entry per record, and the refusal ``wtctl status`` predicts for
    the owned one (``wtctl`` is asked only when nothing else refuses already)."""
    agent_id = row["agent_id"]
    done_already = reached(row, "worktree_done")
    entries: list[dict[str, Any]] = []
    predicted: str | None = None
    try:
        items = td.finished_items(row) if done_already else td.worktree_plan(row)
    except CliError as error:
        if error.code != "refused" or error.reason != "not_held":
            raise
        items = td.finished_items(row)  # the stored record is not the agent's: refuse, plan held
        predicted, ask_wtctl = "not_held", False
    for item in items:
        status: dict[str, Any] | None = None
        action = "none" if item.released else ("removed" if item.owned else "released")
        if item.owned and not item.released and ask_wtctl:
            result = td.wtctl(row).status(item.record_id, agent_id, merged_head=td.merged_head)
            if result.already:
                action = "none"
            else:
                status = result.data
                predicted = predicted or _predicted_refusal(td, status or {})
        entries.append(
            {
                "record_id": item.record_id,
                "path": item.path,
                "owned": item.owned,
                "action": action,
                "wtctl_status": status,
            }
        )
    return entries, predicted


def teardown_command(inv: Any) -> Outcome:
    """The handler of ``teardown`` (``commands.ending``)."""
    args = inv.args
    hook_values = hooklib.parse_invocation_hooks(args.hook or [], TEARDOWN_HOOKS)
    state_dir = inv.state_dir or str(default_state_dir())
    if (args.force or getattr(args, "override_escalation", False)) and not inv.actor.startswith(
        "human:"
    ):
        raise CliError.usage("--force and --override-escalation need a human actor")
    try:
        store = Store.open(state_dir, create=False, gateway=gw.get_gateway())
    except StoreMissing:
        raise CliError.of("not_found", "there are no agents") from None
    except StoreError as error:
        raise _store_failure(error) from None
    agent_id: str | None = None
    td: Teardown | None = None
    try:
        agent_id = resolve_agent_id(args.agent_id, store).agent_id
        td = Teardown(
            store,
            inv.config,
            state_dir=state_dir,
            actor=inv.actor,
            decision_ids=inv.decision_ids,
            merged_head=args.merged_head,
            force=bool(args.force),
            override_escalation=bool(getattr(args, "override_escalation", False)),
            hooks=hook_values,
            gateway=gw.get_gateway(),
        )
        td.warnings.extend(inv.warnings)
        resolved = store.get_agent(agent_id)
        if resolved is not None and resolved["status"] == "torn_down":
            raise td.torn_down(agent_id, False)
        if args.dry_run:
            return dry_run_outcome(td, agent_id)
        return _run(td, agent_id)
    except CasLost:
        raise CliError(
            "failed", "this teardown no longer owns the agent", agent_id=agent_id
        ) from None
    except StoreError as error:
        raise _store_failure(error, agent_id) from None
    finally:
        # A dry run runs no hooks.
        if td is not None and agent_id is not None and not args.dry_run:
            td.deliver(agent_id)
        store.close()


def _run(td: Teardown, agent_id: str) -> Outcome:
    row, retry = td.enter(agent_id)
    done = td.run_from_step_3(row, retry=retry, abort_on_refusal=True)
    payload: dict[str, Any] = {
        "agent": agent_object(td.store, done, warnings=td.warnings),
        "worktrees": td.worktrees,
        "session_forgotten": td.session_forgotten,
        "windows_closed": td.windows_closed,
        "warnings": list(td.warnings),
    }
    return Outcome(0, payload, PayloadKind.COMMAND, f"torn down {agent_id}")


# --- the supervisor's ephemeral teardown ----------------------------------------------------------


def ephemeral(
    store: Store,
    agent_id: str,
    *,
    config: Config,
    state_dir: str,
    backend: SessionBackend,
    log: Callable[..., None],
    gateway: gw.Gateway | None = None,
) -> str | None:
    """Teardown from step 3 under automation rules (no ``--force``, no human-only flag) for an
    agent that is ``tearing_down`` with the calling supervisor as its teardown process.

    Returns the reason it was refused at a step that changed nothing (the supervisor then parks
    the agent ``done`` with ``teardown_refused``), or ``None``: the agent is a tombstone, or a
    failure left it ``tearing_down`` (logged), which ``teardown`` or ``prune`` retries.
    """
    td = Teardown(
        store,
        config,
        state_dir=state_dir,
        actor=SUPERVISOR_ACTOR,
        backend=backend,
        gateway=gateway,
        log=log,
    )
    try:
        row = store.get_agent(agent_id)
        if not td.mine(row) or row is None:
            log("warning", level="warn", message="ephemeral teardown: the agent is not ours")
            return None
        refusal = td.unsupported(row)
        if refusal is not None:
            return UNSUPPORTED
        td.run_from_step_3(row, retry=False, abort_on_refusal=False)
    except CliError as error:
        if td.step4_refused is not None:  # a worktree call refused: nothing of that record changed
            log("warning", level="warn", message=f"ephemeral teardown refused: {error.message}")
            return td.step4_refused
        if error.code == "refused" and not td.past_refusals:
            return error.reason or error.code
        log("warning", level="warn", message=f"ephemeral teardown stopped: {error.message}")
    except (StoreError, ValueError, OSError) as error:
        log("warning", level="warn", message=f"ephemeral teardown stopped: {type(error).__name__}")
    return None
