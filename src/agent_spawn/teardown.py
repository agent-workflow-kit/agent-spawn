"""``teardown`` for agents without worktrees or windows (interface.md, Teardown order).

The steps this module owns: 1 (resolve and usage checks), 2 (the live checks and the lifecycle
entry), 6 (forget the session) and 7 (the tombstone). The named markers (``teardown_step``) make
a retry continue where a dead teardown process stopped. The supervisor of an ephemeral agent runs
the same steps from step 3 (``ephemeral``).

TODO(#32): the ``pre-teardown`` hooks (step 3); until then an agent that has any configured or
given ``pre-teardown`` hook is refused before anything changes (a veto must never be skipped).
TODO(#23): worktrees (step 4) and windows (step 5); until then an agent that holds a record or
had a window is refused before anything changes.

Every lifecycle write is a compare-and-set under the agent's lease; no lower tool and no hook
runs while the lease is held.
"""

from __future__ import annotations

import os
import shlex
import stat
from collections.abc import Callable, Mapping, Sequence
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

TOOL = "claude-session"
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
        hooks: Sequence[Mapping[str, str]] = (),
        backend: SessionBackend | None = None,
        gateway: gw.Gateway | None = None,
        log: Callable[..., None] | None = None,
    ) -> None:
        self.store = store
        self.config = config
        self.state_dir = state_dir
        self.actor = actor
        self.decision_ids = list(decision_ids)
        self.merged_head = merged_head
        self.force = force
        self.hooks = [dict(item) for item in hooks]
        self.backend = backend
        self.gateway = gateway or gw.get_gateway()
        self.warnings: list[str] = []
        self.session_forgotten = False
        self.session_missing = False
        self.past_refusals = False  # step 6 started: a failure there is not a refusal
        self.hooks_before: object = None  # agents.hooks before this call's entry wrote it
        self.hooks_written = False
        self._log = log

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

    def audit(self, row: Mapping[str, Any]) -> None:
        """An audit record for a forced teardown (a removed worktree belongs to issue #23)."""
        try:
            append_audit(
                self.state_dir,
                actor=self.actor,
                decision_ids=self.decision_ids,
                target=row["agent_id"],
                action="teardown",
                inputs={
                    "labels": self.store.get_labels(row["agent_id"]).texts(),
                    "repo": row.get("repo"),
                    "worktree": None,
                    "csession_id": row.get("csession_id"),
                    "merged_head": self.merged_head,
                    "force": self.force,
                },
                result="ok",
            )
        except (RecordsError, ValueError, OSError) as error:
            self.warnings.append(f"could not write the audit log ({type(error).__name__})")

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
        """The pieces of teardown this issue does not implement, refused before any change."""
        agent_id = row["agent_id"]
        what: str | None = None
        if row.get("teardown_step") == "worktree_partial":
            what = "a partially removed worktree"
        elif self.store.get_holds(agent_id) or (
            row.get("wt_record_id") and not reached(row, "worktree_done")
        ):
            what = "worktrees"  # TODO(#23): steps 4 and 5 of the teardown order
        elif not reached(row, "windows_done") and any(
            run.get("term_id") or run.get("socket_path") for run in self.store.get_runs(agent_id)
        ):
            what = "windows"  # TODO(#23)
        elif not reached(row, "hooks_done") and self.pre_hook_entries():
            what = "pre-teardown hooks"  # TODO(#32)
        if what is None:
            return None
        return CliError(
            "internal",
            f"teardown with {what} is not implemented yet (tracked in issues #23 and #32)",
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

    def set_step(self, agent_id: str, marker: str) -> None:
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
                changes={"teardown_step": marker, "updated": self.store.gateway.wall_time()},
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
                        "worktrees": [],  # TODO(#23): one entry per record teardown processed
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
        if self.force:
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


def _dry_run(td: Teardown, agent_id: str) -> Outcome:
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
    payload: dict[str, Any] = {
        "dry_run": True,
        "agent_id": agent_id,
        "would_refuse": would_refuse,
        "worktrees": [],  # TODO(#23): wtctl status of the owned record
    }
    return Outcome(
        20 if would_refuse is not None else 0,
        payload,
        PayloadKind.DRY_RUN,
        f"would refuse: {would_refuse}" if would_refuse else f"would tear down {agent_id}",
    )


def teardown_command(inv: Any) -> Outcome:
    """The handler of ``teardown`` (``commands.ending``)."""
    args = inv.args
    hook_values = hooklib.parse_invocation_hooks(args.hook or [], TEARDOWN_HOOKS)
    state_dir = inv.state_dir or str(default_state_dir())
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
            hooks=hook_values,
            gateway=gw.get_gateway(),
        )
        td.warnings.extend(inv.warnings)
        resolved = store.get_agent(agent_id)
        if resolved is not None and resolved["status"] == "torn_down":
            raise td.torn_down(agent_id, False)
        if args.dry_run:
            return _dry_run(td, agent_id)
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
        "worktrees": [],
        "session_forgotten": td.session_forgotten,
        "windows_closed": [],
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
        if error.code == "refused" and not td.past_refusals:
            return error.reason or error.code
        log("warning", level="warn", message=f"ephemeral teardown stopped: {error.message}")
    except (StoreError, ValueError, OSError) as error:
        log("warning", level="warn", message=f"ephemeral teardown stopped: {type(error).__name__}")
    return None
