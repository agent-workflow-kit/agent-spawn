"""The inline liveness check (interface.md, Liveness).

``list``, ``show``, ``wait``, the launcher's polls, ``resume``, ``teardown``, ``stop``, ``prune``
and ``reconcile`` call ``Liveness.check`` for the agents they touch. The check looks at the
processes the row records (supervisor, launcher) by PID, start time and boot id, and, only when
the recorded supervisor (or, before one exists, the launcher) is dead, at the session
(``claude-session status``) to decide what is true:

* ``launching``, no supervisor, a dead launcher: a session found by its ``agent=<id>`` label is
  recorded, a plain resume whose run does not exist gets its pre-resume row back, an abandoned
  fork is removed, any other launch parks ``launch_failed``;
* a pending run whose supervisor is dead is left to claude-session's own expiry
  (``never_adopted``) and reported (``pending_run``); it never sets ``supervisor_lost``;
* a final run is parked per the outcome table (never with the automatic report resume or the
  promotion: they belong to a supervisor); a run that was not the first run of a launch or
  resume and ended in its launch phase parks per the run that triggered it (the chain rule);
* a ``live`` session whose supervisor is dead stays ``live`` with ``supervisor_lost`` and
  ``needs_attention: lost`` (once); ``agent-spawn stop`` is the remedy;
* a session claude-session no longer knows parks ``session_missing``.

Safety. The check never signals anything and never calls ``claude-session stop``: it reads
(``status``, ``list``) and writes agent-spawn's own state. Every write is a compare-and-set under
the agent's lease whose guard names the exact process and run that were judged, so of several
concurrent checks exactly one applies a recovery and the others find it applied; a row that moved
in between is left alone. No mutating lower-tool call and no hook runs while the lease is held
(the check may read ``claude-session status`` before its compare-and-set), and the check never
delivers hooks: its events wait in the outbox for the next mutating command, the
supervisor or ``reconcile``. In a dry run it only reports (the returned row is the one the real
check would leave).
"""

from __future__ import annotations

import json
import os
import shlex
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from . import chain, lifecycle, procs
from . import gateway as gw
from .agent_view import outcome_object, ts
from .outcome_table import CANCEL_END_REASONS, LAUNCH_PHASE_OUTCOMES, park_row
from .output import CliError
from .procs import ProcessIdentity
from .records import LOG_NAME, RecordsError, append_log, parse_record_ts
from .session_backend import RunOutcome, SessionBackend, SessionStatus
from .store import LeaseHeld, Store
from .worktrees import WtctlBackend, worktrees_containing

__all__ = [
    "Change",
    "Checked",
    "Liveness",
]

#: ``reconcile`` change kinds this module reports (interface.md, ``reconcile``).
PARKED = "parked"
SUPERVISOR_LOST = "supervisor_lost"
LAUNCH_FAILED = "launch_failed"
SESSION_MISSING = "session_missing"
PENDING_RUN = "pending_run"
AMBIGUOUS_SESSION = "ambiguous_session"
FORK_ABANDONED = "fork_abandoned"
#: Not a ``reconcile`` kind of its own in the interface: a restored pre-resume row is reported as
#: ``parked`` (the agent is parked again) with ``details.restored``.
RESTORED = "restored"
#: How much of the end of ``log.jsonl`` is read to find out whether a change was logged already.
_LOG_TAIL = 256 * 1024
WtctlFor = Callable[[Mapping[str, Any]], WtctlBackend | None]


@dataclass
class Change:
    """One recovery a check applied (or, in a dry run, would apply)."""

    agent_id: str
    kind: str
    details: dict[str, Any] = field(default_factory=lambda: {})


@dataclass
class Checked:
    """The result of a check: the agent's row afterwards (``None`` when its record is gone), the
    changes made, and warnings for what could not be judged (a failing lower tool, a busy
    lease). ``removed`` is true for an abandoned fork whose record was deleted."""

    row: dict[str, Any] | None
    changes: list[Change] = field(default_factory=lambda: [])
    warnings: list[str] = field(default_factory=lambda: [])
    removed: bool = False

    @property
    def session_missing(self) -> bool:
        return any(change.kind == SESSION_MISSING for change in self.changes)


@dataclass
class _Ctx:
    """One agent's check: the row judged, and what is collected."""

    row: dict[str, Any]
    changes: list[Change] = field(default_factory=lambda: [])
    warnings: list[str] = field(default_factory=lambda: [])
    removed: bool = False
    #: Dry run only: the fields the real check would have written.
    predicted: dict[str, Any] = field(default_factory=lambda: {})

    @property
    def agent_id(self) -> str:
        return self.row["agent_id"]


def row_moved(before: Mapping[str, Any] | None, after: Mapping[str, Any] | None) -> bool:
    """Whether a check really moved the agent (a polling loop looks again at once only then;
    a report-only change such as ``pending_run`` is not news)."""
    if before is None or after is None:
        return before is not after
    names = ("status", "updated", "supervisor_lost", "csession_id")
    return any(before.get(name) != after.get(name) for name in names)


def unreadable(agent_id: str, what: str) -> CliError:
    """An unreadable process entry is an error, never "dead" (interface.md, Liveness)."""
    return CliError(
        "failed",
        f"the {what}'s process entry could not be read",
        agent_id=agent_id,
        exit_code=12,
    )


class Liveness:
    """The inline liveness check over one open store.

    ``fallback_command`` is claude-session's argv for an agent whose row stores none;
    ``wtctl_for`` builds the wtctl backend for an abandoned fork's hold release (``None``: no
    wtctl on this machine). ``dry_run`` reports only.
    """

    def __init__(
        self,
        store: Store,
        *,
        state_dir: str,
        fallback_command: Sequence[str] | None = None,
        wtctl_for: WtctlFor | None = None,
        dry_run: bool = False,
        gateway: gw.Gateway | None = None,
    ) -> None:
        self.store = store
        self.state_dir = state_dir
        self.fallback_command = list(fallback_command) if fallback_command else None
        self.wtctl_for = wtctl_for
        self.dry_run = dry_run
        self.gateway = gateway if gateway is not None else gw.get_gateway()

    # -- entry points ------------------------------------------------------------------------

    def check(self, agent_id: str) -> Checked:
        """Check one agent; a no-op unless it is ``launching`` or ``live``.

        Raises ``CliError`` (exit 12) when a recorded process entry cannot be read."""
        row = self.store.get_agent(agent_id)
        if row is None:
            return Checked(None)
        if row["status"] not in ("launching", "live"):
            return Checked(row)
        ctx = _Ctx(row)
        try:
            self._check(ctx)
        except LeaseHeld:
            # Another process holds the agent's lease (it is changing the agent right now): it
            # is the one recovering, or a lifecycle entry the next check will see.
            ctx.warnings.append(f"agent {agent_id} is busy: its liveness was not judged")
        return self._result(ctx)

    def session_missing(self, agent_id: str, session: str) -> Checked:
        """A command's own ``claude-session`` call found ``session`` unknown (exit 10): park the
        agent ``session_missing`` (a no-op unless it is ``launching`` or ``live`` and still has
        that session)."""
        row = self.store.get_agent(agent_id)
        if row is None or row["status"] not in ("launching", "live"):
            return Checked(row)
        if row.get("csession_id") != session:
            return Checked(row)
        ctx = _Ctx(row)
        try:
            self.park_session_missing(ctx)
        except LeaseHeld:
            ctx.warnings.append(f"agent {agent_id} is busy: it was not parked")
        return self._result(ctx)

    def check_all(self, rows: Sequence[Mapping[str, Any]] | None = None) -> list[Checked]:
        """Check every ``launching`` or ``live`` agent (of ``rows``, else of the store)."""
        found = rows if rows is not None else self.store.list_agents()
        return [
            self.check(row["agent_id"]) for row in found if row["status"] in ("launching", "live")
        ]

    def _result(self, ctx: _Ctx) -> Checked:
        if ctx.removed:
            return Checked(None, ctx.changes, ctx.warnings, removed=True)
        row = ctx.row
        if ctx.predicted:
            return Checked({**row, **ctx.predicted}, ctx.changes, ctx.warnings)
        fresh = self.store.get_agent(ctx.agent_id)
        return Checked(fresh, ctx.changes, ctx.warnings)

    # -- processes ---------------------------------------------------------------------------

    def alive(self, identity: object, agent_id: str, what: str) -> bool:
        """Whether the recorded process is the very one still running (a zombie is dead)."""
        if not isinstance(identity, ProcessIdentity):
            return False
        try:
            return procs.is_alive(identity, gateway=self.gateway)
        except procs.ProcessIdentityError:
            raise unreadable(agent_id, what) from None

    def _check(self, ctx: _Ctx) -> None:
        row = ctx.row
        supervisor = row.get("supervisor")
        if supervisor is not None:
            # The recorded supervisor decides, whether or not the launcher is alive.
            if self.alive(supervisor, ctx.agent_id, "supervisor"):
                return
        elif row["status"] == "launching":
            if self.alive(row.get("launcher"), ctx.agent_id, "launcher"):
                return
            self._dead_launcher(ctx)
            return
        self._dead_supervisor(ctx)

    # -- lower tools -------------------------------------------------------------------------

    def backend(self, row: Mapping[str, Any]) -> SessionBackend | None:
        stored = row.get("claude_session_path")
        command = self.fallback_command
        if isinstance(stored, str) and stored:
            # The agent's own claude-session while it still runs; a stale path (an upgrade
            # removed it) falls back to the configured command.
            try:
                argv = shlex.split(stored)
            except ValueError:
                argv = []
            if (
                argv
                and os.path.isabs(argv[0])
                and os.path.isfile(argv[0])
                and os.access(argv[0], os.X_OK)
            ):
                command = argv
        if not command:
            return None
        return SessionBackend(command, gateway=self.gateway)

    def _status(self, ctx: _Ctx, session: str) -> SessionStatus | None:
        """The session's ``status``; ``None`` after a warning (or after parking the agent
        ``session_missing`` when claude-session does not know the session)."""
        backend = self.backend(ctx.row)
        if backend is None:
            ctx.warnings.append("claude-session is not known: liveness was not judged")
            return None
        try:
            status = backend.status(session)
        except CliError as error:
            if error.tool == "claude-session" and error.exit_code == 10:
                self.park_session_missing(ctx)
            else:
                ctx.warnings.append(f"claude-session status failed ({error.code})")
            return None
        if status.csession_id != session:
            ctx.warnings.append("claude-session answered for another session")
            return None
        return status

    # -- recoveries --------------------------------------------------------------------------

    def _guard(self, row: Mapping[str, Any]) -> lifecycle.Guard:
        """The row must still be the one judged: the same status, supervisor, launcher, session
        and run (a process or a run that changed means another command moved the agent)."""
        judged = {
            name: row.get(name)
            for name in ("status", "supervisor", "launcher", "csession_id", "run")
        }

        def same(fresh: dict[str, Any]) -> bool:
            return all(fresh.get(name) == value for name, value in judged.items())

        return same

    def _note(self, ctx: _Ctx, kind: str, *, log: bool = True, **details: Any) -> None:
        ctx.changes.append(Change(ctx.agent_id, kind, dict(details)))
        if log:
            self.log("reconcile_change", agent_id=ctx.agent_id, details={"kind": kind, **details})

    def _logged_pending(self, ctx: _Ctx, run: int) -> bool:
        """Whether ``pending_run`` was logged for this run already (the recent log is read:
        a pending run is checked on every poll and is logged once, not on every call).

        Best effort: only the last ``_LOG_TAIL`` bytes (256KB) of the log are read, so the line
        is repeated once per 256KB of other log writes while the run stays pending, and two
        concurrent checks may both log it. Duplicate lines are harmless."""
        path = os.path.join(self.state_dir, LOG_NAME)
        try:
            with open(path, "rb") as handle:
                size = handle.seek(0, os.SEEK_END)
                handle.seek(max(0, size - _LOG_TAIL))
                tail = handle.read().decode("utf-8", "replace")
        except OSError:
            return False
        for line in tail.splitlines():
            try:
                record = json.loads(line)
            except ValueError:
                continue
            details = record.get("details") if isinstance(record, dict) else None
            if (
                isinstance(details, dict)
                and record.get("event") == "reconcile_change"  # pyright: ignore[reportUnknownMemberType]
                and record.get("agent_id") == ctx.agent_id  # pyright: ignore[reportUnknownMemberType]
                and details.get("kind") == PENDING_RUN  # pyright: ignore[reportUnknownMemberType]
                and details.get("run") == run  # pyright: ignore[reportUnknownMemberType]
            ):
                return True
        return False

    def log(self, event: str, *, level: str = "info", **fields: Any) -> None:
        if self.dry_run:
            return
        try:
            append_log(self.state_dir, event, level=level, **fields)
        except (RecordsError, ValueError, OSError):
            pass  # a log that cannot be written never stops a recovery

    def _park(
        self,
        ctx: _Ctx,
        *,
        park_reason: str,
        needs_attention: str | None,
        escalated_by: str | None = None,
        changes: Mapping[str, object] | None = None,
        outcome: Mapping[str, Any] | None = None,
        chain_failure: Mapping[str, Any] | None = None,
        runs: Mapping[int, Mapping[str, object]] | None = None,
    ) -> bool:
        """Park the agent; whether this check parked it (a lost race is not an error)."""
        if self.dry_run:
            ctx.predicted = {
                "status": "parked",
                "park_reason": park_reason,
                "needs_attention": needs_attention,
                "escalated_by": escalated_by,
                "promotion_pending": False,
                "supervisor_lost": False,
                **(changes or {}),
            }
            return True
        parked = lifecycle.park(
            self.store,
            ctx.agent_id,
            park_reason=park_reason,
            needs_attention=needs_attention,
            escalated_by=escalated_by,
            changes=changes,
            outcome=outcome,
            chain_failure=chain_failure,
            guard=self._guard(ctx.row),
            runs=runs,
        )
        if parked is None:
            return False
        self.log(
            "status_change",
            level="warn",
            agent_id=ctx.agent_id,
            details={"to": "parked", "park_reason": park_reason, "cause": "liveness"},
        )
        return True

    def park_session_missing(self, ctx: _Ctx) -> None:
        if self._park(ctx, park_reason="session_missing", needs_attention="session_missing"):
            self._note(ctx, SESSION_MISSING)

    def _launch_failed(self, ctx: _Ctx, **details: Any) -> None:
        if self._park(ctx, park_reason="launch_failed", needs_attention="launch_failed"):
            self._note(ctx, LAUNCH_FAILED, **details)

    # -- launching, no supervisor, a dead launcher -------------------------------------------

    def _dead_launcher(self, ctx: _Ctx) -> None:
        row = ctx.row
        agent_id = ctx.agent_id
        session = row.get("csession_id")
        if isinstance(session, str):
            status = self._status(ctx, session)
            if status is not None:
                self._by_session(ctx, status, driven=False)
            return
        backend = self.backend(row)
        if backend is None:
            ctx.warnings.append("claude-session is not known: liveness was not judged")
            return
        # A crash between `start` and the recording of its session: the session carries the
        # `agent=<id>` label.
        try:
            labeled = backend.list_labeled(f"agent={agent_id}")
        except CliError as error:
            ctx.warnings.append(f"claude-session list failed ({error.code})")
            return
        found = [session for session, _labels in labeled]
        if len(found) > 1:
            self._ambiguous(ctx, found)
            return
        if len(found) == 1:
            session, labels = labeled[0]
            reason = self._not_ours(ctx, session, labels)
            if reason is not None:
                # Not adopted: a session of another agent, or labeled for more than this one.
                ctx.warnings.append(f"the session {session} {reason}: it was not adopted")
                self._ambiguous(ctx, found)
                return
            self._adopt_found(ctx, session)
            return
        if row.get("forked_from") is not None:
            self._abandon_fork(ctx)
        elif row.get("pre_resume") is not None:
            self._restore(ctx, "no session")
        else:
            self._launch_failed(ctx, cause="launcher_died")

    def _ambiguous(self, ctx: _Ctx, sessions: Sequence[str]) -> None:
        """The agent's session cannot be told: ``launch_failed`` without recording any."""
        if self._park(ctx, park_reason="launch_failed", needs_attention="launch_failed"):
            self._note(ctx, AMBIGUOUS_SESSION, sessions=sorted(sessions))
            self._note(ctx, LAUNCH_FAILED, cause="ambiguous_session")

    def _not_ours(self, ctx: _Ctx, session: str, labels: list[str] | None) -> str | None:
        """Why the session found by label is not safely this agent's, else ``None``: it is
        recorded on another agent, or its ``agent=`` labels are not exactly this agent's."""
        for other in self.store.list_agents():
            if other["agent_id"] != ctx.agent_id and other.get("csession_id") == session:
                return f"belongs to agent {other['agent_id']}"
        if labels is not None:
            owners = [label for label in labels if label.startswith("agent=")]
            if owners != [f"agent={ctx.agent_id}"]:
                return "does not carry exactly this agent's label"
        return None

    def _adopt_found(self, ctx: _Ctx, session: str) -> None:
        """Record the one session found by label (and its current run), then judge it."""
        status = self._status(ctx, session)
        if status is None:
            return
        row = ctx.row
        if self.dry_run:
            ctx.predicted = {"csession_id": session, "run": status.run.run}
        else:
            recorded = lifecycle.record_session(
                self.store,
                ctx.agent_id,
                session=session,
                run=status.run.run,
                run_fields=self._new_run_fields(row, status),
                guard=self._guard(row),
            )
            if recorded is None:
                return  # another process recorded it first: the next check judges it
            ctx.row = recorded
            self.log(
                "reconcile_change",
                agent_id=ctx.agent_id,
                csession_id=session,
                details={"kind": "session_recorded"},
            )
        self._by_session(ctx, status, driven=False)

    def _new_run_fields(self, row: Mapping[str, Any], status: SessionStatus) -> dict[str, object]:
        return {
            "run_kind": status.run.run_kind,
            "plan_run_kind": status.run.run_kind,
            "mode": row.get("mode"),
            "placement": row.get("placement"),
            "cwd": row.get("cwd"),
            "reads_third_party": status.run.reads_third_party,
        }

    def _restore(self, ctx: _Ctx, why: str) -> None:
        """The pre-resume park fields come back (a plain resume whose launcher died before its
        run existed): no hooks, logged ``interrupted``."""
        if self.dry_run:
            saved = ctx.row.get("pre_resume")
            ctx.predicted = {
                **(saved if isinstance(saved, dict) else {}),
                "status": "parked",
                "launcher": None,
                "pre_resume": None,
            }
            self._note(ctx, PARKED, restored=True, reason=why)
            return
        restored = lifecycle.restore_pre_resume(
            self.store, ctx.agent_id, guard=self._guard(ctx.row)
        )
        if restored is None:
            return
        self.log(
            "interrupted",
            level="warn",
            agent_id=ctx.agent_id,
            details={"cause": "launcher_died", "restored": True},
        )
        self._note(ctx, PARKED, restored=True, reason=why)

    def _abandon_fork(self, ctx: _Ctx) -> None:
        """A fork whose launcher died before its session existed: release its holds, delete its
        prompt copies and its record (the id stays reserved). A hold that cannot be released
        keeps the record for the next check."""
        agent_id = ctx.agent_id
        if self.dry_run:
            ctx.removed = True
            self._note(ctx, FORK_ABANDONED)
            return
        if not self._still_abandoned(ctx):
            return  # something moved since the first look: no side effect, the next check judges
        failed = self._release_fork_holds(ctx)
        if failed:
            return  # the record and its holds stay: the warnings say why
        from .teardown import remove_agent_files  # late: teardown calls this module

        fresh = self.store.get_agent(agent_id)
        if fresh is None or not self._guard(ctx.row)(fresh):
            return  # moved meanwhile: nothing is deleted
        problems = remove_agent_files(self.state_dir, agent_id)
        if problems:
            ctx.warnings.append("the fork's prompt copies could not all be deleted")
        # The guard is judged again under the lease that deletes the record.
        with self.store.lease(lifecycle.lease_key(agent_id), wait=lifecycle.LEASE_WAIT) as lease:
            fresh = self.store.get_agent(agent_id)
            if fresh is None or not self._guard(ctx.row)(fresh):
                return
            self.store.delete_agent(lease, agent_id)
        ctx.removed = True
        self.log("fork_abandoned", level="warn", agent_id=agent_id)
        self._note(ctx, FORK_ABANDONED)

    def _still_abandoned(self, ctx: _Ctx) -> bool:
        """Right before anything is released: the fork's record is still the one judged and
        claude-session still knows no session for it (a session that appeared meanwhile, or a
        row that moved, aborts the abandonment with no side effect)."""
        backend = self.backend(ctx.row)
        if backend is None:
            ctx.warnings.append("claude-session is not known: liveness was not judged")
            return False
        try:
            found = backend.list_by_label(f"agent={ctx.agent_id}")
        except CliError as error:
            ctx.warnings.append(f"claude-session list failed ({error.code})")
            return False
        if found:
            return False
        fresh = self.store.get_agent(ctx.agent_id)  # read last: nothing can move it unseen
        return fresh is not None and self._guard(ctx.row)(fresh)

    def _release_fork_holds(self, ctx: _Ctx) -> bool:
        """``wtctl forget <record> --holder <fork id>`` for every record the fork holds: its hold
        rows and the records of its cwd that list it as a holder (a crash between ``adopt`` and
        the row would hide one). ``not_a_holder`` (exit 0) and exit 10 count as released. True
        when something could not be released (warned)."""
        row = ctx.row
        agent_id = ctx.agent_id
        record_ids = [hold["record_id"] for hold in self.store.get_holds(agent_id)]
        backend = self.wtctl_for(row) if self.wtctl_for is not None else None
        cwd = row.get("cwd")
        failed = False
        if backend is not None and isinstance(cwd, str) and os.path.isabs(cwd):
            try:
                for record in worktrees_containing(backend.list_records(), cwd):
                    if agent_id in record.holders and record.id not in record_ids:
                        record_ids.append(record.id)
            except CliError as error:  # the holds are unknown: the fork must stay
                ctx.warnings.append(f"wtctl list failed ({error.code}): the fork's holds stay")
                failed = True
        if backend is None and record_ids:
            ctx.warnings.append("wtctl is not available: the fork's holds stay")
            return True
        for record_id in record_ids:
            try:
                assert backend is not None
                backend.forget(record_id, agent_id)
            except CliError as error:
                if error.exit_code == 10:
                    continue
                ctx.warnings.append(f"could not release the fork's hold on {record_id}")
                failed = True
        return failed

    # -- a dead supervisor -------------------------------------------------------------------

    def _dead_supervisor(self, ctx: _Ctx) -> None:
        session = ctx.row.get("csession_id")
        if not isinstance(session, str):
            # `live` or `launching` with a recorded supervisor needs a session; without one the
            # row is damaged: nothing to read, nothing to decide.
            ctx.warnings.append("the agent has no session: liveness was not judged")
            return
        status = self._status(ctx, session)
        if status is not None:
            self._by_session(ctx, status, driven=True)

    # -- by the session's current run --------------------------------------------------------

    def _by_session(self, ctx: _Ctx, status: SessionStatus, *, driven: bool) -> None:
        """Nothing drives the agent. ``driven``: a supervisor was recorded (so the run, if any,
        is the supervisor's); otherwise this is a launch or resume whose launcher died."""
        row = ctx.row
        current = status.run
        if (
            not driven
            and row.get("pre_resume") is not None
            and status.current_run <= (row.get("expect_run_after") or 0)
        ):
            self._restore(ctx, "no run")  # the resume's run does not exist
            return
        if current.outcome is not None:
            self._park_final(ctx, status)
            return
        if current.pending or status.pending:
            # Nothing adopted the run: claude-session's own expiry ends it (`never_adopted`).
            once = not self._logged_pending(ctx, current.run)
            self._note(ctx, PENDING_RUN, log=once, run=current.run)
            return
        # The run is adopted and live.
        if row["status"] == "launching":
            if current.phase != "session":
                return  # adopted but still in its launch phase: stays `launching`, nothing set
            if not self._go_live(ctx, current.run):
                return
        self._supervisor_lost(ctx)

    def _go_live(self, ctx: _Ctx, run: int) -> bool:
        """The started transition (and ``post-launch``) a dead supervisor did not get to apply."""
        if self.dry_run:
            ctx.predicted = {
                **ctx.predicted,
                "status": "live",
                "needs_attention": None,
                "run": run,
            }
            return True
        moved = lifecycle.go_live(
            self.store,
            ctx.agent_id,
            run=run,
            changes={"last_live_at": self.gateway.wall_time(), "run": run},
            guard=self._guard(ctx.row),
        )
        if moved is None:
            return False
        ctx.row = moved
        return True

    def _supervisor_lost(self, ctx: _Ctx) -> None:
        row = ctx.row
        if row.get("supervisor_lost"):
            return  # flagged already: once (`needs_attention` is not set again)
        if self.dry_run:
            ctx.predicted = {
                **ctx.predicted,
                "supervisor_lost": True,
                "needs_attention": "lost",
            }
            self._note(ctx, SUPERVISOR_LOST)
            return
        moved = lifecycle.set_supervisor_lost(
            self.store,
            ctx.agent_id,
            guard=lambda fresh: (
                fresh.get("supervisor") == row.get("supervisor")
                and fresh.get("csession_id") == row.get("csession_id")
            ),
        )
        if moved is not None:
            self.log("attention_set", level="warn", agent_id=ctx.agent_id, details={"to": "lost"})
            self._note(ctx, SUPERVISOR_LOST)

    # -- a final run -------------------------------------------------------------------------

    def _outcome(self, ctx: _Ctx, status: SessionStatus) -> RunOutcome:
        """The final outcome of the session's current run: its outcome file when the run row
        names a readable one, else ``status``."""
        result = status.outcome()
        run_row = self.store.get_run(ctx.agent_id, result.run) or {}
        token = run_row.get("outcome_token")
        backend = self.backend(ctx.row)
        if isinstance(token, str) and backend is not None:
            path = os.path.join(self.state_dir, "outcomes", ctx.agent_id, f"{token}.json")
            if os.path.exists(path):
                try:
                    return backend.run_outcome(path, status.csession_id, result.run)
                except CliError:
                    pass  # the status answer below is the same run
        return result

    def _park_final(self, ctx: _Ctx, status: SessionStatus) -> None:
        """Apply the outcome table to the run that ended while nothing was left to do it."""
        row = ctx.row
        agent_id = ctx.agent_id
        result = self._outcome(ctx, status)
        run = result.run
        run_row = self.store.get_run(agent_id, run)
        chain_run = self._chain_run(row, run_row)
        promoted = chain_run == "promoted"
        auto_resumed = chain_run == "auto_resumed"
        first = ts((run_row or {}).get("first_working_at"))
        outcome = outcome_object(
            agent_id,
            result,
            result.raw if result.raw is not None else status.raw.get("run"),
            promoted=promoted,
            auto_resumed=auto_resumed,
            first_working_at=first,
        )
        report = result.report
        report_status = report.get("status") if isinstance(report, dict) else None  # pyright: ignore[reportUnknownMemberType]
        run_fields: dict[str, object] = {
            "outcome": outcome,
            "end_reason": result.end_reason,
            "exit_code": result.exit_code,
            "ended_at": self.gateway.wall_time(),
        }
        if isinstance(report_status, str):
            run_fields["report_status"] = report_status
        if outcome["first_working_at"]:
            try:
                run_fields["first_working_at"] = parse_record_ts(
                    outcome["first_working_at"]
                ).timestamp()
            except ValueError:
                pass
        if run_row is None:
            run_fields.update(self._new_run_fields(row, status))
            if chain_run is not None:
                run_fields.update(
                    mode="interactive" if promoted else "headless",
                    placement=None,
                    auto_resumed=auto_resumed,
                    promoted=promoted,
                )
        table_outcome, trigger = result, None
        failure: chain.ChainFailure | None = None
        if chain_run is not None and self._ended_in_launch(result):
            trigger = self._trigger(agent_id, run)
            if trigger is not None and not (
                result.outcome == "not-started" and result.end_reason in CANCEL_END_REASONS
            ):
                # The chain rule: a supervisor-initiated run that ended in its launch phase
                # parks per the run that triggered it (a cancel by `stop` is its own row).
                failure = chain.ChainFailure(
                    "claude-session",
                    chain.reason_word(result.outcome),
                    run=run,
                    end_reason=result.end_reason,
                )
        reported = outcome
        if failure is not None and trigger is not None:
            reported = trigger
            table = park_row(str(trigger["outcome"]), trigger.get("end_reason"))
            last_end_reason = trigger.get("end_reason")
        else:
            table = park_row(table_outcome.outcome or "failed", table_outcome.end_reason)
            last_end_reason = result.end_reason
        changes: dict[str, object] = {
            "last_outcome": reported,
            "last_end_reason": last_end_reason,
            "last_live_at": self.gateway.wall_time(),
            "exec_proc": None,
            "run": run,
        }
        if self._park(
            ctx,
            park_reason=table.park_reason,
            needs_attention=table.needs_attention,
            escalated_by=table.escalated_by,
            changes=changes,
            outcome=reported,
            chain_failure=None if failure is None else failure.as_dict(),
            runs=None if self.dry_run else {run: run_fields},
        ):
            self._note(
                ctx,
                PARKED,
                outcome=reported.get("outcome"),
                park_reason=table.park_reason,
                run=run,
            )

    @staticmethod
    def _chain_run(row: Mapping[str, Any], run_row: Mapping[str, Any] | None) -> str | None:
        """``promoted`` or ``auto_resumed`` for a run a supervisor started (recorded as such, or
        unknown to agent-spawn while the agent is ``live``: the supervisor's own next run),
        else ``None`` (the first run of a launch or resume)."""
        if run_row is not None:
            if run_row.get("promoted"):
                return "promoted"
            return "auto_resumed" if run_row.get("auto_resumed") else None
        if row["status"] != "live":
            return None
        return "promoted" if row.get("promotion_pending") else "auto_resumed"

    @staticmethod
    def _ended_in_launch(result: RunOutcome) -> bool:
        return result.phase == "launch" and result.outcome in LAUNCH_PHASE_OUTCOMES

    def _trigger(self, agent_id: str, run: int) -> dict[str, Any] | None:
        """The outcome object of the last recorded final run before ``run`` (the run that
        triggered the chain run), else ``None``."""
        for stored in reversed(self.store.get_runs(agent_id)):
            outcome = stored.get("outcome")
            if stored["run"] < run and isinstance(outcome, dict) and "outcome" in outcome:
                return dict(outcome)  # pyright: ignore[reportUnknownArgumentType]
        return None
