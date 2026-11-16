"""``stop``: end an agent's current run chain deliberately (interface.md, ``stop`` and Signals).

After the liveness check the first action that applies is taken:

* ``torn_down`` or unknown: exit 10; ``parked``: a no-op (``already_parked``);
* a pending or live run (also a promotion's pending run and ``supervisor_lost``, and a
  ``tearing_down`` agent's orphaned run): ``claude-session stop`` (``session_stop``);
* a supervisor between runs: SIGTERM to the verified supervisor (``signal_supervisor``);
* ``launching`` before ``start`` returned: SIGTERM to the verified launcher (``signal_launcher``).

Then it waits for the bookkeeping, at most the grace plus 30 s. ``stop`` records nothing on the
agent: the supervisor (or the launcher) parks it per the outcome table.

Safety. A signal goes only to a process recorded on the agent row, verified by PID, start time
and boot id (``procs.forward_signal``), and only after the row was read again under the agent's
lease and still named the same status and process. A process that is gone, replaced or a zombie
is never signaled; the plan is made again from the new state (so a ``stop`` that races a resume,
a park or a launcher's exit acts on what is true now).
"""

from __future__ import annotations

import shlex
import signal
from dataclasses import dataclass
from typing import Any

from . import gateway as gw
from . import lifecycle, procs
from .agent_view import agent_object, outcome_object
from .ids import resolve_agent_id
from .launch import resolve_tool
from .outcome_table import park_row
from .output import CliError, Outcome, PayloadKind
from .procs import ProcessIdentity
from .records import RecordsError, append_audit, append_log
from .session_backend import SessionBackend, SessionStatus
from .store import LeaseHeld, Store, StoreError, StoreMissing, default_state_dir

WAIT_MARGIN = 30.0  # seconds added to the grace
# claude-session's kill_grace default. claude-session exposes it only as a config key of its own
# (not in status or stop output) and agent-spawn has no such key (decided in #13), so the wait
# budget assumes the default.
# todo(#36): the live suite should check that claude-session's default kill_grace is still 10 s.
DEFAULT_GRACE = 10.0
POLL = 0.1  # seconds between looks at the agent row while waiting
STATUS_EVERY = 1.0  # seconds between looks at `claude-session status` while waiting
ATTEMPTS = 3  # plans made when the state changes under a signal
IN_PROGRESS = ("launching", "live")


@dataclass
class Plan:
    action: str  # session_stop | signal_supervisor | signal_launcher | none
    row: dict[str, Any]
    session: str | None = None
    target: ProcessIdentity | None = None
    field: str | None = None  # the row column the target came from


class Stop:
    def __init__(self, inv: Any) -> None:
        self.inv = inv
        self.args = inv.args
        self.gateway = gw.get_gateway()
        self.state_dir = inv.state_dir or str(default_state_dir())
        self.warnings: list[str] = list(inv.warnings)
        self.dry_run = bool(self.args.dry_run)
        self.store: Store | None = None
        self.backend: SessionBackend | None = None
        self.agent_id = ""
        self.action = "none"  # the planned action, for the audit record
        self.result = "error"  # the audit result if the command ends without finish()
        self.audited = False

    # -- records ------------------------------------------------------------------------------

    def log(self, event: str, *, level: str = "info", **fields: Any) -> None:
        try:
            append_log(self.state_dir, event, level=level, agent_id=self.agent_id, **fields)
        except (RecordsError, ValueError, OSError) as error:
            self.warnings.append(f"could not write the log ({type(error).__name__})")

    def audit(self, row: dict[str, Any], result: str, action: str) -> None:
        assert self.store is not None
        try:
            agent = agent_object(self.store, row)
            append_audit(
                self.state_dir,
                actor=self.inv.actor,
                decision_ids=self.inv.decision_ids,
                target=self.agent_id,
                action="stop",
                inputs={
                    "labels": agent["labels"],
                    "repo": agent["repo"],
                    "worktree": agent["worktree"],
                    "csession_id": agent["csession_id"],
                    "merged_head": None,
                    "force": False,
                },
                result=result,
            )
        except (RecordsError, ValueError, OSError, StoreError) as error:
            self.warnings.append(f"could not write the audit log ({type(error).__name__})")
        self.audited = True
        self.log("stop", details={"action": action, "result": result})

    # -- the command --------------------------------------------------------------------------

    def run(self) -> Outcome:
        try:
            self.store = Store.open(self.state_dir, create=False)
        except StoreMissing:
            raise CliError.of("not_found", "no agents") from None
        try:
            return self.stop()
        finally:
            self.store.close()

    def stop(self) -> Outcome:
        assert self.store is not None
        store = self.store
        resolved = resolve_agent_id(self.args.agent_id, store)
        self.agent_id = resolved.agent_id
        row = store.get_agent(self.agent_id)
        if resolved.torn_down or row is None or row["status"] == "torn_down":
            raise CliError.of(
                "not_found",
                f"agent {self.agent_id} is torn down" if resolved.torn_down else "no such agent",
                agent_id=self.agent_id,
                details={"status": "torn_down"} if resolved.torn_down else None,
            )
        try:
            self.backend = self.make_backend(row)
            return self.act(row)
        except CliError as error:
            if error.agent_id is None:
                error.agent_id = self.agent_id
            self.result = "refused" if error.exit_code == 20 else "error"
            raise
        except LeaseHeld as held:
            raise CliError.of(
                "failed",
                "the agent is busy (its lease is held by another process)",
                exit_code=12,
                agent_id=self.agent_id,
            ) from held
        finally:
            # Whatever ends the command after this point (also a signal or an unexpected error)
            # leaves one audit record with the action that was planned.
            if not self.audited and not self.dry_run:
                try:
                    latest = store.get_agent(self.agent_id) or row
                except Exception:
                    latest = row
                try:
                    self.audit(latest, self.result, self.action)
                except Exception as error:  # never mask the exception that ended the command
                    self.warnings.append(f"could not write the audit log ({type(error).__name__})")

    def make_backend(self, row: dict[str, Any]) -> SessionBackend:
        stored = row.get("claude_session_path")
        argv = (
            shlex.split(stored)
            if isinstance(stored, str) and stored
            else resolve_tool(self.inv.config.claude_session_cmd, "claude-session")
        )
        return SessionBackend(
            argv,
            actor=self.inv.actor,
            decision_ids=self.inv.decision_ids,
            gateway=self.gateway,
        )

    def act(self, row: dict[str, Any]) -> Outcome:
        assert self.backend is not None and self.store is not None
        grace = self.args.grace
        budget = (self.grace_seconds(grace) if grace else DEFAULT_GRACE) + WAIT_MARGIN
        plan = self.plan(row)
        self.action = plan.action
        session_stop: dict[str, Any] | None = None
        settled = False
        # ATTEMPTS plans are acted on after a state change, plus the one made after the last
        # failed attempt, so the final plan is never discarded unseen.
        for attempt in range(ATTEMPTS + 1):
            if self.dry_run or plan.action == "none":
                settled = True
                break
            if plan.action == "session_stop":
                assert plan.session is not None
                session_stop = self.backend.stop(plan.session, grace=grace, dry_run=False)
                if session_stop.get("ended") is False:
                    raise CliError.of(
                        "failed",
                        "the run is still live after claude-session's stop",
                        exit_code=12,
                        agent_id=self.agent_id,
                        tool="claude-session",
                    )
                if not session_stop.get("already_final"):
                    settled = True
                    break
                session_stop = None  # the run ended on its own: what is left to stop is next
            elif self.send(plan):
                settled = True
                break
            if attempt == ATTEMPTS:
                break  # the final plan had its attempt
            fresh = self.store.get_agent(self.agent_id)
            if fresh is None:
                return self.finish(plan, None, None, removed=True)
            plan = self.plan(fresh)
            self.action = plan.action
        if not settled:
            if plan.action != "none":
                raise CliError.of(
                    "failed",
                    "the agent changed while it was being stopped (try again)",
                    exit_code=12,
                    agent_id=self.agent_id,
                )
        if self.dry_run and plan.action == "session_stop":
            assert plan.session is not None
            session_stop = self.backend.stop(plan.session, grace=grace, dry_run=True)
        if plan.action == "none" or self.dry_run:
            return self.finish(plan, session_stop, self.store.get_agent(self.agent_id))
        final = self.wait(plan, budget)
        return self.finish(plan, session_stop, final, removed=final is None)

    @staticmethod
    def grace_seconds(grace: str) -> float:
        from .validate import parse_duration

        return parse_duration(grace)

    # -- choosing the action ------------------------------------------------------------------

    def alive(self, identity: object) -> bool:
        return isinstance(identity, ProcessIdentity) and procs.is_alive(
            identity, gateway=self.gateway
        )

    def plan(self, row: dict[str, Any], depth: int = 0) -> Plan:
        assert self.backend is not None and self.store is not None
        status = row["status"]
        if status == "torn_down":
            raise CliError.of("not_found", "agent is torn down", agent_id=self.agent_id)
        if status == "parked":
            return Plan("none", row)
        session = row.get("csession_id")
        current: SessionStatus | None = None
        if isinstance(session, str):
            # TODO(#31): exit 10 (the session is missing) blocks stop with not_found for an
            # existing agent; the liveness check parks it as session_missing instead.
            current = self.backend.status(session)
            if current.run.outcome is None:  # pending or live
                return Plan("session_stop", row, session=session)
        if status == "tearing_down":
            return Plan("none", row)  # no pending or live run left to stop
        if self.orphan_applies(row, current):
            assert current is not None
            if self.dry_run:
                return Plan("none", row)  # a real run would park it
            parked = self.park_orphan(row, current)
            if parked is not None:
                return Plan("none", parked)
            # Another process changed the agent first: plan again from what is true now.
            fresh = self.store.get_agent(self.agent_id)
            if fresh is None:
                raise CliError.of(
                    "failed", "the agent record disappeared", exit_code=1, agent_id=self.agent_id
                )
            if depth + 1 >= ATTEMPTS:
                raise CliError.of(
                    "failed",
                    "the agent changed while it was being stopped (try again)",
                    exit_code=12,
                    agent_id=self.agent_id,
                )
            return self.plan(fresh, depth + 1)
        supervisor = row.get("supervisor")
        if self.alive(supervisor):
            return Plan("signal_supervisor", row, target=supervisor, field="supervisor")
        launcher = row.get("launcher")
        if status == "launching" and self.alive(launcher):
            return Plan("signal_launcher", row, target=launcher, field="launcher")
        raise CliError.of(
            "failed",
            "the agent has no live process to stop (run reconcile)",
            exit_code=12,
            agent_id=self.agent_id,
        )

    def orphan_applies(self, row: dict[str, Any], current: SessionStatus | None) -> bool:
        """A final run with no live supervisor (and, while ``launching``, no live launcher)."""
        if current is None or current.run.outcome is None or current.run.exit_code is None:
            return False
        if self.alive(row.get("supervisor")):
            return False
        return not (row["status"] == "launching" and self.alive(row.get("launcher")))

    def park_orphan(self, row: dict[str, Any], current: SessionStatus) -> dict[str, Any] | None:
        """The liveness rule for a run that is final while nothing is left to bookkeep it: park
        per the outcome table, without the automatic report resume or promotion. ``None`` when
        another process parked the agent first.

        TODO(#31): this belongs to the liveness check; remove it when that lands.
        """
        assert self.store is not None
        result = current.outcome()
        table = park_row(result.outcome or "failed", result.end_reason)
        outcome = outcome_object(self.agent_id, result, current.raw.get("run"))
        changes: dict[str, object] = {
            "last_outcome": outcome,
            "last_end_reason": result.end_reason,
            "last_live_at": self.gateway.wall_time(),
            "exec_proc": None,
            "supervisor_lost": False,
        }
        parked = lifecycle.park(
            self.store,
            self.agent_id,
            park_reason=table.park_reason,
            needs_attention=table.needs_attention,
            escalated_by=table.escalated_by,
            from_status=row["status"],
            changes=changes,
            outcome=outcome,
        )
        return parked

    # -- signaling ----------------------------------------------------------------------------

    def send(self, plan: Plan) -> bool:
        """SIGTERM to the planned process if the row still names it, under the agent's lease."""
        assert self.store is not None and plan.target is not None and plan.field is not None
        with self.store.lease(lifecycle.lease_key(self.agent_id), wait=lifecycle.LEASE_WAIT):
            row = self.store.get_agent(self.agent_id)
            if (
                row is None
                or row["status"] != plan.row["status"]
                or row.get(plan.field) != plan.target
            ):
                return False
            outcome = procs.forward_signal(plan.target, signal.SIGTERM, gateway=self.gateway)
        if not outcome.sent:
            return False
        self.log(
            "signal_sent",
            pid=outcome.pid,
            signal="SIGTERM",
            details={"to": plan.field},
        )
        return True

    # -- waiting for the bookkeeping ----------------------------------------------------------

    def wait(self, plan: Plan, budget: float) -> dict[str, Any] | None:
        """Wait until the agent parked and its supervisor exited, or (``tearing_down``) its run
        is final. ``None`` when the record disappeared. Exit 12 when the budget runs out."""
        assert self.store is not None and self.backend is not None
        deadline = self.gateway.monotonic() + budget
        planned_tearing = plan.row["status"] == "tearing_down"
        next_status = 0.0
        while True:
            row = self.store.get_agent(self.agent_id)
            if row is None:
                return None
            status = row["status"]
            now = self.gateway.monotonic()
            if status == "tearing_down":
                # Teardown owns the agent: done once its run is final (or it has no session).
                if now >= next_status:
                    next_status = now + STATUS_EVERY
                    session = row.get("csession_id")
                    if not isinstance(session, str) or self.backend.status(session).run.outcome:
                        return row
            elif planned_tearing:
                return row
            elif status not in IN_PROGRESS:
                if lifecycle.run_settled(row, supervisor_alive=self.alive(row.get("supervisor"))):
                    return row
            elif status in IN_PROGRESS and now >= next_status:
                next_status = now + STATUS_EVERY
                session = row.get("csession_id")
                if isinstance(session, str):
                    current = self.backend.status(session)
                    if self.orphan_applies(row, current):
                        parked = self.park_orphan(row, current)
                        if parked is not None:
                            return parked
            if now >= deadline:
                raise CliError.of(
                    "failed",
                    "the agent is still live after the wait",
                    exit_code=12,
                    agent_id=self.agent_id,
                    details={"status": status} if status == "launching" else None,
                )
            self.gateway.sleep(POLL)

    # -- the result ---------------------------------------------------------------------------

    def finish(
        self,
        plan: Plan,
        session_stop: dict[str, Any] | None,
        row: dict[str, Any] | None,
        *,
        removed: bool = False,
    ) -> Outcome:
        assert self.store is not None
        if removed and plan.action != "signal_launcher":
            raise CliError.of(
                "failed", "the agent record disappeared", exit_code=1, agent_id=self.agent_id
            )
        shown = row if row is not None else plan.row
        if not self.dry_run:
            self.audit(shown, "ok", plan.action)
        payload = {
            "agent": None if removed else agent_object(self.store, shown, warnings=self.warnings),
            "action": plan.action,
            "session_stop": session_stop if plan.action == "session_stop" else None,
            "already_parked": plan.action == "none",
            "removed": removed,
            "dry_run": self.dry_run,
            "warnings": self.warnings,
        }
        return Outcome(0, payload, PayloadKind.COMMAND)


def stop(inv: Any) -> Outcome:
    return Stop(inv).run()
