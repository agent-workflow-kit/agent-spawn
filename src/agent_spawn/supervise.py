"""``supervise``: the supervisor of an agent's run chain (interface.md, Supervision).

The supervisor runs ``claude-session exec`` for the agent's current run, notices that the run
started (by polling ``status``; the relay is only an early wake-up), reads the outcome when
``exec`` exits, applies the outcome table and parks the agent. A headless run that ended
``exited-unreported`` gets one automatic report resume (``chain.py``); a supervisor-initiated run
that cannot start, or ends in its launch phase, parks the agent per the run that triggered it
(the chain rule). An ephemeral agent whose run ended ``done`` hands over to teardown.

TODO(#27): windows, markers and the handoff (``--handoff``, interactive hosting).
TODO(#29): promotion and host mode (``--host``).
"""

from __future__ import annotations

import os
import shlex
import signal
import subprocess
import time
from types import FrameType
from typing import Any

from . import chain, lifecycle, procs
from . import gateway as gw
from . import hooks as hooklib
from .agent_view import outcome_object, ts
from .chain import Final, RunCtx
from .config import Config
from .outcome_table import CANCEL_END_REASONS, park_row
from .output import CliError, Outcome, PayloadKind
from .records import RecordsError, append_log, parse_record_ts
from .session_backend import RunOutcome, SessionBackend
from .store import Store, StoreError, StoreMissing

STATUS_EVERY = 1.0  # seconds between looks at `claude-session status` and the outbox
STATUS_TIMEOUT = 5.0  # seconds the watch loop's own `status` poll may take
RELAY_EVERY = 0.25  # seconds between looks at the relay's records (an early wake-up)
TICK = 0.05  # seconds between looks at `exec`
SUPERVISOR_ACTOR = "auto:agent-spawn"
# The supervisor writes nothing to its terminal (interface.md, Supervision): no payload.
DONE = Outcome(0, {}, PayloadKind.COMMAND, quiet=True)
OUTCOME_TRIES = 3  # reads of the final outcome before giving up
OUTCOME_RETRY = 0.2  # seconds between them
EXEC_GRACE = 30.0  # seconds `exec` gets after SIGTERM when this supervisor fails
# TODO(#27): an interactive run ignores SIGINT (the terminal delivers it to `claude`); until
# interactive hosting lands every supervisor is headless, so SIGINT is forwarded.
FORWARDED = (signal.SIGTERM, signal.SIGHUP, signal.SIGINT)
#: Outcomes a run can end with in its launch phase (a supervisor-initiated run that does so
#: stops the chain early).
LAUNCH_PHASE_OUTCOMES = frozenset({"not-started", "vetoed", "escalated"})


class Signals:
    """The supervisor's handlers record the signal for the main loop's bookkeeping and, while an
    ``exec`` child is known (``forwarder`` set), forward it themselves: the loop may be blocked
    in a lower-tool call for a long time, and the signal must reach ``exec`` promptly.

    The forwarder marks a signal before sending it, so a handler that interleaves with the loop's
    own forward cannot send it twice. The handler logs nothing (a log write could re-enter one
    in progress); the loop logs what ``take_sent`` returns.
    """

    def __init__(self) -> None:
        self.received: list[int] = []
        self.forwarder: procs.SignalForwarder | None = None
        self.sent: list[procs.SignalOutcome] = []

    def handle(self, signum: int, frame: FrameType | None) -> None:
        self.received.append(signum)
        forwarder = self.forwarder
        if forwarder is None or signum not in FORWARDED:
            return
        try:
            outcome = forwarder.forward(signum)
        except OSError:
            return  # the loop retries through its own forward
        if outcome is not None:
            self.sent.append(outcome)

    def take(self) -> list[int]:
        taken, self.received = self.received, []
        return taken

    def take_sent(self) -> list[procs.SignalOutcome]:
        taken, self.sent = self.sent, []
        return taken


def unavailable_teardown(store: Store, agent_id: str) -> str | None:
    """The ephemeral teardown (interface.md, Teardown order, from step 3).

    Returns the reason it was refused, or ``None`` when it completed (it then owns the
    tombstone and ``post-teardown``).

    TODO(#18): teardown itself lands with the teardown issue; until then an ephemeral agent
    is parked ``done`` with ``teardown_refused``.
    """
    return "teardown_unavailable"


class Supervisor:
    def __init__(self, inv: Any, signals: Signals) -> None:
        self.inv = inv
        self.signals = signals
        self.config: Config = inv.config
        self.state_dir: str = inv.state_dir
        self.gateway = gw.get_gateway()
        self.store: Store | None = None
        self.agent_id: str = inv.args.agent_id
        self.forwarder: procs.SignalForwarder | None = None
        self.child: subprocess.Popen[bytes] | None = None  # the current `exec`
        self.runner: hooklib.AsyncRunner | None = None
        self.terminated = False  # SIGTERM, SIGHUP or SIGINT arrived during this run chain
        self.teardown = unavailable_teardown
        self.relay_seen: tuple[object, ...] = ()
        self.fallback_sent: set[int] = set()

    def log(self, event: str, *, level: str = "info", **fields: Any) -> None:
        try:
            append_log(self.state_dir, event, level=level, agent_id=self.agent_id, **fields)
        except (RecordsError, ValueError, OSError):
            pass  # a log that cannot be written never stops the supervisor

    # -- startup -----------------------------------------------------------------------------

    def startup_check(self) -> dict[str, Any]:
        """The agent's record must name this process as its supervisor (exit 2 otherwise)."""
        assert self.store is not None
        row = self.store.get_agent(self.agent_id)
        me = self.store.holder_identity()
        problem = None
        if row is None:
            problem = "no such agent"
        elif row.get("supervisor") != me:
            problem = "the agent's record does not name this process as its supervisor"
        elif row["status"] not in ("launching", "live"):
            problem = f"the agent is {row['status']}"
        if problem is not None:
            self.log("warning", level="warn", message=f"supervise refused: {problem}")
            raise CliError.usage(f"supervise: {problem}")
        assert row is not None
        return row

    # -- the run -----------------------------------------------------------------------------

    def run(self) -> Outcome:
        args = self.inv.args
        if args.host or args.handoff is not None:
            raise CliError.of(
                "internal", "supervise --host and --handoff are not implemented yet (issue #27)"
            )  # TODO(#27): windows, the handoff and host mode.
        try:
            self.store = Store.open(self.state_dir, create=False, gateway=self.gateway)
        except StoreMissing:
            self.log("warning", level="warn", message="supervise refused: no state database")
            raise CliError.usage("supervise: there is no state database") from None
        try:
            row = self.startup_check()
            return self.supervise(row)
        finally:
            self.store.close()

    def backend(self, row: dict[str, Any], *, timeout: float | None = None) -> SessionBackend:
        extra: dict[str, Any] = {} if timeout is None else {"timeout": timeout}
        return SessionBackend(
            shlex.split(row["claude_session_path"]),
            actor=SUPERVISOR_ACTOR,
            decision_ids=row.get("decision_ids") or [],
            gateway=self.gateway,
            **extra,
        )

    def supervise(self, row: dict[str, Any]) -> Outcome:
        assert self.store is not None
        store = self.store
        backend = self.backend(row)
        poller = self.backend(row, timeout=STATUS_TIMEOUT)  # the watch loop never blocks long
        session = row["csession_id"]
        run = row["run"]
        if not isinstance(session, str) or not isinstance(run, int):
            raise CliError.of("internal", "the agent has no session to supervise")
        run_row = store.get_run(self.agent_id, run) or {}
        token = run_row.get("outcome_token")
        outcome_file = os.path.join(self.state_dir, "outcomes", self.agent_id, f"{token}.json")
        self.runner = hooklib.AsyncRunner(self.state_dir, gateway=self.gateway)
        ctx = RunCtx(run, outcome_file)

        if self.collect_signals():
            # No `exec` child yet: cancel our own pending run and park (nothing can adopt it).
            return self.park_canceled(backend, session, ctx)

        previous: Final | None = None
        while True:
            try:
                child = self.spawn_exec(row, session)
            except (OSError, StoreError) as error:
                # `exec` could not be started (or was stopped again because it could not be
                # recorded): the run it should have adopted must not stay pending.
                self.log("warning", level="error", message=f"could not start exec: {error}")
                if previous is None:
                    return self.park_canceled(backend, session, ctx)
                run_row = store.get_run(self.agent_id, ctx.run) or {}
                chain.cancel_pending(
                    self,
                    backend,
                    session,
                    ctx.run,
                    hooks=run_row.get("hooks"),
                    token=run_row.get("outcome_token"),
                )
                failure = chain.ChainFailure("agent-spawn", "failed", run=ctx.run)
                return self.park_chain_failure(previous, failure)
            self.watch(child, poller, session, ctx)
            self.collect_signals()
            final = self.end_of_run(backend, session, ctx)
            if previous is not None and self.ended_in_launch(final):
                if (
                    self.terminated
                    and final.result.outcome == "not-started"
                    and final.result.end_reason in CANCEL_END_REASONS
                ):
                    # A user's SIGTERM or SIGHUP canceled the chain run: not a chain failure;
                    # the Signals rule applies (killed, no attention).
                    return self.park_final(final)
                # A supervisor-initiated run that ended in its launch phase stops the chain.
                failure = chain.ChainFailure(
                    "claude-session",
                    chain.reason_word(final.result.outcome),
                    run=final.ctx.run,
                    end_reason=final.result.end_reason,
                )
                return self.park_chain_failure(previous, failure)
            if not self.wants_report_resume(row, final, previous):
                return self.park_final(final)
            try:
                ctx = chain.start_report_resume(self, backend, row, session, final)
            except chain.ChainFailure as failure:
                return self.park_chain_failure(final, failure)
            except chain.ChainCanceled:
                return self.park_final(final)
            previous = final

    # -- watching `exec` ---------------------------------------------------------------------

    def collect_signals(self) -> bool:
        """Note the signals that arrived; True when the chain must end (no resume, no more
        runs). The caller decides what to forward."""
        taken = self.signals.take()
        for received in taken:
            self.note_signal(received)
        return bool(taken)

    def note_signal(self, received: int) -> None:
        # TODO(#27): SIGINT of an interactive run is ignored, not a termination.
        if received in FORWARDED:
            self.terminated = True

    def spawn_exec(self, row: dict[str, Any], session: str) -> subprocess.Popen[bytes]:
        assert self.store is not None
        argv = [*shlex.split(row["claude_session_path"]), "exec", "--session-id", session]
        child = self.gateway.spawn(argv, stdin=gw.DEVNULL)
        self.child = child
        self.forwarder = None
        self.signals.forwarder = None
        try:
            identity = procs.read_identity(child.pid, gateway=self.gateway)
            if identity is not None:
                self.forwarder = procs.SignalForwarder(identity, gateway=self.gateway)
                self.signals.forwarder = self.forwarder  # from now on the handler forwards
                with self.store.lease(lifecycle.lease_key(self.agent_id), wait=30.0) as lease:
                    self.store.update_agent(lease, self.agent_id, {"exec_proc": identity})
        except BaseException:
            # `exec` is running but unrecorded: nothing could adopt it, so end it here.
            self.signals.forwarder = None
            self.stop_child(child)
            raise
        return child

    def watch(
        self, child: subprocess.Popen[bytes], backend: SessionBackend, session: str, ctx: RunCtx
    ) -> None:
        """Until ``exec`` exits: forward signals, poll for the start and deliver the outbox."""
        try:
            next_status = 0.0
            next_relay = 0.0
            while child.poll() is None:
                for sent in self.signals.take_sent():
                    self.log_sent(sent)
                for received in self.signals.take():
                    self.note_signal(received)
                    self.forward(received)
                now = time.monotonic()
                woken = False
                if now >= next_relay and not ctx.started:
                    next_relay = now + RELAY_EVERY
                    woken = self.relay_changed()
                if now >= next_status or woken:
                    next_status = now + STATUS_EVERY
                    if not ctx.started:
                        ctx.started = self.look_started(backend, session, ctx)
                    self.deliver()
                self.gateway.sleep(TICK)
        except BaseException:
            self.stop_child(child)
            raise
        finally:
            self.signals.forwarder = None  # `exec` is gone: nothing to forward to
            for sent in self.signals.take_sent():
                self.log_sent(sent)

    def relay_changed(self) -> bool:
        """The relay recorded something new for the agent: poll `status` now (a wake-up only)."""
        assert self.store is not None
        try:
            row = self.store.get_agent(self.agent_id)
        except StoreError:
            return False
        if row is None:
            return False
        seen = (row.get("run"), row.get("session_state"), row.get("activity"))
        changed = seen != self.relay_seen
        self.relay_seen = seen
        return changed

    def stop_child(self, child: subprocess.Popen[bytes]) -> None:
        """Send SIGTERM to exactly this `exec` (by identity when known), wait, then SIGKILL it."""
        try:
            if self.forwarder is not None:
                self.forwarder.forward(signal.SIGTERM)
            else:
                child.terminate()
            try:
                child.wait(timeout=EXEC_GRACE)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        except OSError as error:
            self.log("warning", level="warn", message=f"could not stop exec: {error.strerror}")

    def forward(self, received: int) -> None:
        """Forward SIGTERM, SIGHUP (and SIGINT, headless) once to the exact `exec` process.

        SIGCONT follows every SIGTERM (``procs.forward_signal``). The next link of the
        cancellation chain, `claude`, is claude-session's: `exec` forwards to its own child.
        """
        if received not in FORWARDED:
            return
        if self.forwarder is None:
            # `exec`'s identity could not be read: signal the child we spawned (once).
            if self.child is not None and received not in self.fallback_sent:
                self.fallback_sent.add(received)
                try:
                    self.child.send_signal(received)
                except OSError as error:
                    self.log(
                        "warning", level="warn", message=f"could not signal exec: {error.strerror}"
                    )
            return
        outcome = self.forwarder.forward(received)
        if outcome is not None:
            self.log_sent(outcome)

    def log_sent(self, outcome: procs.SignalOutcome) -> None:
        self.log("signal_sent", pid=outcome.pid, signal=signal.Signals(outcome.signal).name)

    def look_started(self, backend: SessionBackend, session: str, ctx: RunCtx) -> bool:
        """Poll `status`; when the run is in its session phase (or final) the agent goes live."""
        try:
            status = backend.status(session)
        except CliError as error:
            self.log("warning", level="warn", message=f"status failed: {error.code}")
            return False
        if status.run.run != ctx.run:
            return False
        if status.run.phase == "session" or status.run.outcome is not None:
            try:
                return self.mark_started(ctx, status.run.phase)
            except StoreError as error:
                self.log("warning", level="warn", message=f"could not go live: {error}")
        return False

    def mark_started(self, ctx: RunCtx, phase: str) -> bool:
        """The run is in its session phase: `launching` → `live`, or a later run of the chain
        (status stays `live`); `post-launch` for that run either way."""
        assert self.store is not None
        if phase != "session":
            return False
        row = self.store.get_agent(self.agent_id)
        if row is None:
            return False
        changes = {"last_live_at": time.time()}
        if row["status"] == "launching":
            lifecycle.go_live(self.store, self.agent_id, run=ctx.run, changes=changes)
        else:
            lifecycle.run_started(
                self.store,
                self.agent_id,
                run=ctx.run,
                auto_resumed=ctx.auto_resumed,
                changes=changes,
            )
        return True

    def deliver(self) -> None:
        assert self.store is not None and self.runner is not None
        try:
            hooklib.deliver_outbox(
                self.store,
                self.agent_id,
                self.config.hooks,
                state_dir=self.state_dir,
                runner=self.runner,
                gateway=self.gateway,
            )
            self.runner.poll()
        except (StoreError, OSError) as error:
            self.log(
                "warning", level="warn", message=f"hook delivery failed: {type(error).__name__}"
            )

    # -- the end of a run ----------------------------------------------------------------------

    def park_canceled(self, backend: SessionBackend, session: str, ctx: RunCtx) -> Outcome:
        """A signal arrived before `exec` was spawned: cancel the pending run and park."""
        assert self.store is not None
        try:
            backend.stop(session)
        except CliError as error:
            self.log("warning", level="warn", message=f"cancel failed: {error.code}")
        try:
            final = self.end_of_run(backend, session, ctx)
        except CliError as error:
            self.log("warning", level="error", message=f"error: no outcome: {error.code}")
            lifecycle.park(self.store, self.agent_id, park_reason="killed", needs_attention=None)
            self.deliver()
            return DONE
        return self.park_final(final)

    def end_of_run(self, backend: SessionBackend, session: str, ctx: RunCtx) -> Final:
        """Read the run's outcome, record the run in the agent's history (and mark it started
        when it ended in its session phase). Raises when no final outcome can be read."""
        assert self.store is not None
        try:
            result = self.read_outcome(backend, ctx.outcome_file, session, ctx.run)
        except CliError as error:
            self.log("warning", level="error", message=f"error: no outcome: {error.code}")
            # TODO(#31): the liveness check parks an agent whose outcome cannot be read.
            raise
        run_row = self.store.get_run(self.agent_id, ctx.run) or {}
        # The relay's first `working` activity, else the run's `recent_events` (filled in by
        # `run_outcome`), else null.
        first = ts(run_row.get("first_working_at")) or result.first_working_at
        outcome = outcome_object(
            self.agent_id, result, result.raw, first_working_at=first, auto_resumed=ctx.auto_resumed
        )
        if not ctx.started and result.phase == "session":
            ctx.started = self.mark_started(ctx, result.phase)  # it started, then ended
        self.record_run(ctx, result, outcome)
        return Final(ctx, result, outcome)

    def read_outcome(
        self, backend: SessionBackend, outcome_file: str, session: str, run: int
    ) -> RunOutcome:
        """The final outcome of the run, read again a few times when it is not there yet."""
        failure: CliError | None = None
        for attempt in range(OUTCOME_TRIES):
            if attempt:
                self.gateway.sleep(OUTCOME_RETRY)
            try:
                result = backend.run_outcome(outcome_file, session, run)
            except CliError as error:
                failure = error
                continue
            if result.final:
                return result
            failure = CliError.of("internal", "exec exited but the run is not final")
        assert failure is not None
        # TODO(#31): the liveness check parks an agent whose run never became final.
        raise failure

    def record_run(self, ctx: RunCtx, result: RunOutcome, outcome: dict[str, Any]) -> None:
        assert self.store is not None
        report = result.report
        status = report.get("status") if isinstance(report, dict) else None  # pyright: ignore[reportUnknownMemberType]
        fields: dict[str, object] = {
            "outcome": outcome,
            "end_reason": result.end_reason,
            "exit_code": result.exit_code,
            "ended_at": self.gateway.wall_time(),
        }
        if isinstance(status, str):
            fields["report_status"] = status
        if outcome["first_working_at"]:
            try:
                fields["first_working_at"] = parse_record_ts(
                    outcome["first_working_at"]
                ).timestamp()
            except ValueError:
                pass
        with self.store.lease(lifecycle.lease_key(self.agent_id), wait=30.0) as lease:
            with self.store.transaction():
                # The run is over: the agent row never names a dead `exec` between runs.
                self.store.update_agent(lease, self.agent_id, {"exec_proc": None})
                self.store.put_run(lease, self.agent_id, ctx.run, fields)

    # -- the end of the chain ------------------------------------------------------------------

    @staticmethod
    def ended_in_launch(final: Final) -> bool:
        return final.result.phase == "launch" and final.result.outcome in LAUNCH_PHASE_OUTCOMES

    def wants_report_resume(
        self, row: dict[str, Any], final: Final, previous: Final | None
    ) -> bool:
        """The automatic report resume: a headless run that ended ``exited-unreported`` (not for
        ``no_session_start``), once per chain, when configured; never after a stop, a forwarded
        signal or a SIGTERM or SIGHUP received during the chain."""
        result = final.result
        return (
            result.outcome == "exited-unreported"
            and row.get("mode") == "headless"
            and self.config.auto_report_resume
            and previous is None
            and not final.ctx.auto_resumed
            and result.end_reason != "no_session_start"
            and result.end_reason not in chain.NO_RESUME_END_REASONS
            and not self.terminated
        )

    def park_values(self, final: Final) -> tuple[Any, dict[str, object]]:
        table = park_row(final.result.outcome or "failed", final.result.end_reason)
        changes: dict[str, object] = {
            "last_outcome": final.outcome,
            "last_end_reason": final.result.end_reason,
            "last_live_at": time.time(),
            "exec_proc": None,
        }
        return table, changes

    def park_final(self, final: Final) -> Outcome:
        """Apply the outcome table to a run that ended: park, or tear down an ephemeral agent
        whose run was ``done``."""
        assert self.store is not None
        store = self.store
        table, changes = self.park_values(final)
        row = store.get_agent(self.agent_id)
        if (
            final.result.outcome == "done"
            and row is not None
            and row["status"] == "live"
            and row.get("ephemeral")
        ):
            moved = lifecycle.enter_ephemeral_teardown(
                store, self.agent_id, supervisor=store.holder_identity(), changes=changes
            )
            if moved is not None:
                return self.tear_down(final)
        lifecycle.park(
            store,
            self.agent_id,
            park_reason=table.park_reason,
            needs_attention=table.needs_attention,
            escalated_by=table.escalated_by,
            changes=changes,
            outcome=final.outcome,
        )
        return self.finish()

    def tear_down(self, final: Final) -> Outcome:
        """The agent is ``tearing_down`` with this supervisor as the teardown process."""
        assert self.store is not None
        self.deliver()
        reason = self.teardown(self.store, self.agent_id)
        if reason is not None:
            self.log("warning", level="warn", message=f"ephemeral teardown refused: {reason}")
            moved = lifecycle.refuse_ephemeral_teardown(
                self.store, self.agent_id, outcome=final.outcome
            )
            if moved is None:
                self.log(
                    "warning",
                    level="warn",
                    message="ephemeral teardown refusal not applied "
                    "(the status changed or the teardown left a partial state)",
                )
        return self.finish()

    def park_chain_failure(self, trigger: Final, failure: chain.ChainFailure) -> Outcome:
        """A supervisor-initiated run stopped early: park per the triggering run's row."""
        assert self.store is not None
        table, changes = self.park_values(trigger)
        self.log(
            "warning",
            level="warn",
            message=f"chain run stopped: {failure.tool} {failure.reason}",
        )
        lifecycle.park(
            self.store,
            self.agent_id,
            park_reason=table.park_reason,
            needs_attention=table.needs_attention,
            escalated_by=table.escalated_by,
            changes=changes,
            outcome=trigger.outcome,
            chain_failure=failure.as_dict(),
        )
        return self.finish()

    def finish(self) -> Outcome:
        self.deliver()
        self.drain()
        return DONE

    def drain(self) -> None:
        """Wait (a few seconds) for async hooks this supervisor started to be reaped."""
        assert self.runner is not None
        try:
            self.runner.wait(5.0)
        except (OSError, ValueError):
            pass


def supervise(inv: Any) -> Outcome:
    signals = Signals()
    restore = procs.install_signal_handlers(
        {
            signal.SIGTERM: signals.handle,
            signal.SIGHUP: signals.handle,
            signal.SIGINT: signals.handle,  # headless: forwarded (interface.md, Signals)
            signal.SIGQUIT: signal.SIG_IGN,
        }
    )
    # The launcher exec'd us with these blocked: deliver what arrived meanwhile to the handlers.
    signal.pthread_sigmask(signal.SIG_UNBLOCK, FORWARDED)
    try:
        return Supervisor(inv, signals).run()
    finally:
        restore()
