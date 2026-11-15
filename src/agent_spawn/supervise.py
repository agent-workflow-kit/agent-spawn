"""``supervise``: the supervisor of an agent's run chain (interface.md, Supervision).

Issue #11 owns the start of the detached supervisor and the startup check; this module also holds
the part of the run loop that a headless launch needs end to end: spawn ``exec``, notice that the
run started, wait for ``exec``, read the outcome, park by the outcome table and drain the outbox.

TODO(#12): the full run loop belongs to issue #12: the automatic report resume, promotion, window
markers, the chain-failure park, the ephemeral teardown, host mode and the handoff. They are marked
below. Whatever #12 replaces, the startup check and the signal handling here follow the interface.
"""

from __future__ import annotations

import os
import shlex
import signal
import subprocess
import time
from types import FrameType
from typing import Any

from . import gateway as gw
from . import hooks as hooklib
from . import lifecycle, procs
from .agent_view import outcome_object
from .config import Config
from .outcome_table import park_row
from .output import CliError, Outcome, PayloadKind
from .records import RecordsError, append_log, parse_record_ts
from .session_backend import RunOutcome, SessionBackend, read_outcome_file
from .store import Store, StoreError, StoreMissing

STATUS_EVERY = 1.0  # seconds between looks at `claude-session status` and the outbox
TICK = 0.05  # seconds between looks at `exec`
SUPERVISOR_ACTOR = "auto:agent-spawn"
# The supervisor writes nothing to its terminal (interface.md, Supervision): no payload.
DONE = Outcome(0, {}, PayloadKind.COMMAND, quiet=True)
OUTCOME_TRIES = 3  # reads of the final outcome before giving up
OUTCOME_RETRY = 0.2  # seconds between them
EXEC_GRACE = 30.0  # seconds `exec` gets after SIGTERM when this supervisor fails
FORWARDED = (signal.SIGTERM, signal.SIGHUP, signal.SIGINT)


class Signals:
    """The supervisor's handlers only record the signal; the main loop forwards it."""

    def __init__(self) -> None:
        self.received: list[int] = []

    def handle(self, signum: int, frame: FrameType | None) -> None:
        self.received.append(signum)

    def take(self) -> list[int]:
        taken, self.received = self.received, []
        return taken


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
        self.runner: hooklib.AsyncRunner | None = None
        self.forwarded: set[int] = set()

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

    def backend(self, row: dict[str, Any]) -> SessionBackend:
        return SessionBackend(
            shlex.split(row["claude_session_path"]),
            actor=SUPERVISOR_ACTOR,
            decision_ids=row.get("decision_ids") or [],
            gateway=self.gateway,
        )

    def supervise(self, row: dict[str, Any]) -> Outcome:
        assert self.store is not None
        store = self.store
        backend = self.backend(row)
        session = row["csession_id"]
        run = row["run"]
        if not isinstance(session, str) or not isinstance(run, int):
            raise CliError.of("internal", "the agent has no session to supervise")
        run_row = store.get_run(self.agent_id, run) or {}
        token = run_row.get("outcome_token")
        outcome_file = os.path.join(self.state_dir, "outcomes", self.agent_id, f"{token}.json")
        self.runner = hooklib.AsyncRunner(self.state_dir, gateway=self.gateway)

        if self.signals.take():
            # No `exec` child yet: cancel our own pending run and park (nothing can adopt it).
            return self.finish(backend, row, session, run, outcome_file, cancel=True)

        child = self.spawn_exec(row, session)
        try:
            started = False
            next_status = 0.0
            while child.poll() is None:
                for received in self.signals.take():
                    self.forward(received)
                now = time.monotonic()
                if now >= next_status:
                    next_status = now + STATUS_EVERY
                    if not started:
                        started = self.look_started(backend, row, session, run)
                    self.deliver()
                self.gateway.sleep(TICK)
        except BaseException:
            self.stop_child(child)
            raise
        return self.finish(backend, row, session, run, outcome_file, cancel=False)

    def spawn_exec(self, row: dict[str, Any], session: str) -> subprocess.Popen[bytes]:
        assert self.store is not None
        argv = [*shlex.split(row["claude_session_path"]), "exec", "--session-id", session]
        child = self.gateway.spawn(argv, stdin=gw.DEVNULL)
        try:
            identity = procs.read_identity(child.pid, gateway=self.gateway)
            if identity is not None:
                self.forwarder = procs.SignalForwarder(identity, gateway=self.gateway)
                with self.store.lease(lifecycle.lease_key(self.agent_id), wait=30.0) as lease:
                    self.store.update_agent(lease, self.agent_id, {"exec_proc": identity})
        except BaseException:
            # `exec` is running but unrecorded: nothing could adopt it, so end it here.
            self.stop_child(child)
            raise
        return child

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
        """Forward SIGTERM, SIGHUP (and SIGINT, headless) once to the exact `exec` process."""
        # TODO(#12): `exec`'s own child (the claude process) must get the signal too, plus
        # SIGCONT for a stopped one; today only `exec` itself is signaled.
        if received not in FORWARDED or self.forwarder is None:
            return
        outcome = self.forwarder.forward(received)
        if outcome is not None:
            self.log("signal_sent", pid=outcome.pid, signal=signal.Signals(received).name)

    def look_started(
        self, backend: SessionBackend, row: dict[str, Any], session: str, run: int
    ) -> bool:
        """Poll `status`; when the run is in its session phase (or final) the agent goes live."""
        try:
            status = backend.status(session)
        except CliError as error:
            self.log("warning", level="warn", message=f"status failed: {error.code}")
            return False
        if status.run.run != run:
            return False
        if status.run.phase == "session" or status.run.outcome is not None:
            try:
                return self.mark_started(run, status.run.phase)
            except StoreError as error:
                self.log("warning", level="warn", message=f"could not go live: {error}")
        return False

    def mark_started(self, run: int, phase: str) -> bool:
        assert self.store is not None
        if phase != "session":
            return False
        lifecycle.go_live(self.store, self.agent_id, run=run, changes={"last_live_at": time.time()})
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

    # -- the end of the run ----------------------------------------------------------------------

    def finish(
        self,
        backend: SessionBackend,
        row: dict[str, Any],
        session: str,
        run: int,
        outcome_file: str,
        *,
        cancel: bool,
    ) -> Outcome:
        assert self.store is not None
        store = self.store
        if cancel:
            try:
                backend.stop(session)
            except CliError as error:
                self.log("warning", level="warn", message=f"cancel failed: {error.code}")
        try:
            result = self.read_outcome(backend, outcome_file, session, run)
        except CliError as error:
            self.log("warning", level="error", message=f"error: no outcome: {error.code}")
            if cancel:
                lifecycle.park(store, self.agent_id, park_reason="killed", needs_attention=None)
                self.deliver()
                return DONE
            # TODO(#32): the liveness check parks an agent whose outcome cannot be read.
            raise
        raw = read_outcome_file(outcome_file, session, run)
        first = result.first_working_at
        outcome = outcome_object(self.agent_id, result, raw, first_working_at=first)
        # TODO(#12): the automatic report resume and promotion happen here, before the park.
        current = store.get_agent(self.agent_id)
        if current is not None and current["status"] == "launching":
            self.mark_started(run, result.phase)  # a run that ended in its session phase started
        self.record_run(run, result, outcome)
        table = park_row(result.outcome or "failed", result.end_reason)
        lifecycle.park(
            store,
            self.agent_id,
            park_reason=table.park_reason,
            needs_attention=table.needs_attention,
            escalated_by=table.escalated_by,
            changes={
                "last_outcome": outcome,
                "last_end_reason": result.end_reason,
                "last_live_at": time.time(),
                "exec_proc": None,
            },
            outcome=outcome,
        )
        # TODO(#12): an ephemeral agent whose run ended `done` is torn down here.
        self.deliver()
        self.drain()
        return DONE

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
        # TODO(#32): the liveness check parks an agent whose run never became final.
        raise failure

    def record_run(self, run: int, result: RunOutcome, outcome: dict[str, Any]) -> None:
        assert self.store is not None
        report = result.report
        status = report.get("status") if isinstance(report, dict) else None  # pyright: ignore[reportUnknownMemberType]
        fields: dict[str, object] = {
            "outcome": outcome,
            "end_reason": result.end_reason,
            "exit_code": result.exit_code,
            "ended_at": time.time(),
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
            self.store.put_run(lease, self.agent_id, run, fields)

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
