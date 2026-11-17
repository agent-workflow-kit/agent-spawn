"""``supervise``: the supervisor of an agent's run chain (interface.md, Supervision).

The supervisor runs ``claude-session exec`` for the agent's current run, notices that the run
started (by polling ``status``; the relay is only an early wake-up), reads the outcome when
``exec`` exits, applies the outcome table and parks the agent. A headless run that ended
``exited-unreported`` gets one automatic report resume (``chain.py``); a supervisor-initiated run
that cannot start, or ends in its launch phase, parks the agent per the run that triggered it
(the chain rule). An ephemeral agent whose run ended ``done`` hands over to teardown.

Window hosting (interactive, ``--handoff``): the pane's session leader is this process, so a
closed window hangs up only it. It adopts the launcher's environment over the handoff, ignores
SIGINT (the terminal delivers it to ``claude``) and SIGQUIT, forwards SIGHUP and SIGTERM to
``exec``, applies the window markers from the recorded state and never writes to its terminal
(a write after a hangup fails with EIO): errors are logged, never printed.

Promotion (``promotion.py``): a headless run that ended with a session-phase exit 13 continues in a
window. The supervisor opens it (``supervise --host``, host mode below), waits for the host pane
process (the ``exec`` runs there, not here) and forwards SIGTERM and SIGHUP to that process only.
Host mode runs the ``exec`` argv in its window and does no bookkeeping.
"""

from __future__ import annotations

import contextlib
import dataclasses
import math
import os
import shlex
import signal
import subprocess
import time
from collections.abc import Callable, Iterator
from types import FrameType
from typing import Any

from . import chain, handoff, lifecycle, procs, promotion
from . import gateway as gw
from . import hooks as hooklib
from . import teardown as teardown_module
from .agent_view import outcome_object, ts
from .chain import Final, RunCtx
from .config import Config, load_config
from .outcome_table import CANCEL_END_REASONS, LAUNCH_PHASE_OUTCOMES, park_row
from .output import CliError, Outcome, PayloadKind, error_outcome
from .records import RecordsError, append_log, parse_record_ts
from .session_backend import RunOutcome, SessionBackend
from .store import Store, StoreError, StoreMissing
from .term import ATTENTION_MARKER, TermBackend

STATUS_EVERY = 1.0  # seconds between looks at `claude-session status` and the outbox
STATUS_TIMEOUT = 5.0  # seconds the watch loop's own `status` poll may take
MARKER_BACKOFF = 30.0  # seconds between window marker attempts after a termctl failure
RELAY_EVERY = 0.25  # seconds between looks at the relay's records (an early wake-up)
TICK = 0.05  # seconds between looks at `exec`
SUPERVISOR_ACTOR = "auto:agent-spawn"
# The supervisor writes nothing to its terminal (interface.md, Supervision): no payload.
DONE = Outcome(0, {}, PayloadKind.COMMAND, quiet=True)
OUTCOME_TRIES = 3  # reads of the final outcome before giving up
OUTCOME_RETRY = 0.2  # seconds between them
EXEC_GRACE = 30.0  # seconds `exec` gets after SIGTERM when this supervisor fails
# A headless run forwards SIGINT; an interactive (window-hosted) run ignores it: the terminal
# delivers it to `claude` itself (interface.md, Signals).
FORWARDED = (signal.SIGTERM, signal.SIGHUP, signal.SIGINT)
FORWARDED_INTERACTIVE = (signal.SIGTERM, signal.SIGHUP)
#: The relay states that make a window ask for attention.
ATTENTION_STATES = frozenset({"needs-input", "idle"})


class Signals:
    """The supervisor's handlers record the signal for the main loop's bookkeeping and, while an
    ``exec`` child is known (``forwarder`` set), forward it themselves: the loop may be blocked
    in a lower-tool call for a long time, and the signal must reach ``exec`` promptly.

    The forwarder marks a signal before sending it, so a handler that interleaves with the loop's
    own forward cannot send it twice. The handler logs nothing (a log write could re-enter one
    in progress); the loop logs what ``take_sent`` returns.
    """

    def __init__(self, *, interactive: bool = False) -> None:
        self.forwarded = FORWARDED_INTERACTIVE if interactive else FORWARDED
        self.received: list[int] = []
        self.forwarder: procs.SignalForwarder | None = None
        self.sent: list[procs.SignalOutcome] = []

    def handle(self, signum: int, frame: FrameType | None) -> None:
        self.received.append(signum)
        forwarder = self.forwarder
        if forwarder is None or signum not in self.forwarded:
            return
        try:
            outcome = forwarder.forward(signum)
        except Exception:  # noqa: BLE001 - a handler must not raise; the loop retries
            return  # the signal stays queued and the loop retries through its own forward
        if outcome is not None:
            self.sent.append(outcome)

    def take(self) -> list[int]:
        taken, self.received = self.received, []
        return taken

    def take_sent(self) -> list[procs.SignalOutcome]:
        taken, self.sent = self.sent, []
        return taken


class Supervisor:
    def __init__(
        self,
        inv: Any,
        signals: Signals,
        *,
        agent_id: str | None = None,
        state_dir: str | None = None,
        in_process: bool = False,
    ) -> None:
        self.inv = inv
        self.signals = signals
        self.config: Config = inv.config
        self.state_dir: str = state_dir or inv.state_dir
        self.gateway = gw.get_gateway()
        self.store: Store | None = None
        self.agent_id: str = agent_id or inv.args.agent_id
        #: ``launch|resume --placement here``: this is the launcher's own process (interface.md,
        #: Hosting). The terminal is ``exec``'s stdio; the handlers of the launcher stay in
        #: place until ``exec`` is spawned (``spawn_exec`` swaps them for the supervisor's).
        self.in_process = in_process
        self.restore_handlers: Callable[[], None] | None = None
        self.forwarder: procs.SignalForwarder | None = None
        self.child: subprocess.Popen[bytes] | None = None  # the current `exec`
        self.runner: hooklib.AsyncRunner | None = None
        self.terminated = False  # SIGTERM, SIGHUP or SIGINT arrived during this run chain
        self.teardown = self.ephemeral_teardown
        self.relay_seen: tuple[object, ...] = ()
        self.fallback_sent: set[int] = set()
        self.term: TermBackend | None = None  # window markers: only a window-hosted supervisor
        self.marked: dict[int, bool] = {}  # run -> the marker state last applied (attention on)
        self.marker_settled: set[int] = set()  # runs whose pane is gone: no more marker calls
        self.marker_retry_at: dict[int, float] = {}  # run -> earliest next attempt after a failure
        self.marker_warned: dict[int, str] = {}  # run -> the failure last logged (once per change)
        self.clock: Callable[[], float] = time.monotonic
        self.promoted = False  # this run chain's one promotion was decided
        #: Runs this supervisor canceled itself (``chain.cancel_pending``, a host that died before
        #: its ``exec`` adopted the run): such a cancel is a chain failure, while a promoted run
        #: that ended ``canceled`` without one is somebody's deliberate ``stop``.
        self.self_canceled: set[int] = set()
        self.window_socket: str | None = None  # the tmux server the window preflight named

    def log(self, event: str, *, level: str = "info", **fields: Any) -> None:
        try:
            append_log(self.state_dir, event, level=level, agent_id=self.agent_id, **fields)
        except (RecordsError, ValueError, OSError):
            pass  # a log that cannot be written never stops the supervisor

    def ephemeral_teardown(self, store: Store, agent_id: str) -> str | None:
        """The ephemeral teardown (interface.md, Teardown order, from step 3): the refusal
        reason, or ``None`` when it completed or left the agent ``tearing_down`` (logged)."""
        row = store.get_agent(agent_id)
        if row is None:
            return None
        return teardown_module.ephemeral(
            store,
            agent_id,
            config=self.config,
            state_dir=self.state_dir,
            backend=self.backend(row),
            log=self.log,
            gateway=self.gateway,
        )

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
        if args.host:
            return self.run_host()
        if args.handoff is not None:
            refused = self.adopt_handoff()
            if refused is not None:
                return refused
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

    def run_in_process(self, store: Store, row: dict[str, Any]) -> Outcome:
        """``--placement here``: supervise from the launcher's own process, with the launcher's
        open state database (the agent's record already names this process as supervisor)."""
        self.store = store
        return self.supervise(row)

    # -- host mode -----------------------------------------------------------------------------

    def run_host(self) -> Outcome:
        """``supervise --host``: the window of a promoted run. Adopt the supervisor's environment,
        check that a promotion is pending for this very process, run ``exec`` with the terminal
        as its stdin, stdout and stderr, forward SIGHUP and SIGTERM to it and exit 0 once it
        exited. The supervisor does the bookkeeping."""
        refused = self.adopt_handoff()  # the CLI refuses a --host without --handoff (exit 2)
        if refused is not None:
            return refused
        try:
            self.store = Store.open(self.state_dir, create=False, gateway=self.gateway)
        except StoreMissing:
            self.log("warning", level="warn", message="supervise refused: no state database")
            raise CliError.usage("supervise: there is no state database") from None
        try:
            row = self.host_check()
            self.host_exec(row)
        finally:
            self.store.close()
        return DONE

    def host_check(self) -> dict[str, Any]:
        """A promotion must be pending and its promoted run must name this process as its pane
        process (exit 2 otherwise, logged)."""
        assert self.store is not None
        row = self.store.get_agent(self.agent_id)
        problem = None
        run_row: dict[str, Any] = {}
        if row is not None and isinstance(row.get("run"), int):
            run_row = self.store.get_run(self.agent_id, row["run"]) or {}
        if row is None:
            problem = "no such agent"
        elif row["status"] != "live" or not row.get("promotion_pending"):
            problem = "no promotion is pending"
        elif (
            not run_row.get("promoted") or run_row.get("pane_proc") != self.store.holder_identity()
        ):
            problem = "the promoted run does not name this process as its host"
        elif not isinstance(row.get("csession_id"), str):
            problem = "the agent has no session"
        if problem is not None:
            self.log("warning", level="warn", message=f"supervise --host refused: {problem}")
            raise CliError.usage(f"supervise: {problem}")
        assert row is not None
        return row

    def host_exec(self, row: dict[str, Any]) -> None:
        """Run ``exec`` for the promoted run until it exits; nothing is recorded."""
        if self.collect_signals():
            return  # told to stop before exec ran: the supervisor cancels the pending run
        argv = [
            *shlex.split(row["claude_session_path"]),
            "exec",
            "--session-id",
            row["csession_id"],
        ]
        child = self.gateway.spawn(argv)  # the terminal is its stdin, stdout and stderr
        self.child = child
        try:
            identity = procs.read_identity(child.pid, gateway=self.gateway)
            if identity is not None:
                self.forwarder = procs.SignalForwarder(identity, gateway=self.gateway)
                self.signals.forwarder = self.forwarder
            while child.poll() is None:
                for sent in self.signals.take_sent():
                    self.log_sent(sent)
                for received in self.signals.take():
                    self.note_signal(received)
                    self.forward(received)
                self.gateway.sleep(TICK)
        except BaseException:
            self.stop_child(child)
            raise
        finally:
            self.signals.forwarder = None
            for sent in self.signals.take_sent():
                self.log_sent(sent)

    # -- the window handoff --------------------------------------------------------------------

    def adopt_handoff(self) -> Outcome | None:
        """Receive the launcher's environment over the handoff socket and become its owner: the
        process environment is replaced (the six terminal variables stay the pane's) and the
        config is loaded again from it, so the launcher's ``AGENT_SPAWN_*`` overrides apply
        exactly as they do for a detached supervisor (decision of #11). ``None`` on success; on
        failure the exit-1 outcome (nothing was started; ``handoff_failed`` is in the log)."""
        args = self.inv.args
        try:
            seconds = float(args.handoff_timeout)
        except (TypeError, ValueError):
            seconds = math.nan
        if not math.isfinite(seconds) or seconds <= 0:
            raise CliError.usage("supervise: --handoff-timeout must be a positive number")
        try:
            env = handoff.receive_environment(
                args.handoff,
                timeout=handoff.client_deadline(seconds),
                pane_env=os.environ,
                state_dir=self.state_dir,
                agent_id=self.agent_id,
                gateway=self.gateway,
            )
        except handoff.HandoffFailed:
            return quiet_error(CliError("failed", "the environment handoff failed", exit_code=1))
        os.environ.clear()
        os.environ.update(env)
        loaded = load_config(self.inv.config_file, environ=env)
        self.inv.config = loaded.config
        self.config = loaded.config
        return None

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
        stored_term = row.get("termctl_path")
        if (
            getattr(self.inv.args, "handoff", None) is not None
            and isinstance(stored_term, str)
            and stored_term
        ):
            try:
                self.term = TermBackend(shlex.split(stored_term), gateway=self.gateway)
            except ValueError:
                self.log("warning", level="warn", message="the stored termctl path is unusable")
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
            if ctx.promoted:
                # The promoted run's `exec` runs in the host window: wait for the host process.
                self.watch_host(poller, session, ctx)
                self.collect_signals()
                self.settle_promoted(backend, poller, session, ctx)
            else:
                try:
                    child = self.spawn_exec(row, session)
                except (OSError, StoreError) as error:
                    # `exec` could not be started (or was stopped again because it could not be
                    # recorded): the run it should have adopted must not stay pending.
                    self.log("warning", level="error", message=f"could not start exec: {error}")
                    if previous is None:
                        return self.park_canceled(backend, session, ctx)
                    run_row = store.get_run(self.agent_id, ctx.run) or {}
                    try:
                        chain.cancel_pending(
                            self,
                            backend,
                            session,
                            ctx.run,
                            hooks=run_row.get("hooks"),
                            token=run_row.get("outcome_token"),
                        )
                    except Exception as problem:  # noqa: BLE001 - best effort, as chain._start
                        self.log(
                            "warning", level="warn", message=f"could not cancel the run: {problem}"
                        )
                    failure = chain.ChainFailure("agent-spawn", "failed", run=ctx.run)
                    return self.park_chain_failure(previous, failure)
                self.watch(child, poller, session, ctx)
                self.collect_signals()
            final = self.end_of_run(backend, session, ctx)
            if previous is not None and self.ended_in_launch(final):
                if final.result.outcome == "not-started" and (
                    (self.terminated and final.result.end_reason in CANCEL_END_REASONS)
                    or self.stopped_promotion(final)
                ):
                    # A user's SIGTERM or SIGHUP canceled the chain run, or somebody stopped a
                    # promotion's pending run: not a chain failure (killed, no attention).
                    return self.park_final(final)
                # A supervisor-initiated run that ended in its launch phase stops the chain.
                failure = chain.ChainFailure(
                    "claude-session",
                    chain.reason_word(final.result.outcome),
                    run=final.ctx.run,
                    end_reason=final.result.end_reason,
                )
                return self.park_chain_failure(previous, failure)
            if promotion.wants(self, final):
                try:
                    ctx = promotion.start(self, backend, session, final)
                    promotion.open_window(self, backend, session, ctx, final)
                except chain.ChainFailure as failure:
                    return self.park_chain_failure(final, failure)
                except chain.ChainCanceled:
                    return self.park_final(final)
                previous = final
                continue
            if not self.wants_report_resume(row, final, previous):
                return self.park_final(final)
            try:
                ctx = chain.start_report_resume(self, backend, row, session, final)
            except chain.ChainFailure as failure:
                return self.park_chain_failure(final, failure)
            except chain.ChainCanceled:
                return self.park_final(final)
            previous = final

    def stopped_promotion(self, final: Final) -> bool:
        """The promoted run was canceled by someone else (``stop``): this supervisor did not
        cancel it, so it is a deliberate stop and parks per the canceled run."""
        return (
            final.ctx.promoted
            and final.ctx.run not in self.self_canceled
            and final.result.end_reason == "canceled"
        )

    # -- watching `exec` ---------------------------------------------------------------------

    def collect_signals(self) -> bool:
        """Note the signals that arrived; True when the chain must end (no resume, no more
        runs). The caller decides what to forward."""
        taken = self.signals.take()
        for received in taken:
            self.note_signal(received)
        return bool(taken)

    def note_signal(self, received: int) -> None:
        if received in self.signals.forwarded:
            self.terminated = True

    def spawn_exec(self, row: dict[str, Any], session: str) -> subprocess.Popen[bytes]:
        assert self.store is not None
        argv = [*shlex.split(row["claude_session_path"]), "exec", "--session-id", session]
        # A window: `exec` is in the pane's foreground process group with the terminal as its
        # stdin, stdout and stderr (a closed window hangs up only this process). `--placement
        # here` is the same with the launcher's terminal.
        on_terminal = self.in_process or getattr(self.inv.args, "handoff", None) is not None
        with self.signals_held():
            if on_terminal:
                child = self.gateway.spawn(argv)
            else:
                child = self.gateway.spawn(argv, stdin=gw.DEVNULL)
            self.child = child
            self.forwarder = None
            self.signals.forwarder = None
            self.record_exec(child)
        return child

    @contextlib.contextmanager
    def signals_held(self) -> Iterator[None]:
        """In process only: SIGTERM, SIGHUP and SIGINT stay blocked from before ``exec`` exists
        until the supervisor's handlers are installed and ``exec`` is recorded, so a signal is
        either the launcher's (no ``exec`` yet) or the supervisor's, never lost in between."""
        if not self.in_process:
            yield
            return
        old = signal.pthread_sigmask(signal.SIG_BLOCK, ())  # the current mask, unchanged
        try:
            # A launcher handler that was already flagged runs here (raising the launcher's
            # interrupt) before `exec` exists; the mask is restored either way.
            signal.pthread_sigmask(signal.SIG_BLOCK, FORWARDED)
            yield
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK, old)

    def record_exec(self, child: subprocess.Popen[bytes]) -> None:
        assert self.store is not None
        try:
            if self.in_process and self.restore_handlers is None:
                # The supervisor's rows from now on ("the launcher rows until `exec` is
                # spawned, then as `supervise`"): SIGINT and SIGQUIT are ignored.
                self.restore_handlers = procs.install_signal_handlers(
                    {
                        signal.SIGTERM: self.signals.handle,
                        signal.SIGHUP: self.signals.handle,
                        signal.SIGINT: signal.SIG_IGN,
                        signal.SIGQUIT: signal.SIG_IGN,
                    },
                    gateway=self.gateway,
                )
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

    def watch(
        self, child: subprocess.Popen[bytes], backend: SessionBackend, session: str, ctx: RunCtx
    ) -> None:
        """Until ``exec`` exits: forward signals, poll for the start and deliver the outbox."""
        try:
            self.watch_until(lambda: child.poll() is None, self.sleep_tick, backend, session, ctx)
        except BaseException:
            self.stop_child(child)
            raise
        finally:
            self.signals.forwarder = None  # `exec` is gone: nothing to forward to
            for sent in self.signals.take_sent():
                self.log_sent(sent)

    def sleep_tick(self) -> None:
        self.gateway.sleep(TICK)

    def watch_host(self, backend: SessionBackend, session: str, ctx: RunCtx) -> None:
        """Until the host pane process of the promoted run exits (a pidfd; kqueue on macOS):
        forward SIGTERM and SIGHUP to exactly that process, poll for the start and deliver the
        outbox. The process is verified by PID, start time and boot id every time."""
        host = ctx.host
        assert host is not None
        handle = procs.open_verified(host, gateway=self.gateway)
        if handle is None:
            self.signals.forwarder = None
            return  # the host is gone already: the outcome decides
        try:
            self.watch_until(handle.is_alive, lambda: handle.wait_exit(TICK), backend, session, ctx)
        except BaseException:
            # This supervisor failed while the host lives: tell the host to stop (it forwards
            # to `exec`) and give it the grace `exec` gets; the exact verified process only.
            self.forward_host_stop(host)
            procs.wait_for_exit(host, EXEC_GRACE, gateway=self.gateway)
            raise
        finally:
            handle.close()
            self.signals.forwarder = None
            self.forwarder = None
            for sent in self.signals.take_sent():
                self.log_sent(sent)

    def forward_host_stop(self, host: procs.ProcessIdentity) -> None:
        try:
            outcome = procs.forward_signal(host, signal.SIGTERM, gateway=self.gateway)
            self.log_sent(outcome)
        except OSError as error:
            self.log("warning", level="warn", message=f"could not stop the host: {error.strerror}")

    def settle_promoted(
        self, backend: SessionBackend, poller: SessionBackend, session: str, ctx: RunCtx
    ) -> None:
        """The host exited. A dead pane is closed (``--keep-on-exit failed`` keeps it), and a run
        that is not final (the host died before its `exec` adopted it, or while it ran) is
        stopped so that it ends."""
        try:
            status = poller.status(session)
        except CliError as error:
            self.log("warning", level="warn", message=f"status failed: {error.code}")
            status = None
        if status is not None and status.run.run == ctx.run and status.run.outcome is None:
            # The host is gone but the run is not final: a run nothing adopted is canceled (the
            # supervisor's own cancel), a live one is stopped (its `exec` died with the host).
            # The outcome is read again afterwards.
            if status.pending:
                self.self_canceled.add(ctx.run)
            chain.stop_pending(self, backend, session)
        if self.term is not None:
            promotion.close_dead(self, self.term, ctx, wait=promotion.HOST_STOP_WAIT)

    def watch_until(
        self,
        alive: Callable[[], bool],
        idle: Callable[[], object],
        backend: SessionBackend,
        session: str,
        ctx: RunCtx,
    ) -> None:
        """The watch loop: while ``alive()``, forward signals, poll for the start, apply the
        window marker and deliver the outbox; ``idle()`` waits one tick."""
        next_status = 0.0
        next_relay = 0.0
        while alive():
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
                self.apply_marker(ctx)
                self.deliver()
            idle()

    def apply_marker(self, ctx: RunCtx) -> None:
        """The window marker of this run from the recorded (relayed) state: ``needs-input`` or
        ``idle`` set ``attention``, ``working`` clears it. Only the current run's window is
        touched; a termctl failure is a warning and the next look retries."""
        term = self.term
        if term is None or self.store is None:
            return
        if self.terminated or ctx.run in self.marker_settled:
            return  # a hangup or stop is under way, or the pane is gone: leave the window alone
        if self.clock() < self.marker_retry_at.get(ctx.run, 0.0):
            return
        try:
            run_row = self.store.get_run(self.agent_id, ctx.run) or {}
        except StoreError:
            return
        term_id, socket_path = run_row.get("term_id"), run_row.get("socket_path")
        if not isinstance(term_id, str) or not isinstance(socket_path, str):
            return
        state, activity = run_row.get("session_state"), run_row.get("activity")
        if state in ATTENTION_STATES or activity == "idle":
            wanted = True
        elif state == "working" or activity == "working":
            wanted = False
        else:
            return
        if self.marked.get(ctx.run) == wanted or (ctx.run not in self.marked and not wanted):
            return
        try:
            if wanted:
                term.mark(term_id, socket_path, ATTENTION_MARKER)
            else:
                term.unmark(term_id, socket_path)
        except CliError as error:
            if error.tool == "termctl" and error.exit_code == 10:
                self.marker_settled.add(ctx.run)  # the pane is gone: nothing to mark
                self.log("warning", message="the window is gone: no more markers")
                return
            self.marker_retry_at[ctx.run] = self.clock() + MARKER_BACKOFF
            reason = f"window marker failed: {error.code}"
            if self.marker_warned.get(ctx.run) != reason:
                self.marker_warned[ctx.run] = reason
                self.log("warning", level="warn", message=reason)
            return
        self.marker_warned.pop(ctx.run, None)
        self.marker_retry_at.pop(ctx.run, None)
        self.marked[ctx.run] = wanted

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
        if received not in self.signals.forwarded:
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
                promoted=ctx.promoted,
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
            self.agent_id,
            result,
            result.raw,
            first_working_at=first,
            auto_resumed=ctx.auto_resumed,
            promoted=ctx.promoted,
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


def quiet_error(error: CliError) -> Outcome:
    """The outcome of a failed supervisor: the exit code and envelope, printed nowhere (a window
    supervisor writes nothing to its terminal; the log has the reason)."""
    return dataclasses.replace(error_outcome(error), quiet=True)


def supervise(inv: Any) -> Outcome:
    windowed = inv.args.handoff is not None
    signals = Signals(interactive=windowed)
    handlers: dict[int, Any] = {
        signal.SIGTERM: signals.handle,
        signal.SIGHUP: signals.handle,
        signal.SIGQUIT: signal.SIG_IGN,
    }
    # Headless: SIGINT is forwarded. A window: ignored, the terminal delivers it to `claude`.
    handlers[signal.SIGINT] = signal.SIG_IGN if windowed else signals.handle
    restore = procs.install_signal_handlers(handlers)
    # The launcher exec'd us with these blocked: deliver what arrived meanwhile to the handlers.
    signal.pthread_sigmask(signal.SIG_UNBLOCK, FORWARDED)
    supervisor = Supervisor(inv, signals)
    try:
        return supervisor.run()
    except CliError as error:
        if not windowed and not inv.args.host:
            raise  # a headless supervisor's stderr is its log; a direct run shows the refusal
        supervisor.log("warning", level="error", message=f"supervise failed: {error.code}")
        return quiet_error(error)
    finally:
        restore()
