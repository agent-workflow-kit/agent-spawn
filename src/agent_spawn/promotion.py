"""Promotion: a headless run that ended blocked or needs-input continues in a window
(interface.md, Promotion and Supervision).

The supervisor decides (``wants``), then ``start`` runs the chain's shared pre-start checks
(``chain.start_promotion``: the same five checks as the automatic report resume, so a promotion
fails closed alike), the window preflight (``termctl open --dry-run``) before anything is created,
``claude-session start --resume --mode interactive`` and the classification check. ``open_window``
listens on the handoff socket, ``termctl open``s the host (``supervise --host``), records the pane
on the promoted run and serves the handoff. The supervisor then waits for the host pane process
(``Supervisor.watch_host``) and applies the outcome table to the promoted run.

A failure before the host runs (open, handoff) cancels the pending run (``claude-session stop``),
after the host pane process, if one exists, was told to stop (exactly one process parks: the
supervisor, per the chain rule). A promotion that ended with a dead pane (``--keep-on-exit
failed``) has it closed (``close_dead``).

Host mode (``supervise --host``, ``Supervisor.run_host``) is the window process of a promoted run;
it does no bookkeeping.
"""

from __future__ import annotations

import os
import signal
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from . import chain, handoff, lifecycle, procs, third_party
from .chain import Final, RunCtx
from .launch import resolve_stored_tool
from .output import CliError
from .session_backend import SessionBackend
from .store import StoreError
from .term import Pane, TermBackend, sync_window_labels, window_labels

if TYPE_CHECKING:
    from .supervise import Supervisor

#: Seconds the host pane process gets to exit after a SIGTERM when the open or handoff failed.
HOST_STOP_WAIT = 5.0
HOST_TICK = 0.25  # coarse: a window closes in a moment, nothing waits on this


def wants(sup: Supervisor, final: Final) -> bool:
    """Whether the run that just ended is promoted: a headless run with a session-phase exit 13
    (``blocked`` or ``needs-input``), once per chain, when ``promote`` resolves to true, and
    never after a stop, a forwarded signal or a SIGTERM or SIGHUP received during the chain."""
    result = final.result
    if (
        sup.promoted
        or sup.terminated
        or result.phase != "session"
        or result.exit_code != 13
        or result.outcome not in ("blocked", "needs-input")
        or result.end_reason in chain.NO_RESUME_END_REASONS
    ):
        return False
    assert sup.store is not None
    try:
        row = sup.store.get_agent(sup.agent_id)
        run_row = sup.store.get_run(sup.agent_id, final.ctx.run) or {}
    except StoreError:
        return False  # fail closed: a record that cannot be read promotes nothing
    if row is None or row["status"] != "live" or run_row.get("mode") != "headless":
        return False
    kind = run_row.get("run_kind") or row.get("run_kind")
    return sup.config.resolve_promote(
        kind=kind if isinstance(kind, str) else None,
        repo=row.get("repo") if isinstance(row.get("repo"), str) else None,
        explicit=row.get("promote") if isinstance(row.get("promote"), bool) else None,
    )


def termctl_of(sup: Supervisor, row: dict[str, Any]) -> TermBackend:
    """termctl as the launcher resolved it: config's command, else the absolute path stored
    with the agent (exit 12 ``tool_missing`` when neither resolves). The supervisor is the
    actor of its ``open`` and ``close``."""
    stored = row.get("termctl_path")
    argv = resolve_stored_tool(
        sup.config.termctl_cmd, "termctl", stored if isinstance(stored, str) else None
    )
    return TermBackend(
        argv,
        actor="auto:agent-spawn",
        decision_ids=row.get("decision_ids") or [],
        gateway=sup.gateway,
    )


def _plain_directory(cwd: object) -> str | None:
    """The preflight's ``--cwd``: only a directory that exists (never ``session-dir``)."""
    if isinstance(cwd, str) and cwd and cwd != third_party.SESSION_DIR and os.path.isdir(cwd):
        return os.path.realpath(cwd)
    return None


def start(sup: Supervisor, backend: SessionBackend, session: str, trigger: Final) -> RunCtx:
    """Mark the promotion pending, run the pre-start checks and the window preflight, then start
    the promoted run. Raises ``ChainFailure`` (the supervisor parks per ``trigger``) or
    ``ChainCanceled``. The window is opened by ``open_window``."""
    assert sup.store is not None
    store, agent_id = sup.store, sup.agent_id
    sup.promoted = True
    with store.lease(lifecycle.lease_key(agent_id), wait=lifecycle.LEASE_WAIT) as lease:
        try:
            row = store.compare_and_set(
                lease, agent_id, expect_status="live", changes={"promotion_pending": True}
            )
        except StoreError:
            raise chain.ChainCanceled from None  # the agent is not live any more
    try:
        trigger_row = store.get_run(agent_id, trigger.ctx.run) or {}
        sup.log("promotion", message="promoting the run", run=trigger.ctx.run)

        def preflight(classification: third_party.Classification) -> None:
            assert sup.store is not None
            try:
                term = termctl_of(sup, row)
                labels, notes = window_labels(
                    classification.run_kind, agent_id, sup.store.get_labels(agent_id).texts()
                )
                for note in notes:
                    sup.log("warning", level="warn", message=note)
                reply = term.preflight(
                    name=row["name"],
                    labels=labels,
                    placement=row.get("term_placement"),
                    cwd=_plain_directory(trigger_row.get("cwd")),
                    argv=[row["agent_spawn_path"], "supervise"],
                )
            except CliError as error:
                raise chain.failure_of(error) from None
            except (StoreError, KeyError):
                raise chain.ChainFailure("agent-spawn", "failed") from None
            socket = reply.get("socket_path")
            if isinstance(socket, str) and socket.startswith("/") and "\0" not in socket:
                sup.window_socket = socket  # the tmux server the window will be created on

        return chain.start_promotion(sup, backend, row, session, trigger, preflight)
    except StoreError:
        # promotion_pending is set: one process must park, so a record that cannot be read or
        # written fails the promotion like any other chain failure
        raise chain.ChainFailure("agent-spawn", "failed") from None


@dataclass
class _Opened:
    """What ``open_window`` has created so far (for its cleanup)."""

    pane: Pane | None = None
    identity: procs.ProcessIdentity | None = None


def open_window(
    sup: Supervisor, backend: SessionBackend, session: str, ctx: RunCtx, trigger: Final
) -> None:
    """Open the host window of the promoted run and complete the handoff. On success the host
    pane process is ``ctx.host`` and the supervisor forwards signals to it. A failure cancels the
    pending run and raises ``ChainFailure`` (the supervisor parks per ``trigger``); a SIGTERM,
    SIGHUP or SIGINT that arrives until the host is recorded and verified cancels it too and
    raises ``ChainCanceled`` (the supervisor parks per ``trigger``, without a chain failure)."""
    assert sup.store is not None
    opened = _Opened()
    term: TermBackend | None = None
    try:
        row = sup.store.get_agent(sup.agent_id) or {}
        run_row = sup.store.get_run(sup.agent_id, ctx.run) or {}
    except StoreError:
        _abandon(sup, backend, None, session, ctx, opened, None, None)
        raise chain.ChainFailure("agent-spawn", "failed", run=ctx.run) from None
    hooks, token = run_row.get("hooks"), run_row.get("outcome_token")
    try:
        term = termctl_of(sup, row)
        _open(sup, backend, term, row, session, ctx, opened)
    except chain.ChainFailure as failure:
        _abandon(sup, backend, term, session, ctx, opened, hooks, token)
        _check_stop(sup)  # a signal during the failing open parks per the trigger, no failure
        failure.run = ctx.run
        raise
    except chain.ChainCanceled:
        _abandon(sup, backend, term, session, ctx, opened, hooks, token)
        raise
    except CliError as error:
        _abandon(sup, backend, term, session, ctx, opened, hooks, token)
        _check_stop(sup)
        raise chain.failure_of(error, run=ctx.run) from None
    except BaseException as error:
        sup.log("warning", level="error", message=f"promotion failed: {type(error).__name__}")
        _abandon(sup, backend, term, session, ctx, opened, hooks, token)
        if isinstance(error, Exception):
            _check_stop(sup)
        if isinstance(error, Exception):
            raise chain.ChainFailure("agent-spawn", "failed", run=ctx.run) from None
        raise
    sup.term = term  # the window markers of the promoted run


def _window_cwd(sup: Supervisor, backend: SessionBackend, session: str, cwd: object) -> str | None:
    """The window's cwd: the run's cwd; for ``session-dir`` the realpath of
    ``<csession_dir>/work``. ``None`` leaves it to termctl."""
    if cwd == third_party.SESSION_DIR:
        return os.path.realpath(os.path.join(backend.status(session).csession_dir, "work"))
    return os.path.realpath(cwd) if isinstance(cwd, str) and cwd else None


def host_argv(sup: Supervisor, row: dict[str, Any], socket_path: str) -> list[str]:
    """The window's command: absolute paths only, no secret and no agent text."""
    return [
        row["agent_spawn_path"],
        "supervise",
        sup.agent_id,
        "--host",
        "--handoff",
        socket_path,
        "--handoff-timeout",
        str(sup.config.handoff_timeout),
        "--state-dir",
        sup.state_dir,
        "--config",
        row["config_path"],
    ]


def _open(
    sup: Supervisor,
    backend: SessionBackend,
    term: TermBackend,
    row: dict[str, Any],
    session: str,
    ctx: RunCtx,
    opened: _Opened,
) -> None:
    """Listen, ``termctl open``, record the window, serve the handoff and track the host. Until
    the host is verified the supervisor is still headless: a SIGTERM, SIGHUP or SIGINT that
    arrived while the window was being opened is checked after ``open`` returned and once more
    when the handoff is done, and cancels the promotion (the host, if any, is stopped first)."""
    assert sup.store is not None
    store, agent_id = sup.store, sup.agent_id
    try:
        listener = handoff.HandoffListener.create(
            sup.state_dir, agent_id=agent_id, gateway=sup.gateway
        )
    except handoff.HandoffFailed:
        raise chain.ChainFailure("agent-spawn", "handoff_failed", run=ctx.run) from None
    try:
        run_row = store.get_run(agent_id, ctx.run) or {}
        fresh = store.get_agent(agent_id) or row
        kind = fresh.get("run_kind")
        labels, _ = window_labels(
            kind if isinstance(kind, str) else None, agent_id, store.get_labels(agent_id).texts()
        )
        before = _panes(sup, term)
        try:
            pane = term.open(
                name=row["name"],
                labels=labels,
                placement=row.get("term_placement"),
                cwd=_window_cwd(sup, backend, session, run_row.get("cwd")),
                argv=host_argv(sup, row, str(listener.socket_path)),
            )
        except CliError as error:
            if error.exit_code not in (11, 13):  # a veto or escalation created no pane
                _recover_pane(sup, term, ctx, before)  # `open` may have created its pane unreported
            raise
        opened.pane = pane
        _record_window(sup, ctx, pane)  # the pane first: whatever follows can close it
        opened.identity = _record_identity(sup, ctx, pane)
        _check_stop(sup)  # a signal that arrived while `termctl open` ran
        if opened.identity is None:  # the host pane process could not be read
            raise chain.ChainFailure("agent-spawn", "handoff_failed", run=ctx.run)
        try:
            listener.serve(
                pane_pid=pane.pid,
                pane_state=pane.state,
                timeout=sup.config.handoff_timeout,
                env=dict(os.environ),
            )
        except handoff.HandoffFailed:
            raise chain.ChainFailure("agent-spawn", "handoff_failed", run=ctx.run) from None
        _track_host(sup, ctx, opened.identity)
        term_labels = kind if isinstance(kind, str) else None
        if not sync_window_labels(term, store, agent_id, pane, term_labels, labels):
            sup.log("warning", level="warn", message="the window's labels could not be synced")
    finally:
        listener.close()


def _check_stop(sup: Supervisor) -> None:
    """Raise ``ChainCanceled`` when a SIGTERM, SIGHUP or SIGINT arrived (the supervisor is
    headless until the host is tracked, so all three count)."""
    if sup.collect_signals():
        raise chain.ChainCanceled


def _panes(sup: Supervisor, term: TermBackend) -> set[tuple[str, str]] | None:
    """The agent's labeled panes (an older run's dead pane among them) before the real ``open``,
    so that recovery tells the new pane from them, on the server the preflight named; None when
    unknown. Best effort."""
    try:
        return set(term.find_panes(sup.agent_id, sup.window_socket))
    except CliError:
        return None


def _recover_pane(
    sup: Supervisor, term: TermBackend, ctx: RunCtx, before: set[tuple[str, str]] | None
) -> None:
    """A ``termctl open`` that failed may have created its pane without reporting it: find it by
    its ``agent=`` label among the panes that were not there before, and record it on the run,
    so that ``close_dead`` and teardown find it. Best effort, never raises."""
    assert sup.store is not None
    try:
        found = term.find_panes(sup.agent_id, sup.window_socket)
        if before is not None:
            found = [pane for pane in found if pane not in before]
        if before is None and len(found) != 1 or not found:
            return  # nothing, or nothing certain: no pane id is guessed
        with sup.store.lease(lifecycle.lease_key(sup.agent_id), wait=lifecycle.LEASE_WAIT) as lease:
            sup.store.put_run(
                lease,
                sup.agent_id,
                ctx.run,
                {"term_id": found[0][0], "socket_path": found[0][1], "placement": "window"},
            )
    except (CliError, StoreError, OSError):
        sup.log("warning", level="warn", message="a window that failed to open was not found")


def _record_window(sup: Supervisor, ctx: RunCtx, pane: Pane) -> None:
    """Store the window on the promoted run (``term_id``, ``socket_path``, ``placement:
    window``) before anything else can fail: cleanup closes the pane by this record. The agent
    becomes ``interactive`` in a ``window`` in the same transaction (its mode and placement are
    its last run's): a promotion canceled before its window exists leaves the agent as it was."""
    assert sup.store is not None
    with sup.store.lease(lifecycle.lease_key(sup.agent_id), wait=lifecycle.LEASE_WAIT) as lease:
        with sup.store.transaction():
            sup.store.update_agent(
                lease, sup.agent_id, {"mode": "interactive", "placement": "window"}
            )
            sup.store.put_run(
                lease,
                sup.agent_id,
                ctx.run,
                {"term_id": pane.term_id, "socket_path": pane.socket_path, "placement": "window"},
            )


def _record_identity(sup: Supervisor, ctx: RunCtx, pane: Pane) -> procs.ProcessIdentity | None:
    """Store the host pane process on the promoted run (``pane_proc``) before the handoff
    completes: the host checks this record once it has its environment. The pane process is not
    the agent's supervisor. ``None`` when it cannot be read (the window is recorded already)."""
    assert sup.store is not None
    identity: procs.ProcessIdentity | None = None
    if pane.pid is not None and pane.state != "gone":
        try:
            identity = procs.read_identity(pane.pid, gateway=sup.gateway)
        except (procs.ProcessIdentityError, OSError):
            identity = None
    if identity is not None:
        with sup.store.lease(lifecycle.lease_key(sup.agent_id), wait=lifecycle.LEASE_WAIT) as lease:
            sup.store.put_run(lease, sup.agent_id, ctx.run, {"pane_proc": identity})
    return identity


def _track_host(sup: Supervisor, ctx: RunCtx, identity: procs.ProcessIdentity) -> None:
    """From here a SIGTERM or SIGHUP goes to the host pane process (and only to it); SIGINT and
    SIGQUIT are ignored (Signals, a promotion). A signal that arrived while the window was
    opening (not forwarded yet, the supervisor is headless until now) cancels the promotion
    instead (``ChainCanceled``)."""
    from .supervise import FORWARDED_INTERACTIVE

    _check_stop(sup)  # with the headless set: SIGINT counts here
    ctx.host = identity
    sup.child = None
    sup.forwarder = procs.SignalForwarder(identity, gateway=sup.gateway)
    sup.signals.forwarded = FORWARDED_INTERACTIVE
    sup.signals.forwarder = sup.forwarder
    for received in sup.signals.take():  # arrived since the check: the host gets them
        sup.note_signal(received)
        sup.forward(received)


def _abandon(
    sup: Supervisor,
    backend: SessionBackend,
    term: TermBackend | None,
    session: str,
    ctx: RunCtx,
    opened: _Opened,
    hooks: object,
    token: object,
) -> None:
    """The window or the handoff failed: nothing may adopt the pending run. The host pane
    process, when there is one, is told to stop first (it may have its environment already),
    then the pending run is canceled and a dead pane closed. Best effort, never raises."""
    sup.signals.forwarder = None
    sup.forwarder = None
    gone = True
    if opened.identity is not None:
        try:
            procs.forward_signal(opened.identity, signal.SIGTERM, gateway=sup.gateway)
            gone = procs.wait_for_exit(opened.identity, HOST_STOP_WAIT, gateway=sup.gateway)
        except Exception as problem:  # noqa: BLE001 - cleanup must continue
            sup.log("warning", level="warn", message=f"could not stop the host: {problem}")
            gone = False
    if gone:  # a host that may still adopt the run is never raced by a cancel
        try:
            chain.cancel_pending(
                sup,
                backend,
                session,
                ctx.run,
                hooks=hooks,
                token=token if isinstance(token, str) else None,
                promoted=True,
            )
        except Exception as problem:  # noqa: BLE001 - best effort
            sup.log("warning", level="warn", message=f"could not cancel the run: {problem}")
    else:
        sup.log("warning", level="warn", message="the host did not exit: the run is not canceled")
    if term is not None:
        close_dead(sup, term, ctx, wait=HOST_STOP_WAIT)


def close_dead(sup: Supervisor, term: TermBackend, ctx: RunCtx, *, wait: float = 0.0) -> None:
    """Close the promoted run's pane when it is dead (``--keep-on-exit failed`` keeps a pane
    whose host failed). A running pane is never closed. ``wait`` gives a host that is about to
    exit (it was refused its environment) that many seconds to become a dead pane, whenever the
    run names a window. Best effort."""
    assert sup.store is not None
    try:
        run_row = sup.store.get_run(sup.agent_id, ctx.run) or {}
        term_id, socket_path = run_row.get("term_id"), run_row.get("socket_path")
        if not isinstance(term_id, str) or not isinstance(socket_path, str):
            return
        deadline = sup.gateway.monotonic() + wait
        while True:
            if term_id in term.list_dead(sup.agent_id, socket_path):
                term.close(term_id, socket_path)
                return
            if sup.gateway.monotonic() >= deadline:
                return
            if (term_id, socket_path) not in term.find_panes(sup.agent_id, socket_path):
                return  # the pane is gone (a clean exit closes it): nothing to wait for
            sup.gateway.sleep(HOST_TICK)
    except (CliError, StoreError) as problem:
        sup.log("warning", level="warn", message=f"could not close the dead pane: {problem}")
