"""``agent-spawn wait``: block until an agent is not running, then report its last outcome.

``wait`` is an observer (interface.md, `wait` and Signals): it never writes to the agent's
state, never signals the agent or its supervisor, and decides every poll from the stored record
and a liveness check of the recorded supervisor. The supervisor's pidfd only wakes the poll up
earlier.
"""

from __future__ import annotations

import os
import signal
from collections.abc import Mapping
from types import FrameType
from typing import Any, cast

from . import gateway as gw
from . import launch, lifecycle, procs, third_party
from .agent_view import agent_object, chain_first_working_at
from .commands.base import Invocation
from .ids import resolve_agent_id
from .output import (
    EXIT_SIGINT,
    EXIT_SIGTERM,
    EXIT_TIMED_OUT,
    CliError,
    Outcome,
    PayloadKind,
    result_object,
)
from .session_backend import SessionBackend
from .store import Store, StoreError, StoreMissing, default_state_dir
from .validate import GrammarError, parse_duration

POLL = launch.POLL  # seconds between looks at the record
DEAD_SUPERVISOR_POLLS = launch.DEAD_SUPERVISOR_POLLS


class _Signals:
    """Records the first SIGTERM, SIGHUP or SIGINT; the poll loop acts on it."""

    def __init__(self) -> None:
        self.first: int | None = None

    def handle(self, signum: int, frame: FrameType | None) -> None:
        if self.first is None:
            self.first = signum

    @property
    def exit_code(self) -> int:
        return EXIT_SIGTERM if self.first == signal.SIGTERM else EXIT_SIGINT


def _interrupted(signals: _Signals) -> Outcome:
    return Outcome(signals.exit_code, None, PayloadKind.NONE)  # nothing is printed or written


class _Wait:
    def __init__(self, inv: Invocation, signals: _Signals) -> None:
        self.inv = inv
        self.signals = signals
        self.gateway = gw.get_gateway()
        self.warnings: list[str] = list(inv.warnings)
        self.state_dir = inv.state_dir or str(default_state_dir())
        self.checked_run: tuple[str, int | None] | None = None

    # -- the third-party rule for --result-file ------------------------------------------------

    def check_result_file(self, store: Store, row: Mapping[str, Any]) -> None:
        """Run the check; on any failure forget the file, so ``cli.main`` writes nothing to a
        path that was never checked."""
        try:
            self._check_result_file(store, row)
        except BaseException:
            third_party.drop_result_file(self.inv)
            raise

    def _check_result_file(self, store: Store, row: Mapping[str, Any]) -> None:
        """Exit 2 when the file would lie in a third-party-reading run's cwd or session dir.

        The run is the agent's run in progress, else its latest (``row["run"]``). Anything not
        known to be trusted counts as third-party-reading. A refused path is never written,
        not even with the error.
        """
        target = self.inv.result_file
        run = row.get("run")
        key = (row["agent_id"], run if isinstance(run, int) else None)
        if target is None or not isinstance(run, int) or self.checked_run == key:
            return
        run_row = store.get_run(row["agent_id"], run)
        reads = None if run_row is None else run_row.get("reads_third_party")
        if reads is False:
            self.checked_run = key
            return
        directories = self.run_directories(row)
        if any(third_party.inside(target.parent, os.path.realpath(d)) for d in directories):
            third_party.drop_result_file(self.inv)
            raise CliError.usage(
                "--result-file lies inside the cwd or session directory of a "
                "third-party-reading run"
            )
        self.checked_run = key

    def run_directories(self, row: Mapping[str, Any]) -> list[str]:
        """The run's cwd (a throwaway's root is the cwd) and its session directory."""
        found: list[str] = []
        cwd = row.get("cwd")
        if isinstance(cwd, str) and os.path.isabs(cwd):
            found.append(cwd)
        session = row.get("csession_id")
        if not isinstance(session, str):
            return found
        config = self.inv.config
        assert config is not None
        try:
            status = SessionBackend(
                launch.resolve_tool(config.claude_session_cmd, "claude-session"),
                gateway=self.gateway,
            ).status(session)
        except CliError as error:
            if error.exit_code == 10:  # the session is gone: it has no directory to protect
                return found
            third_party.drop_result_file(self.inv)  # fail closed: the directory is unknown
            raise
        found.append(status.csession_dir)
        if isinstance(status.run.cwd, str) and os.path.isabs(status.run.cwd):
            found.append(status.run.cwd)
        return found

    def guard(self, store: Store, row: Mapping[str, Any]) -> Outcome | None:
        """The result-file check; a signal received meanwhile ends the wait via the signal."""
        try:
            self.check_result_file(store, row)
        except Exception:
            if self.signals.first is not None:
                return _interrupted(self.signals)
            raise
        if self.signals.first is not None:
            return _interrupted(self.signals)
        return None

    # -- the poll loop -------------------------------------------------------------------------

    def run(self) -> Outcome:
        args = self.inv.args
        deadline = None
        if args.timeout is not None:
            try:
                deadline = self.gateway.monotonic() + parse_duration(str(args.timeout))
            except GrammarError as error:
                raise CliError.usage(str(error)) from None
        try:
            store = Store.open(self.state_dir, create=False, gateway=self.gateway)
        except StoreMissing:
            raise CliError.of("not_found", f"no agent matches {args.agent_id}") from None
        except StoreError as error:
            raise CliError.of(
                "failed", f"the state database could not be used ({type(error).__name__})"
            ) from None
        try:
            return self.poll(store, args.agent_id, deadline)
        except StoreError as error:
            raise CliError.of(
                "failed", f"the state database could not be used ({type(error).__name__})"
            ) from None
        finally:
            store.close()

    def poll(self, store: Store, text: str, deadline: float | None) -> Outcome:
        resolved = resolve_agent_id(text, store)
        agent_id = resolved.agent_id
        dead_looks = 0
        handle: procs.ProcHandle | None = None
        handle_identity: procs.ProcessIdentity | None = None
        try:
            while True:
                if self.signals.first is not None:
                    return _interrupted(self.signals)
                row = store.get_agent(agent_id)
                if row is None:
                    raise CliError.of("not_found", f"agent {agent_id} no longer exists")
                stopped = self.guard(store, row)
                if stopped is not None:
                    return stopped
                holder = self.holder(row)
                alive = holder is not None and procs.is_alive(holder, gateway=self.gateway)
                if lifecycle.run_settled(row, supervisor_alive=alive):
                    return self.final(store, row)
                if not alive and (
                    row["status"] in ("launching", "live") or row.get("promotion_pending")
                ):
                    dead_looks += 1
                    if dead_looks >= DEAD_SUPERVISOR_POLLS:
                        # TODO(#31): the liveness check parks an agent whose supervisor died
                        # and settles its promotion; until it exists a dead supervisor of a
                        # running agent, or of a pending promotion, is an error.
                        raise CliError.of(
                            "failed",
                            "the supervisor exited before the agent parked (see reconcile)",
                            agent_id=agent_id,
                        )
                else:
                    dead_looks = 0
                if deadline is not None and self.gateway.monotonic() >= deadline:
                    payload = result_object(
                        agent=agent_object(store, row, warnings=self.warnings),
                        timed_out=True,
                        first_working_at=chain_first_working_at(store, row),
                        warnings=self.warnings,
                    )
                    return Outcome(EXIT_TIMED_OUT, payload, PayloadKind.RESULT)
                if alive and holder is not None and holder != handle_identity:
                    if handle is not None:
                        handle.close()
                    handle = procs.open_verified(holder, gateway=self.gateway)
                    handle_identity = holder
                if handle is not None and alive:
                    handle.wait_exit(POLL)  # a wake-up only: the next look decides
                else:
                    self.gateway.sleep(POLL)
        finally:
            if handle is not None:
                handle.close()

    @staticmethod
    def holder(row: Mapping[str, Any]) -> procs.ProcessIdentity | None:
        """The process whose exit the agent waits for: its supervisor, or while `launching`
        and no supervisor is recorded yet, its launcher."""
        supervisor = row.get("supervisor")
        if isinstance(supervisor, procs.ProcessIdentity):
            return supervisor
        if row["status"] == "launching":
            launcher = row.get("launcher")
            return launcher if isinstance(launcher, procs.ProcessIdentity) else None
        return None

    def final(self, store: Store, row: Mapping[str, Any]) -> Outcome:
        agent_id = row["agent_id"]
        stopped = self.guard(store, row)
        if stopped is not None:
            return stopped
        last = row.get("last_outcome")
        if not isinstance(last, dict):
            if row["status"] == "torn_down":
                raise CliError.of(
                    "not_found",
                    f"agent {agent_id} is torn down and had no run",
                    agent_id=agent_id,
                    details={"status": "torn_down"},
                )
            raise CliError.of("no_run", f"agent {agent_id} has no run", agent_id=agent_id)
        outcome = cast(dict[str, Any], last)
        payload = result_object(
            agent=agent_object(store, row, warnings=self.warnings),
            run=outcome,
            first_working_at=chain_first_working_at(store, row),
            warnings=self.warnings,
        )
        return Outcome(int(outcome["exit_code"]), payload, PayloadKind.RESULT)


def wait(inv: Invocation) -> Outcome:
    signals = _Signals()
    restore = procs.install_signal_handlers(
        {
            signal.SIGTERM: signals.handle,
            signal.SIGHUP: signals.handle,
            signal.SIGINT: signals.handle,
        }
    )
    # Keep the handlers until the result is printed: a signal then cannot lose it.
    launch.defer_restore(restore)
    return _Wait(inv, signals).run()
