"""Process bookkeeping for tests that start processes.

Every test process is recorded (PID plus start time) at launch and killed on teardown by
exactly that identity, through a pidfd (or the ``syscall`` fallback). SIGTERM first, SIGKILL
after a grace period. Nothing is ever killed by pattern. ``tests/conftest.py`` loads this as
a plugin; ``proc_tracker`` is autouse, so a test cannot leave a process behind unnoticed.

On Linux the pytest process is made a child subreaper at session start, so every process a
test leaves behind (a detached grandchild included) is reparented to pytest instead of init.
After each test the tracker walks the pytest process's descendants (``/proc`` ppid links):
a live descendant the test did not record is killed by its verified identity (pidfd), reaped
and named in a test failure; zombies left by recorded processes are reaped quietly. Only
descendants of the pytest process are ever touched.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from collections.abc import Iterator, Sequence

import pytest

from agent_spawn import procs
from agent_spawn.gateway import get_gateway
from agent_spawn.procs import ProcessIdentity, parse_stat

GRACE = 2.0
PR_SET_CHILD_SUBREAPER = 36
_LINUX = sys.platform.startswith("linux")


def _stat(pid: int) -> procs.StatInfo | None:
    try:
        with open(f"/proc/{pid}/stat", "rb") as handle:
            return parse_stat(handle.read().decode("utf-8", "surrogateescape"))
    except (OSError, procs.ProcessIdentityError):
        return None


def descendants(root: int) -> dict[int, procs.StatInfo]:
    """Every process whose parent chain reaches ``root`` (Linux ``/proc`` walk)."""
    if not _LINUX:
        return {}
    table: dict[int, procs.StatInfo] = {}
    for name in os.listdir("/proc"):
        if name.isdigit():
            info = _stat(int(name))
            if info is not None:
                table[info.pid] = info
    found: dict[int, procs.StatInfo] = {}
    frontier = [root]
    while frontier:
        parent = frontier.pop()
        for info in table.values():
            if info.ppid == parent and info.pid not in found and info.pid != root:
                found[info.pid] = info
                frontier.append(info.pid)
    return found


def _reap(pid: int) -> None:
    """Reap ``pid`` if it is a zombie child of this process; never blocks."""
    try:
        os.waitpid(pid, os.WNOHANG)
    except ChildProcessError:
        pass


class ProcTracker:
    """``sweep=False`` for a tracker a test makes itself: only the autouse one sweeps strays."""

    def __init__(self, *, sweep: bool = True) -> None:
        self.sweep = sweep
        self.identities: list[ProcessIdentity] = []
        self.popens: list[subprocess.Popen[bytes]] = []
        self.survivors: list[ProcessIdentity] = []
        self.strays: list[str] = []
        self.errors: list[str] = []

    def record(self, identity: ProcessIdentity) -> ProcessIdentity:
        self.identities.append(identity)
        return identity

    def popen(self, argv: Sequence[str], **kwargs: object) -> subprocess.Popen[bytes]:
        """Start a child and record its identity (pidfd first) before anything else."""
        child = subprocess.Popen(list(argv), **kwargs)  # type: ignore[call-overload]  # noqa: S603
        self.popens.append(child)
        handle = procs.identify(child.pid)
        if handle is not None:
            with handle:
                self.record(handle.identity)
        return child

    def _recorded(self) -> set[tuple[int, int]]:
        return {(i.pid, i.start_time) for i in self.identities}

    def _popen_pids(self) -> set[int]:
        return {child.pid for child in self.popens}

    def _kill_verified(self, identity: ProcessIdentity) -> None:
        handle = procs.open_verified(identity)
        if handle is None:
            return
        with handle:
            handle.send(signal.SIGTERM)
            handle.send(signal.SIGCONT)
            if not handle.wait_exit(GRACE):
                handle.send(signal.SIGKILL)
                if not handle.wait_exit(GRACE * 2):
                    self.survivors.append(identity)

    def settle(self, timeout: float) -> None:
        """Wait until no unrecorded descendant is alive, reaping zombies meanwhile."""
        deadline = time.monotonic() + timeout
        while True:
            live = self._reap_zombies()
            if not live or time.monotonic() >= deadline:
                return
            time.sleep(0.02)

    def _reap_zombies(self) -> list[procs.StatInfo]:
        """Reap zombie children not owned by a Popen; return the live unrecorded descendants."""
        me = os.getpid()
        popen_pids = self._popen_pids()
        recorded = self._recorded()
        live: list[procs.StatInfo] = []
        for info in descendants(me).values():
            if info.state in ("Z", "X", "x"):
                if info.ppid == me and info.pid not in popen_pids:
                    _reap(info.pid)
            elif (info.pid, info.start_time) not in recorded:
                live.append(info)
        return live

    def _sweep_strays(self) -> None:
        me = os.getpid()
        for _ in range(10):  # killing a stray reparents its children to us: rescan
            live = self._reap_zombies()
            if not live:
                return
            for info in live:
                try:
                    handle = procs.identify(info.pid)
                    if handle is None:
                        continue
                    with handle:
                        # The pidfd is bound now: re-verify it is the same descendant.
                        if (
                            handle.identity.start_time != info.start_time
                            or info.pid not in descendants(me)
                        ):
                            continue
                        self.strays.append(f"pid {info.pid} ({info.comm})")
                        handle.send(signal.SIGKILL)
                        if not handle.wait_exit(GRACE * 2):
                            self.survivors.append(handle.identity)
                    _reap(info.pid)
                except Exception as exc:  # keep sweeping: one failure must not stop cleanup
                    self.errors.append(f"sweeping pid {info.pid}: {exc!r}")

    def cleanup(self) -> None:
        for identity in reversed(self.identities):
            try:
                self._kill_verified(identity)
            except Exception as exc:  # keep going: one failure must not stop cleanup
                self.errors.append(f"killing {identity}: {exc!r}")
        for child in self.popens:
            try:
                child.wait(timeout=GRACE)
            except subprocess.TimeoutExpired:
                pass
            except Exception as exc:
                self.errors.append(f"waiting for {child.pid}: {exc!r}")
            for stream in (child.stdin, child.stdout, child.stderr):
                if stream is not None:
                    try:
                        stream.close()
                    except Exception as exc:
                        self.errors.append(f"closing a stream of {child.pid}: {exc!r}")
        if not self.sweep:
            return
        try:
            self._sweep_strays()
        except Exception as exc:
            self.errors.append(f"sweeping strays: {exc!r}")


@pytest.fixture(autouse=True, scope="session")
def child_subreaper() -> None:
    """Orphans of test processes are reparented to pytest, so the sweep can find them."""
    if _LINUX:
        get_gateway().prctl(PR_SET_CHILD_SUBREAPER, 1)


@pytest.fixture(autouse=True)
def proc_tracker() -> Iterator[ProcTracker]:
    tracker = ProcTracker()
    try:
        yield tracker
    finally:
        tracker.cleanup()
    assert not tracker.survivors, f"test processes survived SIGKILL: {tracker.survivors}"
    assert not tracker.strays, f"test left unrecorded processes behind: {tracker.strays}"
    assert not tracker.errors, f"process cleanup failed: {tracker.errors}"
