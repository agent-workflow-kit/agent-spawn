"""Process bookkeeping for tests that start processes.

Every test process is recorded (PID plus start time) at launch and killed on teardown by
exactly that identity, through a pidfd (or the ``syscall`` fallback). SIGTERM first, SIGKILL
after a grace period. Nothing is ever killed by pattern. ``tests/conftest.py`` loads this as
a plugin; ``proc_tracker`` is autouse, so a test cannot leave a process behind unnoticed.
"""

from __future__ import annotations

import signal
import subprocess
from collections.abc import Iterator, Sequence

import pytest

from agent_spawn import procs
from agent_spawn.procs import ProcessIdentity

GRACE = 2.0


class ProcTracker:
    def __init__(self) -> None:
        self.identities: list[ProcessIdentity] = []
        self.popens: list[subprocess.Popen[bytes]] = []
        self.survivors: list[ProcessIdentity] = []

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

    def cleanup(self) -> None:
        for identity in reversed(self.identities):
            handle = procs.open_verified(identity)
            if handle is None:
                continue
            with handle:
                handle.send(signal.SIGTERM)
                handle.send(signal.SIGCONT)
                if not handle.wait_exit(GRACE):
                    handle.send(signal.SIGKILL)
                    if not handle.wait_exit(GRACE * 2):
                        self.survivors.append(identity)
        for child in self.popens:
            try:
                child.wait(timeout=GRACE)
            except subprocess.TimeoutExpired:
                pass
            for stream in (child.stdin, child.stdout, child.stderr):
                if stream is not None:
                    stream.close()


@pytest.fixture(autouse=True)
def proc_tracker() -> Iterator[ProcTracker]:
    tracker = ProcTracker()
    try:
        yield tracker
    finally:
        tracker.cleanup()
    assert not tracker.survivors, f"test processes survived SIGKILL: {tracker.survivors}"
