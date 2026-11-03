"""An in-memory ``Gateway`` for unit tests (see the seam notes in ``agent_spawn.gateway``).

It models ``/proc``, pidfds bound to the process that held a PID when they were opened, a
signal log, a manual clock and canned ``run`` results keyed by ``argv[0]``. Anything a test
does not configure raises, so an unexpected OS access fails loudly.
"""

from __future__ import annotations

import socket
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import IO

from agent_spawn.gateway import PeerCredentials, SignalHandler

BOOT = "11111111-2222-3333-4444-555555555555"


def stat_line(pid: int, comm: str, state: str, start: int, ppid: int = 1) -> str:
    """A realistic ``/proc/<pid>/stat`` line (52 fields)."""
    tail = [state, str(ppid)] + ["0"] * 17 + [str(start)] + ["0"] * 29
    return f"{pid} ({comm}) " + " ".join(tail) + "\n"


@dataclass
class FakeProc:
    comm: str = "fake"
    state: str = "S"
    start: int = 1000
    ppid: int = 1
    token: int = 0


@dataclass
class FakeGateway:
    platform_name: str = "linux"
    procs: dict[int, FakeProc] = field(default_factory=dict)
    files: dict[str, str] = field(default_factory=dict)
    run_results: dict[str, Callable[[Sequence[str]], subprocess.CompletedProcess[str]]] = field(
        default_factory=dict
    )
    signals: list[tuple[int, int]] = field(default_factory=list)  # (pid, signal) delivered
    killed_by_pid: list[tuple[int, int]] = field(default_factory=list)
    run_calls: list[list[str]] = field(default_factory=list)
    now: float = 1000.0
    boot_id: str = BOOT
    pidfd_error: OSError | None = None
    on_pidfd_open: Callable[[int], None] | None = None
    handlers: dict[int, SignalHandler] = field(default_factory=dict)
    _fds: dict[int, tuple[int, int]] = field(default_factory=dict)  # fd -> (pid, token)
    _next_fd: int = 100
    _next_token: int = 1
    peer: PeerCredentials = PeerCredentials(1, 1, 1)
    prctl_calls: list[tuple[int, ...]] = field(default_factory=list)

    # test helpers
    def add(self, pid: int, **kwargs: object) -> FakeProc:
        proc = FakeProc(**kwargs)  # type: ignore[arg-type]
        proc.token = self._next_token
        self._next_token += 1
        self.procs[pid] = proc
        return proc

    def _sync_files(self) -> None:
        self.files["/proc/sys/kernel/random/boot_id"] = self.boot_id + "\n"
        for pid, proc in self.procs.items():
            self.files[f"/proc/{pid}/stat"] = stat_line(
                pid, proc.comm, proc.state, proc.start, proc.ppid
            )

    # Gateway
    def run(
        self,
        argv: Sequence[str],
        *,
        stdin: str | None = None,
        timeout: float | None = None,
        env: Mapping[str, str] | None = None,
        cwd: str | None = None,
    ) -> subprocess.CompletedProcess[str]:
        self.run_calls.append(list(argv))
        return self.run_results[argv[0]](argv)

    def spawn(
        self,
        argv: Sequence[str],
        *,
        stdin: int | IO[bytes] | None = None,
        stdout: int | IO[bytes] | None = None,
        stderr: int | IO[bytes] | None = None,
        env: Mapping[str, str] | None = None,
        cwd: str | None = None,
        start_new_session: bool = False,
    ) -> subprocess.Popen[bytes]:
        raise NotImplementedError

    def read_file(self, path: str) -> str:
        if path.startswith("/proc/"):
            self.files.pop(path, None)
            self._sync_files()
        if path not in self.files:
            raise FileNotFoundError(path)
        return self.files[path]

    def monotonic(self) -> float:
        return self.now

    def wall_time(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds

    def platform(self) -> str:
        return self.platform_name

    def getuid(self) -> int:
        return 1000

    def getpid(self) -> int:
        return 1

    def pidfd_open(self, pid: int) -> int:
        if self.pidfd_error is not None:
            raise self.pidfd_error
        if self.on_pidfd_open is not None:
            self.on_pidfd_open(pid)
        proc = self.procs.get(pid)
        if proc is None:
            raise ProcessLookupError(pid)
        fd = self._next_fd
        self._next_fd += 1
        self._fds[fd] = (pid, proc.token)
        return fd

    def pidfd_send_signal(self, pidfd: int, sig: int) -> None:
        pid, token = self._fds[pidfd]
        proc = self.procs.get(pid)
        if proc is None or proc.token != token:
            raise ProcessLookupError(pid)  # bound to the original process, never a reused PID
        self.signals.append((pid, sig))

    def kill(self, pid: int, sig: int) -> None:
        self.killed_by_pid.append((pid, sig))

    def close(self, fd: int) -> None:
        self._fds.pop(fd)

    def open_fds(self) -> int:
        return len(self._fds)

    def poll_readable(self, fd: int, timeout: float) -> bool:
        pid, token = self._fds[fd]
        proc = self.procs.get(pid)
        return proc is None or proc.token != token or proc.state in ("Z", "X")

    def wait_pid_exit(self, pid: int, timeout: float) -> bool:
        self.now += timeout
        return pid not in self.procs

    def install_signal_handler(self, sig: int, handler: SignalHandler) -> SignalHandler:
        previous = self.handlers.get(sig, 0)
        self.handlers[sig] = handler
        return previous

    def prctl(self, option: int, arg2: int = 0, arg3: int = 0, arg4: int = 0, arg5: int = 0) -> int:
        self.prctl_calls.append((option, arg2, arg3, arg4, arg5))
        return 0

    def peer_credentials(self, sock: socket.socket) -> PeerCredentials:
        return self.peer

    def pipe(self) -> tuple[int, int]:
        raise NotImplementedError

    def fork(self) -> int:
        raise NotImplementedError

    def setsid(self) -> int:
        raise NotImplementedError

    def waitpid(self, pid: int, flags: int = 0) -> tuple[int, int]:
        raise NotImplementedError

    def exit_now(self, code: int) -> None:
        raise NotImplementedError
