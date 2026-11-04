"""An in-memory ``Gateway`` for unit tests (see the seam notes in ``agent_spawn.gateway``).

It models ``/proc``, pidfds bound to the process that held a PID when they were opened, a
signal log, a manual clock and canned ``run`` results keyed by ``argv[0]``. Anything a test
does not configure raises, so an unexpected OS access fails loudly.
"""

from __future__ import annotations

import errno
import socket
import subprocess
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import IO

from agent_spawn.gateway import PeerCredentials, SignalHandler

BOOT = "11111111-2222-3333-4444-555555555555"

# ``sysctl -n kern.boottime`` as macOS prints it: the trailing date is in local time.
DARWIN_BOOTTIME = "{ sec = 1700000000, usec = 123456 } Tue Nov 14 22:13:20 2023\n"


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
    tgid: int | None = None  # None: a thread-group leader (Tgid == PID)
    threads: int = 1
    has_status: bool = True


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
    run_envs: list[dict[str, str] | None] = field(default_factory=list)
    run_timeouts: list[float | None] = field(default_factory=list)
    now: float = 1000.0
    boot_id: str = BOOT
    pidfd_error: OSError | None = None
    on_pidfd_open: Callable[[int], None] | None = None
    after_pidfd_open: Callable[[int], None] | None = None
    before_send_signal: Callable[[int], None] | None = None
    pipe_plan: list[OSError | None] | None = None  # None: pipe() is not configured
    sigmask_calls: list[tuple[int, frozenset[int]]] = field(default_factory=list)
    handlers: dict[int, SignalHandler] = field(default_factory=dict)
    _fds: dict[int, tuple[int, int]] = field(default_factory=dict)  # fd -> (pid, token)
    _next_fd: int = 100
    _next_token: int = 1
    _pipe_fds: set[int] = field(default_factory=set)
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
            status = f"/proc/{pid}/status"
            if proc.has_status:
                tgid = pid if proc.tgid is None else proc.tgid
                self.files[status] = (
                    f"Name:\t{proc.comm}\nState:\t{proc.state}\nTgid:\t{tgid}\n"
                    f"Pid:\t{pid}\nPPid:\t{proc.ppid}\nThreads:\t{proc.threads}\n"
                )
            else:
                self.files.pop(status, None)

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
        self.run_envs.append(None if env is None else dict(env))
        self.run_timeouts.append(timeout)
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
            pid_text = path.split("/")[2]
            if pid_text.isdigit() and int(pid_text) not in self.procs:
                self.files.pop(path, None)
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
        if self.after_pidfd_open is not None:
            self.after_pidfd_open(pid)
        return fd

    def pidfd_send_signal(self, pidfd: int, sig: int) -> None:
        pid, token = self._fds[pidfd]
        if self.before_send_signal is not None:
            self.before_send_signal(pid)
        proc = self.procs.get(pid)
        if proc is None or proc.token != token:
            raise ProcessLookupError(pid)  # bound to the original process, never a reused PID
        self.signals.append((pid, sig))

    def kill(self, pid: int, sig: int) -> None:
        if pid not in self.procs:
            raise ProcessLookupError(errno.ESRCH, "No such process")
        self.killed_by_pid.append((pid, sig))

    def close(self, fd: int) -> None:
        if fd in self._fds:
            del self._fds[fd]
        elif fd in self._pipe_fds:
            self._pipe_fds.discard(fd)
        else:
            raise OSError(errno.EBADF, "Bad file descriptor")

    def open_fds(self) -> int:
        return len(self._fds) + len(self._pipe_fds)

    def poll_readable(self, fd: int, timeout: float) -> bool:
        pid, token = self._fds[fd]
        proc = self.procs.get(pid)
        readable = (
            proc is None
            or proc.token != token
            or proc.state in ("X", "x")
            or (proc.state == "Z" and proc.threads <= 1)
        )
        if not readable:
            self.now += max(0.0, timeout)  # a real poll blocks for the timeout
        return readable

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

    def pthread_sigmask(self, how: int, signals: Iterable[int]) -> set[int]:
        self.sigmask_calls.append((how, frozenset(int(s) for s in signals)))
        return set()

    def pipe(self) -> tuple[int, int]:
        if self.pipe_plan is None:
            raise NotImplementedError
        failure = self.pipe_plan.pop(0)
        if failure is not None:
            raise failure
        read_end, write_end = self._next_fd, self._next_fd + 1
        self._next_fd += 2
        self._pipe_fds.update((read_end, write_end))
        return read_end, write_end

    def fork(self) -> int:
        raise NotImplementedError

    def setsid(self) -> None:
        raise NotImplementedError

    def waitpid(self, pid: int, flags: int = 0) -> tuple[int, int]:
        raise NotImplementedError

    def exit_now(self, code: int) -> None:
        raise NotImplementedError
