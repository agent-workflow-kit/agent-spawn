"""Single boundary for everything that touches the operating system.

Every subprocess, signal, pidfd, ``prctl``, socket peer-credential read, ``/proc``
read, clock read and fork made by agent-spawn goes through a ``Gateway`` object, so
tests can substitute fakes (design guide section 14). Commands are run as argv lists,
never through a shell, and their output is decoded explicitly as UTF-8 with
``surrogateescape`` (never through the locale).

Fake-substitution seam (for ``tests/fakebin`` and every unit test):

* ``Gateway`` is a ``typing.Protocol``; ``SystemGateway`` is the real implementation.
  A fake only needs the methods the code under test calls, but should satisfy the
  protocol so pyright checks it structurally.
* Code takes the gateway as a keyword argument (``gateway=None`` means the current
  default) and resolves it with ``get_gateway()``. Never import ``subprocess``, ``os.kill``
  or ``os.fork`` anywhere else in ``src/agent_spawn``.
* Tests either pass a fake explicitly, or install one for the test with
  ``set_gateway(fake)`` (it returns the previous gateway; restore it in a fixture).
* ``tests/fake_gateway.py`` has ``FakeGateway``: an in-memory ``/proc``, a table of fake
  pidfds, a signal log, a manual clock and canned ``run`` results keyed by ``argv[0]``.
  Fake binaries on ``PATH`` (``tests/fakebin``) are used with the real ``SystemGateway.run``.
"""

from __future__ import annotations

import ctypes
import errno
import os
import select
import selectors
import signal
import socket
import struct
import subprocess
import sys
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import IO, Any, Protocol

# Linux syscall numbers; identical on every architecture since the unified table (5.x).
_SYS_PIDFD_SEND_SIGNAL = 424
_SYS_PIDFD_OPEN = 434

# macOS getsockopt constants for the peer PID.
_SOL_LOCAL = 0
_LOCAL_PEERPID = 0x002

SignalHandler = Callable[[int, Any], Any] | int | signal.Handlers | None

# Raised by ``run`` when its timeout expires (re-exported so callers need not import subprocess).
TimeoutExpired = subprocess.TimeoutExpired

# Re-exported stdio redirections for ``Gateway.spawn`` (callers never import subprocess).
PIPE = subprocess.PIPE
DEVNULL = subprocess.DEVNULL
#: ``SystemGateway.run`` keeps at most this many bytes of each of stdout and stderr; one more
#: byte kills the child and raises ``OutputTooLarge`` (a runaway child cannot exhaust memory).
MAX_RUN_OUTPUT = 8 * 1024 * 1024
_READ_CHUNK = 64 * 1024


class OutputTooLarge(Exception):
    """A command printed more than ``MAX_RUN_OUTPUT`` bytes on one stream; it was killed."""


@dataclass(frozen=True)
class PeerCredentials:
    """Credentials of the process on the other end of a Unix socket."""

    pid: int
    uid: int
    gid: int


def decode(data: bytes) -> str:
    """Decode subprocess output explicitly; never through the locale."""
    return data.decode("utf-8", errors="surrogateescape")


class Gateway(Protocol):
    """Everything agent-spawn needs from the operating system."""

    # subprocesses
    def run(
        self,
        argv: Sequence[str],
        *,
        stdin: str | None = None,
        timeout: float | None = None,
        env: Mapping[str, str] | None = None,
        cwd: str | None = None,
    ) -> subprocess.CompletedProcess[str]: ...

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
    ) -> subprocess.Popen[bytes]: ...

    # files, clocks, platform
    def read_file(self, path: str) -> str: ...
    def monotonic(self) -> float: ...
    def wall_time(self) -> float: ...
    def sleep(self, seconds: float) -> None: ...
    def platform(self) -> str: ...
    def getuid(self) -> int: ...
    def getpid(self) -> int: ...

    # pidfds, signals, waiting on a process that is not our child
    def pidfd_open(self, pid: int) -> int: ...
    def pidfd_send_signal(self, pidfd: int, sig: int) -> None: ...
    def kill(self, pid: int, sig: int) -> None: ...
    def getpgid(self, pid: int) -> int: ...
    def killpg(self, pgid: int, sig: int) -> None: ...
    def close(self, fd: int) -> None: ...
    def poll_readable(self, fd: int, timeout: float) -> bool: ...
    def wait_pid_exit(self, pid: int, timeout: float) -> bool: ...

    # signal handlers and the calling thread's signal mask
    def install_signal_handler(self, sig: int, handler: SignalHandler) -> SignalHandler: ...
    def pthread_sigmask(self, how: int, signals: Iterable[int]) -> set[int]: ...

    # process attributes and sockets
    def prctl(
        self, option: int, arg2: int = 0, arg3: int = 0, arg4: int = 0, arg5: int = 0
    ) -> int: ...
    def peer_credentials(self, sock: socket.socket) -> PeerCredentials: ...

    # fork primitives (double fork)
    def pipe(self) -> tuple[int, int]: ...
    def fork(self) -> int: ...
    def setsid(self) -> None: ...
    def waitpid(self, pid: int, flags: int = 0) -> tuple[int, int]: ...
    def exit_now(self, code: int) -> None: ...


_RESET_SIGNALS = tuple(
    getattr(signal, name)
    for name in ("SIGINT", "SIGQUIT", "SIGHUP", "SIGTERM")
    if hasattr(signal, name)
)


def _reset_child_signals() -> None:
    """Run in a spawned child before ``exec``: give it default signal dispositions and an empty
    blocked mask, so a supervisor that ignores or blocks signals does not pass that on (an
    ignored disposition survives ``exec``). ``subprocess`` already restores ``SIGPIPE``.

    Safe despite the usual ``preexec_fn`` warning: it only calls ``signal.signal`` and
    ``signal.pthread_sigmask`` (async-signal-safe system calls, no locks, no allocation of
    note) and the CLI is single-threaded, so no other thread can hold a lock at fork time.
    """
    for sig in _RESET_SIGNALS:
        signal.signal(sig, signal.SIG_DFL)
    signal.pthread_sigmask(signal.SIG_SETMASK, ())


INTERRUPT_GRACE = 5.0  # seconds a child gets to act on SIGTERM when its caller is interrupted


def _stop_politely(proc: subprocess.Popen[bytes]) -> None:
    """SIGTERM ``proc`` (only it), wait a bounded time, then SIGKILL it."""
    try:
        proc.terminate()
        proc.wait(timeout=INTERRUPT_GRACE)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
    except BaseException:  # a second interrupt while waiting: do not leave the child running
        proc.kill()
        proc.wait()
        raise


def _communicate(
    proc: subprocess.Popen[bytes],
    data: bytes | None,
    timeout: float | None,
    argv: Sequence[str],
) -> tuple[bytes, bytes]:
    """Feed ``data`` and collect stdout and stderr of ``proc`` with a per-stream byte cap.

    On a timeout or an overflow only ``proc`` itself is killed (never a pattern or a group) and
    the pipes are closed without waiting for EOF, so a grandchild holding them cannot block.
    """
    deadline = None if timeout is None else time.monotonic() + timeout
    chunks: dict[int, list[bytes]] = {}
    sizes: dict[int, int] = {}
    selector = selectors.DefaultSelector()
    pending = data
    assert proc.stdout is not None and proc.stderr is not None
    out_fd, err_fd = proc.stdout.fileno(), proc.stderr.fileno()
    streams = [stream for stream in (proc.stdin, proc.stdout, proc.stderr) if stream is not None]
    try:
        if proc.stdin is not None:
            if not data:
                proc.stdin.close()
            else:
                os.set_blocking(proc.stdin.fileno(), False)
                selector.register(proc.stdin, selectors.EVENT_WRITE, "in")
        for name, stream in (("out", proc.stdout), ("err", proc.stderr)):
            assert stream is not None
            os.set_blocking(stream.fileno(), False)
            chunks[stream.fileno()] = []
            sizes[stream.fileno()] = 0
            selector.register(stream, selectors.EVENT_READ, name)
        while selector.get_map():
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                raise subprocess.TimeoutExpired(list(argv), timeout or 0.0)
            for key, _ in selector.select(remaining):
                fd = key.fd
                if key.data == "in":
                    assert pending is not None and proc.stdin is not None
                    try:
                        written = os.write(fd, pending[:_READ_CHUNK])
                        pending = pending[written:]
                    except BlockingIOError:
                        continue
                    except OSError:  # the child closed its stdin: stop feeding it
                        pending = b""
                    if not pending:
                        selector.unregister(proc.stdin)
                        proc.stdin.close()
                    continue
                try:
                    block = os.read(fd, _READ_CHUNK)
                except BlockingIOError:
                    continue
                if not block:
                    selector.unregister(key.fileobj)
                    continue
                sizes[fd] += len(block)
                if sizes[fd] > MAX_RUN_OUTPUT:
                    raise OutputTooLarge(f"a command printed more than {MAX_RUN_OUTPUT} bytes")
                chunks[fd].append(block)
        remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
        proc.wait(timeout=remaining)
    except (subprocess.TimeoutExpired, OutputTooLarge):
        proc.kill()
        proc.wait()
        raise
    except Exception:
        # An ordinary failure (e.g. OSError, MemoryError): stop the child at once.
        proc.kill()
        proc.wait()
        raise
    except BaseException:
        # An interrupt (a signal handler raised): ask the exact child to stop, then force it.
        _stop_politely(proc)
        raise
    finally:
        selector.close()
        for stream in streams:
            try:
                stream.close()
            except OSError:
                pass
    return b"".join(chunks[out_fd]), b"".join(chunks[err_fd])


class SystemGateway:
    """The real implementation.

    ``force_syscall`` makes the pidfd calls use the ``ctypes`` ``syscall`` fallback even
    where ``os.pidfd_open`` exists, so the fallback is exercised on every build.
    """

    def __init__(self, *, force_syscall: bool = False) -> None:
        self._force_syscall = force_syscall
        self._libc: ctypes.CDLL | None = None

    def _c(self) -> ctypes.CDLL:
        if self._libc is None:
            self._libc = ctypes.CDLL(None, use_errno=True)
        return self._libc

    # subprocesses
    def run(
        self,
        argv: Sequence[str],
        *,
        stdin: str | None = None,
        timeout: float | None = None,
        env: Mapping[str, str] | None = None,
        cwd: str | None = None,
    ) -> subprocess.CompletedProcess[str]:
        data = None if stdin is None else stdin.encode("utf-8", errors="surrogateescape")
        # Without input the child gets /dev/null, never this process's own stdin.
        proc = subprocess.Popen(
            list(argv),
            stdin=subprocess.DEVNULL if data is None else subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=None if env is None else dict(env),
            cwd=cwd,
        )
        out, err = _communicate(proc, data, timeout, argv)
        return subprocess.CompletedProcess(list(argv), proc.returncode, decode(out), decode(err))

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
        return subprocess.Popen(
            list(argv),
            stdin=stdin,
            stdout=stdout,
            stderr=stderr,
            env=None if env is None else dict(env),
            cwd=cwd,
            start_new_session=start_new_session,
            preexec_fn=_reset_child_signals,
        )

    # files, clocks, platform
    def read_file(self, path: str) -> str:
        with open(path, "rb") as handle:
            return decode(handle.read())

    def monotonic(self) -> float:
        return time.monotonic()

    def wall_time(self) -> float:
        return time.time()

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)

    def platform(self) -> str:
        return sys.platform

    def getuid(self) -> int:
        return os.getuid()

    def getpid(self) -> int:
        return os.getpid()

    # pidfds and signals
    def pidfd_open(self, pid: int) -> int:
        if not self._force_syscall and hasattr(os, "pidfd_open"):
            return os.pidfd_open(pid)
        libc = self._c()
        result = libc.syscall(ctypes.c_long(_SYS_PIDFD_OPEN), ctypes.c_int(pid), ctypes.c_uint(0))
        if result < 0:
            code = ctypes.get_errno()
            raise OSError(code, os.strerror(code))
        return int(result)

    def pidfd_send_signal(self, pidfd: int, sig: int) -> None:
        if not self._force_syscall and hasattr(signal, "pidfd_send_signal"):
            signal.pidfd_send_signal(pidfd, sig)
            return
        libc = self._c()
        result = libc.syscall(
            ctypes.c_long(_SYS_PIDFD_SEND_SIGNAL),
            ctypes.c_int(pidfd),
            ctypes.c_int(sig),
            ctypes.c_void_p(None),
            ctypes.c_uint(0),
        )
        if result < 0:
            code = ctypes.get_errno()
            raise OSError(code, os.strerror(code))

    def kill(self, pid: int, sig: int) -> None:
        os.kill(pid, sig)

    def getpgid(self, pid: int) -> int:
        return os.getpgid(pid)

    def killpg(self, pgid: int, sig: int) -> None:
        """Signal a process group this tool created (hooks); callers verify it first."""
        os.killpg(pgid, sig)

    def close(self, fd: int) -> None:
        os.close(fd)

    def poll_readable(self, fd: int, timeout: float) -> bool:
        poller = select.poll()
        poller.register(fd, select.POLLIN | select.POLLHUP | select.POLLERR)
        return bool(poller.poll(max(0.0, timeout) * 1000.0))

    def wait_pid_exit(self, pid: int, timeout: float) -> bool:
        """Wait for a non-child to exit with kqueue ``EVFILT_PROC``/``NOTE_EXIT`` (macOS)."""
        kqueue = getattr(select, "kqueue", None)
        kevent = getattr(select, "kevent", None)
        if kqueue is None or kevent is None:
            raise NotImplementedError("kqueue is not available on this platform")
        queue = kqueue()
        try:
            event = kevent(
                pid,
                filter=select.KQ_FILTER_PROC,  # pyright: ignore[reportAttributeAccessIssue]
                flags=select.KQ_EV_ADD | select.KQ_EV_ONESHOT,  # pyright: ignore[reportAttributeAccessIssue]
                fflags=select.KQ_NOTE_EXIT,  # pyright: ignore[reportAttributeAccessIssue]
            )
            try:
                return bool(queue.control([event], 1, max(0.0, timeout)))
            except ProcessLookupError:
                return True
        finally:
            queue.close()

    def install_signal_handler(self, sig: int, handler: SignalHandler) -> SignalHandler:
        return signal.signal(sig, handler)

    def pthread_sigmask(self, how: int, signals: Iterable[int]) -> set[int]:
        """Change the calling thread's signal mask; return the previous one."""
        return {int(sig) for sig in signal.pthread_sigmask(how, signals)}

    # process attributes and sockets
    def prctl(self, option: int, arg2: int = 0, arg3: int = 0, arg4: int = 0, arg5: int = 0) -> int:
        if not sys.platform.startswith("linux"):
            raise NotImplementedError("prctl exists only on Linux")
        libc = self._c()
        result = libc.prctl(
            ctypes.c_int(option),
            ctypes.c_ulong(arg2),
            ctypes.c_ulong(arg3),
            ctypes.c_ulong(arg4),
            ctypes.c_ulong(arg5),
        )
        if result < 0:
            code = ctypes.get_errno()
            raise OSError(code, os.strerror(code))
        return int(result)

    def peer_credentials(self, sock: socket.socket) -> PeerCredentials:
        if sys.platform.startswith("linux"):
            raw = sock.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("3i"))
            pid, uid, gid = struct.unpack("3i", raw)
            return PeerCredentials(pid=pid, uid=uid, gid=gid)
        if sys.platform == "darwin":
            raw = sock.getsockopt(_SOL_LOCAL, _LOCAL_PEERPID, struct.calcsize("i"))
            (pid,) = struct.unpack("i", raw)
            uid = ctypes.c_uint32()
            gid = ctypes.c_uint32()
            libc = self._c()
            if libc.getpeereid(sock.fileno(), ctypes.byref(uid), ctypes.byref(gid)) != 0:
                code = ctypes.get_errno()
                raise OSError(code or errno.EINVAL, os.strerror(code or errno.EINVAL))
            return PeerCredentials(pid=pid, uid=uid.value, gid=gid.value)
        raise NotImplementedError(f"peer credentials are not supported on {sys.platform}")

    # fork primitives
    def pipe(self) -> tuple[int, int]:
        return os.pipe()

    def fork(self) -> int:
        return os.fork()

    def setsid(self) -> None:
        os.setsid()

    def waitpid(self, pid: int, flags: int = 0) -> tuple[int, int]:
        return os.waitpid(pid, flags)

    def exit_now(self, code: int) -> None:
        os._exit(code)


_default: Gateway = SystemGateway()


def get_gateway() -> Gateway:
    """Return the current default gateway."""
    return _default


def set_gateway(gateway: Gateway) -> Gateway:
    """Install ``gateway`` as the default and return the previous one (for restoring)."""
    global _default
    previous = _default
    _default = gateway
    return previous


def reset_sigchld() -> None:
    """Reset an inherited ignored SIGCHLD to the default, once at CLI entry.

    With SIGCHLD ignored the kernel reaps children itself, so ``subprocess`` cannot read
    their exit status and reports a failing child as exit 0.
    """
    get_gateway().install_signal_handler(signal.SIGCHLD, signal.SIG_DFL)


def run(
    argv: list[str],
    *,
    stdin: str | None = None,
    timeout: float | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run ``argv`` without a shell, capturing text stdout and stderr."""
    return get_gateway().run(argv, stdin=stdin, timeout=timeout)
