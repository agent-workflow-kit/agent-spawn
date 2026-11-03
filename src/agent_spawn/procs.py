"""Process identity, pidfds, signal forwarding and the detached double fork.

Normative sources: interface.md "Signals" and "Liveness", design.md spikes C1-C6.

* A process is identified by ``ProcessIdentity``: PID, start time and boot id. A PID alone
  is never trusted: a reused PID has another start time, a reboot has another boot id.
* A zombie counts as dead.
* On Linux a signal is sent only through a pidfd that was opened *before* the identity was
  read and verified (``open_verified``), so the pidfd is bound to the verified process and
  a PID reused later cannot receive the signal. If the identity does not match, nothing is
  sent. Without a pidfd (macOS) the handle verifies the identity immediately before
  ``kill``; that leaves a tiny window the platform cannot close (see interface.md
  "Platform notes").
* Every OS access goes through the gateway (``gateway.py``), so unit tests use fakes.
"""

from __future__ import annotations

import errno
import json
import os
import signal
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import TracebackType
from typing import Literal

from agent_spawn.gateway import Gateway, SignalHandler, get_gateway

LINUX_BOOT_ID_PATH = "/proc/sys/kernel/random/boot_id"
_DEAD_STATES = frozenset({"Z", "X", "x"})
_ACK = b"A"
_MAX_FD = 65536


class ProcessIdentityError(Exception):
    """A process could not be identified reliably (never treated as 'dead')."""


class DetachedForkError(Exception):
    """The double fork failed: a pipe failed or closed before the acknowledgement."""


@dataclass(frozen=True)
class ProcessIdentity:
    """PID plus start time plus boot id: the exact identity recorded for a process."""

    pid: int
    start_time: int
    boot_id: str

    def to_dict(self) -> dict[str, int | str]:
        return {"pid": self.pid, "start_time": self.start_time, "boot_id": self.boot_id}

    @staticmethod
    def from_dict(data: Mapping[str, object]) -> ProcessIdentity:
        pid, start, boot = data.get("pid"), data.get("start_time"), data.get("boot_id")
        if (
            not isinstance(pid, int)
            or isinstance(pid, bool)
            or not isinstance(start, int)
            or isinstance(start, bool)
            or not isinstance(boot, str)
            or pid <= 0
        ):
            raise ValueError(f"malformed process identity: {dict(data)!r}")
        return ProcessIdentity(pid=pid, start_time=start, boot_id=boot)


@dataclass(frozen=True)
class StatInfo:
    pid: int
    comm: str
    state: str
    ppid: int
    start_time: int


@dataclass(frozen=True)
class ProcInfo:
    """One reading of a live-or-zombie process."""

    identity: ProcessIdentity
    state: str

    @property
    def alive(self) -> bool:
        return self.state not in _DEAD_STATES


def _is_uint(text: str) -> bool:
    return text.isascii() and text.isdigit()


def parse_stat(text: str) -> StatInfo:
    """Parse ``/proc/<pid>/stat``.

    ``comm`` is in parentheses and may contain spaces, ``(`` and ``)``; the fields after it
    never contain ``)``, so the *last* ``)`` ends it. ``starttime`` is field 22.
    """
    open_at = text.find(" (")
    close_at = text.rfind(")")
    if open_at <= 0 or close_at < open_at + 2:
        raise ProcessIdentityError(f"malformed stat line: {text[:80]!r}")
    pid_text = text[:open_at]
    rest_text = text[close_at + 1 :]
    if not _is_uint(pid_text) or not rest_text.startswith(" "):
        raise ProcessIdentityError(f"malformed stat line: {text[:80]!r}")
    rest = rest_text.split()
    # rest[0] is field 3 (state); field 22 (starttime) is rest[19].
    if len(rest) < 20 or len(rest[0]) != 1 or not _is_uint(rest[1]) or not _is_uint(rest[19]):
        raise ProcessIdentityError(f"malformed stat line: {text[:80]!r}")
    return StatInfo(
        pid=int(pid_text),
        comm=text[open_at + 2 : close_at],
        state=rest[0],
        ppid=int(rest[1]),
        start_time=int(rest[19]),
    )


def _gw(gateway: Gateway | None) -> Gateway:
    return gateway if gateway is not None else get_gateway()


def read_boot_id(*, gateway: Gateway | None = None) -> str:
    gw = _gw(gateway)
    if gw.platform().startswith("linux"):
        try:
            boot = gw.read_file(LINUX_BOOT_ID_PATH).strip()
        except OSError as exc:
            raise ProcessIdentityError(f"cannot read the boot id: {exc}") from exc
        if not boot:
            raise ProcessIdentityError("empty boot id")
        return boot
    done = gw.run(
        ["sysctl", "-n", "kern.boottime"],
        env={"LC_ALL": "C", "PATH": "/usr/sbin:/sbin:/usr/bin:/bin"},
    )
    boot = done.stdout.strip()
    if done.returncode != 0 or not boot:
        raise ProcessIdentityError("cannot read the boot time")
    return boot


def _inspect_darwin(pid: int, boot_id: str, gw: Gateway) -> ProcInfo | None:
    done = gw.run(
        ["ps", "-o", "lstart=,stat=", "-p", str(pid)],
        env={"LC_ALL": "C", "PATH": "/usr/bin:/bin"},
    )
    out = done.stdout.strip()
    if done.returncode != 0 or not out:
        return None
    parts = out.rsplit(None, 1)
    if len(parts) != 2:
        raise ProcessIdentityError(f"malformed ps output: {out!r}")
    started, stat = parts
    try:
        epoch = int(time.mktime(time.strptime(started, "%a %b %d %H:%M:%S %Y")))
    except ValueError as exc:
        raise ProcessIdentityError(f"malformed ps start time: {started!r}") from exc
    return ProcInfo(ProcessIdentity(pid, epoch, boot_id), stat[:1])


def inspect(pid: int, *, gateway: Gateway | None = None) -> ProcInfo | None:
    """Read the identity and state of ``pid``; ``None`` when no such process exists.

    A process that exists but cannot be parsed raises ``ProcessIdentityError``: an
    unreadable process is never silently treated as dead.
    """
    gw = _gw(gateway)
    if pid <= 0:
        return None
    boot_id = read_boot_id(gateway=gw)
    if not gw.platform().startswith("linux"):
        return _inspect_darwin(pid, boot_id, gw)
    try:
        text = gw.read_file(f"/proc/{pid}/stat")
    except (FileNotFoundError, ProcessLookupError):
        return None
    except OSError as exc:
        if exc.errno in (errno.ENOENT, errno.ESRCH):
            return None
        raise ProcessIdentityError(f"cannot read stat of {pid}: {exc}") from exc
    stat = parse_stat(text)
    if stat.pid != pid:
        raise ProcessIdentityError(f"stat of {pid} names pid {stat.pid}")
    return ProcInfo(ProcessIdentity(pid, stat.start_time, boot_id), stat.state)


def read_identity(pid: int, *, gateway: Gateway | None = None) -> ProcessIdentity | None:
    info = inspect(pid, gateway=gateway)
    return None if info is None else info.identity


def is_alive(identity: ProcessIdentity, *, gateway: Gateway | None = None) -> bool:
    """True only for the very process recorded: same PID, start time and boot id, not a zombie."""
    info = inspect(identity.pid, gateway=gateway)
    return info is not None and info.identity == identity and info.alive


class ProcHandle:
    """A verified handle on one process (a pidfd on Linux)."""

    def __init__(self, identity: ProcessIdentity, pidfd: int | None, gateway: Gateway) -> None:
        self.identity = identity
        self._fd = pidfd
        self._gw = gateway

    def __enter__(self) -> ProcHandle:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        if self._fd is not None:
            fd, self._fd = self._fd, None
            try:
                self._gw.close(fd)
            except OSError:
                pass

    def is_alive(self) -> bool:
        if self._fd is not None and self._gw.poll_readable(self._fd, 0):
            return False  # the pidfd is readable once the process has exited
        return is_alive(self.identity, gateway=self._gw)

    def send(self, sig: int) -> bool:
        """Send ``sig`` to the verified process. False when it is already gone."""
        if not self.is_alive():
            return False
        try:
            if self._fd is not None:
                self._gw.pidfd_send_signal(self._fd, sig)
            else:
                # No pidfd (macOS): verify immediately before the kill.
                if not is_alive(self.identity, gateway=self._gw):
                    return False
                self._gw.kill(self.identity.pid, sig)
        except ProcessLookupError:
            return False
        return True

    def wait_exit(self, timeout: float) -> bool:
        """Wait until the process is dead (exited or zombie). True when it is dead."""
        gw = self._gw
        deadline = gw.monotonic() + timeout
        while True:
            remaining = deadline - gw.monotonic()
            if self._fd is not None:
                gw.poll_readable(self._fd, min(max(remaining, 0.0), 0.05))
            elif gw.platform() == "darwin":
                gw.wait_pid_exit(self.identity.pid, min(max(remaining, 0.0), 0.05))
            else:
                gw.sleep(min(max(remaining, 0.0), 0.02))
            if not self.is_alive():
                return True
            if remaining <= 0:
                return False


def _open_fd(pid: int, gw: Gateway) -> int | None:
    """Open a pidfd for ``pid``; ``None`` when the process does not exist."""
    try:
        return gw.pidfd_open(pid)
    except ProcessLookupError:
        return None
    except OSError as exc:
        if exc.errno == errno.ESRCH:
            return None
        raise


def _uses_pidfd(gw: Gateway) -> bool:
    return gw.platform().startswith("linux")


def open_verified(
    identity: ProcessIdentity, *, gateway: Gateway | None = None
) -> ProcHandle | None:
    """A handle on exactly ``identity``, or ``None`` if it is gone, reused or a zombie.

    The pidfd is opened first and the identity read afterwards: the pidfd is bound to
    whichever process held the PID at open time, so if that was not the recorded process
    the start time differs and nothing is returned.
    """
    gw = _gw(gateway)
    fd: int | None = None
    if _uses_pidfd(gw):
        fd = _open_fd(identity.pid, gw)
        if fd is None:
            return None
    try:
        info = inspect(identity.pid, gateway=gw)
        if info is None or info.identity != identity or not info.alive:
            if fd is not None:
                gw.close(fd)
            return None
    except BaseException:
        if fd is not None:
            gw.close(fd)
        raise
    return ProcHandle(identity, fd, gw)


def identify(pid: int, *, gateway: Gateway | None = None) -> ProcHandle | None:
    """Obtain the identity of ``pid`` together with a handle bound to it (pidfd first).

    For processes this tool just started: the caller records ``handle.identity``.
    """
    gw = _gw(gateway)
    fd: int | None = None
    if _uses_pidfd(gw):
        fd = _open_fd(pid, gw)
        if fd is None:
            return None
    try:
        info = inspect(pid, gateway=gw)
        if info is None or not info.alive:
            if fd is not None:
                gw.close(fd)
            return None
    except BaseException:
        if fd is not None:
            gw.close(fd)
        raise
    return ProcHandle(info.identity, fd, gw)


SignalOutcomeReason = Literal["sent", "gone"]


@dataclass(frozen=True)
class SignalOutcome:
    pid: int
    signal: int
    reason: SignalOutcomeReason

    @property
    def sent(self) -> bool:
        return self.reason == "sent"


def forward_signal(
    identity: ProcessIdentity, sig: int, *, gateway: Gateway | None = None
) -> SignalOutcome:
    """Send ``sig`` once to exactly ``identity``; SIGCONT follows every SIGTERM.

    A reused PID, a different boot and a zombie all read as ``gone``: nothing is sent.
    """
    handle = open_verified(identity, gateway=gateway)
    if handle is None:
        return SignalOutcome(identity.pid, sig, "gone")
    with handle:
        if not handle.send(sig):
            return SignalOutcome(identity.pid, sig, "gone")
        if sig == signal.SIGTERM:
            handle.send(signal.SIGCONT)
    return SignalOutcome(identity.pid, sig, "sent")


class SignalForwarder:
    """Forwards each distinct signal at most once to one recorded process (idempotent)."""

    def __init__(self, identity: ProcessIdentity, *, gateway: Gateway | None = None) -> None:
        self.identity = identity
        self._gateway = gateway
        self._forwarded: set[int] = set()

    def forward(self, sig: int) -> SignalOutcome | None:
        """Forward ``sig``; ``None`` when this signal was already forwarded."""
        if sig in self._forwarded:
            return None
        self._forwarded.add(sig)
        return forward_signal(self.identity, sig, gateway=self._gateway)


def install_signal_handlers(
    handlers: Mapping[int, SignalHandler], *, gateway: Gateway | None = None
) -> Callable[[], None]:
    """Install ``handlers`` (``signal.SIG_IGN``, ``SIG_DFL`` or a callable); return a restorer."""
    gw = _gw(gateway)
    previous: dict[int, SignalHandler] = {}
    for sig, handler in handlers.items():
        previous[sig] = gw.install_signal_handler(sig, handler)

    def restore() -> None:
        for sig, old in previous.items():
            gw.install_signal_handler(sig, old)

    return restore


@dataclass
class DetachedStart:
    """The launcher's side of a detached start, waiting for its acknowledgement."""

    identity: ProcessIdentity
    _ack_fd: int | None
    _gw: Gateway

    def acknowledge(self) -> None:
        """Tell the grandchild the identity is recorded: it closes descriptors and execs."""
        fd, self._ack_fd = self._ack_fd, None
        if fd is None:
            raise DetachedForkError("already acknowledged or aborted")
        try:
            os.write(fd, _ACK)
        except OSError as exc:
            raise DetachedForkError(f"acknowledgement failed: {exc}") from exc
        finally:
            self._gw.close(fd)

    def abort(self) -> None:
        """Close the acknowledgement pipe without acknowledging: the grandchild exits."""
        fd, self._ack_fd = self._ack_fd, None
        if fd is not None:
            self._gw.close(fd)


_RESET_SIGNALS = ("SIGPIPE", "SIGXFSZ", "SIGINT", "SIGQUIT", "SIGHUP", "SIGTERM")


def _grandchild(
    gw: Gateway,
    argv: Sequence[str],
    env: Mapping[str, str] | None,
    log_fd: int | None,
    id_w: int,
    ack_r: int,
    ack_timeout: float,
) -> None:
    """Runs in the grandchild; never returns."""
    keep = {id_w, ack_r}
    # Drop every inherited descriptor but the two pipe ends right away: the caller's pipes
    # must not stay open (spike C3).
    lows = sorted(fd for fd in keep | ({log_fd} if log_fd is not None else set()))
    start = 3
    for fd in lows + [_MAX_FD]:
        if fd > start:
            os.closerange(start, fd)
        start = max(start, fd + 1)
    devnull = os.open(os.devnull, os.O_RDWR)
    os.dup2(devnull, 0)
    if log_fd is not None:
        os.dup2(log_fd, 1)
        os.dup2(log_fd, 2)
    else:
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
    for fd in (devnull, log_fd):
        if fd is not None and fd > 2:
            os.close(fd)
    for name in _RESET_SIGNALS:
        gw.install_signal_handler(getattr(signal, name), signal.SIG_DFL)
    signal.pthread_sigmask(signal.SIG_SETMASK, [])
    info = inspect(gw.getpid(), gateway=gw)
    if info is None:
        gw.exit_now(70)
        return
    os.write(id_w, json.dumps(info.identity.to_dict()).encode("ascii") + b"\n")
    os.close(id_w)
    if not gw.poll_readable(ack_r, ack_timeout) or os.read(ack_r, 1) != _ACK:
        gw.exit_now(1)
        return
    os.closerange(3, _MAX_FD)
    try:
        os.execve(argv[0], list(argv), dict(os.environ) if env is None else dict(env))
    except OSError as exc:
        os.write(2, f"agent-spawn: exec of {argv[0]!r} failed: {exc}\n".encode("utf-8", "replace"))
        gw.exit_now(127)


def _intermediate(
    gw: Gateway,
    argv: Sequence[str],
    env: Mapping[str, str] | None,
    log_fd: int | None,
    id_r: int,
    id_w: int,
    ack_r: int,
    ack_w: int,
    ack_timeout: float,
) -> None:
    """Runs in the first child; never returns."""
    code = 70
    try:
        os.close(id_r)
        os.close(ack_w)
        gw.setsid()
        if gw.fork() == 0:  # the grandchild is not a session leader: no controlling tty ever
            try:
                _grandchild(gw, argv, env, log_fd, id_w, ack_r, ack_timeout)
            finally:
                gw.exit_now(70)
        code = 0
    finally:
        gw.exit_now(code)


def _read_line(gw: Gateway, fd: int, timeout: float) -> bytes:
    deadline = gw.monotonic() + timeout
    data = b""
    while b"\n" not in data:
        remaining = deadline - gw.monotonic()
        if remaining <= 0 or not gw.poll_readable(fd, remaining):
            raise DetachedForkError("timed out waiting for the detached process to report")
        chunk = os.read(fd, 4096)
        if not chunk:
            raise DetachedForkError("the detached process closed the pipe before reporting")
        data += chunk
        if len(data) > 4096:
            raise DetachedForkError("oversized report from the detached process")
    return data


def spawn_detached(
    argv: Sequence[str],
    *,
    log_path: str | None = None,
    env: Mapping[str, str] | None = None,
    timeout: float = 10.0,
    ack_timeout: float = 60.0,
    gateway: Gateway | None = None,
) -> DetachedStart:
    """Double fork a detached process (``setsid``; stdin ``/dev/null``; stdout and stderr to
    ``log_path`` or ``/dev/null``) that exec's ``argv`` after the launcher acknowledges.

    The grandchild reports its PID, start time and boot id over a pipe and waits for
    ``DetachedStart.acknowledge()``; only then does it close every inherited descriptor
    except stdio and exec ``argv`` (same PID and start time). A pipe failure, or a pipe
    closed before the acknowledgement, is a failed detached fork (``DetachedForkError``)
    and no exec happens. ``argv[0]`` must be an absolute path.
    """
    gw = _gw(gateway)
    if not argv or not os.path.isabs(argv[0]):
        raise ValueError("argv[0] must be an absolute path")
    log_fd: int | None = None
    if log_path is not None:
        log_fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_CLOEXEC, 0o600)
    id_r, id_w = gw.pipe()
    ack_r, ack_w = gw.pipe()
    try:
        pid = gw.fork()
        if pid == 0:
            try:
                _intermediate(gw, argv, env, log_fd, id_r, id_w, ack_r, ack_w, ack_timeout)
            finally:
                gw.exit_now(70)
        # The launcher: keep only the identity read end and the acknowledgement write end.
        gw.close(id_w)
        gw.close(ack_r)
        id_w = ack_r = -1
        gw.waitpid(pid)
        try:
            payload = _read_line(gw, id_r, timeout)
            identity = ProcessIdentity.from_dict(json.loads(payload))
        except (DetachedForkError, ValueError):
            gw.close(ack_w)
            ack_w = -1
            raise
        started = DetachedStart(identity, ack_w, gw)
        ack_w = -1
        return started
    finally:
        for fd in (id_r, id_w, ack_r, ack_w):
            if fd >= 0:
                try:
                    gw.close(fd)
                except OSError:
                    pass
        if log_fd is not None:
            os.close(log_fd)


def wait_for_exit(
    identity: ProcessIdentity, timeout: float, *, gateway: Gateway | None = None
) -> bool:
    """Wait for a non-child process to die (pidfd; kqueue on macOS). True when it is dead.

    An identity that is already dead, reused or a zombie returns True at once.
    """
    handle = open_verified(identity, gateway=gateway)
    if handle is None:
        return True
    with handle:
        return handle.wait_exit(timeout)
