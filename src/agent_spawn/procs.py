"""Process identity, pidfds, signal forwarding and the detached double fork.

Normative sources: interface.md "Signals" and "Liveness", design.md spikes C1-C6.

* A process is identified by ``ProcessIdentity``: PID, start time and boot id. A PID alone
  is never trusted: a reused PID has another start time, a reboot has another boot id.
* A zombie counts as dead once every thread of it exited (a leader that exited while
  another thread runs is alive).
* On Linux a signal is sent only through a pidfd that was opened *before* the identity was
  read and verified (``open_verified``), so the pidfd is bound to the verified process and
  a PID reused later cannot receive the signal. If the identity does not match, nothing is
  sent. A recorded PID that now names a non-leader thread reads as gone. Without a pidfd
  (macOS) the handle verifies the identity immediately before ``kill``; a PID reused in that
  window could receive the signal, a residual risk the platform cannot close (interface.md
  "Platform notes", macOS).
* Every OS access goes through the gateway (``gateway.py``), so unit tests use fakes.
"""

from __future__ import annotations

import calendar
import errno
import fcntl
import json
import os
import re
import resource
import signal
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import TracebackType
from typing import Literal

from agent_spawn.gateway import Gateway, SignalHandler, TimeoutExpired, get_gateway

LINUX_BOOT_ID_PATH = "/proc/sys/kernel/random/boot_id"
_DEAD_STATES = frozenset({"Z", "X", "x"})
_ACK = b"A"
_PID_MAX = 2**31 - 1
_FD_CAP = 2**31 - 1
_TOOL_TIMEOUT = 10.0  # seconds for each ps or sysctl run (macOS)
_BOOTTIME = re.compile(r"\{\s*sec\s*=\s*(\d+)\s*,")


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
            or not 0 < pid <= _PID_MAX
            or start < 0
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
    threads_left: bool = False  # a zombie leader whose thread group still has live threads

    @property
    def alive(self) -> bool:
        if self.state == "Z":
            return self.threads_left
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


def _run_tool(gw: Gateway, argv: list[str], env: dict[str, str]) -> str | None:
    """Run a macOS identity tool with a timeout; its stdout, or ``None`` when it failed."""
    try:
        done = gw.run(argv, env=env, timeout=_TOOL_TIMEOUT)
    except TimeoutExpired as exc:
        raise ProcessIdentityError(f"{argv[0]} timed out") from exc
    except OSError as exc:
        raise ProcessIdentityError(f"cannot run {argv[0]}: {exc}") from exc
    out = done.stdout.strip()
    return out if done.returncode == 0 and out else None


def read_boot_id(*, gateway: Gateway | None = None) -> str:
    """Linux: ``/proc/sys/kernel/random/boot_id``. macOS: the ``sec`` value of
    ``kern.boottime`` (the rest of that line is a local-time date and is ignored)."""
    gw = _gw(gateway)
    if gw.platform().startswith("linux"):
        try:
            boot = gw.read_file(LINUX_BOOT_ID_PATH).strip()
        except OSError as exc:
            raise ProcessIdentityError(f"cannot read the boot id: {exc}") from exc
        if not boot:
            raise ProcessIdentityError("empty boot id")
        return boot
    out = _run_tool(
        gw,
        ["sysctl", "-n", "kern.boottime"],
        {"LC_ALL": "C", "PATH": "/usr/sbin:/sbin:/usr/bin:/bin"},
    )
    match = _BOOTTIME.match(out or "")
    if match is None:
        raise ProcessIdentityError(f"cannot read the boot time: {out!r}")
    return match.group(1)


def _inspect_darwin(pid: int, boot_id: str, gw: Gateway) -> ProcInfo | None:
    # TZ=UTC0: lstart is printed in the zone of ps, so parse it as UTC whatever ours is.
    out = _run_tool(
        gw,
        ["ps", "-o", "lstart=,stat=", "-p", str(pid)],
        {"LC_ALL": "C", "TZ": "UTC0", "PATH": "/usr/bin:/bin"},
    )
    if out is None:
        # ps also fails for reasons other than a missing process: confirm the absence.
        try:
            gw.kill(pid, 0)
        except ProcessLookupError:
            return None
        except PermissionError:
            pass  # exists, owned by someone else
        except OSError as exc:
            raise ProcessIdentityError(f"cannot check process {pid}: {exc}") from exc
        raise ProcessIdentityError(f"ps cannot read process {pid}, which exists")
    parts = out.rsplit(None, 1)
    if len(parts) != 2:
        raise ProcessIdentityError(f"malformed ps output: {out!r}")
    started, stat = parts
    try:
        epoch = calendar.timegm(time.strptime(started, "%a %b %d %H:%M:%S %Y"))
    except ValueError as exc:
        raise ProcessIdentityError(f"malformed ps start time: {started!r}") from exc
    return ProcInfo(ProcessIdentity(pid, epoch, boot_id), stat[:1])


def _read_status(pid: int, gw: Gateway) -> dict[str, str] | None:
    """``/proc/<pid>/status`` as a mapping; ``None`` when the entry is gone."""
    try:
        text = gw.read_file(f"/proc/{pid}/status")
    except (FileNotFoundError, ProcessLookupError):
        return None
    except OSError as exc:
        if exc.errno in (errno.ENOENT, errno.ESRCH):
            return None
        raise
    fields: dict[str, str] = {}
    for line in text.splitlines():
        key, sep, value = line.partition(":")
        if sep:
            fields[key] = value.strip()
    return fields


def _zombie_has_threads(pid: int, gw: Gateway) -> bool:
    """A ``Z`` leader still counts as alive while other threads of its group run."""
    try:
        status = _read_status(pid, gw)
    except OSError:
        return False  # unreadable: the entry is going away
    if status is None:
        return False
    threads = status.get("Threads", "")
    if not _is_uint(threads):
        raise ProcessIdentityError(f"malformed Threads in the status of {pid}: {threads!r}")
    return int(threads) > 1


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
    threads_left = stat.state == "Z" and _zombie_has_threads(pid, gw)
    return ProcInfo(ProcessIdentity(pid, stat.start_time, boot_id), stat.state, threads_left)


def read_identity(pid: int, *, gateway: Gateway | None = None) -> ProcessIdentity | None:
    info = inspect(pid, gateway=gateway)
    return None if info is None else info.identity


def is_alive(identity: ProcessIdentity, *, gateway: Gateway | None = None) -> bool:
    """True only for the very process recorded: same PID, start time and boot id, and not a
    zombie whose every thread exited."""
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
            return False  # the pidfd is readable once the whole thread group has exited
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
    """Open a pidfd for ``pid``; ``None`` when no such process (thread-group leader) exists.

    ``EINVAL`` with a valid PID means the PID now names a thread that is not a leader (a
    reused PID): that reads as gone. Every other failure propagates, and nothing is sent.
    """
    if not 0 < pid <= _PID_MAX:
        return None
    try:
        return gw.pidfd_open(pid)
    except ProcessLookupError:
        return None
    except OSError as exc:
        if exc.errno == errno.ESRCH:
            return None
        if exc.errno == errno.EINVAL:
            status = _read_status(pid, gw)
            if status is None or status.get("Tgid") != str(pid):
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
        outcome = forward_signal(self.identity, sig, gateway=self._gateway)
        self._forwarded.add(sig)  # only a completed forward (sent or gone) counts
        return outcome


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


def _fd_limit() -> int:
    """One past the highest descriptor number that can be open (for ``closerange``)."""
    _soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if hard == resource.RLIM_INFINITY:
        hard = os.sysconf("SC_OPEN_MAX")
    return max(3, min(int(hard), _FD_CAP))


def _close_fds_except(keep: frozenset[int], limit: int) -> None:
    """Child side: close every descriptor >= 3 not in ``keep``, whatever its number.

    On Linux the open descriptors are listed from ``/proc/self/fd`` (some Python builds lack
    ``close_range``, and closing up to a hard limit of 2**20 one by one takes seconds);
    elsewhere, or without ``/proc``, ``closerange`` runs up to ``limit``.
    """
    try:
        names = os.listdir("/proc/self/fd") if sys.platform.startswith("linux") else None
    except OSError:
        names = None
    if names is not None:
        for name in names:
            fd = int(name)
            if fd >= 3 and fd not in keep:
                try:
                    os.close(fd)
                except OSError:
                    pass  # the listing's own handle, already closed
        return
    start = 3
    for fd in sorted(keep) + [limit]:
        if fd > start:
            os.closerange(start, fd)
        start = max(start, fd + 1)


def _above_stdio(fd: int) -> int:
    """Child side: a close-on-exec copy of ``fd`` numbered 3 or more (``fd`` itself if it is)."""
    return fd if fd > 2 else fcntl.fcntl(fd, fcntl.F_DUPFD_CLOEXEC, 3)


def _reset_dispositions(gw: Gateway) -> None:
    """Child side: every catchable signal back to ``SIG_DFL`` (ignores survive exec)."""
    for sig in signal.valid_signals():
        if sig in (signal.SIGKILL, signal.SIGSTOP):
            continue
        try:
            gw.install_signal_handler(sig, signal.SIG_DFL)
        except (OSError, ValueError, RuntimeError):
            pass  # a signal the platform does not let us set


def _grandchild(
    gw: Gateway,
    argv: Sequence[str],
    env: Mapping[str, str] | None,
    log_fd: int | None,
    id_w: int,
    ack_r: int,
    ack_timeout: float,
    fd_limit: int,
) -> None:
    """Runs in the grandchild with every signal blocked; never returns."""
    # Move the descriptors we keep above stdio first: with the caller's 0-2 closed, the log
    # or a pipe end can be 0, 1 or 2 and would be overwritten by the redirections below.
    id_w = _above_stdio(id_w)
    ack_r = _above_stdio(ack_r)
    if log_fd is not None:
        log_fd = _above_stdio(log_fd)
    # Drop every other inherited descriptor right away: the caller's pipes must not stay
    # open (spike C3).
    kept = {id_w, ack_r} | ({log_fd} if log_fd is not None else set())
    _close_fds_except(frozenset(kept), fd_limit)
    devnull = _above_stdio(os.open(os.devnull, os.O_RDWR | os.O_CLOEXEC))
    out = log_fd if log_fd is not None else devnull
    os.dup2(devnull, 0)
    os.dup2(out, 1)
    os.dup2(out, 2)
    for fd in (0, 1, 2):
        os.set_inheritable(fd, True)
    os.close(devnull)
    if log_fd is not None:
        os.close(log_fd)
    _reset_dispositions(gw)
    info = inspect(gw.getpid(), gateway=gw)
    if info is None:
        gw.exit_now(70)
        return
    os.write(id_w, json.dumps(info.identity.to_dict()).encode("ascii") + b"\n")
    os.close(id_w)
    if not gw.poll_readable(ack_r, ack_timeout) or os.read(ack_r, 1) != _ACK:
        gw.exit_now(1)
        return
    _close_fds_except(frozenset(), fd_limit)
    # Every disposition is SIG_DFL now: clear the mask just before exec (a signal pending
    # since the fork is delivered here, with its default action).
    signal.pthread_sigmask(signal.SIG_SETMASK, [])
    try:
        os.execve(argv[0], list(argv), dict(os.environ) if env is None else dict(env))
    except (OSError, ValueError) as exc:
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
    fd_limit: int,
) -> None:
    """Runs in the first child with every signal blocked; never returns."""
    code = 70
    try:
        os.close(id_r)
        os.close(ack_w)
        gw.setsid()
        if gw.fork() == 0:  # the grandchild is not a session leader: no controlling tty ever
            try:
                _grandchild(gw, argv, env, log_fd, id_w, ack_r, ack_timeout, fd_limit)
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


def _check_exec_args(argv: Sequence[str], env: Mapping[str, str] | None) -> None:
    """Refuse what ``execve`` would refuse, before forking (the grandchild cannot report it)."""
    if not argv or not os.path.isabs(argv[0]):
        raise ValueError("argv[0] must be an absolute path")
    for arg in argv:
        if "\0" in arg:
            raise ValueError("an argv string contains a NUL byte")
    for key, value in (env or {}).items():
        if not key or "=" in key or "\0" in key:
            raise ValueError(f"invalid environment variable name: {key!r}")
        if "\0" in value:
            raise ValueError(f"the value of {key!r} contains a NUL byte")


def _close_quietly(gw: Gateway, fd: int) -> None:
    try:
        gw.close(fd)
    except OSError:
        pass


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

    Right after the fork the grandchild closes every inherited descriptor except its two pipe
    ends, redirects stdio and resets every signal disposition. It reports its PID, start time
    and boot id over a pipe and waits for ``DetachedStart.acknowledge()``; only then does it
    clear the signal mask and exec ``argv`` (same PID and start time). Without an
    acknowledgement within ``ack_timeout`` it exits without exec. A pipe failure, or a pipe
    closed before the acknowledgement, is a failed detached fork (``DetachedForkError``) and
    no exec happens. ``argv[0]`` must be an absolute path; argv and env strings must not
    contain NUL and env names must be non-empty without ``=`` (``ValueError``).

    Every signal is blocked from just before the fork until the grandchild's exec, so no
    Python signal handler of the caller ever runs in a forked child.
    """
    # todo(#11): add a fake spawn_detached (procs-level) when the launcher's unit tests need it
    gw = _gw(gateway)
    _check_exec_args(argv, env)
    fd_limit = _fd_limit()
    log_fd: int | None = None
    id_r = id_w = ack_r = ack_w = -1
    pid = -1
    reaped = False
    try:
        if log_path is not None:
            log_fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_CLOEXEC, 0o600)
        id_r, id_w = gw.pipe()
        ack_r, ack_w = gw.pipe()
        old_mask = gw.pthread_sigmask(signal.SIG_BLOCK, ())  # read it before changing it
        try:
            gw.pthread_sigmask(signal.SIG_BLOCK, signal.valid_signals())
            pid = gw.fork()
            if pid == 0:
                try:
                    _intermediate(
                        gw, argv, env, log_fd, id_r, id_w, ack_r, ack_w, ack_timeout, fd_limit
                    )
                finally:
                    gw.exit_now(70)
        finally:
            if pid != 0:  # the launcher (or a failed fork): restore the caller's mask
                gw.pthread_sigmask(signal.SIG_SETMASK, old_mask)
        # The launcher: keep only the identity read end and the acknowledgement write end.
        gw.close(id_w)
        gw.close(ack_r)
        id_w = ack_r = -1
        try:
            gw.waitpid(pid)
        except ChildProcessError:
            pass  # SIGCHLD is ignored: the kernel reaped it already
        except OSError as exc:
            raise DetachedForkError(f"waiting for the intermediate process failed: {exc}") from exc
        reaped = True
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
        if pid > 0 and not reaped:
            try:
                gw.waitpid(pid)  # the intermediate exits right after its fork
            except OSError:
                pass
        for fd in (id_r, id_w, ack_r, ack_w):
            if fd >= 0:
                _close_quietly(gw, fd)
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
