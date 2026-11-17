"""The one-shot environment handoff to a process in a terminal window.

Normative source: interface.md "Environment handoff"; decisions D4 and D21 in design.md.

A tmux window inherits the tmux server's environment, not the caller's, so the launching
process (the *listener*: the launcher, or the supervisor for a promotion) hands its environment
to the window process (the *client*, ``supervise --handoff``) over a Unix stream socket:

1. ``HandoffListener.create`` makes a fresh ``<state>/handoff/<random>/`` (0700) holding an
   ``owner`` file and the listening socket ``s``. It runs *before* ``termctl open``.
2. After ``termctl open`` returned, ``listener.serve`` reads the pane PID and its start time and
   ``accept``s; connections made earlier waited in the backlog. A connection is answered only
   if its peer uid is ours, its peer PID is the pane PID and the peer's start time is the
   recorded one. Every other connection is closed without data and logged
   (``handoff_rejected``). The peer check is the authentication: there is no nonce.
3. The answer is one JSON object ``{"env": {...}}`` (the listener's environment minus the
   terminal variables), at most 4 MiB, ended by ``shutdown(SHUT_WR)``. The client sends nothing.
4. ``serve`` always closes the listener at its end (success, rejection storm, expiry, error):
   the socket, the ``owner`` file and the directory are removed, so a second connection never
   gets the environment.
5. ``receive_environment`` is the client: it connects, reads to EOF with a size cap and builds
   the new environment from ``env`` plus the six terminal variables of its own (pane) environment.

The environment is never put in argv, a log record, an exception message or a file. Messages
and log details name only a cause (``HandoffFailed.cause``), never a value.

Users and seams of later issues:

* The launcher/resume wiring (``launch.Launch.place_window``) calls ``create`` before ``termctl
  open`` and ``serve`` after it (``pane_pid`` and ``pane_state`` come from termctl's JSON);
  ``supervise --handoff`` calls ``receive_environment``, exits 1 on ``HandoffFailed`` and loads
  its config again from the received environment, so the launcher's ``AGENT_SPAWN_*`` overrides
  apply (``supervise.Supervisor.adopt_handoff``).
* ``supervise --host`` (a promotion window, ``supervise.Supervisor.run_host``) uses the same client.
* ``sweep_dead`` is ``reconcile``'s housekeeping: it removes the handoff directories whose
  ``owner`` is dead (a listener killed before its ``close``).
"""

from __future__ import annotations

import errno
import json
import os
import re
import secrets
import socket
import stat
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent_spawn import procs
from agent_spawn.gateway import Gateway, get_gateway
from agent_spawn.records import RecordsError, append_log

__all__ = [
    "CLIENT_SLACK",
    "MAX_PAYLOAD",
    "TERMINAL_VARIABLES",
    "HandoffFailed",
    "HandoffListener",
    "build_environment",
    "client_deadline",
    "encode_payload",
    "handoff_root",
    "parse_payload",
    "read_owner",
    "receive_environment",
    "sweep_dead",
]

#: At most this many bytes of payload, enforced by the writer and by the reader.
MAX_PAYLOAD = 4 * 1024 * 1024
#: The client's deadline is the handoff timeout plus this (termctl's post-open hooks may delay
#: the launcher).
CLIENT_SLACK = 60.0
#: Variables the pane supplies; the payload never carries them.
TERMINAL_VARIABLES = (
    "TERM",
    "TERM_PROGRAM",
    "TERM_PROGRAM_VERSION",
    "COLORTERM",
    "TMUX",
    "TMUX_PANE",
)
SOCKET_NAME = "s"
OWNER_NAME = "owner"
HANDOFF_DIR = "handoff"
_BACKLOG = 128
_READ_CHUNK = 64 * 1024
_MAX_REJECTED_LOGS = 20  # a flood of thief connections cannot flood the log
# sun_path sizes including the terminating NUL.
_SUN_PATH_LINUX = 108
_SUN_PATH_DARWIN = 104


class HandoffFailed(Exception):
    """The handoff failed. ``cause`` is a fixed token (never data from the environment).

    Causes: ``no_pane_pid``, ``pane_gone``, ``timeout``, ``socket_path_too_long``,
    ``unsafe_directory``, ``invalid_env``, ``payload_too_large``, ``send_failed``,
    ``connect_failed``, ``empty_payload``, ``malformed_payload``, ``listen_failed``,
    ``pane_unreadable``.

    ``handoff_rejected`` records carry one of ``peer_unreadable``, ``wrong_uid``, ``wrong_pid``,
    ``peer_gone``, ``wrong_start_time``.
    """

    def __init__(self, cause: str, message: str | None = None) -> None:
        super().__init__(message or cause)
        self.cause = cause


def client_deadline(handoff_timeout: float) -> float:
    """The window process's deadline to connect and read."""
    return handoff_timeout + CLIENT_SLACK


# --- the payload ---------------------------------------------------------------------------------


# Lone surrogates other than U+DC80..U+DCFF (os.fsdecode's escapes for undecodable bytes) cannot
# be turned back into bytes by os.fsencode.
_BAD_SURROGATE = re.compile("[\ud800-\udc7f\udd00-\udfff]")


def _valid_pair(name: object, value: object) -> bool:
    return (
        isinstance(name, str)
        and isinstance(value, str)
        and name != ""
        and "=" not in name
        and "\0" not in name
        and "\0" not in value
        and _BAD_SURROGATE.search(name) is None
        and _BAD_SURROGATE.search(value) is None
    )


def encode_payload(env: Mapping[str, str]) -> bytes:
    """The wire bytes for ``env`` minus the terminal variables.

    ``ensure_ascii`` keeps lone surrogates (undecodable bytes in ``os.environ``) intact and the
    result pure ASCII. An entry that cannot be exported (empty name, ``=`` or NUL in the name, NUL
    in the value, a surrogate that is not an undecodable-byte escape) is skipped silently, never
    named. Raises ``HandoffFailed`` for a non-string name or value, or a payload over
    ``MAX_PAYLOAD``; the message never contains a value.
    """
    cleaned: dict[str, str] = {}
    for name, value in env.items():
        if not isinstance(name, str) or not isinstance(value, str):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise HandoffFailed("invalid_env", "the environment holds a malformed variable")
        if not _valid_pair(name, value):
            continue
        if name not in TERMINAL_VARIABLES:
            cleaned[name] = value
    data = json.dumps({"env": cleaned}, separators=(",", ":")).encode("ascii")
    if len(data) > MAX_PAYLOAD:
        raise HandoffFailed("payload_too_large", "the environment exceeds 4 MiB")
    return data


def _no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate key")
        result[key] = value
    return result


def parse_payload(data: bytes) -> dict[str, str]:
    """The ``env`` mapping of a payload, strictly; ``HandoffFailed`` otherwise."""
    if not data:
        raise HandoffFailed("empty_payload", "the handoff payload is empty")
    if len(data) > MAX_PAYLOAD:
        raise HandoffFailed("payload_too_large", "the handoff payload exceeds 4 MiB")
    try:
        parsed = json.loads(data.decode("utf-8"), object_pairs_hook=_no_duplicates)
    except (ValueError, RecursionError):
        raise HandoffFailed("malformed_payload", "the handoff payload is not valid JSON") from None
    if not isinstance(parsed, dict) or set(parsed) != {"env"}:  # pyright: ignore[reportUnknownArgumentType]
        raise HandoffFailed("malformed_payload", "the handoff payload has the wrong shape")
    env: Any = parsed["env"]  # pyright: ignore[reportUnknownVariableType]
    if not isinstance(env, dict):
        raise HandoffFailed("malformed_payload", "the handoff payload has the wrong shape")
    result: dict[str, str] = {}
    for name, value in env.items():  # pyright: ignore[reportUnknownVariableType]
        if not _valid_pair(name, value):
            raise HandoffFailed("malformed_payload", "the handoff payload holds a bad variable")
        result[name] = value  # pyright: ignore[reportUnknownVariableType]
    return result


def build_environment(
    payload_env: Mapping[str, str], pane_env: Mapping[str, str]
) -> dict[str, str]:
    """``payload_env`` (terminal variables dropped) plus the six terminal variables of the pane."""
    result = {k: v for k, v in payload_env.items() if k not in TERMINAL_VARIABLES}
    for name in TERMINAL_VARIABLES:
        if name in pane_env:
            result[name] = pane_env[name]
    return result


# --- logging -------------------------------------------------------------------------------------


def _log(
    state_dir: str | os.PathLike[str] | None,
    event: str,
    cause: str,
    agent_id: str | None,
    level: str = "error",
) -> None:
    """Log a handoff event with a fixed message and the cause only; never raises."""
    if state_dir is None:
        return
    try:
        append_log(
            state_dir,
            event,
            level=level,
            agent_id=agent_id,
            message=f"environment handoff: {cause}",
            details={"cause": cause},
        )
    except (RecordsError, ValueError, OSError):
        pass


# --- the listener --------------------------------------------------------------------------------


def handoff_root(state_dir: str | os.PathLike[str]) -> Path:
    return Path(state_dir) / HANDOFF_DIR


def read_owner(directory: str | os.PathLike[str]) -> procs.ProcessIdentity | None:
    """The ``owner`` identity of a handoff directory, or None if absent or unreadable."""
    try:
        data = json.loads((Path(directory) / OWNER_NAME).read_text(encoding="utf-8"))
        return procs.ProcessIdentity.from_dict(data)
    except (OSError, ValueError, AttributeError, TypeError):
        return None


def _open_private_dir(path: Path, gw: Gateway) -> None:
    """Make sure ``path`` is a real directory of ours with mode 0700 (created if missing)."""
    try:
        os.mkdir(path, 0o700)
    except FileExistsError:
        pass
    except OSError as exc:
        raise HandoffFailed(
            "unsafe_directory", f"cannot create the handoff directory: {exc.strerror}"
        ) from None
    try:
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    except OSError:
        raise HandoffFailed(
            "unsafe_directory", "the handoff directory is not a plain directory"
        ) from None
    try:
        info = os.fstat(fd)
        if info.st_uid != gw.getuid():
            raise HandoffFailed("unsafe_directory", "the handoff directory belongs to another user")
        if stat.S_IMODE(info.st_mode) != 0o700:
            os.fchmod(fd, 0o700)
    except OSError:
        raise HandoffFailed("unsafe_directory", "cannot secure the handoff directory") from None
    finally:
        os.close(fd)


class HandoffListener:
    """The listening side: a fresh 0700 directory, its ``owner`` file and the socket ``s``."""

    def __init__(
        self,
        directory: Path,
        sock: socket.socket,
        *,
        state_dir: Path,
        agent_id: str | None,
        gateway: Gateway,
    ) -> None:
        self.directory = directory
        self.socket_path = directory / SOCKET_NAME
        self._sock: socket.socket | None = sock
        self._state_dir = state_dir
        self._agent_id = agent_id
        self._gw = gateway
        self._served = False

    @classmethod
    def create(
        cls,
        state_dir: str | os.PathLike[str],
        *,
        agent_id: str | None = None,
        gateway: Gateway | None = None,
    ) -> HandoffListener:
        """Create the directory, write ``owner`` and listen. Raises ``HandoffFailed``.

        The state directory itself must be an absolute path. Nothing is left behind on failure.
        """
        gw = gateway or get_gateway()
        state = Path(state_dir)
        if not state.is_absolute():
            raise HandoffFailed("unsafe_directory", "the state directory must be absolute")
        root = handoff_root(state)
        _open_private_dir(root, gw)
        name = secrets.token_hex(6)
        directory = root / name
        sock_path = directory / SOCKET_NAME
        limit = _SUN_PATH_DARWIN if gw.platform() == "darwin" else _SUN_PATH_LINUX
        if len(os.fsencode(sock_path)) >= limit:
            raise HandoffFailed("socket_path_too_long", "the handoff socket path is too long")
        try:
            os.mkdir(directory, 0o700)  # fresh: EEXIST fails, so nothing can pre-exist here
        except OSError:
            raise HandoffFailed("unsafe_directory", "cannot create the handoff directory") from None
        sock: socket.socket | None = None
        try:
            os.chmod(directory, 0o700)  # not subject to the umask
            try:
                me = procs.read_identity(gw.getpid(), gateway=gw)
            except procs.ProcessIdentityError:
                me = None
            if me is None:
                raise HandoffFailed("listen_failed", "cannot identify the listening process")
            fd = os.open(
                directory / OWNER_NAME,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600,
            )
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(me.to_dict(), handle)
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.bind(str(sock_path))
            os.chmod(sock_path, 0o600)
            info = os.lstat(sock_path)
            if not stat.S_ISSOCK(info.st_mode) or info.st_uid != gw.getuid():
                raise HandoffFailed("unsafe_directory", "the handoff socket is not ours")
            sock.listen(_BACKLOG)
        except BaseException as exc:
            if sock is not None:
                sock.close()
            _remove_tree(directory)
            if isinstance(exc, OSError):
                raise HandoffFailed("listen_failed", f"cannot listen: {exc.strerror}") from None
            raise
        return cls(directory, sock, state_dir=state, agent_id=agent_id, gateway=gw)

    # -- lifecycle --

    def close(self) -> None:
        """Stop listening and remove the socket, the ``owner`` file and the directory.

        Idempotent; a missing file counts as removed.
        """
        sock, self._sock = self._sock, None
        if sock is not None:
            sock.close()
        _remove_tree(self.directory)

    def __enter__(self) -> HandoffListener:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # -- serving --

    def serve(
        self,
        *,
        pane_pid: int | None,
        pane_state: str | None,
        timeout: float,
        env: Mapping[str, str],
    ) -> None:
        """Answer exactly one authenticated connection, then close. Raises ``HandoffFailed``.

        Call this right after ``termctl open --json`` returned: ``pane_pid`` and ``pane_state``
        are its ``pid`` and ``state``; ``timeout`` (``handoff_timeout``) runs from this call.
        The listener is closed whatever happens.
        """
        try:
            self._serve(pane_pid, pane_state, timeout, env)
        except HandoffFailed as exc:
            _log(self._state_dir, "handoff_failed", exc.cause, self._agent_id)
            raise
        finally:
            self.close()

    def _serve(
        self, pane_pid: int | None, pane_state: str | None, timeout: float, env: Mapping[str, str]
    ) -> None:
        if self._served or self._sock is None:
            raise HandoffFailed("listen_failed", "the handoff listener is closed or used")
        self._served = True
        gw = self._gw
        if pane_pid is None or isinstance(pane_pid, bool) or pane_pid <= 0:
            raise HandoffFailed("no_pane_pid", "termctl reported no pane PID")
        if pane_state == "gone":
            raise HandoffFailed("pane_gone", "the pane is gone")
        deadline = gw.monotonic() + timeout
        data = encode_payload(env)  # before accepting: an oversized environment fails at once
        try:
            pane = procs.inspect(pane_pid, gateway=gw)
        except (procs.ProcessIdentityError, OSError):
            raise HandoffFailed("pane_unreadable", "the pane process cannot be read") from None
        if pane is None or not pane.alive:
            raise HandoffFailed("pane_gone", "the pane process is gone")
        expected = pane.identity
        rejected = 0
        sock = self._sock
        while True:
            remaining = deadline - gw.monotonic()
            if remaining <= 0:
                raise HandoffFailed("timeout", "no authorised connection before the deadline")
            sock.settimeout(remaining)
            try:
                conn, _ = sock.accept()
            except TimeoutError:
                raise HandoffFailed(
                    "timeout", "no authorised connection before the deadline"
                ) from None
            except OSError as exc:
                if exc.errno in (errno.ECONNABORTED, errno.EINTR):
                    continue
                raise HandoffFailed("listen_failed", f"accept failed: {exc.strerror}") from None
            with conn:
                reason = self._check_peer(conn, expected)
                if reason is not None:
                    rejected += 1
                    if rejected <= _MAX_REJECTED_LOGS:
                        _log(self._state_dir, "handoff_rejected", reason, self._agent_id, "warn")
                    continue
                self._send(conn, data, deadline)
                return

    def _check_peer(self, conn: socket.socket, expected: procs.ProcessIdentity) -> str | None:
        """None for the pane process; otherwise the fixed reason for refusing the connection."""
        gw = self._gw
        try:
            creds = gw.peer_credentials(conn)
        except (OSError, NotImplementedError):
            return "peer_unreadable"
        if creds.uid != gw.getuid():
            return "wrong_uid"
        if creds.pid != expected.pid:
            return "wrong_pid"
        try:
            now = procs.inspect(creds.pid, gateway=gw)
        except procs.ProcessIdentityError:
            return "peer_unreadable"
        if now is None or not now.alive:
            return "peer_gone"
        if now.identity != expected:
            return "wrong_start_time"
        return None

    def _send(self, conn: socket.socket, data: bytes, deadline: float) -> None:
        remaining = deadline - self._gw.monotonic()
        if remaining <= 0:
            raise HandoffFailed("timeout", "the deadline passed before the environment was sent")
        conn.settimeout(remaining)
        try:
            conn.sendall(data)
            conn.shutdown(socket.SHUT_WR)
        except TimeoutError:
            raise HandoffFailed(
                "timeout", "the window process did not read before the deadline"
            ) from None
        except OSError:
            raise HandoffFailed("send_failed", "the environment could not be sent") from None


def _remove_tree(directory: Path) -> None:
    """Remove the files a listener creates and the directory; missing files are fine."""
    for name in (SOCKET_NAME, OWNER_NAME):
        try:
            os.unlink(directory / name)
        except FileNotFoundError:
            pass
        except OSError:
            pass
    try:
        os.rmdir(directory)
    except OSError:
        pass


# --- the sweep of dead listeners ---------------------------------------------------------------

#: A directory without a readable ``owner`` (a listener that died between its ``mkdir`` and the
#: write of the file) is removed only when it is older than this: a live listener writes the file
#: within milliseconds.
OWNERLESS_AGE = 3600.0
_NAME = re.compile(r"[0-9a-f]{12}")  # what ``HandoffListener.create`` names its directories
_OWNER_LIMIT = 4096


@dataclass
class Sweep:
    """What ``sweep_dead`` removed (or, in a dry run, would remove), and what it could not judge."""

    removed: list[str] = field(default_factory=lambda: [])
    warnings: list[str] = field(default_factory=lambda: [])


def _owner_at(dir_fd: int) -> procs.ProcessIdentity | None:
    """``owner`` of the directory open as ``dir_fd``, read without following a symlink."""
    try:
        fd = os.open(OWNER_NAME, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=dir_fd)
    except OSError:
        return None
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            return None
        data = os.read(fd, _OWNER_LIMIT + 1)
    except OSError:
        return None
    finally:
        os.close(fd)
    if len(data) > _OWNER_LIMIT:
        return None
    try:
        return procs.ProcessIdentity.from_dict(json.loads(data.decode("utf-8")))
    except (ValueError, AttributeError, TypeError, KeyError):
        return None


def _owner_dead(owner: procs.ProcessIdentity, gw: Gateway) -> bool:
    """Whether the recorded listener is gone: another boot id, or no process with that PID,
    start time and boot id that is not a zombie. Raises ``ProcessIdentityError`` when the process
    entry cannot be read (never "dead")."""
    if owner.boot_id != procs.read_boot_id(gateway=gw):
        return True
    return not procs.is_alive(owner, gateway=gw)


def sweep_dead(
    state_dir: str | os.PathLike[str], *, dry_run: bool = False, gateway: Gateway | None = None
) -> Sweep:
    """Remove the handoff directories of dead listeners (``reconcile``, step 6).

    A directory goes only when it is a real directory of ours named like the ones ``create``
    makes, its ``owner`` file names a process that is verifiably dead (PID, start time and boot
    id), and it holds nothing but ``s`` and ``owner``; only those two files and the directory
    are removed, through descriptors opened without following symlinks. A live listener's
    directory, one whose owner cannot be judged, and one with anything else in it stay. A
    directory without a readable ``owner`` is judged by its age (``OWNERLESS_AGE``).
    """
    gw = gateway or get_gateway()
    sweep = Sweep()
    try:
        root = os.open(
            handoff_root(state_dir), os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
        )
    except FileNotFoundError:
        return sweep
    except OSError as exc:
        sweep.warnings.append(f"cannot open the handoff directory ({exc.strerror})")
        return sweep
    try:
        if os.fstat(root).st_uid != gw.getuid():
            sweep.warnings.append("the handoff directory belongs to another user: not swept")
            return sweep
        for name in sorted(os.listdir(root)):
            if _NAME.fullmatch(name) is None:
                continue
            try:
                if _sweep_one(root, name, gw, dry_run, sweep):
                    sweep.removed.append(name)
            except procs.ProcessIdentityError:
                sweep.warnings.append(f"handoff {name}: its owner could not be judged: kept")
            except OSError as exc:
                sweep.warnings.append(f"handoff {name}: not removed ({exc.strerror})")
    finally:
        os.close(root)
    return sweep


def _sweep_one(root: int, name: str, gw: Gateway, dry_run: bool, sweep: Sweep) -> bool:
    info = os.lstat(name, dir_fd=root)
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != gw.getuid():
        return False
    fd = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=root)
    try:
        opened = os.fstat(fd)
        if (opened.st_dev, opened.st_ino) != (info.st_dev, info.st_ino):
            return False  # swapped while it was being looked at
        owner = _owner_at(fd)
        if owner is None:
            if gw.wall_time() - opened.st_mtime <= OWNERLESS_AGE:
                return False
        elif not _owner_dead(owner, gw):
            return False
        if set(os.listdir(fd)) - {SOCKET_NAME, OWNER_NAME}:
            sweep.warnings.append(f"handoff {name}: holds other files: kept")
            return False
        if dry_run:
            return True
        for leaf in (SOCKET_NAME, OWNER_NAME):
            try:
                os.unlink(leaf, dir_fd=fd)  # a symlink is unlinked itself, never followed
            except FileNotFoundError:
                pass
    finally:
        os.close(fd)
    os.rmdir(name, dir_fd=root)
    return True


# --- the client ----------------------------------------------------------------------------------


def receive_environment(
    socket_path: str | os.PathLike[str],
    *,
    timeout: float,
    pane_env: Mapping[str, str],
    state_dir: str | os.PathLike[str] | None = None,
    agent_id: str | None = None,
    gateway: Gateway | None = None,
) -> dict[str, str]:
    """Connect to the listener, read its answer and build the environment to run in.

    ``timeout`` is the whole deadline to connect and read (use ``client_deadline``). The
    caller (``supervise``) exits 1 on ``HandoffFailed`` without starting anything; the failure
    is logged as ``handoff_failed`` (cause only) when ``state_dir`` is given. Nothing is sent.
    """
    gw = gateway or get_gateway()
    try:
        return _receive(Path(socket_path), timeout, pane_env, gw)
    except HandoffFailed as exc:
        _log(state_dir, "handoff_failed", exc.cause, agent_id)
        raise


def _receive(
    path: Path, timeout: float, pane_env: Mapping[str, str], gw: Gateway
) -> dict[str, str]:
    deadline = gw.monotonic() + timeout
    try:
        info = os.lstat(path)
    except OSError:
        raise HandoffFailed("connect_failed", "the handoff socket does not exist") from None
    if not stat.S_ISSOCK(info.st_mode) or info.st_uid != gw.getuid():
        raise HandoffFailed("connect_failed", "the handoff socket is not a socket of ours")
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    with sock:
        sock.settimeout(max(deadline - gw.monotonic(), 0.001))
        try:
            sock.connect(str(path))
        except TimeoutError:
            raise HandoffFailed("timeout", "could not connect before the deadline") from None
        except OSError:
            raise HandoffFailed("connect_failed", "cannot connect to the handoff socket") from None
        chunks: list[bytes] = []
        total = 0
        while True:
            remaining = deadline - gw.monotonic()
            if remaining <= 0:
                raise HandoffFailed(
                    "timeout", "the handoff payload did not arrive before the deadline"
                )
            sock.settimeout(remaining)
            try:
                chunk = sock.recv(_READ_CHUNK)
            except TimeoutError:
                raise HandoffFailed(
                    "timeout", "the handoff payload did not arrive before the deadline"
                ) from None
            except OSError:
                raise HandoffFailed("connect_failed", "the handoff connection failed") from None
            if not chunk:
                break
            total += len(chunk)
            if total > MAX_PAYLOAD:
                raise HandoffFailed("payload_too_large", "the handoff payload exceeds 4 MiB")
            chunks.append(chunk)
    return build_environment(parse_payload(b"".join(chunks)), pane_env)
