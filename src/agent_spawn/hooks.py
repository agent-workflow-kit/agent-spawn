"""Hook processes, selectors, payloads and at-least-once outbox delivery.

Normative sources: interface.md "Hooks" and "Hook delivery", design guide section 8,
design.md D9, D34 and D42.

Security rules this module keeps (CLAUDE.md, interface.md "Signals"):

* A hook is ``/bin/sh -c <cmd>`` in a process group of its own (``start_new_session``), with the
  JSON payload on stdin (an unlinked temporary file, so a hook that never reads stdin cannot block
  us). The ``cmd`` comes from the user's config or ``--hook`` flag. Nothing else is ever on the
  command line, and the environment gets only ``AGENT_SPAWN_HOOK``, ``AGENT_SPAWN_ID`` (a
  validated agent id) and ``AGENT_SPAWN_OBJECT_TYPE`` on top of the caller's own. Agent and
  third-party text (report summaries, reasons, labels) travels only in the stdin payload.
* The only processes ever signalled are the process groups this module created: the group leader
  is still unreaped (so its PID cannot have been reused), its identity (PID plus start time) is
  re-verified and ``getpgid(pid) == pid`` is checked before each group signal. Never by pattern.
* Captured stdout and stderr are capped (``CAPTURE_LIMIT``); the rest is read and dropped so a
  hook that floods a pipe can neither exhaust memory nor block on a full pipe.
* ``report.summary`` and ``report.reason`` are nulled in every post-hook payload unless the run
  is known to be trusted and untainted (``reads_third_party`` and ``tainted`` both ``false``).
* Post-hook failures are logged (``hook_failed``, ``hook_timeout``) and never raised; a log that
  cannot be written never aborts the command either.

Stored contracts other issues rely on:

* An outbox event (``OutboxEvent(event=<hook name>, payload=<body>)``) stores the payload body
  without ``tool``, ``hook``, ``object_type``, ``object_id``, ``event_id`` and ``ts``; delivery
  adds them (``event_id`` is the ``events`` row id, ``ts`` its creation time). Build bodies with
  ``agent_event`` so the common fields are right.
* Per-invocation ``--hook NAME=CMD`` entries are stored as a JSON list of
  ``{"hook": NAME, "cmd": CMD}`` (``parse_invocation_hooks``): in ``agent_runs.hooks`` for the
  run hooks (``post-launch``, ``post-state-change``, ``post-escalate``, ``post-park``; the run is
  the payload's ``run``, else its outcome's, else the agent's current run) and in
  ``agents.hooks`` for ``pre-teardown`` and ``post-teardown``. They have a fixed 30 s timeout,
  are never async and have no selectors.
"""

from __future__ import annotations

import copy
import json
import math
import os
import re
import secrets
import selectors
import signal
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import IO, Any, Literal

from . import procs, records
from .config import Hooks
from .gateway import DEVNULL, PIPE, Gateway, get_gateway
from .output import CliError, cut_utf8
from .store import LeaseHeld, LeaseLost, OutboxEvent, Store, StoreError

__all__ = [
    "ASYNC",
    "CAPTURE_LIMIT",
    "DEFAULT_TIMEOUT",
    "POST_HOOKS",
    "PRE_HOOKS",
    "AsyncRunner",
    "DeliveryResult",
    "HookEntry",
    "PreHookVerdict",
    "ProcessResult",
    "agent_event",
    "common_fields",
    "config_entries",
    "deliver_outbox",
    "entry_matches",
    "finalize_payload",
    "invocation_entries",
    "parse_invocation_hooks",
    "redact_payload",
    "run_hook_process",
    "run_pre_hooks",
    "verdict_error",
]

PRE_HOOKS = ("pre-launch", "pre-teardown")
POST_HOOKS = (
    "post-launch",
    "post-state-change",
    "post-escalate",
    "post-park",
    "post-teardown",
)
ASYNC = "async"
DEFAULT_TIMEOUT = 30.0
INVOCATION_TIMEOUT = 30.0
CAPTURE_LIMIT = 64 * 1024  # bytes kept of each of a hook's stdout and stderr
LOG_TEXT_LIMIT = 4096  # bytes of hook stderr put in a log record
TERM_GRACE = 2.0  # seconds between SIGTERM and SIGKILL of a timed-out hook's group
DRAIN_GRACE = 0.5  # seconds to keep reading pipes after the hook's leader exited
POLL = 0.05
REAP_TIMEOUT = 5.0
MAX_ASYNC = 16  # async hooks one runner keeps running at once
ASYNC_CAP_WAIT = 10.0  # seconds ``AsyncRunner.start`` waits for a free slot
WATCHDOG_GRACE = 2  # whole seconds between the watchdog's SIGTERM and SIGKILL
_KILL_CODES = frozenset({-15, -9, 143, 137})  # SIGTERM/SIGKILL as a signal or as 128 + signal
ESCALATE_EXIT = 13
MAX_DECISION_IDS = 32
_AGENT_ID = re.compile(r"ag-[a-z2-7]{8}")
_DECISION_ID = re.compile(r"[A-Za-z0-9._:@/-]{1,128}")
_RUN_HOOKS = frozenset({"post-launch", "post-state-change", "post-escalate", "post-park"})
_TEARDOWN_HOOKS = frozenset({"pre-teardown", "post-teardown"})
_SH = "/bin/sh"
# Runs an async hook: $1 the command (never interpolated into this text), $2 the timeout and $3
# the SIGTERM-to-SIGKILL grace, both in whole seconds. The wrapper is the group leader; the
# watchdog subshell signals the whole group (``kill 0``); its USR1 trap is installed before
# anything else so an early signal from the wrapper cannot orphan the watchdog's ``sleep`` (the
# trap kills ``$!``, the most recent background job, which is set at the fork). The timeout
# bounds the command itself only: the watchdog stops when the command exits, so background
# children it leaves behind afterwards are not bounded.
_ASYNC_WRAPPER = """
(
  trap '' TERM
  trap 'kill -s KILL $! 2>/dev/null; exit 0' USR1
  sleep "$2" & wait $!
  kill -s TERM 0
  sleep "$3" & wait $!
  kill -s KILL 0
) &
w=$!
/bin/sh -c "$1"
rc=$?
kill -s USR1 "$w" 2>/dev/null
wait "$w" 2>/dev/null
exit "$rc"
"""


# --- entries and selectors -----------------------------------------------------------------------


@dataclass(frozen=True)
class HookEntry:
    """One hook to run: a config entry or a per-invocation ``--hook``."""

    cmd: str
    timeout: float = DEFAULT_TIMEOUT
    async_: bool = False
    kinds: tuple[str, ...] | None = None
    exclude_kinds: tuple[str, ...] | None = None
    labels: tuple[str, ...] | None = None
    states: tuple[str, ...] | None = None
    statuses: tuple[str, ...] | None = None
    source: Literal["config", "invocation"] = "config"


def _tuple(value: Iterable[str] | None) -> tuple[str, ...] | None:
    return None if value is None else tuple(value)


def config_entries(hooks: Hooks, name: str) -> list[HookEntry]:
    """The configured entries of hook ``name`` in config order (drop-ins already appended)."""
    if name not in PRE_HOOKS and name not in POST_HOOKS:
        raise ValueError(f"unknown hook {name!r}")
    result: list[HookEntry] = []
    for item in getattr(hooks, name.replace("-", "_")):
        result.append(
            HookEntry(
                cmd=item.cmd,
                timeout=item.timeout,
                async_=bool(getattr(item, "async_", False)),
                kinds=_tuple(item.kinds),
                exclude_kinds=_tuple(item.exclude_kinds),
                labels=_tuple(item.labels),
                states=_tuple(getattr(item, "states", None)),
                statuses=_tuple(getattr(item, "statuses", None)),
            )
        )
    return result


def parse_invocation_hooks(values: Sequence[str], allowed: Iterable[str]) -> list[dict[str, str]]:
    """Parse repeated ``--hook NAME=CMD`` values into the stored form; a bad one is a usage error.

    ``allowed`` names the hooks the command accepts (``launch``: ``pre-launch``, ``post-launch``).
    The command is split at the first ``=``; it may itself contain ``=``.
    """
    names = tuple(allowed)
    parsed: list[dict[str, str]] = []
    for value in values:
        name, sep, cmd = value.partition("=")
        if not sep or name not in names:
            raise CliError.usage(f"--hook must be NAME=CMD with NAME one of {', '.join(names)}")
        if not cmd.strip():
            raise CliError.usage(f"--hook {name}: the command is empty")
        if "\0" in cmd:
            raise CliError.usage(f"--hook {name}: the command contains a NUL byte")
        parsed.append({"hook": name, "cmd": cmd})
    return parsed


def invocation_entries(stored: object, name: str) -> list[HookEntry]:
    """Entries of hook ``name`` from a stored per-invocation list; malformed items are skipped."""
    if not isinstance(stored, list):
        return []
    result: list[HookEntry] = []
    for item in stored:  # pyright: ignore[reportUnknownVariableType]
        if not isinstance(item, dict):
            continue
        item_name = item.get("hook")  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        cmd = item.get("cmd")  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        if item_name != name or not isinstance(cmd, str) or not cmd or "\0" in cmd:
            continue
        result.append(HookEntry(cmd=cmd, timeout=INVOCATION_TIMEOUT, source="invocation"))
    return result


def _strings(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]  # pyright: ignore[reportUnknownVariableType]


def entry_matches(entry: HookEntry, payload: Mapping[str, Any]) -> bool:
    """Selectors are ANDed. A payload without the field a selector names never matches a
    positive selector (``kinds``, ``labels``, ``states``, ``statuses``) and is never excluded by
    ``exclude_kinds``."""
    run_kind = payload.get("run_kind")
    if entry.kinds is not None and run_kind not in entry.kinds:
        return False
    if entry.exclude_kinds is not None and run_kind in entry.exclude_kinds:
        return False
    if entry.labels is not None and not set(entry.labels) <= set(_strings(payload.get("labels"))):
        return False
    if entry.states is not None and (
        payload.get("hook") != "post-state-change"
        or payload.get("change") != "session"
        or payload.get("state") not in entry.states
    ):
        return False
    return entry.statuses is None or payload.get("status") in entry.statuses


# --- payloads ------------------------------------------------------------------------------------


def common_fields(agent: Mapping[str, Any], reads_third_party: bool | None) -> dict[str, Any]:
    """The top-level fields every post-hook payload shares, from the agent object
    (``schemas/defs.json#/$defs/agent``) and the last run's classification."""
    return {
        "repo": agent["repo"],
        "labels": list(agent["labels"]),
        "kind": agent["kind"],
        "run_kind": agent["run_kind"],
        "reads_third_party": reads_third_party,
        "tainted": agent["tainted"],
        "mode": agent["mode"],
        "status": agent["status"],
        "needs_attention": agent["needs_attention"],
    }


def agent_event(
    hook: str,
    agent: Mapping[str, Any],
    *,
    reads_third_party: bool | None,
    extra: Mapping[str, Any] | None = None,
) -> OutboxEvent:
    """The outbox event of ``hook`` for an agent: common fields, ``agent`` and ``extra``."""
    if hook not in POST_HOOKS:
        raise ValueError(f"{hook!r} is not a post-hook")
    body = common_fields(agent, reads_third_party)
    body["agent"] = copy.deepcopy(dict(agent))
    body.update(copy.deepcopy(dict(extra or {})))
    return OutboxEvent(hook, body)


def _scrub(value: object, depth: int = 0) -> None:
    if depth > 64:
        raise ValueError("payload nests too deeply")
    if isinstance(value, dict):
        for key, item in value.items():  # pyright: ignore[reportUnknownVariableType]
            if key == "report" and isinstance(item, dict):
                for field_name in ("summary", "reason"):
                    if field_name in item:
                        item[field_name] = None
            else:
                _scrub(item, depth + 1)
    elif isinstance(value, list):
        for item in value:  # pyright: ignore[reportUnknownVariableType]
            _scrub(item, depth + 1)


def redact_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """A copy with every ``report.summary`` and ``report.reason`` nulled (top level and inside
    outcome objects) unless ``reads_third_party`` and ``tainted`` are both ``false``."""
    copied = copy.deepcopy(dict(payload))
    if copied.get("reads_third_party") is False and copied.get("tainted") is False:
        return copied
    _scrub(copied)
    return copied


def _format_ts(moment: float) -> str:
    return records.format_ts(datetime.fromtimestamp(moment, UTC))


def finalize_payload(
    hook: str,
    body: Mapping[str, Any],
    *,
    event_id: int,
    ts: str,
    agent_id: str | None,
    redact: bool,
) -> dict[str, Any]:
    """The document written to a hook's stdin: the body plus the identifying fields."""
    payload = redact_payload(body) if redact else copy.deepcopy(dict(body))
    payload.update(
        tool="agent-spawn",
        hook=hook,
        object_type="agent",
        object_id=agent_id,
        event_id=event_id,
        ts=ts,
    )
    return payload


def _encode_payload(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(payload, ensure_ascii=True, allow_nan=False, separators=(",", ":")).encode(
        "ascii"
    )


# --- logging -------------------------------------------------------------------------------------


def _log(
    state_dir: str | os.PathLike[str],
    event: str,
    *,
    level: str = "info",
    agent_id: str | None = None,
    hook: str | None = None,
    pid: int | None = None,
    signal_name: str | None = None,
    message: str | None = None,
    details: Mapping[str, Any] | None = None,
) -> None:
    """Append a log record; a log that cannot be written never aborts a hook run."""
    try:
        records.append_log(
            state_dir,
            event,
            level=level,
            agent_id=agent_id,
            hook=hook,
            pid=pid,
            signal=signal_name,
            message=message,
            details=details,
        )
    except (ValueError, records.RecordsError, OSError):
        pass


# --- running one hook process --------------------------------------------------------------------


@dataclass
class ProcessResult:
    """What one hook process did. ``exit_code`` is None when it was killed (``signal`` is set)
    or never started (``error`` is set)."""

    exit_code: int | None = None
    signal: int | None = None
    timed_out: bool = False
    stdout: bytes = b""
    stderr: bytes = b""
    stdout_truncated: bool = False
    stderr_truncated: bool = False
    duration: float = 0.0
    error: str | None = None
    unreaped: bool = False

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out and self.error is None


class _Group:
    """The process group of one hook this module started, signalled only when verified.

    ``signal`` re-checks, immediately before each group signal, that the leader is still
    unreaped (so its PID cannot belong to anything else), still has the identity recorded at
    spawn (PID plus start time) and still leads a group of its own.
    """

    def __init__(self, child: Any, gateway: Gateway) -> None:
        self.child = child
        self.pid: int = child.pid
        self.gw = gateway
        self.identity: procs.ProcessIdentity | None = None
        try:
            handle = procs.identify(self.pid, gateway=gateway)
        except (OSError, procs.ProcessIdentityError):
            handle = None  # unidentified: only the leader, through ``child``, is ever signalled
        if handle is not None:
            self.identity = handle.identity
            handle.close()

    def leader_done(self) -> bool:
        """The leader exited. Without a recorded identity this asks the (still unreaped, so
        safe) ``Popen`` and reaps; otherwise it reads the process table and does not reap."""
        if self.child.returncode is not None:
            return True
        if self.identity is None:
            return self.child.poll() is not None
        try:
            info = procs.inspect(self.pid, gateway=self.gw)
        except procs.ProcessIdentityError:
            return False
        return info is None or info.identity != self.identity or not info.alive

    def signal(self, sig: int) -> bool:
        if self.child.returncode is not None or self.identity is None:
            return False
        try:
            info = procs.inspect(self.pid, gateway=self.gw)
            if info is None or info.identity != self.identity:
                return False
            if self.gw.getpgid(self.pid) != self.pid:
                return False
            self.gw.killpg(self.pid, sig)
        except (ProcessLookupError, PermissionError, procs.ProcessIdentityError):
            return False
        return True

    def _signal_leader(self, sig: int) -> bool:
        """Signal only the leader, through its unreaped ``Popen`` (its PID cannot be reused)."""
        if self.child.returncode is not None:
            return False
        try:
            self.child.send_signal(sig)
        except (ProcessLookupError, PermissionError):
            return False
        return True

    def terminate(self, state_dir: str | os.PathLike[str] | None, hook: str | None) -> None:
        """SIGTERM (with SIGCONT), wait for the leader up to ``TERM_GRACE``, then SIGKILL.

        A group whose identity was never recorded is not signalled as a group: only the leader
        is (unverified members stay unsignalled)."""
        if self.identity is None:
            self._terminate_leader(state_dir, hook)
            return
        sent = self.signal(signal.SIGTERM)
        if sent:
            self.signal(signal.SIGCONT)
            self._logged(state_dir, hook, signal.SIGTERM)
        end = self.gw.monotonic() + TERM_GRACE
        while sent and self.gw.monotonic() < end and not self.leader_done():
            self.gw.sleep(0.02)
        if self.signal(signal.SIGKILL):
            self._logged(state_dir, hook, signal.SIGKILL)

    def _terminate_leader(self, state_dir: str | os.PathLike[str] | None, hook: str | None) -> None:
        sent = self._signal_leader(signal.SIGTERM)
        if sent:
            self._signal_leader(signal.SIGCONT)
            self._logged(state_dir, hook, signal.SIGTERM, group=False)
        end = self.gw.monotonic() + TERM_GRACE
        while sent and self.gw.monotonic() < end and not self.leader_done():
            self.gw.sleep(0.02)
        if self._signal_leader(signal.SIGKILL):
            self._logged(state_dir, hook, signal.SIGKILL, group=False)

    def _logged(
        self,
        state_dir: str | os.PathLike[str] | None,
        hook: str | None,
        sig: int,
        group: bool = True,
    ) -> None:
        if state_dir is not None:
            _log(
                state_dir,
                "signal_sent",
                hook=hook,
                pid=self.pid,
                signal_name=signal.Signals(sig).name,
                details={"group": group},
            )


def hook_environment(
    environ: Mapping[str, str] | None, hook: str, agent_id: str | None
) -> dict[str, str]:
    """The caller's environment plus the three documented variables."""
    if agent_id is not None and _AGENT_ID.fullmatch(agent_id) is None:
        raise ValueError(f"invalid agent id {agent_id!r}")
    env = dict(os.environ if environ is None else environ)
    env["AGENT_SPAWN_HOOK"] = hook
    env["AGENT_SPAWN_ID"] = agent_id or ""
    env["AGENT_SPAWN_OBJECT_TYPE"] = "agent"
    return env


def _stdin_file(data: bytes, state_dir: str | os.PathLike[str]) -> IO[bytes]:
    """An unlinked temporary file holding ``data``, rewound (the state dir is private)."""
    handle = tempfile.TemporaryFile(dir=state_dir)  # an OSError is the hook's error, no fallback
    try:
        handle.write(data)
        handle.flush()
        handle.seek(0)
    except BaseException:
        handle.close()
        raise
    return handle


def _spawn(
    cmd: str,
    payload: bytes,
    *,
    state_dir: str | os.PathLike[str],
    env: Mapping[str, str],
    gateway: Gateway,
    capture: bool,
    wrapper: tuple[str, str] | None = None,
) -> Any:
    """Start ``/bin/sh -c cmd`` (or, with ``wrapper=(timeout, grace)``, the async wrapper that
    gets ``cmd`` as its ``$1``)."""
    stdin = _stdin_file(payload, state_dir)
    argv = (
        [_SH, "-c", cmd]
        if wrapper is None
        else [_SH, "-c", _ASYNC_WRAPPER, "agent-spawn-hook", cmd, *wrapper]
    )
    try:
        return gateway.spawn(
            argv,
            stdin=stdin,
            stdout=PIPE if capture else DEVNULL,
            stderr=PIPE if capture else DEVNULL,
            env=env,
            cwd=str(state_dir),
            start_new_session=True,
        )
    finally:
        stdin.close()


def _read_capped(
    selector: selectors.BaseSelector,
    buffers: dict[str, bytearray],
    truncated: dict[str, bool],
    wait: float,
) -> None:
    for key, _ in selector.select(wait):
        name: str = key.data
        try:
            chunk = os.read(key.fd, 65536)
        except BlockingIOError:
            continue
        except OSError:
            chunk = b""
        if not chunk:
            selector.unregister(key.fileobj)
            continue
        room = CAPTURE_LIMIT - len(buffers[name])
        if room > 0:
            buffers[name].extend(chunk[:room])
        if len(chunk) > room:
            truncated[name] = True


def _kill_quietly(child: Any) -> None:
    """SIGKILL the leader through its ``Popen`` (safe while unreaped); never raises."""
    try:
        if child.poll() is None:
            child.kill()
    except Exception:  # noqa: BLE001 - best effort
        pass


def _stop(child: Any, group: _Group | None, state_dir: Any, hook: str | None) -> None:
    """Stop a hook after an error: its verified group first, then the leader itself."""
    if group is not None:
        try:
            if not group.leader_done():
                group.terminate(state_dir, hook)
        except Exception:  # noqa: BLE001 - best effort, the leader is killed below
            pass
    _kill_quietly(child)


def run_hook_process(
    cmd: str,
    payload: bytes,
    *,
    hook: str,
    agent_id: str | None,
    timeout: float,
    state_dir: str | os.PathLike[str],
    environ: Mapping[str, str] | None = None,
    gateway: Gateway | None = None,
) -> ProcessResult:
    """Run one hook to completion (or ``timeout``) and return what happened. Never raises for a
    failure of the hook itself: a spawn error is ``error``, a timeout kills the hook's group."""
    gw = gateway or get_gateway()
    env = hook_environment(environ, hook, agent_id)
    result = ProcessResult()
    if "\0" in cmd:
        result.error = "the command contains a NUL byte"
        return result
    started = gw.monotonic()
    try:
        child = _spawn(cmd, payload, state_dir=state_dir, env=env, gateway=gw, capture=True)
    except (OSError, ValueError) as exc:
        result.error = f"could not start the hook: {exc.__class__.__name__}"
        return result
    group: _Group | None = None
    selector: selectors.BaseSelector | None = None
    buffers = {"stdout": bytearray(), "stderr": bytearray()}
    truncated = {"stdout": False, "stderr": False}
    clean = False
    try:
        group = _Group(child, gw)
        selector = selectors.DefaultSelector()
        for name, stream in (("stdout", child.stdout), ("stderr", child.stderr)):
            os.set_blocking(stream.fileno(), False)
            selector.register(stream, selectors.EVENT_READ, name)
        deadline = started + max(timeout, 0.0)
        done_at: float | None = None
        while True:
            now = gw.monotonic()
            if done_at is None and group.leader_done():
                done_at = now  # checked before the deadline: a hook that finished is not late
            if done_at is None and now >= deadline:
                result.timed_out = True
                break
            if done_at is not None and (not selector.get_map() or now - done_at >= DRAIN_GRACE):
                break
            _read_capped(selector, buffers, truncated, min(POLL, max(deadline - now, 0.0)))
        if result.timed_out:
            group.terminate(state_dir, hook)
            end = gw.monotonic() + 0.2
            while selector.get_map() and gw.monotonic() < end:
                _read_capped(selector, buffers, truncated, POLL)
        clean = True
    except Exception as exc:  # noqa: BLE001 - supervision failed: report it, clean up below
        result.error = f"the hook could not be supervised: {exc.__class__.__name__}"
    finally:
        if not clean:
            _stop(child, group, state_dir, hook)  # an exception is leaving: leave no hook behind
        code: int | None
        try:
            code = child.wait(timeout=REAP_TIMEOUT)
        except Exception:  # noqa: BLE001 - a hook stuck in the kernel is reported, not waited for
            code = None
            result.unreaped = True
            result.error = result.error or "the hook did not exit and could not be reaped"
            _kill_quietly(child)
        if selector is not None:
            selector.close()
        for stream in (child.stdout, child.stderr):
            if stream is not None:
                stream.close()
    result.duration = gw.monotonic() - started
    result.stdout = bytes(buffers["stdout"])
    result.stderr = bytes(buffers["stderr"])
    result.stdout_truncated = truncated["stdout"]
    result.stderr_truncated = truncated["stderr"]
    if code is not None:
        if code < 0:
            result.signal = -code
        else:
            result.exit_code = code
    return result


# --- asynchronous hooks --------------------------------------------------------------------------


@dataclass
class _Async:
    group: _Group
    hook: str
    agent_id: str | None
    deadline: float


@dataclass
class AsyncRunner:
    """Tracks the ``async: true`` hooks this process started.

    An async hook is started detached (own process group, stdout and stderr discarded) under a
    small ``sh`` wrapper (``_ASYNC_WRAPPER``) and its event counts as delivered once it started.
    The wrapper leads the group and starts a watchdog in it: after the timeout (rounded up to
    whole seconds) the watchdog sends SIGTERM to its own group and SIGKILL after a grace, and the
    wrapper stops the watchdog when the command ends. So the timeout holds even after the
    process that started the hook has exited. While this process lives, ``poll`` additionally
    reaps finished hooks (logging failures) and kills overdue ones the same verified way as a
    synchronous timeout (a backstop with the exact deadline). At most ``MAX_ASYNC`` hooks run at
    once: ``start`` waits up to ``ASYNC_CAP_WAIT`` for a slot and then refuses.

    The watchdog is a plain ``sh`` subshell: it ignores SIGTERM (so the group's SIGTERM does not
    stop it before the SIGKILL) and stops on SIGUSR1; it needs no GNU ``timeout``.
    """

    state_dir: str | os.PathLike[str]
    gateway: Gateway | None = None
    running: list[_Async] = field(default_factory=lambda: [])

    def start(
        self,
        cmd: str,
        payload: bytes,
        *,
        hook: str,
        agent_id: str | None,
        timeout: float,
        environ: Mapping[str, str] | None = None,
    ) -> str | None:
        """Start the hook; returns an error text, or None when it started."""
        gw = self.gateway or get_gateway()
        if "\0" in cmd:
            return "the command contains a NUL byte"
        env = hook_environment(environ, hook, agent_id)
        end = gw.monotonic() + ASYNC_CAP_WAIT
        while self.poll() >= MAX_ASYNC:
            if gw.monotonic() >= end:
                return f"too many async hooks are running ({MAX_ASYNC})"
            gw.sleep(0.02)
        seconds = max(1, math.ceil(max(timeout, 0.0)))
        try:
            child = _spawn(
                cmd,
                payload,
                state_dir=self.state_dir,
                env=env,
                gateway=gw,
                capture=False,
                wrapper=(str(seconds), str(WATCHDOG_GRACE)),
            )
        except (OSError, ValueError) as exc:
            return f"could not start the hook: {exc.__class__.__name__}"
        try:
            self.running.append(
                _Async(_Group(child, gw), hook, agent_id, gw.monotonic() + max(timeout, 0.0))
            )
        except Exception as exc:  # noqa: BLE001 - never leave a started hook unattended
            _kill_quietly(child)
            return f"could not track the hook: {exc.__class__.__name__}"
        return None

    def poll(self) -> int:
        """Reap finished hooks, kill overdue ones; returns how many still run."""
        gw = self.gateway or get_gateway()
        still: list[_Async] = []
        for item in self.running:
            child = item.group.child
            if child.poll() is not None:
                code: int = child.returncode
                if code != 0 and gw.monotonic() >= item.deadline and code in _KILL_CODES:
                    # The watchdog fired before this poll: the wrapper died of its SIGTERM/SIGKILL
                    # (negative code) or the command did and ``exit "$rc"`` passed 128 + signal on.
                    _log(
                        self.state_dir,
                        "hook_timeout",
                        level="warn",
                        agent_id=item.agent_id,
                        hook=item.hook,
                        message="async hook timed out",
                        details={"async": True},
                    )
                elif code != 0:
                    _log(
                        self.state_dir,
                        "hook_failed",
                        level="warn",
                        agent_id=item.agent_id,
                        hook=item.hook,
                        message="async hook failed",
                        details={
                            "exit_code": code if code > 0 else None,
                            "async": True,
                            "signal": -code if code < 0 else (code - 128 if code > 128 else None),
                        },
                    )
            elif gw.monotonic() >= item.deadline:
                item.group.terminate(self.state_dir, item.hook)
                try:
                    child.wait(timeout=REAP_TIMEOUT)
                except Exception:  # noqa: BLE001 - reported below, never waited for
                    pass
                _log(
                    self.state_dir,
                    "hook_timeout",
                    level="warn",
                    agent_id=item.agent_id,
                    hook=item.hook,
                    message="async hook timed out",
                    details={"async": True},
                )
            else:
                still.append(item)
        self.running = still
        return len(still)

    def wait(self, timeout: float) -> int:
        """``poll`` until nothing runs or ``timeout`` passed; returns how many still run."""
        gw = self.gateway or get_gateway()
        end = gw.monotonic() + timeout
        while self.poll() and gw.monotonic() < end:
            gw.sleep(0.02)
        return self.poll()


# --- pre-hooks -----------------------------------------------------------------------------------


def _decision_ids(stdout: bytes) -> list[str]:
    """``decision_id`` (a string or a list of strings) from a hook's JSON stdout; anything that
    is not valid is ignored."""
    try:
        data = json.loads(stdout.decode("utf-8", errors="replace"))
    except (ValueError, RecursionError):  # hostile nesting is just invalid output
        return []
    if not isinstance(data, dict):
        return []
    raw = data.get("decision_id")  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    items: list[object] = list(raw) if isinstance(raw, list) else [raw]  # pyright: ignore[reportUnknownArgumentType]
    found: list[str] = []
    for item in items[:MAX_DECISION_IDS]:
        if isinstance(item, str) and _DECISION_ID.fullmatch(item):
            found.append(item)
    return found


@dataclass(frozen=True)
class PreHookVerdict:
    """The outcome of a pre-hook run: ``ok``, ``veto`` (exit 11) or ``escalate`` (exit 13).

    ``hook`` is the hook name, ``stderr`` the deciding hook's stderr (untrusted; cut to 4096
    bytes by ``CliError``), ``cause`` how it was decided and ``decision_ids`` every valid
    ``decision_id`` any hook that ran printed (the caller records them).
    """

    verdict: Literal["ok", "veto", "escalate"]
    hook: str
    stderr: bytes = b""
    cause: Literal["exit", "timeout", "signal", "spawn"] | None = None
    exit_code: int | None = None
    decision_ids: tuple[str, ...] = ()


def run_pre_hooks(
    hook: str,
    body: Mapping[str, Any],
    entries: Sequence[HookEntry],
    *,
    state_dir: str | os.PathLike[str],
    agent_id: str | None = None,
    environ: Mapping[str, str] | None = None,
    gateway: Gateway | None = None,
) -> PreHookVerdict:
    """Run the matching entries in order (design guide section 8).

    Exit 0 passes; exit 13 escalates but the remaining hooks still run; any other non-zero
    exit, a signal death, a timeout or a spawn failure vetoes at once (later hooks do not run).
    ``body`` is the payload without ``tool``, ``hook``, ``object_type``, ``object_id``,
    ``event_id`` and ``ts``. A pre-hook runs once and is never redelivered, so its ``event_id``
    is a random positive integer below 2**53 (the schema requires one; there is no outbox row).
    """
    if hook not in PRE_HOOKS:
        raise ValueError(f"{hook!r} is not a pre-hook")
    payload = finalize_payload(
        hook,
        body,
        event_id=secrets.randbelow(2**53 - 1) + 1,
        ts=records.format_ts(datetime.now(UTC)),
        agent_id=agent_id,
        redact=False,
    )
    data = _encode_payload(payload)
    decisions: list[str] = []
    escalated: PreHookVerdict | None = None
    for index, entry in enumerate(entries):
        if not entry_matches(entry, payload):
            continue
        result = run_hook_process(
            entry.cmd,
            data,
            hook=hook,
            agent_id=agent_id,
            timeout=entry.timeout,
            state_dir=state_dir,
            environ=environ,
            gateway=gateway,
        )
        decisions.extend(_decision_ids(result.stdout))
        decided = tuple(dict.fromkeys(decisions))
        details = _result_details(index, entry, result)
        if result.ok:
            _log(state_dir, "hook_delivered", agent_id=agent_id, hook=hook, details=details)
            continue
        if result.exit_code == ESCALATE_EXIT and not result.timed_out:
            _log(
                state_dir,
                "hook_escalate",
                level="warn",
                agent_id=agent_id,
                hook=hook,
                details=details,
            )
            if escalated is None:
                escalated = PreHookVerdict(
                    "escalate", hook, result.stderr, "exit", ESCALATE_EXIT, decided
                )
            continue
        cause: Literal["exit", "timeout", "signal", "spawn"] = (
            "timeout"
            if result.timed_out
            else "spawn"
            if result.error is not None
            else "signal"
            if result.exit_code is None
            else "exit"
        )
        _log(
            state_dir,
            "hook_timeout" if result.timed_out else "hook_veto",
            level="warn",
            agent_id=agent_id,
            hook=hook,
            details=details,
        )
        stderr = result.stderr or (result.error or "").encode()
        return PreHookVerdict("veto", hook, stderr, cause, result.exit_code, decided)
    decided = tuple(dict.fromkeys(decisions))
    if escalated is not None:
        return PreHookVerdict("escalate", hook, escalated.stderr, "exit", ESCALATE_EXIT, decided)
    return PreHookVerdict("ok", hook, decision_ids=decided)


def verdict_error(verdict: PreHookVerdict, *, agent_id: str | None = None) -> CliError | None:
    """The error envelope of a veto (exit 11) or an escalation (exit 13); None when it passed.

    Left out on purpose: ``decision_ids``. The envelope then carries the command's own
    ``--decision-id`` values; the caller adds ``verdict.decision_ids`` to what it records.
    """
    if verdict.verdict == "ok":
        return None
    if verdict.verdict == "veto":
        message = f"the {verdict.hook} hook vetoed"
        if verdict.cause == "timeout":
            message = f"the {verdict.hook} hook timed out (a timeout vetoes)"
        return CliError.of(
            "vetoed", message, agent_id=agent_id, hook=verdict.hook, hook_stderr=verdict.stderr
        )
    return CliError.of(
        "escalated",
        f"the {verdict.hook} hook escalated",
        agent_id=agent_id,
        hook=verdict.hook,
        hook_stderr=verdict.stderr,
    )


def _result_details(index: int, entry: HookEntry, result: ProcessResult) -> dict[str, Any]:
    """Log details of one run: never the command (it may hold a credential) or the payload."""
    details: dict[str, Any] = {
        "entry": index,
        "source": entry.source,
        "async": entry.async_,
        "exit_code": result.exit_code,
        "duration": round(result.duration, 3),
    }
    if result.signal is not None:
        details["signal"] = result.signal
    if result.timed_out:
        details["timeout"] = entry.timeout
    if result.error is not None:
        details["error"] = result.error
    if result.stderr:
        details["stderr"] = cut_utf8(result.stderr, LOG_TEXT_LIMIT)
    return details


# --- outbox delivery -----------------------------------------------------------------------------


@dataclass
class DeliveryResult:
    """What one ``deliver_outbox`` call did."""

    delivered: int = 0  # events marked delivered by this call
    hooks_run: int = 0
    hooks_failed: int = 0
    lease_held_elsewhere: bool = False  # another live process delivers; its events are its own
    error: str | None = None  # a store failure that stopped delivery (logged, not raised)


def _invocation_source(
    store: Store, agent_id: str, agent: Mapping[str, Any], hook: str, payload: Mapping[str, Any]
) -> object:
    if hook in _TEARDOWN_HOOKS:
        return agent.get("hooks")
    run: object = payload.get("run")
    if not isinstance(run, int) or isinstance(run, bool):
        outcome = payload.get("outcome")
        run = outcome.get("run") if isinstance(outcome, dict) else None  # pyright: ignore[reportUnknownMemberType]
    if not isinstance(run, int) or isinstance(run, bool):
        snapshot = payload.get(
            "agent"
        )  # the stored snapshot, not the live row (the live row may have moved on)
        run = snapshot.get("run") if isinstance(snapshot, dict) else None  # pyright: ignore[reportUnknownMemberType]
    if not isinstance(run, int) or isinstance(run, bool):
        return None
    row = store.get_run(agent_id, run)
    return None if row is None else row.get("hooks")


class _Deliverer:
    def __init__(
        self,
        store: Store,
        hooks: Hooks,
        state_dir: str | os.PathLike[str],
        runner: AsyncRunner,
        environ: Mapping[str, str] | None,
        gateway: Gateway | None,
    ) -> None:
        self.store = store
        self.hooks = hooks
        self.state_dir = state_dir
        self.runner = runner
        self.environ = environ
        self.gateway = gateway

    def deliver(self, agent_id: str, event: Mapping[str, Any], result: DeliveryResult) -> None:
        hook = event["event"]
        stored = event["payload"]
        if hook not in POST_HOOKS or not isinstance(stored, dict):
            _log(
                self.state_dir,
                "hook_failed",
                level="warn",
                agent_id=agent_id,
                message="malformed outbox event skipped",
                details={"event_id": event["event_id"]},
            )
            return
        try:
            payload = finalize_payload(
                hook,
                stored,  # pyright: ignore[reportUnknownArgumentType]
                event_id=event["event_id"],
                ts=_format_ts(event["created"]),
                agent_id=agent_id,
                redact=True,
            )
            data = _encode_payload(payload)
        except (ValueError, TypeError, RecursionError, OverflowError, OSError):
            _log(
                self.state_dir,
                "hook_failed",
                level="warn",
                agent_id=agent_id,
                hook=hook,
                message="unencodable outbox event skipped",
                details={"event_id": event["event_id"]},
            )
            return
        agent = self.store.get_agent(agent_id) or {}
        entries = config_entries(self.hooks, hook) + invocation_entries(
            _invocation_source(self.store, agent_id, agent, hook, payload), hook
        )
        for index, entry in enumerate(entries):
            if entry_matches(entry, payload):
                self._run(agent_id, hook, index, entry, data, result)

    def _run(
        self,
        agent_id: str,
        hook: str,
        index: int,
        entry: HookEntry,
        data: bytes,
        result: DeliveryResult,
    ) -> None:
        result.hooks_run += 1
        try:
            outcome = self._execute(agent_id, hook, entry, data)
        except Exception as exc:  # noqa: BLE001 - one hook never aborts the command
            outcome = ProcessResult(error=f"{exc.__class__.__name__} while running the hook")
        # a malformed id (never stored by this tool) must not make the log record itself fail
        logged_id = agent_id if _AGENT_ID.fullmatch(agent_id) else None
        details = _result_details(index, entry, outcome)
        if outcome.ok:
            _log(self.state_dir, "hook_delivered", agent_id=logged_id, hook=hook, details=details)
            return
        result.hooks_failed += 1
        _log(
            self.state_dir,
            "hook_timeout" if outcome.timed_out else "hook_failed",
            level="warn",
            agent_id=logged_id,
            hook=hook,
            message="hook timed out" if outcome.timed_out else "hook failed",
            details=details,
        )

    def _execute(self, agent_id: str, hook: str, entry: HookEntry, data: bytes) -> ProcessResult:
        if entry.async_ and entry.source == "config":
            error = self.runner.start(
                entry.cmd,
                data,
                hook=hook,
                agent_id=agent_id,
                timeout=entry.timeout,
                environ=self.environ,
            )
            return ProcessResult(exit_code=0 if error is None else None, error=error)
        return run_hook_process(
            entry.cmd,
            data,
            hook=hook,
            agent_id=agent_id,
            timeout=entry.timeout,
            state_dir=self.state_dir,
            environ=self.environ,
            gateway=self.gateway,
        )


def deliver_outbox(
    store: Store,
    agent_id: str,
    hooks: Hooks,
    *,
    state_dir: str | os.PathLike[str] | None = None,
    runner: AsyncRunner | None = None,
    environ: Mapping[str, str] | None = None,
    gateway: Gateway | None = None,
    rounds: int = 10,
) -> DeliveryResult:
    """Deliver the agent's undelivered events in ``event_id`` order under ``delivery:<id>``.

    The holder runs, for each event, every matching hook (config entries in config order, then
    the run's ``--hook`` entries), then marks the event delivered; a hook that fails, times out
    or cannot start is logged and does not stop the rest, and the event still counts as
    delivered (a failing hook is not retried). A process killed between running a hook and
    marking the event leaves it undelivered: the next deliverer runs it again (at least once).

    A process that cannot take the lease returns at once and leaves its events to the holder.
    After releasing, the holder looks once more: an event queued while it was finishing is
    delivered by it (up to ``rounds`` times) rather than waiting for the next deliverer.
    ``async: true`` hooks start under ``runner`` (a private ``AsyncRunner`` when none is given,
    left running when this returns, bounded by their watchdog); their event counts as delivered
    once they started. Read-only commands (``list``, ``show``, ``wait``) must not call this.
    """
    result = DeliveryResult()
    directory = state_dir if state_dir is not None else store.path.parent
    if runner is None:  # async hooks need a runner; a private one is bounded by the watchdog
        runner = AsyncRunner(directory, gateway=gateway)
    deliverer = _Deliverer(store, hooks, directory, runner, environ, gateway)
    key = f"delivery:{agent_id}"
    try:
        for _ in range(rounds):
            try:
                lease = store.acquire_lease(key)
            except LeaseHeld:
                result.lease_held_elsewhere = True
                return result
            try:
                while True:
                    pending = store.pending_events(agent_id, limit=100)
                    if not pending:
                        break
                    for event in pending:
                        deliverer.deliver(agent_id, event, result)
                        try:
                            runner.poll()  # reap async hooks that finished meanwhile
                        except Exception:  # noqa: BLE001 - reaping never aborts delivery
                            pass
                        if store.mark_delivered(lease, agent_id, event["event_id"]):
                            result.delivered += 1
            finally:
                store.release_lease(lease)
            if not store.pending_events(agent_id, limit=1):
                break
    except LeaseLost:
        result.error = "the delivery lease was lost"
    except StoreError as exc:
        result.error = f"{exc.__class__.__name__}: {exc}"
    if result.error is not None:
        _log(
            directory,
            "warning",
            level="warn",
            agent_id=agent_id,
            message="hook delivery stopped",
            details={"error": cut_utf8(result.error, LOG_TEXT_LIMIT)},
        )
    return result
