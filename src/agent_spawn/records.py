"""``log.jsonl`` and ``audit.jsonl``: append-only, private, one JSON object per line.

Records follow ``schemas/log.json`` and ``schemas/audit.json``. Writers take only the fields those
schemas define (keyword arguments, no free-form extras), so secrets, environment values and
prompt text have no field to travel in; ``message`` and ``details`` hold short event text that
callers keep clean. Every append is a single ``write`` of one complete line under an exclusive
``flock``, so concurrent processes never interleave records; files are opened without following
symlinks and kept at 0600.

Nothing here logs on its own: callers decide what to record (the store does no logging).
"""

from __future__ import annotations

import fcntl
import json
import os
import re
import stat
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

__all__ = [
    "AUDIT_ACTIONS",
    "AUDIT_INPUT_KEYS",
    "AUDIT_RESULTS",
    "LOG_EVENTS",
    "LOG_LEVELS",
    "MAX_RECORD_BYTES",
    "AuditRead",
    "RecordsError",
    "append_audit",
    "append_log",
    "format_ts",
    "read_audit",
]

DIR_MODE = 0o700
FILE_MODE = 0o600
LOG_NAME = "log.jsonl"
AUDIT_NAME = "audit.jsonl"
MAX_RECORD_BYTES = 256 * 1024

LOG_LEVELS = frozenset({"debug", "info", "warn", "error"})
LOG_EVENTS = frozenset(
    """launch resume fork adopt holder_added status_change attention_set attention_cleared ack
    stop label_change relay_event relay_event_ignored promotion auto_report_resume
    third_party_refused classification_mismatch classification_unknown fork_abandoned teardown
    teardown_retry prune reconcile_change handoff_failed handoff_rejected signal_sent interrupted
    hook_delivered hook_failed hook_timeout hook_veto hook_escalate schema_older warning
    migration""".split()
)
AUDIT_ACTIONS = frozenset(
    """teardown stop launch_escalation_override resume_escalation_override
    teardown_escalation_override resume_tainted""".split()
)
AUDIT_RESULTS = frozenset({"ok", "refused", "error"})
AUDIT_INPUT_KEYS = frozenset({"labels", "repo", "worktree", "csession_id", "merged_head", "force"})
_AUDIT_KEYS = frozenset({"ts", "actor", "decision_id", "target", "action", "inputs", "result"})
_HOOKS = frozenset(
    {
        "pre-launch",
        "pre-teardown",
        "post-launch",
        "post-state-change",
        "post-escalate",
        "post-park",
        "post-teardown",
    }
)

_AGENT_ID = re.compile(r"ag-[a-z2-7]{8}")
_CSESSION_ID = re.compile(r"cs-[a-z2-7]{8}")
_WT_ID = re.compile(r"wt-[a-z2-7]{8}")
_ACTOR = re.compile(r"(?:human|auto):[A-Za-z0-9._@-]{1,64}")
_DECISION_ID = re.compile(r"[A-Za-z0-9._:@/-]{1,128}")
_LABEL = re.compile(r"[a-z0-9][a-z0-9_.-]{0,63}=[A-Za-z0-9._:/#@+=,-]{0,256}")
_SHA = re.compile(r"[0-9a-f]{40}")
_ABS_PATH = re.compile(r"/[^\n]*")


class RecordsError(Exception):
    """A log or audit file that cannot be used safely (symlink, wrong owner, I/O failure)."""


def format_ts(moment: datetime) -> str:
    """UTC with millisecond precision and a ``Z``, as the schemas' ``date-time`` expects."""
    return moment.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _parse_ts(text: str) -> datetime:
    moment = datetime.fromisoformat(text)
    return moment.replace(tzinfo=UTC) if moment.tzinfo is None else moment.astimezone(UTC)


# --- validation ----------------------------------------------------------------------------------


def _fail(message: str) -> ValueError:
    return ValueError(message)


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _match(pattern: re.Pattern[str], value: object, what: str) -> None:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise _fail(f"invalid {what}")


def _optional(pattern: re.Pattern[str], value: object, what: str) -> None:
    if value is not None:
        _match(pattern, value, what)


def _check_decision_ids(value: object) -> None:
    if not isinstance(value, list | tuple) or len(value) > 32:  # pyright: ignore[reportUnknownArgumentType]
        raise _fail("invalid decision ids")
    for item in value:  # pyright: ignore[reportUnknownVariableType]
        _match(_DECISION_ID, item, "decision id")


def _encode(record: Mapping[str, Any]) -> bytes:
    try:
        text = json.dumps(record, ensure_ascii=True, allow_nan=False, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise _fail(f"record is not JSON: {exc}") from None
    line = text.encode("ascii") + b"\n"
    if len(line) > MAX_RECORD_BYTES:
        raise _fail("record too large")
    return line


def _check_log(record: Mapping[str, Any]) -> None:
    if record["event"] not in LOG_EVENTS:
        raise _fail(f"unknown log event {record['event']!r}")
    if record["level"] not in LOG_LEVELS:
        raise _fail(f"unknown log level {record['level']!r}")
    _optional(_AGENT_ID, record["agent_id"], "agent id")
    _optional(_CSESSION_ID, record.get("csession_id"), "session id")
    _optional(_ACTOR, record.get("actor"), "actor")
    run = record.get("run")
    if run is not None and (not _is_int(run) or run < 1):
        raise _fail("invalid run")
    if "decision_ids" in record:
        _check_decision_ids(record["decision_ids"])
    hook = record.get("hook")
    if hook is not None and hook not in _HOOKS:
        raise _fail(f"unknown hook {hook!r}")
    pid = record.get("pid")
    if pid is not None and not _is_int(pid):
        raise _fail("invalid pid")
    for key in ("signal", "message"):
        if record.get(key) is not None and not isinstance(record[key], str):
            raise _fail(f"invalid {key}")
    if "details" in record and not isinstance(record["details"], dict):
        raise _fail("details must be an object")


def _check_inputs(inputs: object) -> None:
    if not isinstance(inputs, dict) or set(inputs) != AUDIT_INPUT_KEYS:  # pyright: ignore[reportUnknownArgumentType]
        raise _fail(
            "audit inputs need exactly labels, repo, worktree, csession_id, merged_head, force"
        )
    labels = inputs["labels"]
    if not isinstance(labels, list | tuple) or len(labels) > 64:  # pyright: ignore[reportUnknownArgumentType]
        raise _fail("invalid labels")
    for label in labels:  # pyright: ignore[reportUnknownVariableType]
        _match(_LABEL, label, "label")
    repo = inputs["repo"]
    if repo is not None and (not isinstance(repo, str) or not repo or "\n" in repo):
        raise _fail("invalid repo")
    worktree = inputs["worktree"]
    if worktree is not None:
        if not isinstance(worktree, dict) or set(worktree) != {"record_id", "path", "owned"}:  # pyright: ignore[reportUnknownArgumentType]
            raise _fail("invalid worktree")
        _match(_WT_ID, worktree["record_id"], "worktree record id")
        _match(_ABS_PATH, worktree["path"], "worktree path")
        if not isinstance(worktree["owned"], bool):
            raise _fail("invalid worktree owned")
    _optional(_CSESSION_ID, inputs["csession_id"], "session id")
    _optional(_SHA, inputs["merged_head"], "merged head")
    if not isinstance(inputs["force"], bool):
        raise _fail("force must be a boolean")


def check_audit(record: Mapping[str, Any]) -> None:
    """Raise ``ValueError`` unless ``record`` has the shape of ``schemas/audit.json``."""
    if set(record) != _AUDIT_KEYS:
        raise _fail("audit record needs exactly " + ", ".join(sorted(_AUDIT_KEYS)))
    ts = record["ts"]
    if not isinstance(ts, str):
        raise _fail("invalid ts")
    try:
        _parse_ts(ts)
    except ValueError:
        raise _fail("invalid ts") from None
    _match(_ACTOR, record["actor"], "actor")
    _check_decision_ids(record["decision_id"])
    _match(_AGENT_ID, record["target"], "target")
    if record["action"] not in AUDIT_ACTIONS:
        raise _fail(f"unknown audit action {record['action']!r}")
    if record["result"] not in AUDIT_RESULTS:
        raise _fail(f"unknown audit result {record['result']!r}")
    _check_inputs(record["inputs"])


# --- appending -----------------------------------------------------------------------------------


def append_log(
    state_dir: str | os.PathLike[str],
    event: str,
    *,
    level: str = "info",
    agent_id: str | None = None,
    csession_id: str | None = None,
    run: int | None = None,
    actor: str | None = None,
    decision_ids: Sequence[str] | None = None,
    hook: str | None = None,
    pid: int | None = None,
    signal: str | None = None,
    message: str | None = None,
    details: Mapping[str, Any] | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Append one ``log.jsonl`` record and return it. Fields left ``None`` are omitted."""
    record: dict[str, Any] = {
        "ts": format_ts(now or datetime.now(UTC)),
        "level": level,
        "event": event,
        "agent_id": agent_id,
    }
    optional: dict[str, Any] = {
        "csession_id": csession_id,
        "run": run,
        "actor": actor,
        "decision_ids": None if decision_ids is None else list(decision_ids),
        "hook": hook,
        "pid": pid,
        "signal": signal,
        "message": message,
        "details": None if details is None else dict(details),
    }
    record.update({key: value for key, value in optional.items() if value is not None})
    _check_log(record)
    _append(Path(state_dir), LOG_NAME, _encode(record))
    return record


def append_audit(
    state_dir: str | os.PathLike[str],
    *,
    actor: str,
    decision_ids: Sequence[str],
    target: str,
    action: str,
    inputs: Mapping[str, Any],
    result: str,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Append one ``audit.jsonl`` record and return it."""
    if isinstance(decision_ids, str):
        raise _fail("decision ids must be a list")
    record: dict[str, Any] = {
        "ts": format_ts(now or datetime.now(UTC)),
        "actor": actor,
        "decision_id": list(decision_ids),
        "target": target,
        "action": action,
        "inputs": dict(inputs),
        "result": result,
    }
    check_audit(record)
    _append(Path(state_dir), AUDIT_NAME, _encode(record))
    return record


_APPEND_FLAGS = os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC
_READ_FLAGS = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC


def _open_checked(path: Path, flags: int) -> int:
    """Open ``path`` without following a link; it must be a regular file of this user, one link."""
    try:
        fd = os.open(path, flags, FILE_MODE)
    except OSError as exc:
        if exc.errno == 40:  # ELOOP
            raise RecordsError(f"{path} is a symbolic link: refusing to follow it") from exc
        raise
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise RecordsError(f"{path} is not a regular file")
        if info.st_uid != os.geteuid():
            raise RecordsError(f"{path} is not owned by this user")
        if info.st_nlink != 1:
            raise RecordsError(f"{path} has more than one link")
        if flags & os.O_RDWR and stat.S_IMODE(info.st_mode) & 0o077:
            os.fchmod(fd, FILE_MODE)
    except BaseException:
        os.close(fd)
        raise
    return fd


def _append(directory: Path, name: str, line: bytes) -> None:
    try:
        os.makedirs(directory, mode=DIR_MODE, exist_ok=True)
        fd = _open_checked(directory / name, _APPEND_FLAGS)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            size = os.fstat(fd).st_size
            # A crash can leave a line without its newline; start ours on a fresh line.
            if size and os.pread(fd, 1, size - 1) != b"\n":
                line = b"\n" + line
            view = memoryview(line)
            while view:
                view = view[os.write(fd, view) :]
        finally:
            os.close(fd)  # drops the lock
    except OSError as exc:
        raise RecordsError(f"cannot append to {directory / name}: {exc}") from exc


# --- reading -------------------------------------------------------------------------------------


@dataclass(frozen=True)
class AuditRead:
    """Valid audit records oldest first, and how many lines were not records."""

    records: list[dict[str, Any]]
    skipped: int


def _read_lines(path: Path) -> Iterator[bytes]:
    try:
        fd = _open_checked(path, _READ_FLAGS)
    except FileNotFoundError:
        return
    except OSError as exc:
        if isinstance(exc, RecordsError):
            raise
        raise RecordsError(f"cannot read {path}: {exc}") from exc
    try:
        fcntl.flock(fd, fcntl.LOCK_SH)
        with os.fdopen(fd, "rb", closefd=False) as handle:
            yield from handle
    except OSError as exc:
        raise RecordsError(f"cannot read {path}: {exc}") from exc
    finally:
        os.close(fd)


def read_audit(state_dir: str | os.PathLike[str]) -> AuditRead:
    """Every valid record of ``audit.jsonl``, oldest first (ties keep file order).

    A missing file or directory reads as empty and is not created. Blank lines are ignored;
    lines that are torn, not JSON or not shaped like an audit record are counted in ``skipped``.
    """
    found: list[tuple[datetime, dict[str, Any]]] = []
    skipped = 0
    for raw in _read_lines(Path(state_dir) / AUDIT_NAME):
        if not raw.strip():
            continue
        try:
            record = json.loads(raw.decode("utf-8"))
            if not isinstance(record, dict):
                raise ValueError("not an object")
            check_audit(record)  # pyright: ignore[reportUnknownArgumentType]
            found.append((_parse_ts(record["ts"]), record))  # pyright: ignore[reportUnknownArgumentType]
        except (ValueError, KeyError, TypeError):
            skipped += 1
    found.sort(key=lambda item: item[0])
    return AuditRead([record for _, record in found], skipped)


def parse_record_ts(text: str) -> datetime:
    """The instant of a record's ``ts`` (already checked by :func:`check_audit`)."""
    return _parse_ts(text)
