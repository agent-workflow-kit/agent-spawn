"""Exit codes, the error envelope, result objects and ``--result-file``.

Everything a command prints goes through here so that the output-by-exit-code table, the error
envelope (``schemas/cli/error.json``) and the result file (exactly what ``--json`` prints) cannot
drift apart. Handlers return an :class:`Outcome` or raise a :class:`CliError`; they never print.
"""

from __future__ import annotations

import copy
import enum
import json
import os
import re
import secrets
import stat
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, TextIO

# --- exit codes -------------------------------------------------------------------------------

EXIT_OK = 0
EXIT_INTERNAL = 1
EXIT_USAGE = 2
EXIT_NOT_FOUND = 10
EXIT_VETOED = 11
EXIT_FAILED = 12
EXIT_ESCALATED = 13
EXIT_REFUSED = 20
EXIT_AMBIGUOUS = 21
EXIT_TIMED_OUT = 22
EXIT_SIGINT = 130
EXIT_SIGTERM = 143

ERROR_CODES: tuple[str, ...] = (
    "usage",
    "config_invalid",
    "not_found",
    "vetoed",
    "escalated",
    "confirmation_required",
    "refused",
    "ambiguous",
    "tool_missing",
    "handoff_failed",
    "no_window",
    "no_run",
    "worktree_gone",
    "partial",
    "launch_failed",
    "failed",
    "schema_newer",
    "internal",
)

DEFAULT_EXIT: dict[str, int] = {
    "usage": EXIT_USAGE,
    "config_invalid": EXIT_USAGE,
    "not_found": EXIT_NOT_FOUND,
    "vetoed": EXIT_VETOED,
    "escalated": EXIT_ESCALATED,
    "confirmation_required": EXIT_ESCALATED,
    "refused": EXIT_REFUSED,
    "ambiguous": EXIT_AMBIGUOUS,
    "tool_missing": EXIT_FAILED,
    "handoff_failed": EXIT_FAILED,
    "no_window": EXIT_FAILED,
    "no_run": EXIT_FAILED,
    "worktree_gone": EXIT_FAILED,
    "partial": EXIT_FAILED,
    "launch_failed": EXIT_FAILED,
    "failed": EXIT_FAILED,
    "schema_newer": EXIT_FAILED,
    "internal": EXIT_INTERNAL,
}

# Exit codes an error envelope may accompany (interface.md, Output by exit code).
ENVELOPE_EXITS = frozenset({1, 2, 10, 11, 12, 13, 20, 21, 130, 143})

TOOLS = ("agent-spawn", "claude-session", "termctl", "wtctl")
LOWER_TOOLS = TOOLS[1:]
HOOKS = ("pre-launch", "pre-teardown")
DETAIL_KEYS = frozenset({"status", "cause", "retryable", "hint"})
HOOK_STDERR_LIMIT = 4096
_REASON = re.compile(r"^[a-z][a-z0-9_]*$")


# --- text helpers -----------------------------------------------------------------------------


def cut_utf8(value: str | bytes, limit: int = HOOK_STDERR_LIMIT) -> str:
    """Return ``value`` cut to at most ``limit`` UTF-8 bytes, never inside a character.

    Invalid input (undecodable bytes, lone surrogates) is replaced with U+FFFD or ``?`` *before*
    the cut, so the result is valid Unicode and its encoding is at most ``limit`` bytes.
    """
    if isinstance(value, bytes):
        text = value.decode("utf-8", errors="replace")
    else:
        text = value.encode("utf-8", errors="replace").decode("utf-8")
    encoded = text.encode("utf-8")
    if len(encoded) <= limit:
        return text
    cut = limit
    # encoded[cut] is the first byte dropped; while it is a continuation byte we are inside a
    # character, so back up to that character's first byte and drop it whole.
    while cut > 0 and (encoded[cut] & 0xC0) == 0x80:
        cut -= 1
    return encoded[:cut].decode("utf-8")


_CONTROL = re.compile("[\x00-\x08\x0b-\x1f\x7f-\x9f]")


def safe_text(text: str) -> str:
    """Make ``text`` safe to print on a terminal.

    Lone surrogates (undecodable argv bytes) and control characters other than tab and newline
    (escape sequences) are shown as ``\\x..`` escapes.
    """
    text = text.encode("utf-8", errors="backslashreplace").decode("utf-8")
    return _CONTROL.sub(lambda match: f"\\x{ord(match.group()):02x}", text)


def render_json(payload: object) -> str:
    """The one-line ASCII JSON that ``--json`` prints and ``--result-file`` stores.

    ASCII keeps stdout valid whatever untrusted text (even lone surrogates) the payload holds.
    """
    return json.dumps(payload, ensure_ascii=True)


# --- the error envelope -----------------------------------------------------------------------


class CliError(Exception):
    """A command failure that is printed as the error envelope with ``exit_code``."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        exit_code: int | None = None,
        reason: str | None = None,
        agent_id: str | None = None,
        tool: str = "agent-spawn",
        lower_code: str | None = None,
        lower_error: Mapping[str, Any] | None = None,
        details: Mapping[str, Any] | None = None,
        candidates: Sequence[str] = (),
        hook: str | None = None,
        hook_stderr: str | bytes | None = None,
        decision_ids: Sequence[str] | None = None,
    ) -> None:
        super().__init__(message)
        if code not in ERROR_CODES:
            raise ValueError(f"unknown error code {code!r}")
        if exit_code is None:
            exit_code = DEFAULT_EXIT[code]
        if exit_code not in ENVELOPE_EXITS:
            raise ValueError(f"exit {exit_code} cannot carry an error envelope")
        if tool not in TOOLS:
            raise ValueError(f"unknown tool {tool!r}")
        if code == "refused":
            if reason is None or not _REASON.match(reason):
                raise ValueError(f"code refused needs a reason matching {_REASON.pattern}")
        elif reason is not None:
            raise ValueError("reason is only set with code refused")
        if tool == "agent-spawn" and lower_code is not None:
            raise ValueError("lower_code must be null when tool is agent-spawn")
        if lower_code is not None and lower_error is None:
            raise ValueError("lower_code needs lower_error")
        if hook is None:
            if hook_stderr is not None:
                raise ValueError("hook_stderr needs hook")
        else:
            if hook not in HOOKS:
                raise ValueError(f"unknown hook {hook!r}")
            if code not in ("vetoed", "escalated") or tool != "agent-spawn":
                raise ValueError("hook is only set for agent-spawn's own vetoed/escalated")
        if details is not None and not set(details) <= DETAIL_KEYS:
            raise ValueError(f"unknown details keys {sorted(set(details) - DETAIL_KEYS)}")
        if tool == "agent-spawn" and code == "ambiguous":
            if len(candidates) < 2:
                raise ValueError("ambiguous needs at least two candidates")
        elif candidates:
            raise ValueError("candidates are only set for agent-spawn's own ambiguous")
        if reason in ("third_party_cwd", "third_party_run") and tool != "agent-spawn":
            raise ValueError("third-party refusals come from agent-spawn")
        cause = (details or {}).get("cause")
        if cause in ("classification_mismatch", "classification_unknown") and (
            code != "launch_failed" or tool != "agent-spawn"
        ):
            raise ValueError("classification causes are agent-spawn's launch_failed")
        if cause == "worktree_state" and (code != "failed" or tool != "agent-spawn"):
            raise ValueError("worktree_state is agent-spawn's failed")
        self.code = code
        self.message = message
        self.exit_code = exit_code
        self.reason = reason
        self.agent_id = agent_id
        self.tool = tool
        self.lower_code = lower_code
        self.lower_error: dict[str, Any] | None = (
            None if lower_error is None else copy.deepcopy(dict(lower_error))
        )
        self.details: dict[str, Any] | None = None if details is None else dict(details)
        self.candidates = list(candidates)
        self.hook = hook
        self.hook_stderr = None if hook_stderr is None else cut_utf8(hook_stderr)
        self.decision_ids: list[str] | None = None if decision_ids is None else list(decision_ids)

    @classmethod
    def of(cls, code: str, message: str, **fields: Any) -> CliError:
        return cls(code, message, **fields)

    @classmethod
    def usage(cls, message: str) -> CliError:
        return cls("usage", message)

    def envelope(self, default_decision_ids: Iterable[str] | None = None) -> dict[str, Any]:
        """The envelope with every field present (``schemas/cli/error.json``)."""
        if self.decision_ids is not None:
            decision_ids = list(self.decision_ids)
        else:
            decision_ids = list(default_decision_ids or ())
        return {
            "error": {
                "code": self.code,
                "message": self.message,
                "reason": self.reason,
                "agent_id": self.agent_id,
                "tool": self.tool,
                "lower_code": self.lower_code,
                "lower_error": copy.deepcopy(self.lower_error),
                "details": None if self.details is None else dict(self.details),
                "candidates": list(self.candidates),
                "hook": self.hook,
                "hook_stderr": self.hook_stderr,
                "decision_ids": decision_ids,
            }
        }


def from_lower_tool(
    tool: str,
    exit_code: int,
    lower_error: object,
    *,
    agent_id: str | None = None,
    decision_ids: Sequence[str] | None = None,
) -> CliError:
    """Map a lower tool's failed call to agent-spawn's envelope (interface.md, Error envelope).

    ``lower_error`` is the lower tool's whole ``error`` object and is carried verbatim. Its text is
    untrusted, so the message is written here and never quotes it.
    """
    if tool not in LOWER_TOOLS:
        raise ValueError(f"{tool!r} is not a lower tool")
    lower = copy.deepcopy(lower_error) if isinstance(lower_error, dict) else None
    raw_code = lower.get("code") if lower is not None else None
    lower_code = raw_code if isinstance(raw_code, str) else None
    exit_override: int | None = None
    reason: str | None = None
    if exit_code == EXIT_NOT_FOUND:
        code, message = "not_found", f"{tool}: not found"
    elif exit_code == EXIT_VETOED:
        code, message = "vetoed", f"{tool} vetoed the request"
    elif exit_code == EXIT_ESCALATED:
        if lower_code == "confirmation_required":
            code, message = "confirmation_required", f"{tool} requires confirmation"
        else:
            code, message = "escalated", f"{tool} escalated the request"
    elif exit_code == EXIT_REFUSED:
        raw_reason = lower.get("reason") if lower is not None else None
        reason = (
            raw_reason if isinstance(raw_reason, str) and _REASON.match(raw_reason) else "unknown"
        )
        code, message = "refused", f"{tool} refused the request: {reason}"
    elif exit_code == EXIT_FAILED:
        if tool == "claude-session" and lower_code == "launch_failed":
            code, message = "launch_failed", f"{tool} could not start the run"
        else:
            code, message = "failed", f"{tool} failed (see lower_error)"
    elif exit_code == EXIT_AMBIGUOUS:
        code, message = "ambiguous", f"{tool} reported an ambiguous selector (see lower_error)"
    elif exit_code == EXIT_INTERNAL and tool == "wtctl" and lower_code == "partial":
        code, message, exit_override = "partial", "wtctl removed only part of the worktree", 12
    elif exit_code == EXIT_USAGE:
        code, message = "usage", f"{tool} rejected the request as a usage error"
    else:
        code, message = "internal", f"{tool} exited with {exit_code}"
    return CliError(
        code,
        message,
        exit_code=exit_override,
        reason=reason,
        agent_id=agent_id,
        tool=tool,
        lower_code=lower_code,
        lower_error=lower,
        decision_ids=decision_ids,
    )


# --- outcomes and the output-by-exit-code table ------------------------------------------------


class PayloadKind(enum.Enum):
    RESULT = "result object"
    COMMAND = "command schema"
    DRY_RUN = "dry-run plan"
    ENVELOPE = "error envelope"
    NONE = "nothing"


def allowed_payload_kinds(exit_code: int) -> frozenset[PayloadKind]:
    """What ``--json`` may print for ``exit_code`` (interface.md, Output by exit code)."""
    kinds = PayloadKind
    if exit_code == EXIT_OK:
        return frozenset({kinds.RESULT, kinds.COMMAND, kinds.DRY_RUN})
    if exit_code in (11, 12, 13):
        return frozenset({kinds.RESULT, kinds.ENVELOPE})
    if exit_code in (14, 15, 16, 17, 18, EXIT_TIMED_OUT):
        return frozenset({kinds.RESULT})
    if exit_code == EXIT_REFUSED:
        return frozenset({kinds.DRY_RUN, kinds.ENVELOPE})
    if exit_code in (EXIT_SIGINT, EXIT_SIGTERM):
        return frozenset({kinds.ENVELOPE, kinds.NONE})
    return frozenset({kinds.ENVELOPE})


@dataclass(frozen=True)
class Outcome:
    """What a command returns: the exit code and what ``--json`` prints for it.

    ``payload`` is None exactly for ``PayloadKind.NONE`` (an interrupted waiter). ``human`` is
    optional text for the default (non ``--json``) output of non-error payloads.
    """

    exit_code: int
    payload: dict[str, Any] | None
    kind: PayloadKind
    human: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in allowed_payload_kinds(self.exit_code):
            raise ValueError(f"exit {self.exit_code} cannot carry a {self.kind.value}")
        if (self.payload is None) != (self.kind is PayloadKind.NONE):
            raise ValueError("payload is None exactly for PayloadKind.NONE")


def error_outcome(error: CliError, default_decision_ids: Iterable[str] | None = None) -> Outcome:
    envelope = error.envelope(default_decision_ids)
    return Outcome(error.exit_code, envelope, PayloadKind.ENVELOPE)


def result_object(
    *,
    agent: Mapping[str, Any] | None,
    run: Mapping[str, Any] | None = None,
    first_working_at: str | None = None,
    timed_out: bool = False,
    warnings: Iterable[str] = (),
) -> dict[str, Any]:
    """The result object of launch, resume and wait (design guide 4.2, schemas/cli/launch.json)."""
    if timed_out and run is not None:
        raise ValueError("a timed_out result has run null")
    return {
        "agent": agent,
        "phase": None if run is None else run["phase"],
        "outcome": None if run is None else run["outcome"],
        "end_reason": None if run is None else run["end_reason"],
        "exit_code": None if run is None else run["exit_code"],
        "run": run,
        "first_working_at": first_working_at,
        "timed_out": timed_out,
        "warnings": list(warnings),
    }


def emit(outcome: Outcome, *, as_json: bool, stdout: TextIO, stderr: TextIO) -> None:
    """Print ``outcome``: errors always on stderr, JSON (or human text) on stdout."""
    payload = outcome.payload
    if outcome.kind is PayloadKind.NONE or payload is None:
        return
    if outcome.kind is PayloadKind.ENVELOPE:
        error = payload["error"]
        line = f"agent-spawn: error: {error['message']}"
        hint = (error.get("details") or {}).get("hint")
        if hint:
            line += f" (hint: {hint})"
        stderr.write(safe_text(line) + "\n")
    if as_json:
        stdout.write(render_json(payload) + "\n")
    elif outcome.kind is not PayloadKind.ENVELOPE:
        human = outcome.human or json.dumps(payload, indent=2, ensure_ascii=True)
        stdout.write(safe_text(human) + "\n")


# --- --result-file -----------------------------------------------------------------------------


@dataclass(frozen=True)
class ResultFileTarget:
    """A checked ``--result-file``: the parent's realpath, resolved once, and the file name."""

    parent: str
    name: str


def resolve_result_file(path: str) -> ResultFileTarget:
    """Check ``path`` and resolve its parent's realpath once (interface.md, Result file)."""
    if not path or "\0" in path:
        raise CliError.usage("--result-file needs a path without NUL bytes")
    name = os.path.basename(path)
    if name in ("", ".", ".."):
        raise CliError.usage("--result-file must name a file, not a directory")
    parent = os.path.realpath(os.path.dirname(path) or ".")
    return ResultFileTarget(parent=parent, name=name)


def _temp_name(name: str) -> str:
    return f".agent-spawn-{secrets.token_hex(8)}.tmp"


_DIR_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)


def _open_parent(parent: str) -> int:
    """Open ``parent`` component by component from ``/``, never following a symlink."""
    fd = os.open("/", _DIR_FLAGS)
    try:
        for component in parent.split("/"):
            if not component:
                continue
            try:
                next_fd = os.open(component, _DIR_FLAGS, dir_fd=fd)
            except OSError as exc:
                if _is_symlink(component, fd):
                    raise OSError(
                        exc.errno, f"path component {component!r} is a symbolic link"
                    ) from None
                raise
            os.close(fd)
            fd = next_fd
        return fd
    except BaseException:
        os.close(fd)
        raise


def _is_symlink(name: str, dir_fd: int) -> bool:
    try:
        return stat.S_ISLNK(os.stat(name, dir_fd=dir_fd, follow_symlinks=False).st_mode)
    except OSError:
        return False


def write_result_file(target: ResultFileTarget, text: str) -> str | None:
    """Write ``text`` atomically (mode 0600) without following links.

    Returns a warning for the caller to print, or None on success. It never raises: a result-file
    failure must not change the command's exit code.
    """
    try:
        _write(target, text)
    except OSError as exc:
        reason = exc.strerror or str(exc)
        return f"--result-file {target.parent}/{target.name} not written: {reason}"
    return None


def _write(target: ResultFileTarget, text: str) -> None:
    data = text.encode("utf-8")
    dir_fd = _open_parent(target.parent)
    temp = _temp_name(target.name)
    created = False
    try:
        if _is_symlink(target.name, dir_fd):
            raise OSError(0, f"{target.name!r} is a symbolic link")
        fd = os.open(
            temp,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
            0o600,
            dir_fd=dir_fd,
        )
        created = True
        try:
            os.fchmod(fd, 0o600)
            view = memoryview(data)
            while view:
                view = view[os.write(fd, view) :]
            os.fsync(fd)
        finally:
            os.close(fd)
        os.rename(temp, target.name, src_dir_fd=dir_fd, dst_dir_fd=dir_fd)
        created = False
        try:
            os.fsync(dir_fd)
        except OSError:
            pass  # the rename is done; durability of the directory entry is best effort
    finally:
        if created:
            try:
                os.unlink(temp, dir_fd=dir_fd)
            except OSError:
                pass
        os.close(dir_fd)
