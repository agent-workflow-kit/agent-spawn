"""The agent session role over claude-session.

Every call goes through the gateway as an argv list (no shell), with ``--json`` where the
command has it. Prompts and inputs travel as files (``--prompt-file``, ``--input-file``), never
as argv or environment text; the only free text in argv are the caller's validated flag values.
Everything claude-session prints is untrusted: it is parsed defensively, never quoted in a
message, and carried to the caller only as ``lower_error``.

This module knows no agent state. The launch, resume and supervise code (later issues) owns the
order of calls, the parking and the logging; it uses these functions for the calls themselves:

* ``classify``: the dry run before every ``start`` and its refusal (``would_refuse``);
* ``start``: ``start [--resume [--fork]]`` with the relay hook, the outcome file and the labels;
* ``check_classification``: ``status`` right after ``start``; cancels a run it cannot accept;
* ``run_outcome``: the outcome file of a finished run, with the ``status --run`` fallback;
* ``status``, ``stop``, ``forget``, ``label_add``/``label_remove``, ``verify_secret``,
  ``list_by_label`` and ``reconcile``.

TODO(#11): launch order, TODO(#13): resume and fork; supervision (#12) uses these calls.
"""

from __future__ import annotations

import json
import os
import re
import secrets
import shlex
import stat
import subprocess
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Literal, TypeGuard

from . import gateway as gw
from .ids import is_agent_id
from .labels import RESERVED_KEYS, parse_label
from .output import CliError, from_lower_tool
from .third_party import (
    CheckDecision,
    Classification,
    PlanError,
    check_error,
    decide_check,
    parse_plan,
    refusal,
)
from .validate import GrammarError, parse_duration

TOOL = "claude-session"
SESSION_ID = re.compile(r"cs-[a-z2-7]{8}")
MODES = ("interactive", "headless")
#: Output beyond this is treated as unusable (the gateway reads it whole, so this bounds parsing).
MAX_OUTPUT = 8 * 1024 * 1024
#: An outcome file is a small JSON object; anything larger is not one.
MAX_OUTCOME_FILE = 1024 * 1024
DEFAULT_TIMEOUT = 120.0
#: ``stop`` waits for the grace, claude-session's ``kill_grace`` and 5 s; this is the margin.
STOP_MARGIN = 15.0
RECONCILE_TIMEOUT = 300.0

OUTCOMES = frozenset(
    {
        "done",
        "blocked",
        "needs-input",
        "failed",
        "exited-unreported",
        "killed",
        "timeout",
        "lost",
        "not-started",
        "vetoed",
        "escalated",
    }
)
PHASES = ("launch", "session")
STATES = frozenset(
    {
        "starting",
        "working",
        "idle",
        "needs-input",
        "reported",
        "exited",
        "exited-unreported",
        "killed",
        "timeout",
        "lost",
        "not-started",
    }
)
#: The outcome -> exit code table (claude-session interface.md, Outcome mapping).
OUTCOME_EXIT_CODES = {
    "done": 0,
    "blocked": 13,
    "needs-input": 13,
    "failed": 18,
    "exited-unreported": 14,
    "killed": 15,
    "timeout": 16,
    "lost": 17,
    "not-started": 12,
    "vetoed": 11,
    "escalated": 13,
}
END_REASONS = frozenset(
    {
        "normal", "auto_exit", "user_exit", "forwarded_term", "forwarded_hup", "forwarded_int",
        "stopped", "timeout", "max_turns", "api_error", "no_result", "no_report",
        "result_rejected", "no_session_start", "wrapper_lost", "lost", "internal_error",
        "hook_veto", "hook_escalate", "hook_timeout", "canceled", "never_adopted",
        "launch_interrupted", "config_invalid", "claude_not_found", "sandbox_unavailable",
        "socket_path_too_long", "invalid_permission_profile", "unsafe_cwd", "unsafe_path",
        "unsafe_plugin_dir", "unsafe_read_dir", "plugin_dir_missing", "git_dir_changed",
        "no_git_cwd", "fork_cwd_unsupported", "no_tty", "tmpdir_unsafe", "session_dir_unsafe",
        "unsafe_claude_args", "unsafe_setting_sources", "claude_too_old",
    }
)  # fmt: skip
_END_REASON_PATTERN = re.compile(r"exit_[0-9]+|signal_[a-z0-9]+")
_LONE_SURROGATE = re.compile("[\ud800-\udfff]")
#: The events window of ``status`` holds the last 20 events.
EVENT_WINDOW = 20


# --- the run specification ----------------------------------------------------------------------


@dataclass(frozen=True)
class RunSpec:
    """Everything one ``claude-session start`` (or its dry run) is built from.

    ``mode`` is always passed explicitly. ``resume`` is a ``csession_id``; ``fork`` needs it.
    ``labels`` are the agent's ``key=value`` labels; ``agent=<agent_id>`` is added here.
    ``prompt_file`` and ``input_files`` are agent-spawn's private copies (never the caller's).
    ``setting_sources`` may be the empty string (none); ``env_allowlist_only`` may be the empty
    tuple (an empty list) or None (not given); ``auto_exit`` is None, True or False.
    """

    agent_id: str
    mode: str
    resume: str | None = None
    fork: bool = False
    kind: str | None = None
    skill: str | None = None
    plugin_dirs: tuple[str, ...] = ()
    model: str | None = None
    timeout: str | None = None
    max_turns: int | None = None
    permission_profile: str | None = None
    sandbox: str | None = None
    setting_sources: str | None = None
    reads_third_party: bool = False
    env_allow: tuple[str, ...] = ()
    env_allowlist_only: tuple[str, ...] | None = None
    auto_exit: bool | None = None
    reset: tuple[str, ...] = ()
    allow_tainted: bool = False
    override_escalation: bool = False
    repo: str | None = None
    cwd: str | None = None
    labels: tuple[str, ...] = ()
    prompt_file: str | None = None
    input_files: tuple[str, ...] = ()


def _in(value: object, options: Collection[str]) -> bool:
    """``value`` is one of ``options`` (an unhashable or non-string value never is)."""
    return isinstance(value, str) and value in options


def _clean(value: object) -> str | None:
    """``value`` when it is a string without NUL and without a lone surrogate, else None."""
    if isinstance(value, str) and "\0" not in value and _LONE_SURROGATE.search(value) is None:
        return value
    return None


def _absolute(value: object) -> bool:
    """Whether ``value`` is an absolute path string: a relative or empty session directory would
    be resolved against agent-spawn's own cwd by the result-file checks, so it is unusable."""
    return isinstance(value, str) and value.startswith("/")


def _session_id(value: object) -> str:
    """A positional session id: exactly a csession_id, so it can never be read as a flag."""
    if not isinstance(value, str) or SESSION_ID.fullmatch(value) is None:
        raise CliError.usage("the session id is not a csession_id")
    return value


def _arg(value: str, name: str) -> str:
    """A flag value that can travel in argv: text without NUL and with a strict UTF-8 form."""
    try:
        if "\0" in value:
            raise ValueError
        value.encode("utf-8")
    except (ValueError, UnicodeEncodeError):
        raise CliError.usage(f"{name} cannot be passed to claude-session") from None
    return value


def _flag(name: str, value: str) -> str:
    # The `--flag=value` form: a value that starts with `-` stays a value.
    return f"{name}={_arg(value, name)}"


def _check_spec(spec: RunSpec) -> None:
    if not is_agent_id(spec.agent_id):
        raise ValueError("RunSpec.agent_id is not an agent id")
    if spec.mode not in MODES:
        raise ValueError(f"RunSpec.mode must be one of {MODES}")
    if spec.resume is None:
        if spec.fork or spec.reset or spec.allow_tainted or spec.override_escalation:
            raise ValueError(
                "--fork, --reset, --allow-tainted and --override-escalation need resume"
            )
    else:
        if SESSION_ID.fullmatch(spec.resume) is None:
            raise ValueError("RunSpec.resume is not a csession_id")
        if spec.fork and spec.override_escalation:
            raise ValueError("--override-escalation cannot be combined with --fork")
        if not spec.fork and spec.cwd is not None:
            raise ValueError("a plain resume runs in the stored cwd; --cwd needs --fork")
    if spec.max_turns is not None and (
        isinstance(spec.max_turns, bool) or not isinstance(spec.max_turns, int)
    ):
        raise ValueError("RunSpec.max_turns must be an int")
    paths = [*spec.plugin_dirs, *spec.input_files]
    if spec.cwd is not None and spec.cwd != "session-dir":
        paths.append(spec.cwd)
    if spec.prompt_file is not None:
        paths.append(spec.prompt_file)
    if not all(isinstance(path, str) and os.path.isabs(path) for path in paths):
        raise ValueError("RunSpec cwd, prompt_file, input_files and plugin_dirs must be absolute")
    for pair in spec.labels:
        try:
            label = parse_label(pair)
        except GrammarError as error:
            raise CliError.usage(str(error)) from None
        if label.key in RESERVED_KEYS:
            raise CliError.usage(f"label key {label.key!r} is reserved for agent-spawn")


def launch_flags(spec: RunSpec) -> list[str]:
    """The flags shared by the dry run and the real ``start`` (all but hook and outcome file)."""
    _check_spec(spec)
    argv: list[str] = []
    if spec.resume is not None:
        argv.append(_flag("--resume", spec.resume))
        if spec.fork:
            argv.append("--fork")
    argv.append(_flag("--mode", spec.mode))
    for name, value in (
        ("--kind", spec.kind),
        ("--skill", spec.skill),
        ("--model", spec.model),
        ("--timeout", spec.timeout),
        ("--permission-profile", spec.permission_profile),
        ("--sandbox", spec.sandbox),
        ("--setting-sources", spec.setting_sources),
    ):
        if value is not None:
            argv.append(_flag(name, value))
    for directory in spec.plugin_dirs:
        argv.append(_flag("--plugin-dir", directory))
    if spec.max_turns is not None:
        argv.append(_flag("--max-turns", str(spec.max_turns)))
    if spec.reads_third_party:
        argv.append("--reads-third-party")
    for name in spec.env_allow:
        argv.append(_flag("--env-allow", name))
    if spec.env_allowlist_only is not None:
        if spec.env_allowlist_only:
            for name in spec.env_allowlist_only:
                argv.append(_flag("--env-allowlist-only", name))
        else:
            argv.append("--env-allowlist-only=")
    if spec.auto_exit is not None:
        argv.append("--auto-exit" if spec.auto_exit else "--no-auto-exit")
    for setting in spec.reset:
        argv.append(_flag("--reset", setting))
    if spec.allow_tainted:
        argv.append("--allow-tainted")
    if spec.override_escalation:
        argv.append("--override-escalation")
    if spec.repo is not None:
        argv.append(_flag("--repo", spec.repo))
    if spec.cwd is not None:
        argv.append(_flag("--cwd", spec.cwd))
    for pair in (*spec.labels, f"agent={spec.agent_id}"):
        argv.append(_flag("--label", pair))
    if spec.prompt_file is not None:
        argv.append(_flag("--prompt-file", spec.prompt_file))
    for path in spec.input_files:
        argv.append(_flag("--input-file", path))
    return argv


def relay_hook(agent_spawn: str, state_dir: str, config: str) -> str:
    """The ``--hook`` value of the state relay, every argv element POSIX-quoted.

    claude-session runs the hook with ``sh -c``, so each element is quoted for ``sh``.
    """
    for path in (agent_spawn, state_dir, config):
        if not os.path.isabs(path):
            raise ValueError("the relay hook takes absolute paths")
        _arg(path, "relay hook path")
    argv = [agent_spawn, "--state-dir", state_dir, "--config", config, "_session-event"]
    return "post-state-change=" + " ".join(shlex.quote(part) for part in argv)


# --- the outcome file ---------------------------------------------------------------------------


def _private_dir(path: str) -> None:
    """Create ``path`` (mode 0700) or accept an existing real directory of the user."""
    try:
        os.mkdir(path, 0o700)
    except FileExistsError:
        pass
    info = os.lstat(path)
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid():
        raise CliError("internal", "the outcomes directory is not a directory of the user")
    if info.st_mode & 0o077:
        os.chmod(path, 0o700)


def new_outcome_file(state_dir: str, agent_id: str) -> str:
    """A fresh ``<state>/outcomes/<agent_id>/<random>.json`` path (its directory is created)."""
    if not os.path.isabs(state_dir) or not is_agent_id(agent_id):
        raise ValueError("new_outcome_file needs an absolute state dir and an agent id")
    base = os.path.join(state_dir, "outcomes")
    directory = os.path.join(base, agent_id)
    try:
        _private_dir(base)
        _private_dir(directory)
    except OSError as error:
        raise CliError(
            "failed", f"cannot create the outcomes directory ({error.strerror})"
        ) from None
    return os.path.join(directory, secrets.token_hex(16) + ".json")


def read_outcome_file(path: str, csession_id: str, run: int) -> dict[str, Any] | None:
    """The outcome object in ``path``, or None when it is missing, unreadable, malformed, not
    for exactly this ``csession_id`` and ``run``, or not a final outcome.

    The file is opened without following a symlink and read at most ``MAX_OUTCOME_FILE`` bytes.
    """
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC)
    except OSError:
        return None
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_OUTCOME_FILE:
            return None
        with os.fdopen(fd, "rb", closefd=False) as handle:
            raw = handle.read(MAX_OUTCOME_FILE + 1)
    except OSError:
        return None
    finally:
        os.close(fd)
    if len(raw) > MAX_OUTCOME_FILE:
        return None
    data = _loads(gw.decode(raw))
    if data is None:
        return None
    if data.get("csession_id") != csession_id or not _is_int(data.get("run")) or data["run"] != run:
        return None
    return data if _outcome_fields(data) is not None else None


# --- parsing helpers ----------------------------------------------------------------------------


def _is_int(value: object) -> TypeGuard[int]:
    return isinstance(value, int) and not isinstance(value, bool)


def _loads(text: str) -> dict[str, Any] | None:
    """A JSON object from ``text``, or None (bad JSON, too deep, not an object, too big)."""
    if len(text) > MAX_OUTPUT:
        return None
    try:
        data = json.loads(text)
    except (ValueError, RecursionError):
        return None
    return data if isinstance(data, dict) else None


@dataclass(frozen=True)
class RunOutcome:
    """The final outcome of one run, from its outcome file or from ``status --run``.

    ``outcome`` is None for a run that is still live (only possible from ``status``).
    """

    csession_id: str
    run: int
    phase: str
    outcome: str | None
    end_reason: str | None
    exit_code: int | None
    state: str | None
    report: dict[str, Any] | None
    source: Literal["file", "status"]
    first_working_at: str | None = None
    #: The parsed outcome file this was read from (the extras of ``outcome_object``), if any.
    raw: dict[str, Any] | None = field(default=None, repr=False, compare=False)

    @property
    def final(self) -> bool:
        return self.outcome is not None


def _end_reason_ok(value: object) -> bool:
    return _in(value, END_REASONS) or (
        isinstance(value, str) and _END_REASON_PATTERN.fullmatch(value) is not None
    )


def _exit_code_ok(code: object, outcome: object) -> TypeGuard[int]:
    if not _is_int(code) or not 0 <= code <= 255:
        return False
    expected = OUTCOME_EXIT_CODES.get(outcome) if isinstance(outcome, str) else None
    return expected is None or code == expected


def _outcome_fields(data: Mapping[str, Any]) -> dict[str, Any] | None:
    """Validated outcome fields of an outcome object (a final one), else None."""
    phase, outcome, reason = data.get("phase"), data.get("outcome"), data.get("end_reason")
    code, state, report = data.get("exit_code"), data.get("state"), data.get("report")
    if not _in(phase, PHASES) or not _in(outcome, OUTCOMES) or not _exit_code_ok(code, outcome):
        return None
    if reason is not None and not _end_reason_ok(reason):
        return None
    if state is not None and not _in(state, STATES):
        return None
    if report is not None and not isinstance(report, dict):
        return None
    return {
        "phase": phase,
        "outcome": outcome,
        "end_reason": reason,
        "exit_code": code,
        "state": state,
        "report": report,
    }


# --- status -------------------------------------------------------------------------------------


@dataclass(frozen=True)
class RunStatus:
    run: int
    run_kind: str | None
    reads_third_party: bool
    phase: str
    pending: bool
    cwd: str | None
    outcome: str | None
    end_reason: str | None
    exit_code: int | None
    state: str | None
    report: dict[str, Any] | None
    first_working_at: str | None


@dataclass(frozen=True)
class SessionStatus:
    """The parts of ``claude-session status --json`` that agent-spawn relies on."""

    csession_id: str
    csession_dir: str
    tainted: bool
    tainted_by_run: int | None
    current_run: int
    pending: bool
    repo: str | None
    mode: str | None
    state: str | None
    run: RunStatus
    raw: dict[str, Any] = field(repr=False)

    def outcome(self) -> RunOutcome:
        """This run as a ``RunOutcome`` (final or not)."""
        run = self.run
        return RunOutcome(
            self.csession_id,
            run.run,
            run.phase,
            run.outcome,
            run.end_reason,
            run.exit_code,
            run.state,
            run.report,
            "status",
            run.first_working_at,
        )


def _str_or_none(value: object) -> str | None:
    return _clean(value)


def _field(value: object, ok: bool) -> Any:
    return value if ok else None


def _window_reaches_back(events: list[object], run: int) -> bool:
    """Whether a full events window shows what precedes the run: an event of an earlier run, or
    event 1 (the window lists the last 20 events, oldest first)."""
    for event in events:
        if not isinstance(event, dict):
            continue
        number, other = event.get("event_id"), event.get("run")
        if (_is_int(other) and other < run) or (_is_int(number) and number <= 1):
            return True
    return False


def _first_working(events: object, run: int) -> str | None:
    """The ``event_ts`` of the run's first event with ``to_activity: working``, else None."""
    if not isinstance(events, list):
        return None
    if len(events) >= EVENT_WINDOW and not _window_reaches_back(events, run):
        return None  # the window may have cut the run's first events off
    best: tuple[int, str] | None = None
    for event in events:
        if not isinstance(event, dict) or event.get("to_activity") != "working":
            continue
        stamp, number = event.get("event_ts"), event.get("event_id")
        if event.get("run") != run or not isinstance(stamp, str) or not _is_int(number):
            continue
        if best is None or number < best[0]:
            best = (number, stamp)
    return None if best is None else best[1]


def parse_status(data: Mapping[str, Any]) -> SessionStatus | None:
    """A ``SessionStatus`` from a status object, or None when its essentials are unusable."""
    session_id, directory = data.get("csession_id"), data.get("csession_dir")
    current, tainted = data.get("current_run"), data.get("tainted")
    run = data.get("run")
    if (
        not isinstance(session_id, str)
        or SESSION_ID.fullmatch(session_id) is None
        or not isinstance(directory, str)
        or not _absolute(directory)
        or not _is_int(current)
        or current < 1
        or not isinstance(tainted, bool)
        or not isinstance(run, dict)
    ):
        return None
    number, reads, phase = run.get("run"), run.get("reads_third_party"), run.get("phase")
    if (
        not _is_int(number)
        or number < 1
        or not isinstance(reads, bool)
        or not isinstance(phase, str)
        or phase not in PHASES
    ):
        return None
    by_run = data.get("tainted_by_run")
    if by_run is not None and not _is_int(by_run):
        return None
    if _clean(session_id) is None or _clean(directory) is None:
        return None
    outcome, code = run.get("outcome"), run.get("exit_code")
    if outcome is not None and not _in(outcome, OUTCOMES):
        return None
    report = run.get("report")
    reason, state = run.get("end_reason"), run.get("state")
    run_status = RunStatus(
        run=number,
        run_kind=_str_or_none(run.get("run_kind")),
        reads_third_party=reads,
        phase=phase,
        pending=run.get("pending") is True,
        cwd=_str_or_none(run.get("cwd")),
        outcome=outcome,
        end_reason=_field(reason, _end_reason_ok(reason)),
        exit_code=_field(code, _exit_code_ok(code, outcome)),
        state=_field(state, _in(state, STATES)),
        report=report if isinstance(report, dict) else None,
        first_working_at=_first_working(data.get("recent_events"), number),
    )
    return SessionStatus(
        csession_id=session_id,
        csession_dir=directory,
        tainted=tainted,
        tainted_by_run=by_run,
        current_run=current,
        pending=data.get("pending") is True,
        repo=_str_or_none(data.get("repo")),
        mode=_str_or_none(data.get("mode")),
        state=_str_or_none(data.get("state")),
        run=run_status,
        raw=dict(data),
    )


# --- start --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class StartResult:
    """``claude-session start --json`` on success."""

    csession_id: str
    csession_dir: str
    run: int
    argv: tuple[str, ...]
    forked_from: str | None
    warnings: tuple[str, ...]
    required_env_names: tuple[str, ...]
    env_inherit: bool
    expires_at: str | None


class StartError(CliError):
    """A failed ``start``. ``csession_id`` and ``run`` are set when a run was allocated (the
    exit-12 ``launch_failed`` envelope, or a success the output of which is malformed): the
    caller records them and parks the agent. Both are None when nothing is known.

    ``maybe_allocated`` is True when claude-session may have allocated a session although the
    error does not name it (a timeout, an unrunnable or signaled process, an unusable success
    output, exit 1 or 12 other than a clean ``schema_newer``, too much output). It is False for
    a usage error (exit 2), the clean refusals and lookups (10, 11, 20, 21) and ``schema_newer``,
    and when the tool could not be run at all. **Recovery** when it is True and ``csession_id``
    is None: ``SessionBackend.list_by_label("agent=<agent_id>")`` finds the orphan; ``stop`` it
    (a pending run is canceled) and ``forget`` it, or adopt it.
    """

    def __init__(
        self,
        code: str,
        message: str,
        *,
        csession_id: str | None = None,
        run: int | None = None,
        maybe_allocated: bool = False,
        **fields: Any,
    ) -> None:
        super().__init__(code, message, **fields)
        self.csession_id = csession_id
        self.run = run
        self.maybe_allocated = maybe_allocated or csession_id is not None


def _allocated(error: object) -> tuple[str | None, int | None]:
    """The session and run an error envelope names, when they are well-formed."""
    if not isinstance(error, dict):
        return None, None
    session = error.get("csession_id")
    run = error.get("run")
    return (
        session if isinstance(session, str) and SESSION_ID.fullmatch(session) else None,
        run if _is_int(run) and run >= 1 else None,
    )


def _maybe_allocated(code: int, envelope: object) -> bool:
    """Whether a failed ``start`` may have left a session behind: not for a usage error, a clean
    refusal or lookup (2, 10, 11, 20, 21) or a clean ``schema_newer``; yes for a signal, exit 1,
    12 and every other code."""
    if code in (2, 10, 11, 20, 21):
        return False
    return not (
        code == 12 and isinstance(envelope, dict) and envelope.get("code") == "schema_newer"
    )


def _str_tuple(value: object) -> tuple[str, ...] | None:
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return tuple(value)
    return None


def parse_start(data: Mapping[str, Any]) -> StartResult | None:
    session, directory, run = data.get("csession_id"), data.get("csession_dir"), data.get("run")
    argv = _str_tuple(data.get("argv"))
    warnings = _str_tuple(data.get("warnings", []))
    names = _str_tuple(data.get("required_env_names", []))
    forked = data.get("forked_from")
    if (
        not isinstance(session, str)
        or SESSION_ID.fullmatch(session) is None
        or not isinstance(directory, str)
        or not _absolute(directory)
        or not _is_int(run)
        or run < 1
        or argv is None
        or warnings is None
        or names is None
        or not (forked is None or isinstance(forked, str))
    ):
        return None
    return StartResult(
        csession_id=session,
        csession_dir=directory,
        run=run,
        argv=argv,
        forked_from=forked,
        warnings=warnings,
        required_env_names=names,
        env_inherit=data.get("env_inherit") is True,
        expires_at=_str_or_none(data.get("expires_at")),
    )


@dataclass(frozen=True)
class ClassificationCheck:
    """The post-start check: the decision, what ``status`` said and what the cancel did."""

    decision: CheckDecision
    status: SessionStatus | None
    cancel: dict[str, Any] | None = None
    cancel_error: CliError | None = None
    #: True only when the cancel was sent and the run ended (``stop`` answered ``ended: true``).
    cancelled: bool = False

    @property
    def verdict(self) -> Literal["accept", "warn", "cancel"]:
        return self.decision.verdict

    def error(self, agent_id: str | None) -> CliError:
        """The exit-12 ``launch_failed`` error of a cancelled run."""
        return check_error(self.decision, agent_id=agent_id)


# --- the backend --------------------------------------------------------------------------------


@dataclass
class _Reply:
    code: int
    data: dict[str, Any] | None


class SessionBackend:
    """claude-session as agent-spawn uses it.

    ``command`` is the resolved argv prefix (config ``claude_session_cmd`` or the stored
    absolute path). ``actor`` and ``decision_ids`` are added to the commands that accept them
    (``start``, ``stop``, ``forget``).
    """

    def __init__(
        self,
        command: Sequence[str],
        *,
        actor: str | None = None,
        decision_ids: Sequence[str] = (),
        gateway: gw.Gateway | None = None,
        timeout: float | None = DEFAULT_TIMEOUT,
        kill_grace: float = 30.0,
    ) -> None:
        if not command:
            raise ValueError("an empty claude-session command")
        self.command = list(command)
        self.actor = actor
        self.decision_ids = list(decision_ids)
        self._gateway = gateway
        self.timeout = timeout
        self.kill_grace = kill_grace

    # -- plumbing

    @property
    def gateway(self) -> gw.Gateway:
        return self._gateway if self._gateway is not None else gw.get_gateway()

    def audit_flags(self) -> list[str]:
        flags: list[str] = []
        if self.actor is not None:
            flags.append(_flag("--actor", self.actor))
        for value in self.decision_ids:
            flags.append(_flag("--decision-id", value))
        return flags

    def call(
        self, args: Sequence[str], *, stdin: str | None = None, timeout: float | None = None
    ) -> _Reply:
        """Run one command. Lower-tool exit codes are returned, not raised; a missing tool, a
        timeout and an unusable (huge or non-JSON) output are the only failures raised here
        or reported as ``data=None``."""
        argv = [*self.command, *args]
        try:
            done = self.gateway.run(argv, stdin=stdin, timeout=self._timeout_for(timeout))
        except FileNotFoundError:
            raise CliError(
                "tool_missing",
                "claude-session is not installed or not executable",
                tool=TOOL,
                details={"hint": "install claude-session"},
            ) from None
        except PermissionError:
            raise CliError(
                "tool_missing",
                "claude-session is not executable",
                tool=TOOL,
                details={"hint": "install claude-session"},
            ) from None
        except subprocess.TimeoutExpired:
            raise CliError("failed", "claude-session did not answer in time", tool=TOOL) from None
        except gw.OutputTooLarge:
            raise CliError("failed", "claude-session printed too much output", tool=TOOL) from None
        except OSError as error:
            raise CliError(
                "failed", f"claude-session could not run ({error.strerror})", tool=TOOL
            ) from None
        return _Reply(done.returncode, _loads(done.stdout))

    def _timeout_for(self, wanted: float | None) -> float | None:
        """The call timeout: the default, or a longer one a slow command needs."""
        if self.timeout is None or wanted is None:
            return self.timeout
        return max(self.timeout, wanted)

    def _check_actor(self, spec: RunSpec) -> None:
        # claude-session accepts these human-only flags only with an actor; fail before a call.
        if self.actor is None and (spec.allow_tainted or spec.override_escalation):
            raise CliError.usage("--allow-tainted and --override-escalation need an actor")

    def _fail(self, reply: _Reply, *, agent_id: str | None = None) -> CliError:
        """The pass-through error of a failed call (its envelope verbatim in ``lower_error``)."""
        error = reply.data.get("error") if reply.data is not None else None
        return from_lower_tool(TOOL, reply.code, error, agent_id=agent_id)

    def _unusable(self, what: str) -> CliError:
        return CliError("internal", f"claude-session printed unusable {what}", tool=TOOL)

    # -- classification

    def dry_run_argv(self, spec: RunSpec) -> list[str]:
        """``start ... --dry-run --json``: the real command without hook and outcome file."""
        flags = launch_flags(spec)
        self._check_actor(spec)
        return ["start", *flags, *self.audit_flags(), "--dry-run", "--json"]

    def classify(self, spec: RunSpec, *, refuse: bool = True) -> Classification:
        """The classification of ``classify_with_plan`` (see there)."""
        return self.classify_with_plan(spec, refuse=refuse)[0]

    def classify_with_plan(
        self, spec: RunSpec, *, refuse: bool = True
    ) -> tuple[Classification, dict[str, Any]]:
        """Run the dry run and read the classification; also returns the plan, verbatim.

        Exits 0, 12 and 20 carry a plan and are all used. ``would_refuse`` raises the exit-20
        refusal (``tool: claude-session``) unless ``refuse`` is false; ``would_fail`` is not
        acted on. Any other result fails the command and starts nothing.
        """
        reply = self.call(self.dry_run_argv(spec))
        plan = reply.data
        if reply.code in (0, 12, 20) and plan is not None and "error" not in plan:
            try:
                classification = parse_plan(plan, requested_cwd=spec.cwd)
            except PlanError:
                raise self._unusable("a dry-run plan") from None
            if spec.reads_third_party and not classification.reads_third_party:
                classification = replace(classification, reads_third_party=True)
            if refuse:
                problem = refusal(classification)
                if problem is None and reply.code == 20:
                    problem = refusal(replace(classification, would_refuse="unknown"))
                if problem is not None:
                    raise problem
            return classification, plan
        if reply.code == 0:
            raise self._unusable("output for a dry run")
        failure = self._fail(reply)
        if reply.code == 2:
            # claude-session rejected a flag value: the caller's input, not our bug.
            raise CliError(
                "usage",
                "claude-session rejected the launch arguments (see lower_error)",
                tool=TOOL,
                lower_code=failure.lower_code,
                lower_error=failure.lower_error,
            )
        raise failure

    # -- start

    def start(
        self,
        spec: RunSpec,
        *,
        outcome_file: str,
        hook: str,
        extra_hooks: Sequence[str] = (),
    ) -> StartResult:
        """``claude-session start`` with the relay ``hook`` and the ``outcome_file``.

        Raises ``StartError``: with ``csession_id`` and ``run`` set when the exit-12
        ``launch_failed`` envelope allocated a run.
        """
        if not os.path.isabs(outcome_file):
            raise ValueError("the outcome file must be an absolute path")
        flags = launch_flags(spec)
        self._check_actor(spec)
        args = [
            "start",
            *flags,
            *(_flag("--hook", value) for value in (hook, *extra_hooks)),
            _flag("--outcome-file", outcome_file),
            *self.audit_flags(),
            "--json",
        ]
        try:
            reply = self.call(args)
        except CliError as error:
            raise StartError(
                error.code,
                str(error),
                exit_code=error.exit_code,
                tool=error.tool,
                details=error.details,
                maybe_allocated=error.code != "tool_missing",
            ) from None
        data = reply.data
        if reply.code == 0:
            if data is not None:
                result = parse_start(data)
                if result is not None:
                    return result
            session, run = _allocated(data)
            raise StartError(
                "internal",
                "claude-session printed an unusable start result",
                csession_id=session,
                run=run,
                maybe_allocated=True,
                tool=TOOL,
            )
        failure = self._fail(reply)
        envelope = data.get("error") if data is not None else None
        session, run = _allocated(envelope) if failure.code == "launch_failed" else (None, None)
        raise StartError(
            failure.code,
            str(failure),
            csession_id=session,
            run=run,
            maybe_allocated=_maybe_allocated(reply.code, envelope),
            exit_code=failure.exit_code,
            reason=failure.reason,
            tool=TOOL,
            lower_code=failure.lower_code,
            lower_error=failure.lower_error,
        )

    # -- status and the post-start check

    def status(self, csession_id: str, run: int | None = None) -> SessionStatus:
        """``status <id> [--run n] --json``; raises a pass-through error or ``internal``."""
        args = ["status", _session_id(csession_id)]
        if run is not None:
            args.append(_flag("--run", str(run)))
        reply = self.call([*args, "--json"])
        if reply.code != 0:
            raise self._fail(reply)
        parsed = parse_status(reply.data) if reply.data is not None else None
        if parsed is None:
            raise self._unusable("a status")
        return parsed

    def check_classification(
        self,
        started: StartResult,
        expected: Classification,
        *,
        rule_allows: Callable[[bool, str | None], bool],
        result_file_allows: Callable[[], bool] = lambda: True,
    ) -> ClassificationCheck:
        """Compare the dry-run classification with ``status`` right after ``start``.

        A mismatch the cwd rule allows continues (``warn``: the caller logs
        ``classification_mismatch`` at ``warn``); otherwise, and when ``status`` fails or is
        unusable or names another run, the pending run is cancelled here (``claude-session
        stop``: no process can adopt it yet) and the verdict is ``cancel``. Parking the agent
        and the exit-12 error (``ClassificationCheck.error``) are the caller's.

        ``rule_allows(actual, cwd)`` is the caller's cwd rule (it owns the records of the cwd:
        #11, #13, #14); ``cwd`` is the run's actual cwd from ``status``. A third-party-reading
        run without a string cwd, and any exception from a callback, cancel as
        ``classification_unknown``. The cancel is reported in ``cancelled`` (True only when
        ``stop`` answered ``ended: true``); a stop that left the run live, or failed, sets
        ``cancel_error`` and the caller must not treat the run as gone.
        """
        status: SessionStatus | None
        actual: bool | None
        try:
            status = self.status(started.csession_id)
        except CliError:
            status, actual = None, None
        else:
            if (
                status.csession_id != started.csession_id
                or status.current_run != started.run
                or status.run.run != started.run
            ):
                status, actual = None, None
            else:
                actual = status.run.reads_third_party or status.tainted_by_run == started.run
        try:
            decision = decide_check(
                expected.reads_third_party,
                actual,
                cwd=status.run.cwd if status is not None else None,
                rule_allows=rule_allows,
                result_file_allows=result_file_allows,
            )
        except Exception:  # noqa: BLE001 - fail closed, whatever the callers' code did
            decision = CheckDecision(
                "cancel", "classification_unknown", expected.reads_third_party, actual
            )
        if decision.verdict != "cancel":
            return ClassificationCheck(decision, status)
        cancel: dict[str, Any] | None = None
        cancel_error: CliError | None = None
        try:
            cancel = self.stop(started.csession_id)
        except CliError as error:
            cancel_error = error
        else:
            if cancel.get("ended") is not True:
                cancel_error = CliError(
                    "launch_failed",
                    "the run could not be canceled and is still live",
                    details={"cause": decision.cause},
                )
        return ClassificationCheck(
            decision, status, cancel, cancel_error, cancelled=cancel_error is None
        )

    # -- outcome

    def run_outcome(self, outcome_file: str, csession_id: str, run: int) -> RunOutcome:
        """The outcome of a finished run: its outcome file, else ``status --run <n>``.

        The file is used only when it parses and names exactly this session and run; a missing
        file, a parse error or a mismatch falls back to ``status``. A null ``first_working_at``
        is filled from ``status`` ``recent_events`` (one extra ``status`` call).
        """
        data = read_outcome_file(outcome_file, csession_id, run)
        if data is not None:
            fields = _outcome_fields(data)
            if fields is not None:
                status: SessionStatus | None = None
                try:
                    status = self.status(csession_id, run)
                except CliError:
                    pass
                first = status.run.first_working_at if status is not None else None
                return RunOutcome(
                    csession_id, run, source="file", first_working_at=first, raw=data, **fields
                )
        status = self.status(csession_id, run)
        if status.csession_id != csession_id or status.run.run != run:
            raise self._unusable("a status for another run")
        return replace(status.outcome(), raw=data)

    # -- other commands

    def stop(
        self, csession_id: str, *, grace: str | None = None, dry_run: bool = False
    ) -> dict[str, Any]:
        """``stop <id> [--grace D] [--dry-run]``. The stop object verbatim for exit 0, and for exit
        12 with ``ended: false`` (still live: the caller checks ``["ended"]``); else raises."""
        args = ["stop", _session_id(csession_id), *self.audit_flags()]
        wanted: float | None = None
        if grace is not None:
            try:
                wanted = parse_duration(grace) + self.kill_grace + STOP_MARGIN
            except GrammarError as error:
                raise CliError.usage(str(error)) from None
            args.append(_flag("--grace", grace))
        if dry_run:
            args.append("--dry-run")
        reply = self.call([*args, "--json"], timeout=wanted)
        data = reply.data
        if reply.code == 0 or (
            reply.code == 12
            and data is not None
            and "error" not in data
            and data.get("ended") is False
        ):
            if data is None or not isinstance(data.get("ended"), bool):
                raise self._unusable("a stop result")
            return data
        raise self._fail(reply)

    def forget(self, csession_id: str) -> dict[str, Any]:
        """``forget <id>``; exit 10 (unknown) raises ``not_found`` (callers count it as done)."""
        reply = self.call(["forget", _session_id(csession_id), *self.audit_flags(), "--json"])
        if reply.code != 0:
            raise self._fail(reply)
        if reply.data is None:
            raise self._unusable("a forget result")
        return reply.data

    def _label(self, action: str, csession_id: str, pairs: Sequence[str]) -> dict[str, Any]:
        if not pairs:
            raise ValueError("label needs at least one pair")
        for pair in pairs:
            try:
                parse_label(pair)
            except GrammarError as error:
                raise CliError.usage(str(error)) from None
        reply = self.call(["label", action, _session_id(csession_id), *pairs, "--json"])
        if reply.code != 0:
            raise self._fail(reply)
        if reply.data is None:
            raise self._unusable("a label result")
        return reply.data

    def label_add(self, csession_id: str, pairs: Sequence[str]) -> dict[str, Any]:
        return self._label("add", csession_id, pairs)

    def label_remove(self, csession_id: str, pairs: Sequence[str]) -> dict[str, Any]:
        return self._label("remove", csession_id, pairs)

    def verify_secret(self, csession_id: str, secret: str) -> bool:
        """``verify-secret <id>`` with the secret on stdin (never argv). True only for exit 0
        with ``{"match": true}``; exit 11 is False; anything else raises."""
        reply = self.call(["verify-secret", _session_id(csession_id), "--json"], stdin=secret)
        if reply.code == 11:
            return False
        if reply.code != 0:
            raise self._fail(reply)
        return reply.data is not None and reply.data.get("match") is True

    def list_by_label(self, pair: str) -> list[str]:
        """The ``csession_id`` of every session with the label (``list --label k=v --json``)."""
        try:
            parse_label(pair)
        except GrammarError as error:
            raise CliError.usage(str(error)) from None
        reply = self.call(["list", _flag("--label", pair), "--json"])
        if reply.code != 0:
            raise self._fail(reply)
        sessions = reply.data.get("sessions") if reply.data is not None else None
        if not isinstance(sessions, list):
            raise self._unusable("a session list")
        found: list[str] = []
        for item in sessions:
            session = item.get("csession_id") if isinstance(item, dict) else None
            if not isinstance(session, str) or SESSION_ID.fullmatch(session) is None:
                raise self._unusable("a session list")
            found.append(session)
        return found

    def reconcile(self, *, dry_run: bool = False) -> dict[str, Any]:
        args = ["reconcile", *(["--dry-run"] if dry_run else []), "--json"]
        reply = self.call(args, timeout=RECONCILE_TIMEOUT)
        if reply.code != 0:
            raise self._fail(reply)
        if reply.data is None:
            raise self._unusable("a reconcile result")
        return reply.data
