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

TODO(#11): launch order, TODO(#13): resume and fork, TODO(#14): supervision use these calls.
"""

from __future__ import annotations

import json
import os
import re
import secrets
import shlex
import stat
import subprocess
from collections.abc import Callable, Mapping, Sequence
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
from .validate import GrammarError

TOOL = "claude-session"
SESSION_ID = re.compile(r"cs-[a-z2-7]{8}")
MODES = ("interactive", "headless")
#: Output beyond this is treated as unusable (the gateway reads it whole, so this bounds parsing).
MAX_OUTPUT = 8 * 1024 * 1024
#: An outcome file is a small JSON object; anything larger is not one.
MAX_OUTCOME_FILE = 1024 * 1024
DEFAULT_TIMEOUT = 120.0

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
    if spec.max_turns is not None and (
        isinstance(spec.max_turns, bool) or not isinstance(spec.max_turns, int)
    ):
        raise ValueError("RunSpec.max_turns must be an int")
    for pair in spec.labels:
        try:
            label = parse_label(pair)
        except GrammarError as error:
            raise CliError.usage(str(error)) from None
        if label.key in RESERVED_KEYS:
            raise CliError.usage(f"label key {label.key!r} is reserved for agent-spawn")


def _launch_flags(spec: RunSpec) -> list[str]:
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

    @property
    def final(self) -> bool:
        return self.outcome is not None


def _outcome_fields(data: Mapping[str, Any]) -> dict[str, Any] | None:
    """Validated outcome fields of an outcome object (a final one), else None."""
    phase, outcome, reason = data.get("phase"), data.get("outcome"), data.get("end_reason")
    code, state, report = data.get("exit_code"), data.get("state"), data.get("report")
    if phase not in PHASES or outcome not in OUTCOMES or not _is_int(code):
        return None
    if reason is not None and not isinstance(reason, str):
        return None
    if state is not None and not isinstance(state, str):
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
    return value if isinstance(value, str) else None


def _first_working(events: object, run: int) -> str | None:
    """The ``event_ts`` of the run's first event with ``to_activity: working``, else None."""
    if not isinstance(events, list):
        return None
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
        or not _is_int(current)
        or current < 1
        or not isinstance(tainted, bool)
        or not isinstance(run, dict)
    ):
        return None
    number, reads, phase = run.get("run"), run.get("reads_third_party"), run.get("phase")
    if not _is_int(number) or number < 1 or not isinstance(reads, bool) or phase not in PHASES:
        return None
    by_run = data.get("tainted_by_run")
    outcome, code = run.get("outcome"), run.get("exit_code")
    if outcome is not None and outcome not in OUTCOMES:
        return None
    report = run.get("report")
    run_status = RunStatus(
        run=number,
        run_kind=_str_or_none(run.get("run_kind")),
        reads_third_party=reads,
        phase=phase,
        pending=run.get("pending") is True,
        cwd=_str_or_none(run.get("cwd")),
        outcome=outcome,
        end_reason=_str_or_none(run.get("end_reason")),
        exit_code=code if _is_int(code) else None,
        state=_str_or_none(run.get("state")),
        report=report if isinstance(report, dict) else None,
        first_working_at=_first_working(data.get("recent_events"), number),
    )
    return SessionStatus(
        csession_id=session_id,
        csession_dir=directory,
        tainted=tainted,
        tainted_by_run=by_run if _is_int(by_run) else None,
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
    caller records them and parks the agent. Both are None when nothing is known."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        csession_id: str | None = None,
        run: int | None = None,
        **fields: Any,
    ) -> None:
        super().__init__(code, message, **fields)
        self.csession_id = csession_id
        self.run = run


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
    ) -> None:
        if not command:
            raise ValueError("an empty claude-session command")
        self.command = list(command)
        self.actor = actor
        self.decision_ids = list(decision_ids)
        self._gateway = gateway
        self.timeout = timeout

    # -- plumbing

    @property
    def gateway(self) -> gw.Gateway:
        return self._gateway if self._gateway is not None else gw.get_gateway()

    def _audit_flags(self) -> list[str]:
        flags: list[str] = []
        if self.actor is not None:
            flags.append(_flag("--actor", self.actor))
        for value in self.decision_ids:
            flags.append(_flag("--decision-id", value))
        return flags

    def _call(self, args: Sequence[str], *, stdin: str | None = None) -> _Reply:
        """Run one command. Lower-tool exit codes are returned, not raised; a missing tool, a
        timeout and an unusable (huge or non-JSON) output are the only failures raised here
        or reported as ``data=None``."""
        argv = [*self.command, *args]
        try:
            done = self.gateway.run(argv, stdin=stdin, timeout=self.timeout)
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
        except OSError as error:
            raise CliError(
                "failed", f"claude-session could not run ({error.strerror})", tool=TOOL
            ) from None
        return _Reply(done.returncode, _loads(done.stdout))

    def _fail(self, reply: _Reply, *, agent_id: str | None = None) -> CliError:
        """The pass-through error of a failed call (its envelope verbatim in ``lower_error``)."""
        error = reply.data.get("error") if reply.data is not None else None
        return from_lower_tool(TOOL, reply.code, error, agent_id=agent_id)

    def _unusable(self, what: str) -> CliError:
        return CliError("internal", f"claude-session printed unusable {what}", tool=TOOL)

    # -- classification

    def dry_run_argv(self, spec: RunSpec) -> list[str]:
        """``start ... --dry-run --json``: the real command without hook and outcome file."""
        return ["start", *_launch_flags(spec), *self._audit_flags(), "--dry-run", "--json"]

    def classify(self, spec: RunSpec, *, refuse: bool = True) -> Classification:
        """Run the dry run and read the classification.

        Exits 0, 12 and 20 carry a plan and are all used. ``would_refuse`` raises the exit-20
        refusal (``tool: claude-session``) unless ``refuse`` is false; ``would_fail`` is not
        acted on. Any other result fails the command and starts nothing.
        """
        reply = self._call(self.dry_run_argv(spec))
        plan = reply.data
        if reply.code in (0, 12, 20) and plan is not None and "error" not in plan:
            try:
                classification = parse_plan(plan, requested_cwd=spec.cwd)
            except PlanError:
                raise self._unusable("a dry-run plan") from None
            if refuse:
                problem = refusal(classification)
                if problem is None and reply.code == 20:
                    problem = refusal(replace(classification, would_refuse="unknown"))
                if problem is not None:
                    raise problem
            return classification
        if reply.code == 0:
            raise self._unusable("output for a dry run")
        raise self._fail(reply)

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
        args = [
            "start",
            *_launch_flags(spec),
            *(_flag("--hook", value) for value in (hook, *extra_hooks)),
            _flag("--outcome-file", outcome_file),
            *self._audit_flags(),
            "--json",
        ]
        try:
            reply = self._call(args)
        except CliError as error:
            raise StartError(
                error.code,
                str(error),
                exit_code=error.exit_code,
                tool=error.tool,
                details=error.details,
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
            exit_code=failure.exit_code,
            reason=failure.reason,
            tool=TOOL,
            lower_code=failure.lower_code,
            lower_error=failure.lower_error,
        )

    # -- status and the post-start check

    def status(self, csession_id: str, run: int | None = None) -> SessionStatus:
        """``status <id> [--run n] --json``; raises a pass-through error or ``internal``."""
        args = ["status", _arg(csession_id, "session id")]
        if run is not None:
            args.append(_flag("--run", str(run)))
        reply = self._call([*args, "--json"])
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
        rule_allows: Callable[[bool], bool],
        result_file_allows: Callable[[], bool] = lambda: True,
    ) -> ClassificationCheck:
        """Compare the dry-run classification with ``status`` right after ``start``.

        A mismatch the cwd rule allows continues (``warn``: the caller logs
        ``classification_mismatch`` at ``warn``); otherwise, and when ``status`` fails or is
        unusable or names another run, the pending run is cancelled here (``claude-session
        stop``: no process can adopt it yet) and the verdict is ``cancel``. Parking the agent
        and the exit-12 error (``ClassificationCheck.error``) are the caller's.
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
        decision = decide_check(
            expected.reads_third_party,
            actual,
            rule_allows=rule_allows,
            result_file_allows=result_file_allows,
        )
        if decision.verdict != "cancel":
            return ClassificationCheck(decision, status)
        cancel: dict[str, Any] | None = None
        cancel_error: CliError | None = None
        try:
            cancel = self.stop(started.csession_id)
        except CliError as error:
            cancel_error = error
        return ClassificationCheck(decision, status, cancel, cancel_error)

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
                return RunOutcome(csession_id, run, source="file", first_working_at=first, **fields)
        status = self.status(csession_id, run)
        if status.csession_id != csession_id or status.run.run != run:
            raise self._unusable("a status for another run")
        return status.outcome()

    # -- other commands

    def stop(self, csession_id: str, *, grace: str | None = None) -> dict[str, Any]:
        """``stop <id> [--grace D]``. Returns the stop object verbatim for exit 0, and for exit
        12 with ``ended: false`` (still live: the caller checks ``["ended"]``); else raises."""
        args = ["stop", _arg(csession_id, "session id"), *self._audit_flags()]
        if grace is not None:
            args.append(_flag("--grace", grace))
        reply = self._call([*args, "--json"])
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
        reply = self._call(
            ["forget", _arg(csession_id, "session id"), *self._audit_flags(), "--json"]
        )
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
        reply = self._call(["label", action, _arg(csession_id, "session id"), *pairs, "--json"])
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
        reply = self._call(
            ["verify-secret", _arg(csession_id, "session id"), "--json"], stdin=secret
        )
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
        reply = self._call(["list", _flag("--label", pair), "--json"])
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
        reply = self._call(args)
        if reply.code != 0:
            raise self._fail(reply)
        if reply.data is None:
            raise self._unusable("a reconcile result")
        return reply.data
