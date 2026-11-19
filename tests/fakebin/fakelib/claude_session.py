"""Fake ``claude-session`` (interface 0.1.0-design).

Implemented commands: ``start`` (also ``--resume``/``--fork`` and ``--dry-run``), ``exec``
(adopting a pending run with ``--session-id``), ``resume``, ``stop``, ``wait``, ``status``,
``list``, ``label add|remove``, ``forget``, ``reconcile`` and ``verify-secret``. Other commands
(``report``, ``transcript``, ``export``, ``kinds``, ``cron``, ``hook-event``) exit 1 with
``fake_unsupported``: agent-spawn does not call them. Sessions created by ``start`` are kept in
``<FAKEBIN_LOG>.sessions.json`` so that ``exec --session-id`` can adopt them, prefixes resolve
and ``list`` shows them. See ``tests/fakebin/FIDELITY.md`` for what is simpler than the real tool.
"""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from . import grammars as g
from .core import (
    BOOL,
    FIXED_TS,
    MULTI,
    VALUE,
    Context,
    Exit,
    Parsed,
    base32_id,
    counter,
    drop_signal,
    locked,
    log_event,
    parse_args,
    read_json,
    require_pattern,
    run_fake,
)

VERSION = "0.1.0"
GLOBAL = {
    "--json": BOOL,
    "--config": VALUE,
    "--verbose": BOOL,
    "--quiet": BOOL,
    "--yes": BOOL,
}
LAUNCH = {
    "--kind": VALUE,
    "--mode": VALUE,
    "--skill": VALUE,
    "--prompt-file": VALUE,
    "--input-file": MULTI,
    "--plugin-dir": MULTI,
    "--repo": VALUE,
    "--label": MULTI,
    "--hook": MULTI,
    "--hook-timeout": MULTI,
    "--cwd": VALUE,
    "--timeout": VALUE,
    "--max-turns": VALUE,
    "--model": VALUE,
    "--permission-profile": VALUE,
    "--sandbox": VALUE,
    "--setting-sources": VALUE,
    "--reads-third-party": BOOL,
    "--auto-exit": BOOL,
    "--no-auto-exit": BOOL,
    "--implicit-done": BOOL,
    "--no-implicit-done": BOOL,
    "--env-allow": MULTI,
    "--env-allowlist-only": MULTI,
    "--reset": MULTI,
    "--outcome-file": VALUE,
    "--allow-tainted": BOOL,
    "--override-escalation": BOOL,
    "--actor": VALUE,
    "--decision-id": MULTI,
    "--dry-run": BOOL,
}
HOOK_NAMES = ("pre-launch", "post-state-change")
# Observed on the real parser: `--reset kind|mode|skill|repo|cwd` is an invalid choice.
RESET_NAMES = (
    "model",
    "plugin-dir",
    "env-allow",
    "env-allowlist-only",
    "setting-sources",
    "sandbox",
    "timeout",
    "max-turns",
    "permission-profile",
    "auto-exit",
    "implicit-done",
)
STATES = (
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
)
REPORT_STATUSES = ("done", "blocked", "failed", "needs-input")
SETTING_SOURCES = ("user", "project", "local")
MAX_INTEGER = 9007199254740991
DURATION_UNITS = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0, "d": 86400.0}
MAX_SECONDS = 36500 * 86400.0
# claude-session's own repo id grammar (differs from the shared one in grammars.py).
CS_REPO_ID = r"[a-z0-9]([a-z0-9.-]*[a-z0-9])?(:[0-9]{1,5})?(/(?!\.\.?(/|$))[A-Za-z0-9._-]+)+"
DURATION = re.compile(r"[0-9]+(?:\.[0-9]+)?(?:ms|s|m|h|d)?")
ACTOR_MAX_ESCAPED = 4 * 1024
DECISION_IDS_MAX_ESCAPED = 256 * 1024
SPECS: dict[str, dict[str, str]] = {
    "exec": {**GLOBAL, **LAUNCH, "--session-id": VALUE, "--run": VALUE},
    "start": {**GLOBAL, **LAUNCH, "--resume": VALUE, "--fork": BOOL},
    "resume": {**GLOBAL, **LAUNCH, "--fork": BOOL},
    "stop": {
        **GLOBAL,
        "--grace": VALUE,
        "--actor": VALUE,
        "--decision-id": MULTI,
        "--dry-run": BOOL,
    },
    "wait": {**GLOBAL, "--run": VALUE, "--timeout": VALUE, "--outcome-file": VALUE},
    "status": {**GLOBAL, "--run": VALUE},
    "list": {
        **GLOBAL,
        "--label": MULTI,
        "--state": MULTI,
        "--kind": VALUE,
        "--launch-kind": VALUE,
        "--run-kind": VALUE,
        "--mode": VALUE,
        "--needs-input": BOOL,
        "--pending": BOOL,
        "--repo": VALUE,
        "--since": VALUE,
        "--limit": VALUE,
        "--forked-from": VALUE,
        "--report-status": MULTI,
        "--live": BOOL,
        "--tainted": BOOL,
    },
    "label": {**GLOBAL, "--dry-run": BOOL},
    "forget": {
        **GLOBAL,
        "--older-than": VALUE,
        "--state": MULTI,
        "--kind": VALUE,
        "--label": MULTI,
        "--actor": VALUE,
        "--decision-id": MULTI,
        "--dry-run": BOOL,
    },
    "reconcile": {**GLOBAL, "--dry-run": BOOL},
    "verify-secret": {"--json": BOOL},  # accepted and ignored: the output is always JSON
}
UNSUPPORTED = ("report", "transcript", "export", "kinds", "cron", "hook-event")
ADOPT_FLAGS = {
    "--session-id",
    "--run",
    "--json",
    "--dry-run",
    "--outcome-file",
    "--verbose",
    "--quiet",
    "--yes",
}

# outcome -> (exit code, state, end_reason, phase, report status)
OUTCOMES: dict[str, tuple[int, str, str, str, str | None]] = {
    "done": (0, "exited", "normal", "session", "done"),
    "blocked": (13, "exited", "normal", "session", "blocked"),
    "needs-input": (13, "exited", "normal", "session", "needs-input"),
    "failed": (18, "exited", "normal", "session", "failed"),
    "exited-unreported": (14, "exited-unreported", "no_report", "session", None),
    "killed": (15, "killed", "stopped", "session", None),
    "timeout": (16, "timeout", "timeout", "session", None),
    "lost": (17, "lost", "lost", "session", None),
    "not-started": (12, "not-started", "launch_interrupted", "launch", None),
    "vetoed": (11, "not-started", "hook_veto", "launch", None),
    "escalated": (13, "not-started", "hook_escalate", "launch", None),
}


def outcome_object(name: str, csession_id: str, run: int = 1, **overrides: Any) -> dict[str, Any]:
    """A schema-valid outcome object (``outcome.json#/$defs/outcome``) for outcome ``name``."""
    code, state, end_reason, phase, report_status = OUTCOMES[name]
    report: dict[str, Any] | None = None
    if report_status is not None:
        implicit = report_status == "done"
        report = {
            "status": report_status,
            "implicit": implicit,
            "summary": None if implicit else "fake summary",
            "reason": None if report_status in ("done", "failed") else "fake reason",
            "artifact": {
                "name": "result.md",
                "path": f"/fake/state/sessions/{csession_id}/artifacts/result.md",
                "sha256": "0" * 64,
                "size": 0,
            },
            "reported_at": FIXED_TS,
        }
    obj: dict[str, Any] = {
        "csession_id": csession_id,
        "run": run,
        "forked_from": None,
        "claude_session_id": None if phase == "launch" else "00000000-0000-4000-8000-000000000000",
        "phase": phase,
        "outcome": name,
        "end_reason": end_reason,
        "exit_code": code,
        "state": state,
        "report": report,
        "session_started_at": None if phase == "launch" else FIXED_TS,
        "cost_usd": None if phase == "launch" else 0.0,
        "message": None,
        "warnings": [],
    }
    obj.update(overrides)
    return obj


# --- state ---------------------------------------------------------------------------------


def _store_path() -> Path:
    return Path(os.environ["FAKEBIN_LOG"] + ".sessions.json")


def load_sessions() -> dict[str, dict[str, Any]]:
    return read_json(_store_path(), "sessions file", dict, {})


def update_sessions(change: Callable[[dict[str, dict[str, Any]]], Any]) -> Any:
    path = _store_path()
    with locked(path):
        data: dict[str, dict[str, Any]] = read_json(path, "sessions file", dict, {})
        result = change(data)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data))
        os.replace(tmp, path)
        return result


def state_dir() -> str:
    base = os.environ.get("XDG_STATE_HOME") or os.path.join(os.path.expanduser("~"), ".local/state")
    return os.path.join(base, "claude-session")


def self_path() -> str:
    return str(Path(sys.argv[0]).resolve())


def _abs(path: str | None) -> str | None:
    return None if path is None else os.path.join(os.getcwd(), path)


def parse_duration(
    ctx: Context, flag: str, text: str, *, bound: bool = False, positive: bool = False
) -> float | None:
    """The real duration grammar: seconds or number+unit, at most 36500d; ``bound`` also takes
    ``none`` and refuses zero, ``positive`` refuses zero."""
    if bound and text == "none":
        return None
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)(ms|s|m|h|d)", text)
    if match:
        seconds = float(match.group(1)) * DURATION_UNITS[match.group(2)]
    elif re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", text):
        seconds = float(text)
    else:
        ctx.usage(f"{flag}: invalid duration: expected seconds or <number><unit>")
    if seconds > MAX_SECONDS:
        ctx.usage(f"{flag}: invalid duration: must be at most 36500d")
    if seconds <= 0 and (bound or positive):
        ctx.usage(f"{flag} must be positive, got {text!r}")
    return seconds


def python_int(ctx: Context, flag: str, text: str) -> int:
    """argparse's ``type=int``: Python's own ``int()`` (``+5``, `` 5``, ``٥``, ``1_0`` all pass)."""
    try:
        return int(text)
    except ValueError:
        ctx.usage(f"argument {flag}: invalid int value: {text!r}")


def escaped_size(value: str) -> int:
    return len(json.dumps(value, ensure_ascii=True))


# --- argument checks -----------------------------------------------------------------------


def session_ref(ctx: Context, raw: str) -> str:
    """Resolve an id input like the real tool: ASCII case-insensitive, ``cs-`` optional.

    Fewer than 4 characters after ``cs-`` exits 2. Anything else that cannot be an id (not
    4 to 8 characters of ``[a-z2-7]``, e.g. a longer or malformed id) is unknown (10), like a
    full id or prefix no known, non-forgotten session starts with; several matches exit 21.
    """
    text = raw.translate(str.maketrans("ABCDEFGHIJKLMNOPQRSTUVWXYZ", "abcdefghijklmnopqrstuvwxyz"))
    body = text[3:] if text.startswith("cs-") else text
    if len(body) < 4:
        ctx.usage(f"session id {raw!r} is too short: give at least 4 characters after cs-")
    if not re.fullmatch(r"[a-z2-7]{4,8}", body):
        ctx.fail(10, "not_found", f"unknown session: {raw}")
    matches = sorted(i for i in load_sessions() if i.startswith(f"cs-{body}"))
    if not matches:
        ctx.fail(10, "not_found", f"unknown session: {raw}")
    if len(matches) > 1:
        ctx.fail(21, "ambiguous", f"ambiguous session id: {raw}", matches=matches)
    return matches[0]


def known_run(ctx: Context, session_id: str, run: int | None) -> int:
    """The run number to report: the current one, or ``run`` if the session has it (else 10)."""
    current = int(load_sessions()[session_id].get("run", 1))
    if run is None:
        return current
    if not 1 <= run <= current:
        ctx.fail(10, "not_found", f"session {session_id} has no run {run}")
    return run


def check_kind(ctx: Context, flag: str, value: str) -> None:
    if re.fullmatch(g.KIND, value) is None:
        ctx.usage(f"{flag}: invalid kind name {value!r}")


def check_launch_flags(ctx: Context, p: Parsed) -> None:
    if (mode := p.get("--mode")) is not None and mode not in ("interactive", "headless"):
        ctx.usage(f"invalid --mode: {mode!r}")
    if (kind := p.get("--kind")) is not None:
        check_kind(ctx, "--kind", kind)
    if (skill := p.get("--skill")) is not None:
        require_pattern(skill, g.SKILL, "--skill")
    if (repo := p.get("--repo")) is not None:
        require_pattern(repo, CS_REPO_ID, "--repo")
    sandbox = p.get("--sandbox")
    if sandbox is not None and sandbox not in ("required", "preferred", "off"):
        ctx.usage(f"invalid --sandbox: {sandbox!r}")
    for label in p.multi("--label"):
        require_pattern(label, g.LABEL, "label")
    # --actor and --decision-id have no grammar here, only caps (interface.md, launch flags):
    # the actor at most 4 KiB, the decision ids at most 256 KiB together, escaped as JSON.
    check_audit_values(ctx, p)
    if (timeout := p.get("--timeout")) is not None:
        parse_duration(ctx, "--timeout", timeout, bound=True)
    if (turns := p.get("--max-turns")) is not None:
        if not 1 <= python_int(ctx, "--max-turns", turns) <= MAX_INTEGER:
            ctx.usage(f"--max-turns must be between 1 and {MAX_INTEGER}")
    model = p.get("--model")
    if model is not None and model.startswith("-"):
        ctx.usage("--model must not start with '-'")
    hook_names: list[str] = []
    for hook in p.multi("--hook"):
        name, eq, _cmd = hook.partition("=")
        if name not in HOOK_NAMES or not eq:
            ctx.usage(f"invalid --hook: {hook!r}")
        hook_names.append(name)
    for entry in p.multi("--hook-timeout"):
        name, eq, duration = entry.partition("=")
        if name not in hook_names or not eq:
            ctx.usage(f"--hook-timeout {entry!r} without a matching --hook")
        parse_duration(ctx, f"--hook-timeout {name}", duration, positive=True)
    for resettable in p.multi("--reset"):
        if resettable not in RESET_NAMES:
            ctx.usage(f"argument --reset: invalid choice: {resettable!r}")
        if f"--{resettable}" in p.flags or (
            resettable in ("auto-exit", "implicit-done") and f"--no-{resettable}" in p.flags
        ):
            ctx.usage(f"--reset {resettable} together with --{resettable}")
    for var in p.multi("--env-allow"):
        require_pattern(var, g.ENV_NAME, "environment name")
    for var in p.multi("--env-allowlist-only"):
        if var != "":  # `--env-allowlist-only ''` is an empty list
            require_pattern(var, g.ENV_NAME, "environment name")
    sources = p.get("--setting-sources")
    if sources:  # '' means none
        for source in sources.split(","):
            if source not in SETTING_SOURCES:
                ctx.usage(f"--setting-sources: unknown source {source!r}")
    seen: set[str] = set()
    for path in p.multi("--input-file"):
        if path == "-":
            ctx.usage("--input-file - is not allowed")
        base = os.path.basename(path)
        if base in seen or base in (".", "..") or not re.fullmatch(r"[A-Za-z0-9._-]{1,128}", base):
            ctx.usage(f"invalid or duplicate input file name: {base!r}")
        seen.add(base)
    prompt = p.get("--prompt-file")
    if prompt is not None and prompt != "-" and not os.path.isfile(prompt):
        ctx.usage(f"--prompt-file {prompt!r} is not a readable file")
    if prompt == "-" and p.get("--mode", "interactive") == "interactive" and ctx.stdin_is_tty():
        ctx.usage("--prompt-file - with a terminal on stdin needs a headless run")
    # Human-only flags: an explicit actor must be human:<non-blank user>; without --actor,
    # stdin must be a TTY.
    for flag in ("--allow-tainted", "--override-escalation"):
        if not p.has(flag):
            continue
        actor = p.get("--actor")
        if actor is not None:
            if not (actor.startswith("human:") and actor[6:].strip()):
                ctx.usage(f"{flag} needs a human actor")
        elif not ctx.stdin_is_tty():
            ctx.usage(f"{flag} needs a human actor (stdin is not a terminal)")
    if prompt == "-":
        ctx.read_stdin()


def check_audit_values(ctx: Context, p: Parsed) -> None:
    actor = p.get("--actor")
    if actor is not None and (
        len(actor) > ACTOR_MAX_ESCAPED or escaped_size(actor) > ACTOR_MAX_ESCAPED
    ):
        ctx.usage(f"--actor is too long (at most {ACTOR_MAX_ESCAPED} bytes escaped)")
    budget = DECISION_IDS_MAX_ESCAPED
    for item in p.multi("--decision-id"):
        size = escaped_size(item) + 1
        if len(item) >= budget or size > budget:
            ctx.usage("the --decision-id values are too long")
        budget -= size


# --- dispatch ------------------------------------------------------------------------------


def new_entry(**fields: Any) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "run": 1,
        "adopted": False,
        "final": False,
        "outcome": None,
        "end_reason": None,
        "stop_requested": False,
        "outcome_file": None,
        "labels": [],
        "kind": "implement",
        "mode": "headless",
        "repo": None,
        "forked_from": None,
    }
    entry.update(fields)
    return entry


def entry_state(entry: dict[str, Any]) -> str:
    if entry.get("final"):
        return OUTCOMES[entry.get("outcome") or "done"][1]
    return "working" if entry.get("adopted") else "starting"


def dispatch(ctx: Context) -> tuple[int, Any, str]:
    argv = ctx.argv
    before = argv[: argv.index("--")] if "--" in argv else argv
    if "--version" in argv and "--" not in argv[: argv.index("--version")]:
        raise Exit(0, f"claude-session {os.environ.get('FAKEBIN_VERSION', VERSION)}\n")
    ctx.json_mode = "--json" in before
    if not argv:
        ctx.usage("the following arguments are required: <command>")
    if argv[0] in ("-h", "--help"):
        raise Exit(0, "usage: claude-session [-h] [--version] <command> ...\n")
    command, rest = argv[0], list(argv[1:])
    if command.startswith("-"):
        ctx.usage(f"unrecognized arguments: {command}")  # global flags follow the command
    if command in UNSUPPORTED:
        ctx.fail(1, "fake_unsupported", f"the fake does not implement {command!r}")
    if command not in SPECS:
        ctx.usage(f"argument <command>: invalid choice: {command!r}")
    if any(token in ("-h", "--help") for token in rest):
        raise Exit(0, f"usage: claude-session {command} ...\n")
    if command == "label":
        # `label` is a parser of its own: the action comes first, the flags after it.
        if not rest or rest[0] not in ("add", "remove"):
            ctx.usage("label needs add or remove first")
        command = f"label {rest.pop(0)}"
    p = parse_args(rest, SPECS[command.split()[0]], argparse_values=True)
    ctx.json_mode = p.has("--json")
    if p.has("--verbose") and p.has("--quiet"):
        ctx.usage("--verbose and --quiet are mutually exclusive")
    return HANDLERS[command](ctx, p)


def do_start(ctx: Context, p: Parsed) -> tuple[int, Any, str]:
    if p.positionals:
        ctx.usage(f"unexpected argument: {p.positionals[0]!r}")
    if not p.has("--json"):
        ctx.usage("start requires --json")
    resume = p.get("--resume")
    for flag in ("--reset", "--fork", "--allow-tainted", "--override-escalation"):
        if p.has(flag) and resume is None:
            ctx.usage(f"{flag} needs --resume")
    if p.has("--override-escalation") and p.has("--fork"):
        ctx.usage("--override-escalation cannot be combined with --fork")
    check_launch_flags(ctx, p)
    mode = p.get("--mode", "interactive")
    if (
        resume is None
        and mode == "headless"
        and not p.has("--skill")
        and not p.has("--prompt-file")
    ):
        ctx.usage("a headless new session needs --skill or --prompt-file")
    existing = session_ref(ctx, resume) if resume is not None else None
    fork = p.has("--fork")
    if p.has("--dry-run"):
        return 0, dry_run_plan(p, existing, fork), "start"
    ctx.check_scripted_error("start")
    new_id = existing if (existing is not None and not fork) else base32_id("cs-", counter("cs"))
    labels = sorted(set(p.multi("--label")))

    def record(data: dict[str, dict[str, Any]]) -> int:
        previous = data.get(new_id, {})
        run = int(previous.get("run", 0)) + 1
        data[new_id] = new_entry(
            run=run,
            outcome_file=_abs(p.get("--outcome-file")),
            labels=sorted(set(previous.get("labels", [])) | set(labels)),
            kind=p.get("--kind", previous.get("kind", "implement")),
            mode=p.get("--mode", previous.get("mode", mode)),
            repo=p.get("--repo", previous.get("repo")),
            forked_from=existing if fork else previous.get("forked_from"),
        )
        return run

    run = update_sessions(record)
    # the real tool creates the session directory and its `work/` at `start`
    os.makedirs(os.path.join(state_dir(), "sessions", new_id, "work"), exist_ok=True)
    out = {
        "csession_id": new_id,
        "csession_dir": os.path.join(state_dir(), "sessions", new_id),
        "argv": [self_path(), "exec", "--session-id", new_id, "--run", str(run)],
        "run": run,
        "forked_from": existing if fork else None,
        "expires_at": "2026-11-03T10:01:00Z",
        "required_env_names": [],
        "env_inherit": False,
        "warnings": [],
    }
    return 0, out, "start"


def implicit_done_value(p: Parsed) -> bool:
    """The effective implicit_done: the last of the two flags wins, default true."""
    given = [t for t in p.tokens if t in ("--implicit-done", "--no-implicit-done")]
    return given[-1] == "--implicit-done" if given else True


def dry_run_plan(p: Parsed, existing: str | None, fork: bool) -> dict[str, Any]:
    kind = p.get("--kind", "implement")
    run = 1
    if existing is not None and not fork:
        run = int(load_sessions().get(existing, {}).get("run", 0)) + 1
    return {
        "dry_run": True,
        "csession_id": None if fork else existing,
        "run": run,
        "fork": fork,
        "forked_from": existing if fork else None,
        "launch_kind": kind,
        "run_kind": kind,
        "labels": sorted(set(p.multi("--label"))),
        "taints": False,
        "mode": p.get("--mode", "interactive"),
        # A resume runs in the previous run's cwd (interface.md, --cwd); the fake records none,
        # so it plans the session dir. A launch or fork without --cwd uses the caller's.
        "cwd": p.get("--cwd")
        or ("session-dir" if existing is not None and not fork else os.getcwd()),
        "repo": p.get("--repo"),
        "skill": p.get("--skill"),
        "has_prompt": p.has("--prompt-file"),
        "input_files": [os.path.basename(f) for f in p.multi("--input-file")],
        "implicit_done": implicit_done_value(p),
        "permission_profile": None,
        "timeout": None,
        "max_turns": None,
        "actor": p.get("--actor", "auto:unknown"),
        "override_escalation": p.has("--override-escalation"),
        "allow_tainted": p.has("--allow-tainted"),
        "sandbox_lowered": False,
        "config_hash_changed": None,
        "profile": {
            "name": kind,
            "builtin": True,
            "fallback": False,
            "permission_profile": "default",
            "sandbox": p.get("--sandbox", "off"),
            "sandbox_available": False,
            "sandbox_problems": [],
            "reads_third_party": p.has("--reads-third-party"),
            "implicit_done": implicit_done_value(p),
            "setting_sources": [],
            "mcp": "none",
            "cwd": "session-dir",
            "env": {"inherit": False, "allow": [], "set_names": [], "path_prepend": []},
            "required_env_names": [],
            "path": "/usr/bin:/bin",
            "read_dirs": [],
            "write_dirs": [],
            "timeout": None,
            "interactive_timeout": None,
            "max_turns": None,
            "auto_exit": True,
            "backstop": False,
            "notes_context": False,
            "model": p.get("--model"),
            "claude_args": [],
            "warnings": [],
        },
        "origins": {},
        "settings": {},
        "claude_argv": [],
        "env_names": [],
        "sandbox": {
            "mode": p.get("--sandbox", "off"),
            "available": False,
            "problems": [],
            "allow_read": [],
            "allow_write": [],
            "deny_read": [],
            "deny_write": [],
        },
        "hooks": [],
        "would_refuse": None,
        "would_fail": None,
        "warnings": [],
    }


def do_exec(ctx: Context, p: Parsed) -> tuple[int, Any, str]:
    if p.positionals:
        ctx.usage(f"unexpected argument: {p.positionals[0]!r}")
    adopting = p.get("--session-id")
    if adopting is None:
        if p.has("--run"):
            ctx.usage("--run needs --session-id")
        for flag in ("--reset", "--fork", "--allow-tainted", "--override-escalation"):
            if p.has(flag):
                ctx.usage(f"{flag} is not accepted by exec")
        check_launch_flags(ctx, p)
        require_terminal(ctx, p, "exec")
        if p.has("--dry-run"):
            return 0, dry_run_plan(p, None, False), "exec"
        ctx.check_scripted_error("exec")
        return run_session(
            ctx,
            "exec",
            base32_id("cs-", counter("cs")),
            1,
            _abs(p.get("--outcome-file")),
            mode=p.get("--mode", "interactive"),
            fields={
                "kind": p.get("--kind", "implement"),
                "labels": sorted(set(p.multi("--label"))),
                "repo": p.get("--repo"),
            },
        )
    extra = [flag for flag in p.flags if flag not in ADOPT_FLAGS]
    if extra:
        ctx.usage(f"{extra[0]} is not accepted with --session-id")
    run_arg = p.get("--run")
    if run_arg is not None and not 1 <= python_int(ctx, "--run", run_arg) <= MAX_INTEGER:
        ctx.usage(f"--run must be a positive integer (at most {MAX_INTEGER})")
    session_id = session_ref(ctx, adopting)
    entry = load_sessions()[session_id]
    if run_arg is not None and int(run_arg) != entry["run"]:
        ctx.fail(20, "refused", "the pending run expired", reason="expired")
    if entry.get("final") and not entry.get("adopted"):
        ctx.fail(20, "refused", "the pending run was canceled", reason="expired")
    if p.has("--dry-run"):
        return 0, dry_run_plan(p, session_id, False), "exec"

    def adopt(data: dict[str, dict[str, Any]]) -> bool:
        if data[session_id]["adopted"]:
            return False
        data[session_id]["adopted"] = True
        return True

    ctx.check_scripted_error("exec")
    if not update_sessions(adopt):
        ctx.fail(20, "refused", "the run was already adopted", reason="already_adopted")
    outcome_file = _abs(p.get("--outcome-file")) or entry.get("outcome_file")
    return run_session(
        ctx,
        "exec",
        session_id,
        entry["run"],
        outcome_file,
        mode=entry.get("mode", "headless"),
        forked_from=entry.get("forked_from"),
    )


def require_terminal(ctx: Context, p: Parsed, command: str) -> None:
    interactive = p.get("--mode", "interactive") == "interactive"
    if interactive and not (ctx.stdin_is_tty() and sys.stdout.isatty()):
        ctx.usage(f"interactive {command} needs a terminal on stdin and stdout")


SIGNAL_CAUSES = {
    signal.SIGTERM: "forwarded_term",
    signal.SIGHUP: "forwarded_hup",
    signal.SIGINT: "forwarded_int",
}


def begin_run(
    session_id: str,
    run: int,
    mode: str,
    forked_from: str | None,
    fields: dict[str, Any] | None,
) -> None:
    """Record the run as live (and the session as known, when ``exec`` created it)."""

    def change(data: dict[str, dict[str, Any]]) -> None:
        entry = data.setdefault(session_id, new_entry(**(fields or {})))
        entry.update(
            run=run,
            adopted=True,
            final=False,
            outcome=None,
            end_reason=None,
            stop_requested=False,
            mode=mode,
        )
        if forked_from is not None:
            entry["forked_from"] = forked_from

    update_sessions(change)


def end_run(session_id: str, outcome: str) -> None:
    def change(data: dict[str, dict[str, Any]]) -> None:
        if session_id in data:
            data[session_id].update(final=True, outcome=outcome)

    update_sessions(change)


CHILD_SCRIPT = """
import os, signal, sys, time
path, parent = sys.argv[1], int(sys.argv[2])
def note(signum, _frame):
    with open(path, "a") as handle:
        handle.write(signal.Signals(signum).name + "\\n")
    if signum == signal.SIGTERM:
        os._exit(0)
for number in (signal.SIGTERM, signal.SIGCONT, signal.SIGHUP):
    signal.signal(number, note)
deadline = time.monotonic() + 120
while time.monotonic() < deadline and os.getppid() == parent:
    time.sleep(0.05)
"""


def spawn_child(ctx: Context) -> subprocess.Popen[bytes]:
    """A real child of the fake ``exec`` (a stand-in for ``claude``): it records the signals it
    gets in a file named by the ``child_started`` record, and ends on SIGTERM."""
    path = os.path.join(os.environ.get("FAKEBIN_LOG", "/tmp") + ".child-signals")
    child = subprocess.Popen(
        [sys.executable, "-c", CHILD_SCRIPT, path, str(os.getpid())], stdin=subprocess.DEVNULL
    )
    log_event("child_started", id=ctx.invoke_id, pid=child.pid, signals_file=path)
    return child


def run_session(
    ctx: Context,
    command: str,
    session_id: str,
    run: int,
    outcome_file: str | None,
    *,
    mode: str,
    forked_from: str | None = None,
    fields: dict[str, Any] | None = None,
) -> tuple[int, Any, str]:
    """Run the fake session to its scripted outcome, as ``exec`` does in the foreground.

    A scripted ``hang`` without ``on_signal`` follows the interface's signal table: SIGTERM and
    SIGHUP end the run ``killed`` (exit 15, end reason ``forwarded_term``/``forwarded_hup``, or
    ``stopped`` after a ``stop``), SIGINT likewise in a headless run (``forwarded_int``) and
    is ignored in an interactive one. Only the first signal counts: the handlers are replaced
    by a dropping handler at once, so a second signal cannot interrupt the finalization.
    """
    begin_run(session_id, run, mode, forked_from, fields)
    response = ctx.response(command)
    name = response.get("outcome", "done")
    numbers = (signal.SIGTERM, signal.SIGHUP, signal.SIGINT)
    killing = bool(response.get("hang")) and "on_signal" not in response
    child = spawn_child(ctx) if response.get("child") else None
    if killing:

        def killed(signum: int, _frame: Any) -> None:
            for number in numbers:
                signal.signal(number, drop_signal)
            ctx.signaled.append(signum)
            if child is not None:  # like claude-session's exec: pass the signal to `claude`
                child.send_signal(signum)
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
            stopped = load_sessions().get(session_id, {}).get("stop_requested")
            reason = "stopped" if stopped else SIGNAL_CAUSES[signal.Signals(signum)]
            obj = outcome_object("killed", session_id, run, end_reason=reason)
            obj["forked_from"] = forked_from
            write_outcome_file(outcome_file, obj)
            end_run(session_id, "killed")
            raise Exit(15, json.dumps(obj) + "\n" if ctx.json_mode else "")

        for number in numbers:
            if number == signal.SIGINT and mode == "interactive":
                signal.signal(number, signal.SIG_IGN)
            else:
                signal.signal(number, killed)
    try:
        ctx.behave(command)
    finally:
        if killing:  # the run is over: a late signal must not overwrite the outcome below
            for number in numbers:
                signal.signal(number, drop_signal)
    obj = outcome_object(name, session_id, run)
    if isinstance(response.get("end_reason"), str):  # e.g. a `not-started` ended never_adopted
        obj["end_reason"] = response["end_reason"]
    obj["forked_from"] = forked_from
    write_outcome_file(outcome_file, obj)
    end_run(session_id, name)
    return int(obj["exit_code"]), obj, command


def write_outcome_file(path: str | None, obj: dict[str, Any]) -> None:
    """Temporary file plus rename, mode 0600 (the real tool also walks parents with O_NOFOLLOW).

    A failed write never changes the outcome or the exit code: it is a warning on stderr and an
    ``outcome_file_failed`` record in the invocation log, and the temporary file is removed.
    """
    if path is None:
        return
    tmp: str | None = None
    try:
        if path == "" or path.endswith("/"):
            raise OSError("the path is empty or names a directory")
        tmp = os.path.join(os.path.dirname(path), f".{os.path.basename(path)}.{os.getpid()}.tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as handle:
            handle.write(json.dumps(obj) + "\n")
        os.replace(tmp, path)
        tmp = None
    except (OSError, ValueError) as problem:
        sys.stderr.write(f"claude-session: warning: outcome file not written: {problem}\n")
        log_event("outcome_file_failed", path=path, error=str(problem))
    finally:
        if tmp is not None:
            try:
                os.unlink(tmp)
            except OSError:
                pass


def do_resume(ctx: Context, p: Parsed) -> tuple[int, Any, str]:
    if len(p.positionals) != 1:
        ctx.usage("resume takes exactly one session id")
    if p.has("--override-escalation") and p.has("--fork"):
        ctx.usage("--override-escalation cannot be combined with --fork")
    check_launch_flags(ctx, p)
    session_id = session_ref(ctx, p.positionals[0])
    fork = p.has("--fork")
    if p.has("--dry-run"):
        return 0, dry_run_plan(p, session_id, fork), "resume"
    require_terminal(ctx, p, "resume")
    entry = load_sessions()[session_id]
    ctx.check_scripted_error("resume")
    if fork:
        new_id = base32_id("cs-", counter("cs"))
        run = 1
        forked_from: str | None = session_id
    else:
        new_id = session_id
        run = int(entry.get("run", 0)) + 1
        forked_from = entry.get("forked_from")
    return run_session(
        ctx,
        "resume",
        new_id,
        run,
        _abs(p.get("--outcome-file")),
        mode=p.get("--mode", entry.get("mode", "interactive")),
        forked_from=forked_from,
        fields={
            "kind": p.get("--kind", entry.get("kind", "implement")),
            "labels": sorted(set(p.multi("--label"))),
            "repo": p.get("--repo", entry.get("repo")),
        },
    )


def do_stop(ctx: Context, p: Parsed) -> tuple[int, Any, str]:
    if len(p.positionals) != 1:
        ctx.usage("stop takes exactly one session id")
    check_audit_values(ctx, p)
    if (grace := p.get("--grace")) is not None:
        parse_duration(ctx, "--grace", grace)
    session_id = session_ref(ctx, p.positionals[0])
    entry = load_sessions()[session_id]
    dry = p.has("--dry-run")
    final = bool(entry.get("final"))
    pending = not entry.get("adopted") and not final
    out: dict[str, Any] = {
        "csession_id": session_id,
        "run": int(entry.get("run", 1)),
        "already_final": final,
        "canceled": pending,
        "dry_run": dry,
        "ended": True,
        "run_exit_code": None,
        "signals": [],
        "state": "",
        "outcome": None,
        "end_reason": None,
        "recorded_by_stop": False,
    }
    ctx.check_scripted_error("stop")
    if final:
        name = entry.get("outcome") or "done"
        code, state, end_reason, _phase, _report = OUTCOMES[name]
        out.update(run_exit_code=code, state=state, outcome=name, end_reason=end_reason)
    elif pending:
        out.update(
            run_exit_code=12, state="not-started", outcome="not-started", end_reason="canceled"
        )
        if not dry:

            def cancel(data: dict[str, dict[str, Any]]) -> None:
                data[session_id].update(
                    final=True, outcome="not-started", adopted=False, end_reason="canceled"
                )

            update_sessions(cancel)
    elif dry:
        out.update(ended=False, state="working")
    else:
        out.update(run_exit_code=15, state="killed", outcome="killed", end_reason="stopped")

        def request(data: dict[str, dict[str, Any]]) -> None:
            data[session_id].update(final=True, outcome="killed", stop_requested=True)

        update_sessions(request)
    return 0, out, "stop"


def do_wait(ctx: Context, p: Parsed) -> tuple[int, Any, str]:
    if len(p.positionals) != 1:
        ctx.usage("wait takes exactly one session id")
    if (timeout := p.get("--timeout")) is not None:
        parse_duration(ctx, "--timeout", timeout, bound=True)
    run_arg = p.get("--run")
    run_number = None if run_arg is None else python_int(ctx, "--run", run_arg)
    session_id = session_ref(ctx, p.positionals[0])
    entry = load_sessions()[session_id]
    run = known_run(ctx, session_id, run_number)
    ctx.behave("wait")
    name = ctx.response("wait").get("outcome") or entry.get("outcome") or "done"
    obj = {
        **outcome_object(name, session_id, run),
        "timed_out": False,
        "run_exit_code": OUTCOMES[name][0],
        "pending": False,
    }
    obj["forked_from"] = entry.get("forked_from")
    write_outcome_file(_abs(p.get("--outcome-file")), obj)
    return OUTCOMES[name][0], obj, "wait"


def do_status(ctx: Context, p: Parsed) -> tuple[int, Any, str]:
    if len(p.positionals) != 1:
        ctx.usage("status takes exactly one session id")
    run_arg = p.get("--run")
    run_number = None if run_arg is None else python_int(ctx, "--run", run_arg)
    session_id = session_ref(ctx, p.positionals[0])
    entry = load_sessions()[session_id]
    run = known_run(ctx, session_id, run_number)
    kind = entry.get("kind", "implement")
    mode = entry.get("mode", "headless")
    name = ctx.response("status").get("outcome") or entry.get("outcome") or "done"
    code, state, end_reason, phase, _ = OUTCOMES[name]
    outcome = outcome_object(name, session_id, run)
    if entry.get("final") and entry.get("end_reason") and not ctx.response("status").get("outcome"):
        end_reason = outcome["end_reason"] = entry["end_reason"]  # e.g. a pending run's `stop`
    out: dict[str, Any] = {
        "csession_id": session_id,
        "created_at": FIXED_TS,
        "launch_kind": kind,
        "run_kind": kind,
        "state": state,
        "activity": None,
        "needs_input_reason": None,
        "live": False,
        "tainted": False,
        "tainted_at": None,
        "tainted_by_run": None,
        "forked_from": entry.get("forked_from"),
        "labels": entry.get("labels", []),
        "repo": None,
        "mode": mode,
        "current_run": run,
        "pending": False,
        "expires_at": None,
        "updated_at": FIXED_TS,
        "csession_dir": os.path.join(state_dir(), "sessions", session_id),
        "claude_session_id": outcome["claude_session_id"],
        "claude_session_ids": [],
        "run": {
            "run": run,
            "run_kind": kind,
            "reads_third_party": False,
            "implicit_done": True,
            "mode": mode,
            "cwd": None,
            "repo": None,
            "model": None,
            "skill": None,
            "has_prompt": False,
            "sandbox": "off",
            "sandbox_active": False,
            "permission_profile": None,
            "decision_ids": [],
            "input_files": [],
            "pending": False,
            "expires_at": None,
            "started_at": FIXED_TS,
            "session_started_at": outcome["session_started_at"],
            "ended_at": FIXED_TS,
            "state": state,
            "phase": phase,
            "message": None,
            "stop_requested_at": None,
            "warnings": [],
            "state_since": FIXED_TS,
            "wrapper_pid": None,
            "wrapper_alive": False,
            "child_pid": None,
            "child_alive": False,
            "backstop_used": False,
            "report": outcome["report"],
            "outcome": name,
            "end_reason": end_reason,
            "session_end_reason": None,
            "exit_code": code,
            "result_path": None,
            "cost_usd": outcome["cost_usd"],
            "num_turns": None,
            "is_error": None,
        },
        "runs": run,
        "recent_events": [],
        "warnings": [],
    }
    if (
        ctx.response("status").get("pending_unadopted") is True
        and not entry.get("adopted")
        and not entry.get("final")
    ):
        # Opt-in (a scripted ``pending_unadopted``): a run nothing adopted yet is reported
        # pending, in its launch phase, as the interface says (unverified, FIDELITY.md).
        out.update(state="starting", pending=True, expires_at="2026-11-03T10:01:00Z")
        out["run"].update(
            state="starting",
            phase="launch",
            pending=True,
            expires_at="2026-11-03T10:01:00Z",
            session_started_at=None,
            ended_at=None,
            report=None,
            outcome=None,
            end_reason=None,
            exit_code=None,
        )
    return 0, out, "status"


def list_item(session_id: str, entry: dict[str, Any]) -> dict[str, Any]:
    pending = not entry.get("adopted", False) and not entry.get("final", False)
    final = bool(entry.get("final", False))
    outcome = entry.get("outcome") if final else None
    return {
        "csession_id": session_id,
        "created_at": FIXED_TS,
        "launch_kind": entry.get("kind", "implement"),
        "run_kind": entry.get("kind", "implement"),
        "state": entry_state(entry),
        "activity": None,
        "needs_input_reason": None,
        "live": not final,
        "tainted": False,
        "tainted_at": None,
        "tainted_by_run": None,
        "forked_from": entry.get("forked_from"),
        "labels": entry.get("labels", []),
        "repo": entry.get("repo"),
        "mode": entry.get("mode", "headless"),
        "current_run": entry.get("run", 1),
        "pending": pending,
        "expires_at": "2026-11-03T10:01:00Z" if pending else None,
        "updated_at": FIXED_TS,
        "phase": "launch" if pending else "session",
        "last_outcome": outcome,
        "last_end_reason": OUTCOMES[outcome][2] if outcome else None,
        "exit_code": OUTCOMES[outcome][0] if outcome else None,
        "report_status": OUTCOMES[outcome][4] if outcome else None,
    }


def check_choices(ctx: Context, flag: str, values: list[str], choices: tuple[str, ...]) -> None:
    for value in values:
        if value not in choices:
            ctx.usage(f"argument {flag}: invalid choice: {value!r}")


def do_list(ctx: Context, p: Parsed) -> tuple[int, Any, str]:
    if p.positionals:
        ctx.usage("list takes no positional arguments")
    for label in p.multi("--label"):
        require_pattern(label, g.LABEL, "label")
    check_choices(ctx, "--state", p.multi("--state"), STATES)
    check_choices(ctx, "--report-status", p.multi("--report-status"), (*REPORT_STATUSES, "none"))
    mode = p.get("--mode")
    if mode is not None:
        check_choices(ctx, "--mode", [mode], ("interactive", "headless"))
    for flag in ("--kind", "--launch-kind", "--run-kind"):
        if (value := p.get(flag)) is not None:
            check_kind(ctx, flag, value)
    if (repo := p.get("--repo")) is not None:
        require_pattern(repo, CS_REPO_ID, "--repo")
    forked = p.get("--forked-from")
    if forked is not None and len(forked.lower().removeprefix("cs-")) < 4:
        ctx.usage(f"--forked-from: session id {forked!r} is too short")
    since = p.get("--since")
    if since is not None and not DURATION.fullmatch(since):
        try:
            datetime.fromisoformat(since)
        except ValueError:
            ctx.usage("--since: expected an ISO 8601 time or a duration")
        # a duration too large is refused by parse_duration below
    elif since is not None:
        parse_duration(ctx, "--since", since)
    limit = p.get("--limit")
    if limit is not None and python_int(ctx, "--limit", limit) < 1:
        ctx.usage("--limit must be at least 1")
    forked_prefix = "cs-" + forked.lower().removeprefix("cs-") if forked is not None else ""
    wanted = set(p.multi("--label"))
    states = set(p.multi("--state"))
    reports = set(p.multi("--report-status"))
    items: list[dict[str, Any]] = []
    for session_id, entry in sorted(load_sessions().items()):
        item = list_item(session_id, entry)
        kinds = {item["launch_kind"], item["run_kind"]}
        checks = [
            wanted <= set(item["labels"]),
            not states or item["state"] in states,
            p.get("--kind") is None or p.get("--kind") in kinds,
            p.get("--launch-kind") in (None, item["launch_kind"]),
            p.get("--run-kind") in (None, item["run_kind"]),
            mode is None or item["mode"] == mode,
            not p.has("--pending") or item["pending"],
            not p.has("--live") or item["live"],
            not p.has("--needs-input") or item["state"] == "needs-input",
            not p.has("--tainted") or item["tainted"],
            repo is None or item["repo"] == repo,
            not reports or (item["report_status"] or "none") in reports,
            forked is None
            or (bool(item["forked_from"]) and item["forked_from"].startswith(forked_prefix)),
        ]
        if all(checks):
            items.append(item)
    if limit is not None:
        items = items[: int(limit)]
    return 0, {"sessions": items}, "list"


def label_handler(sub: str) -> Callable[[Context, Parsed], tuple[int, Any, str]]:
    def handler(ctx: Context, p: Parsed) -> tuple[int, Any, str]:
        if len(p.positionals) < 2:
            ctx.usage(f"label {sub} needs a session id and at least one key=value")
        pairs = p.positionals[1:]
        for pair in pairs:
            require_pattern(pair, g.LABEL, "label")
        session_id = session_ref(ctx, p.positionals[0])
        dry = p.has("--dry-run")

        def change(data: dict[str, dict[str, Any]]) -> tuple[list[str], list[str]]:
            entry = data[session_id]  # the id resolved: a label command never creates a session
            current = set(entry.get("labels", []))
            wanted = set(pairs)
            changed = sorted(wanted - current) if sub == "add" else sorted(wanted & current)
            result = current | wanted if sub == "add" else current - wanted
            if not dry:
                entry["labels"] = sorted(result)
            return sorted(result), changed

        ctx.check_scripted_error(f"label {sub}")
        labels, changed = update_sessions(change)
        out = {
            "csession_id": session_id,
            "labels": labels,
            "added": changed if sub == "add" else [],
            "removed": changed if sub == "remove" else [],
            "dry_run": dry,
        }
        return 0, out, f"label {sub}"

    return handler


def do_forget(ctx: Context, p: Parsed) -> tuple[int, Any, str]:
    bulk = [f for f in ("--older-than", "--state", "--kind", "--label") if p.has(f)]
    if p.positionals and bulk:
        ctx.usage(f"{bulk[0]} cannot be used with a session id")
    if not p.positionals and not p.has("--older-than"):
        ctx.usage("forget needs a session id or --older-than")
    if len(p.positionals) > 1:
        ctx.usage("forget takes one session id")
    check_choices(ctx, "--state", p.multi("--state"), STATES)
    if (kind := p.get("--kind")) is not None:
        check_kind(ctx, "--kind", kind)
    for label in p.multi("--label"):
        require_pattern(label, g.LABEL, "label")
    if (older := p.get("--older-than")) is not None:
        parse_duration(ctx, "--older-than", older)
    check_audit_values(ctx, p)
    ids = [session_ref(ctx, p.positionals[0])] if p.positionals else []
    dry = p.has("--dry-run")
    ctx.check_scripted_error("forget")
    if ids and not dry:
        update_sessions(lambda data: data.pop(ids[0], None))
    out = {
        "dry_run": dry,
        "forgotten": [
            {"csession_id": i, "forgotten_at": FIXED_TS, "delete_after": FIXED_TS} for i in ids
        ],
        "skipped": [],
    }
    return 0, out, "forget"


def do_reconcile(ctx: Context, p: Parsed) -> tuple[int, Any, str]:
    if p.positionals:
        ctx.usage("reconcile takes no arguments")
    out = {
        "already_running": False,
        "dry_run": p.has("--dry-run"),
        "changes": [],
        "orphans": [],
        "warnings": [],
    }
    return 0, out, "reconcile"


def do_verify_secret(ctx: Context, p: Parsed) -> tuple[int, Any, str]:
    ctx.json_mode = True
    if len(p.positionals) != 1:
        ctx.usage("verify-secret takes exactly one session id")
    if ctx.stdin_is_tty():
        ctx.usage("verify-secret reads the secret from a pipe, not a terminal")
    secret = ctx.read_stdin()
    if len(secret.encode()) > 4096:
        ctx.usage("secret too long")
    expected = ctx.response("verify-secret").get("secret")
    # A short or unknown id is simply no match (exit 11): the real tool gives no oracle.
    full = re.fullmatch(g.CS_ID, p.positionals[0].lower()) is not None
    match = full and expected is not None and secret.rstrip() == expected
    return (0 if match else 11), {"match": match}, "verify-secret"


HANDLERS: dict[str, Callable[[Context, Parsed], tuple[int, Any, str]]] = {
    "start": do_start,
    "exec": do_exec,
    "resume": do_resume,
    "stop": do_stop,
    "wait": do_wait,
    "status": do_status,
    "list": do_list,
    "label add": label_handler("add"),
    "label remove": label_handler("remove"),
    "forget": do_forget,
    "reconcile": do_reconcile,
    "verify-secret": do_verify_secret,
}


def main(argv: list[str]) -> int:
    return run_fake("claude-session", argv, dispatch)
