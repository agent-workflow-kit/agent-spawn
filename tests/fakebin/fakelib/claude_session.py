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
import sys
from collections.abc import Callable
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
    Usage,
    base32_id,
    counter,
    locked,
    parse_args,
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
RESET_NAMES = (
    "kind",
    "mode",
    "skill",
    "repo",
    "cwd",
    "timeout",
    "max-turns",
    "model",
    "permission-profile",
    "sandbox",
    "setting-sources",
    "auto-exit",
    "plugin-dir",
)
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
    path = _store_path()
    return json.loads(path.read_text()) if path.exists() else {}


def update_sessions(change: Callable[[dict[str, dict[str, Any]]], Any]) -> Any:
    path = _store_path()
    with locked(path):
        data: dict[str, dict[str, Any]] = json.loads(path.read_text()) if path.exists() else {}
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


def _seconds(duration: str) -> float:
    match = re.match(r"[0-9.]+", duration)
    return float(match.group()) if match else 0.0


# --- argument checks -----------------------------------------------------------------------


def session_ref(ctx: Context, raw: str) -> str:
    """Resolve an id input: case-insensitive, ``cs-`` optional, 4 to 8 characters of ``[a-z2-7]``.

    A full id is accepted as is (the fake does not know every session); a prefix resolves
    against the sessions ``start`` created: unknown exits 10, several exit 21.
    """
    text = raw.lower()
    body = text[3:] if text.startswith("cs-") else text
    if not re.fullmatch(r"[a-z2-7]{4,8}", body):
        ctx.usage(f"invalid session id: {raw!r}")
    if len(body) == 8:
        return f"cs-{body}"
    matches = sorted(i for i in load_sessions() if i.startswith(f"cs-{body}"))
    if not matches:
        ctx.fail(10, "not_found", f"no session matches {raw!r}")
    if len(matches) > 1:
        ctx.fail(21, "ambiguous", f"{raw!r} matches several sessions", matches=matches)
    return matches[0]


def check_launch_flags(ctx: Context, p: Parsed) -> None:
    if (mode := p.get("--mode")) is not None and mode not in ("interactive", "headless"):
        ctx.usage(f"invalid --mode: {mode!r}")
    if (kind := p.get("--kind")) is not None:
        require_pattern(kind, g.KIND, "--kind")
    if (skill := p.get("--skill")) is not None:
        require_pattern(skill, g.SKILL, "--skill")
    sandbox = p.get("--sandbox")
    if sandbox is not None and sandbox not in ("required", "preferred", "off"):
        ctx.usage(f"invalid --sandbox: {sandbox!r}")
    for label in p.multi("--label"):
        require_pattern(label, g.LABEL, "label")
    for decision in p.multi("--decision-id"):
        require_pattern(decision, g.DECISION_ID, "--decision-id")
    if (actor := p.get("--actor")) is not None:
        require_pattern(actor, g.ACTOR, "--actor")
    if (timeout := p.get("--timeout")) is not None and timeout != "none":
        require_pattern(timeout, g.CS_DURATION, "--timeout")
        if _seconds(timeout) == 0:
            ctx.usage("--timeout 0 is invalid (use none)")
    if (turns := p.get("--max-turns")) is not None:
        if not re.fullmatch(r"[0-9]+", turns) or not 1 <= int(turns) <= 9007199254740991:
            ctx.usage("--max-turns must be at least 1")
    model = p.get("--model")
    if model is not None and (model == "" or model.startswith("-")):
        ctx.usage("--model must not start with '-'")
    if p.has("--auto-exit") and p.has("--no-auto-exit"):
        ctx.usage("--auto-exit and --no-auto-exit are mutually exclusive")
    hook_names: list[str] = []
    for hook in p.multi("--hook"):
        name, eq, cmd = hook.partition("=")
        if name not in HOOK_NAMES or not eq or not cmd:
            ctx.usage(f"invalid --hook: {hook!r}")
        hook_names.append(name)
    for entry in p.multi("--hook-timeout"):
        name, eq, duration = entry.partition("=")
        if name not in hook_names or not eq:
            ctx.usage(f"--hook-timeout {entry!r} without a matching --hook")
        require_pattern(duration, g.CS_DURATION, "--hook-timeout")
    for resettable in p.multi("--reset"):
        if resettable not in RESET_NAMES:
            ctx.usage(f"unknown --reset name: {resettable!r}")
        if f"--{resettable}" in p.flags:
            ctx.usage(f"--reset {resettable} together with --{resettable}")
    for var in p.multi("--env-allow") + p.multi("--env-allowlist-only"):
        require_pattern(var, g.ENV_NAME + r"\*?", "environment name")
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


# --- dispatch ------------------------------------------------------------------------------


def split_command(argv: list[str]) -> tuple[str, list[str]]:
    index = next((i for i, token in enumerate(argv) if not token.startswith("-")), None)
    if index is None:
        raise Usage("a command is required")
    return argv[index], argv[:index] + argv[index + 1 :]


def dispatch(ctx: Context) -> tuple[int, Any, str]:
    argv = ctx.argv
    if "--version" in argv and "--" not in argv[: argv.index("--version")]:
        raise Exit(0, f"claude-session {os.environ.get('FAKEBIN_VERSION', VERSION)}\n")
    ctx.json_mode = "--json" in argv
    command, rest = split_command(argv)
    if command in UNSUPPORTED:
        ctx.fail(1, "fake_unsupported", f"the fake does not implement {command!r}")
    if command not in SPECS:
        ctx.usage(f"unknown command: {command!r}")
    if any(token in ("-h", "--help") for token in rest):
        raise Exit(0, f"usage: claude-session {command} ...\n")
    if command == "label":
        sub = next((t for t in rest if not t.startswith("-")), "")
        if sub not in ("add", "remove"):
            ctx.usage("label needs add or remove")
        rest.remove(sub)
        command = f"label {sub}"
    p = parse_args(rest, SPECS[command.split()[0]])
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
    new_id = existing if (existing is not None and not fork) else base32_id("cs-", counter("cs"))
    labels = sorted(set(p.multi("--label")))

    def record(data: dict[str, dict[str, Any]]) -> int:
        previous = data.get(new_id, {})
        run = int(previous.get("run", 0)) + 1
        data[new_id] = {
            "run": run,
            "adopted": False,
            "outcome_file": _abs(p.get("--outcome-file")),
            "labels": sorted(set(previous.get("labels", [])) | set(labels)),
            "kind": p.get("--kind", previous.get("kind", "implement")),
            "mode": p.get("--mode", previous.get("mode", mode)),
            "forked_from": existing if fork else previous.get("forked_from"),
        }
        return run

    ctx.check_scripted_error("start")
    run = update_sessions(record)
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
        "cwd": p.get("--cwd") or os.getcwd(),
        "repo": p.get("--repo"),
        "skill": p.get("--skill"),
        "has_prompt": p.has("--prompt-file"),
        "input_files": [os.path.basename(f) for f in p.multi("--input-file")],
        "implicit_done": True,
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
            "implicit_done": True,
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
        return run_session(
            ctx, "exec", base32_id("cs-", counter("cs")), 1, _abs(p.get("--outcome-file"))
        )
    extra = [flag for flag in p.flags if flag not in ADOPT_FLAGS]
    if extra:
        ctx.usage(f"{extra[0]} is not accepted with --session-id")
    run_arg = p.get("--run")
    if run_arg is not None and (not re.fullmatch(r"[0-9]+", run_arg) or int(run_arg) < 1):
        ctx.usage("--run must be a positive integer")
    session_id = session_ref(ctx, adopting)
    entry = load_sessions().get(session_id)
    if entry is None:
        ctx.fail(10, "not_found", f"no session {session_id}")
    if run_arg is not None and int(run_arg) != entry["run"]:
        ctx.fail(20, "refused", "the pending run expired", reason="expired")
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
    return run_session(ctx, "exec", session_id, entry["run"], outcome_file)


def require_terminal(ctx: Context, p: Parsed, command: str) -> None:
    interactive = p.get("--mode", "interactive") == "interactive"
    if interactive and not (ctx.stdin_is_tty() and sys.stdout.isatty()):
        ctx.usage(f"interactive {command} needs a terminal on stdin and stdout")


def run_session(
    ctx: Context, command: str, session_id: str, run: int, outcome_file: str | None
) -> tuple[int, Any, str]:
    """Run the fake session to its scripted outcome, as ``exec`` does in the foreground.

    A scripted ``hang`` without ``on_signal`` ends on SIGTERM/SIGHUP like the real wrapper
    does: the run is finalized ``killed`` (exit 15), the outcome file written, and nothing
    else signaled (the real wrapper forwards the signal to ``claude``).
    """
    response = ctx.response(command)
    name = response.get("outcome", "done")
    if response.get("hang") and "on_signal" not in response:

        def killed(signum: int, _frame: Any) -> None:
            ctx.signaled.append(signum)
            obj = outcome_object("killed", session_id, run, end_reason="forwarded_term")
            write_outcome_file(outcome_file, obj)
            raise Exit(15, json.dumps(obj) + "\n" if ctx.json_mode else "")

        for number in (signal.SIGTERM, signal.SIGHUP):
            signal.signal(number, killed)
    ctx.behave(command)
    obj = outcome_object(name, session_id, run)
    write_outcome_file(outcome_file, obj)
    return int(obj["exit_code"]), obj, command


def write_outcome_file(path: str | None, obj: dict[str, Any]) -> None:
    """Temporary file plus rename, mode 0600 (the real tool also walks parents with O_NOFOLLOW)."""
    if path is None:
        return
    tmp = os.path.join(os.path.dirname(path), f".{os.path.basename(path)}.{os.getpid()}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as handle:
        handle.write(json.dumps(obj) + "\n")
    os.replace(tmp, path)


def do_resume(ctx: Context, p: Parsed) -> tuple[int, Any, str]:
    if len(p.positionals) != 1:
        ctx.usage("resume takes exactly one session id")
    if p.has("--override-escalation") and p.has("--fork"):
        ctx.usage("--override-escalation cannot be combined with --fork")
    check_launch_flags(ctx, p)
    session_id = session_ref(ctx, p.positionals[0])
    if p.has("--dry-run"):
        return 0, dry_run_plan(p, session_id, p.has("--fork")), "resume"
    require_terminal(ctx, p, "resume")
    run = int(load_sessions().get(session_id, {}).get("run", 0)) + 1
    return run_session(ctx, "resume", session_id, run, _abs(p.get("--outcome-file")))


def do_stop(ctx: Context, p: Parsed) -> tuple[int, Any, str]:
    if len(p.positionals) != 1:
        ctx.usage("stop takes exactly one session id")
    for decision in p.multi("--decision-id"):
        require_pattern(decision, g.DECISION_ID, "--decision-id")
    if (actor := p.get("--actor")) is not None:
        require_pattern(actor, g.ACTOR, "--actor")
    if (grace := p.get("--grace")) is not None:
        require_pattern(grace, g.CS_DURATION, "--grace")
    session_id = session_ref(ctx, p.positionals[0])
    dry = p.has("--dry-run")
    out: dict[str, Any] = {
        "csession_id": session_id,
        "run": int(load_sessions().get(session_id, {}).get("run", 1)),
        "already_final": False,
        "canceled": False,
        "dry_run": dry,
        "ended": not dry,
        "run_exit_code": None if dry else 15,
        "signals": [],
        "state": "working" if dry else "killed",
        "outcome": None if dry else "killed",
        "end_reason": None if dry else "stopped",
        "recorded_by_stop": False,
    }
    return 0, out, "stop"


def do_wait(ctx: Context, p: Parsed) -> tuple[int, Any, str]:
    if len(p.positionals) != 1:
        ctx.usage("wait takes exactly one session id")
    if (timeout := p.get("--timeout")) is not None:
        require_pattern(timeout, g.CS_DURATION, "--timeout")
        if _seconds(timeout) == 0:
            ctx.usage("wait --timeout 0 is invalid")
    session_id = session_ref(ctx, p.positionals[0])
    entry = load_sessions().get(session_id, {})
    run_arg = p.get("--run")
    if run_arg is not None and not re.fullmatch(r"[1-9][0-9]*", run_arg):
        ctx.usage("--run must be a positive integer")
    run = int(run_arg or entry.get("run", 1))
    ctx.behave("wait")
    name = ctx.response("wait").get("outcome", "done")
    obj = {
        **outcome_object(name, session_id, run),
        "timed_out": False,
        "run_exit_code": OUTCOMES[name][0],
        "pending": False,
    }
    write_outcome_file(_abs(p.get("--outcome-file")), obj)
    return OUTCOMES[name][0], obj, "wait"


def do_status(ctx: Context, p: Parsed) -> tuple[int, Any, str]:
    if len(p.positionals) != 1:
        ctx.usage("status takes exactly one session id")
    session_id = session_ref(ctx, p.positionals[0])
    entry = load_sessions().get(session_id, {})
    run = int(entry.get("run", 1))
    kind = entry.get("kind", "implement")
    mode = entry.get("mode", "headless")
    name = ctx.response("status").get("outcome", "done")
    code, state, end_reason, phase, _ = OUTCOMES[name]
    outcome = outcome_object(name, session_id, run)
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
    return 0, out, "status"


def list_item(session_id: str, entry: dict[str, Any]) -> dict[str, Any]:
    pending = not entry.get("adopted", False)
    return {
        "csession_id": session_id,
        "created_at": FIXED_TS,
        "launch_kind": entry.get("kind", "implement"),
        "run_kind": entry.get("kind", "implement"),
        "state": "starting",
        "activity": None,
        "needs_input_reason": None,
        "live": True,
        "tainted": False,
        "tainted_at": None,
        "tainted_by_run": None,
        "forked_from": entry.get("forked_from"),
        "labels": entry.get("labels", []),
        "repo": None,
        "mode": entry.get("mode", "headless"),
        "current_run": entry.get("run", 1),
        "pending": pending,
        "expires_at": "2026-11-03T10:01:00Z" if pending else None,
        "updated_at": FIXED_TS,
        "phase": "launch" if pending else "session",
        "last_outcome": None,
        "last_end_reason": None,
        "exit_code": None,
        "report_status": None,
    }


def do_list(ctx: Context, p: Parsed) -> tuple[int, Any, str]:
    if p.positionals:
        ctx.usage("list takes no positional arguments")
    for label in p.multi("--label"):
        require_pattern(label, g.LABEL, "label")
    limit = p.get("--limit")
    if limit is not None and not re.fullmatch(r"[0-9]+", limit):
        ctx.usage("--limit must be an integer")
    wanted = set(p.multi("--label"))
    items = [
        list_item(session_id, entry)
        for session_id, entry in sorted(load_sessions().items())
        if wanted <= set(entry.get("labels", []))
    ]
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
            entry = data.setdefault(session_id, {"run": 1, "adopted": True, "labels": []})
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
    for decision in p.multi("--decision-id"):
        require_pattern(decision, g.DECISION_ID, "--decision-id")
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
