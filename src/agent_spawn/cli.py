"""Command-line entry point: the frozen argparse surface, global flags, the human-only gate.

``main`` parses, checks usage, dispatches through ``commands.table.HANDLERS`` and prints through
``output``. Every usage failure, including argparse's own, exits 2 with the error envelope.
Argument values that have a grammar owned by another module (labels, durations, ids, ...) are kept
raw here; their handlers validate them.
"""

from __future__ import annotations

import argparse
import getpass
import re
import sys
import traceback
from collections.abc import Sequence
from typing import Any, NoReturn

from . import __version__
from .commands.base import Invocation
from .commands.table import HANDLERS
from .config import load_config
from .output import (
    CliError,
    Outcome,
    emit,
    error_outcome,
    render_json,
    resolve_result_file,
    safe_text,
    write_result_file,
)

PROG = "agent-spawn"
DESCRIPTION = (
    "Launch, resume, park and tear down Claude agents: worktree + terminal + Claude session."
)

_ACTOR = re.compile(r"(human|auto):[A-Za-z0-9._@-]{1,64}")
_DECISION_ID = re.compile(r"[A-Za-z0-9._:@/-]{1,128}")
_ME_NAME_INVALID = re.compile(r"[^A-Za-z0-9._@-]")
MAX_DECISION_IDS = 32

INTERNAL_COMMANDS = ("supervise", "_session-event")
# (namespace attribute, flag) of the flags that need a human (interface.md, Global flags).
HUMAN_ONLY_FLAGS = (
    ("force", "--force"),
    ("override_escalation", "--override-escalation"),
    ("allow_tainted", "--allow-tainted"),
)


class UsageError(Exception):
    """A usage failure; ``usage`` is the usage text of the parser that found it."""

    def __init__(self, message: str, usage: str = "") -> None:
        super().__init__(message)
        self.message = message
        self.usage = usage


class _ParserExit(Exception):
    """argparse finished with ``--help`` or ``--version`` (already printed on stdout)."""

    def __init__(self, status: int) -> None:
        super().__init__(status)
        self.status = status


class SpawnParser(argparse.ArgumentParser):
    """argparse that raises instead of printing and exiting, so ``main`` owns all output."""

    command_parsers: dict[str, SpawnParser]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("allow_abbrev", False)
        super().__init__(*args, **kwargs)
        self.command_parsers = {}

    def error(self, message: str) -> NoReturn:
        raise UsageError(message, self.format_usage())

    def exit(self, status: int = 0, message: str | None = None) -> NoReturn:
        raise _ParserExit(status)


# --- argument types ---------------------------------------------------------------------------


def _actor(value: str) -> str:
    if not _ACTOR.fullmatch(value):
        raise argparse.ArgumentTypeError(
            f"invalid actor {value!r}: expected human:<name> or auto:<name> "
            "(letters, digits and . _ @ -, at most 64 characters)"
        )
    return value


def _decision_id(value: str) -> str:
    if not _DECISION_ID.fullmatch(value):
        raise argparse.ArgumentTypeError(f"invalid decision id {value!r}")
    return value


def _interval(value: str) -> int:
    if not re.fullmatch(r"[0-9]{1,2}", value) or not 1 <= int(value) <= 59:
        raise argparse.ArgumentTypeError(f"invalid interval {value!r}: expected 1 to 59 minutes")
    return int(value)


# --- the parser -------------------------------------------------------------------------------


def _add_global_flags(parser: SpawnParser, *, top: bool) -> None:
    """Global flags are accepted before and after the command.

    The command's parser must not overwrite what the top-level parser stored, so its defaults are
    suppressed; ``--decision-id`` is collected in a second list and merged by ``_decision_ids``.
    """
    unset: dict[str, Any] = {} if top else {"default": argparse.SUPPRESS}
    parser.add_argument("--json", action="store_true", help="machine-readable output", **unset)
    parser.add_argument("--config", metavar="PATH", help="config file", **unset)
    parser.add_argument("--verbose", action="store_true", help="more diagnostics", **unset)
    parser.add_argument("--quiet", action="store_true", help="fewer messages", **unset)
    parser.add_argument(
        "--yes", action="store_true", help="accepted and ignored: no step prompts", **unset
    )
    parser.add_argument(
        "--actor", type=_actor, metavar="ACTOR", help="human:<user> or auto:<name>", **unset
    )
    decision_default: dict[str, Any] = {"default": None} if top else unset
    parser.add_argument(
        "--decision-id",
        type=_decision_id,
        action="append",
        dest="decision_id" if top else "decision_id_after",
        metavar="ID",
        help="audit decision id (repeatable)",
        **decision_default,
    )
    parser.add_argument("--state-dir", metavar="PATH", help=argparse.SUPPRESS, **unset)


def _repeatable(parser: Any, name: str, metavar: str, help_text: str) -> None:
    parser.add_argument(name, action="append", metavar=metavar, help=help_text)


def _add_pass_throughs(parser: Any) -> None:
    """claude-session launch flags passed through unchanged (claude-session validates them)."""
    parser.add_argument("--kind", metavar="KIND", help="run kind")
    parser.add_argument("--skill", metavar="SKILL", help="skill invoked by the bootstrap prompt")
    _repeatable(parser, "--plugin-dir", "DIR", "plugin directory")
    parser.add_argument("--model", metavar="MODEL")
    parser.add_argument("--timeout", metavar="DURATION|none", help="wall-clock bound of the run")
    parser.add_argument("--max-turns", metavar="N", help="headless turn bound")
    parser.add_argument("--permission-profile", metavar="NAME|FILE")
    parser.add_argument("--sandbox", metavar="MODE", help="required, preferred or off")
    parser.add_argument("--setting-sources", metavar="LIST")
    parser.add_argument("--reads-third-party", action="store_true", default=None)
    _repeatable(parser, "--env-allow", "VAR", "add a name to the environment allowlist")
    _repeatable(parser, "--env-allowlist-only", "VAR", "replace the environment allowlist")
    auto_exit = parser.add_mutually_exclusive_group()
    auto_exit.add_argument("--auto-exit", dest="auto_exit", action="store_const", const=True)
    auto_exit.add_argument("--no-auto-exit", dest="auto_exit", action="store_const", const=False)
    parser.set_defaults(auto_exit=None)


def _add_run_flags(parser: Any, *, with_launch_only: bool) -> None:
    """Flags shared by launch and resume."""
    parser.add_argument("--mode", choices=("interactive", "headless"))
    parser.add_argument("--cwd", metavar="PATH|session-dir")
    parser.add_argument("--placement", choices=("window", "here"))
    parser.add_argument("--term-placement", metavar="NAME")
    parser.add_argument("--name", metavar="TEXT", help="window name")
    _repeatable(parser, "--label", "KEY=VALUE", "add a label")
    prompt = parser.add_mutually_exclusive_group()
    prompt.add_argument("--prompt", metavar="TEXT", help="trusted text only")
    prompt.add_argument("--prompt-file", metavar="PATH|-")
    _repeatable(parser, "--input-file", "PATH", "input file for the session")
    promote = parser.add_mutually_exclusive_group()
    promote.add_argument("--promote", dest="promote", action="store_const", const=True)
    promote.add_argument("--no-promote", dest="promote", action="store_const", const=False)
    parser.set_defaults(promote=None)
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--wait-timeout", metavar="DURATION")
    parser.add_argument("--result-file", metavar="PATH")
    parser.add_argument("--override-escalation", action="store_true", help="human only")
    parser.add_argument("--dry-run", action="store_true")
    _add_pass_throughs(parser)


def _add_launch(parser: Any) -> None:
    _add_run_flags(parser, with_launch_only=True)
    parser.add_argument("--repo", metavar="REPO-ID")
    parser.add_argument("--worktree", metavar="new|PATH|none")
    parser.add_argument("--branch", metavar="NAME")
    parser.add_argument("--base", metavar="BRANCH")
    parser.add_argument("--ref", metavar="SHA")
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--worktree-name", metavar="NAME")
    parser.add_argument("--no-bootstrap", action="store_true")
    parser.add_argument("--copy-env", action="store_true")
    parser.add_argument("--ephemeral", action="store_true")
    _repeatable(parser, "--hook", "NAME=CMD", "per-invocation hook (pre-launch, post-launch)")


def _add_resume(parser: Any) -> None:
    _add_run_flags(parser, with_launch_only=False)
    parser.add_argument("agent_id", nargs="?", metavar="AGENT_ID")
    _repeatable(parser, "--select", "KEY=VALUE", "select the agent by label (ANDed)")
    parser.add_argument("--fork", action="store_true")
    _repeatable(parser, "--reset", "SETTING", "drop a stored launch setting")
    parser.add_argument("--allow-tainted", action="store_true", help="human only")
    _repeatable(parser, "--hook", "NAME=CMD", "per-invocation hook (post-launch)")


def _agent_id(parser: Any) -> None:
    parser.add_argument("agent_id", metavar="AGENT_ID")


def _add_wait(parser: Any) -> None:
    _agent_id(parser)
    parser.add_argument("--timeout", metavar="DURATION")
    parser.add_argument("--result-file", metavar="PATH")


def _add_list(parser: Any) -> None:
    _repeatable(parser, "--label", "KEY=VALUE", "match a label (ANDed)")
    parser.add_argument("--needs-attention", action="store_true")
    _repeatable(parser, "--status", "STATUS", "match a status (ORed)")
    _repeatable(parser, "--session-state", "STATE", "match a session state (ORed)")
    parser.add_argument("--kind", metavar="KIND")
    parser.add_argument("--repo", metavar="REPO-ID")
    parser.add_argument("--mode", choices=("interactive", "headless"))
    parser.add_argument("--forked-from", metavar="AGENT_ID")
    parser.add_argument("--since", metavar="TIME|DURATION")


def _add_attach(parser: Any) -> None:
    _agent_id(parser)
    parser.add_argument("--client", metavar="CLIENT")
    parser.add_argument("--no-attach", action="store_true")


def _add_label(parser: Any, *, session: bool) -> None:
    parser.add_argument("items", nargs="*", metavar="[AGENT_ID] KEY=VALUE")
    if session:
        parser.add_argument("--session", metavar="CSESSION_ID", help="self-labeling (add only)")
    parser.add_argument("--dry-run", action="store_true")


def _add_stop(parser: Any) -> None:
    _agent_id(parser)
    parser.add_argument("--grace", metavar="DURATION")
    parser.add_argument("--dry-run", action="store_true")


def _add_teardown(parser: Any) -> None:
    _agent_id(parser)
    parser.add_argument("--merged-head", metavar="SHA")
    parser.add_argument("--force", action="store_true", help="human only")
    parser.add_argument("--override-escalation", action="store_true", help="human only")
    _repeatable(parser, "--hook", "NAME=CMD", "per-invocation hook (pre-teardown, post-teardown)")
    parser.add_argument("--dry-run", action="store_true")


def _add_prune(parser: Any) -> None:
    parser.add_argument("--older-than", metavar="DURATION")
    parser.add_argument("--include-owned", action="store_true")
    parser.add_argument("--include-attention", action="store_true")
    parser.add_argument("--dry-run", action="store_true")


def _add_dry_run(parser: Any) -> None:
    parser.add_argument("--dry-run", action="store_true")


def _add_cron_install(parser: Any) -> None:
    parser.add_argument("--interval", type=_interval, metavar="MINUTES", help="1 to 59")
    parser.add_argument("--env-file", metavar="PATH")
    parser.add_argument("--dry-run", action="store_true")


def _add_export(parser: Any) -> None:
    parser.add_argument("--since", metavar="TIME|DURATION")
    _repeatable(parser, "--action", "ACTION", "match an action (ORed)")


def _add_browse(parser: Any) -> None:
    parser.add_argument("--focus", metavar="AGENT_ID")
    parser.add_argument("--result-file", metavar="PATH")


def _add_supervise(parser: Any) -> None:
    _agent_id(parser)
    parser.add_argument("--host", action="store_true")
    parser.add_argument("--handoff", metavar="SOCKET")
    parser.add_argument("--handoff-timeout", metavar="SECONDS")


def build_parser() -> SpawnParser:
    """The whole command surface. ``command_parsers`` maps ``label add`` etc. to its parser."""
    parser = SpawnParser(prog=PROG, description=DESCRIPTION)
    parser.add_argument("--version", action="version", version=f"{PROG} {__version__}")
    _add_global_flags(parser, top=True)
    commands = parser.add_subparsers(dest="command", metavar="<command>", required=True)

    def add(group: Any, key: str, name: str, fill: Any, help_text: str | None) -> SpawnParser:
        # No `help` keyword at all hides a command from the listing (SUPPRESS prints a marker).
        listing: dict[str, Any] = {} if help_text is None else {"help": help_text}
        child = group.add_parser(name, allow_abbrev=False, **listing)
        _add_global_flags(child, top=False)
        fill(child)
        parser.command_parsers[key] = child
        return child

    add(commands, "launch", "launch", _add_launch, "launch an agent")
    add(commands, "resume", "resume", _add_resume, "resume or fork an agent")
    add(commands, "wait", "wait", _add_wait, "wait for an agent to stop running")
    add(commands, "list", "list", _add_list, "list agents")
    add(commands, "show", "show", _agent_id, "show one agent")
    add(commands, "attach", "attach", _add_attach, "attach to an agent's window")
    add(commands, "ack", "ack", _agent_id, "clear an agent's needs_attention")
    add(commands, "stop", "stop", _add_stop, "end an agent's current run")
    add(commands, "teardown", "teardown", _add_teardown, "tear an agent down")
    add(commands, "prune", "prune", _add_prune, "tear down old parked agents")
    add(commands, "reconcile", "reconcile", _add_dry_run, "recover from crashes and races")
    add(commands, "export", "export", _add_export, "print audit records")
    add(commands, "browse", "browse", _add_browse, "browse agents in a TUI")
    add(commands, "supervise", "supervise", _add_supervise, None)
    add(commands, "_session-event", "_session-event", lambda p: None, None)

    for group_name, verbs in (
        ("label", (("add", True), ("remove", False))),
        ("cron", (("install", None), ("uninstall", None))),
    ):
        group_parser = commands.add_parser(
            group_name,
            help="add or remove labels" if group_name == "label" else "manage the cron entry",
            allow_abbrev=False,
        )
        _add_global_flags(group_parser, top=False)
        verb_group = group_parser.add_subparsers(dest="subcommand", metavar="<verb>", required=True)
        for verb, session in verbs:
            if group_name == "label":
                fill = lambda p, s=session: _add_label(p, session=bool(s))  # noqa: E731
            elif verb == "install":
                fill = _add_cron_install
            else:
                fill = _add_dry_run
            add(verb_group, f"{group_name} {verb}", verb, fill, f"{group_name} {verb}")
    return parser


# --- environment seams --------------------------------------------------------------------------


def stdin_is_tty() -> bool:
    """Whether stdin is a terminal; an unusable stdin counts as no terminal."""
    try:
        return bool(sys.stdin is not None and sys.stdin.isatty())
    except (ValueError, OSError, AttributeError):
        return False


def me_name() -> str:
    """The user name for the default actor ``human:<me.name>``, fitted to the actor grammar."""
    # The config key me.name, when set, takes precedence over this (see parse_invocation).
    try:
        name = getpass.getuser()
    except (OSError, KeyError, ImportError):
        name = ""
    name = _ME_NAME_INVALID.sub("_", name)[:64]
    return name or "unknown"


# --- usage checks -------------------------------------------------------------------------------


def _command_key(args: argparse.Namespace) -> str:
    sub = getattr(args, "subcommand", None)
    return f"{args.command} {sub}" if sub else str(args.command)


def _check_usage(args: argparse.Namespace, key: str) -> None:
    """Checks that need the whole namespace. Raises ``ValueError`` with the usage message."""
    internal = key in INTERNAL_COMMANDS
    state_dir: str | None = getattr(args, "state_dir", None)
    config: str | None = getattr(args, "config", None)
    if internal:
        for flag, value in (("--state-dir", state_dir), ("--config", config)):
            if value is None or not value.startswith("/") or "\0" in value:
                raise ValueError(f"{key} needs an absolute {flag}")
    elif state_dir is not None:
        raise ValueError("--state-dir is only for internal commands")

    if key == "resume" and (args.agent_id is None) == (not args.select):
        raise ValueError("resume needs exactly one of AGENT_ID or --select")

    if args.command == "label":
        items: list[str] = args.items
        if getattr(args, "session", None) is not None:
            args.agent_id, args.pairs = None, items
            if not items:
                raise ValueError("label add --session needs at least one KEY=VALUE")
        else:
            if len(items) < 2:
                raise ValueError(
                    f"label {args.subcommand} needs an AGENT_ID and at least one KEY=VALUE"
                )
            args.agent_id, args.pairs = items[0], items[1:]
            args.session = None


def _check_human_only(args: argparse.Namespace, actor: str, explicit: bool, tty: bool) -> None:
    used = [flag for dest, flag in HUMAN_ONLY_FLAGS if getattr(args, dest, False)]
    if not used or tty or (explicit and actor.startswith("human:")):
        return
    raise ValueError(
        f"{', '.join(used)} needs a human: run it from a terminal or pass an explicit "
        "--actor human:<user>"
    )


def parse_invocation(argv: Sequence[str]) -> Invocation:
    """Parse ``argv`` and run every usage check. Raises ``UsageError`` (and ``_ParserExit``)."""
    parser = build_parser()
    args = parser.parse_args(list(argv))
    key = _command_key(args)
    child = parser.command_parsers[key]
    decision_ids: list[str] = list(args.decision_id or []) + list(
        getattr(args, "decision_id_after", None) or []
    )
    try:
        if len(decision_ids) > MAX_DECISION_IDS:
            raise ValueError(f"at most {MAX_DECISION_IDS} --decision-id values")
        _check_usage(args, key)
    except ValueError as exc:
        raise UsageError(str(exc), child.format_usage()) from None
    # Fail fast on an invalid config (CliError config_invalid, exit 2) before anything else runs.
    internal = key in INTERNAL_COMMANDS
    named: str | None = getattr(args, "config", None)
    loaded = load_config(named, warn_missing=named is not None and not internal)
    try:
        tty = stdin_is_tty()
        explicit = args.actor is not None
        actor: str = args.actor or (
            f"human:{loaded.config.me.name or me_name()}" if tty else "auto:unknown"
        )
        _check_human_only(args, actor, explicit, tty)
    except ValueError as exc:
        raise UsageError(str(exc), child.format_usage()) from None
    return Invocation(
        command=key,
        args=args,
        as_json=bool(args.json),
        actor=actor,
        actor_explicit=explicit,
        decision_ids=decision_ids,
        config_path=getattr(args, "config", None),
        state_dir=getattr(args, "state_dir", None),
        verbose=bool(args.verbose),
        quiet=bool(args.quiet),
        config=loaded.config,
        config_file=loaded.path,
        warnings=list(loaded.warnings),
    )


# --- main ---------------------------------------------------------------------------------------


def _json_requested(argv: Sequence[str]) -> bool:
    """Whether ``--json`` was given, for errors raised before parsing finished."""
    head = list(argv)
    if "--" in head:
        head = head[: head.index("--")]
    return "--json" in head


def _write(stream: Any, text: str) -> None:
    try:
        stream.write(text)
        stream.flush()
    except BrokenPipeError:
        pass


def main(argv: Sequence[str] | None = None) -> int:
    argv_list = list(sys.argv[1:] if argv is None else argv)
    try:
        inv = parse_invocation(argv_list)
    except _ParserExit as done:
        return done.status
    except CliError as error:  # config_invalid
        emit(
            error_outcome(error),
            as_json=_json_requested(argv_list),
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
        return error.exit_code
    except UsageError as exc:
        if exc.usage:
            _write(sys.stderr, safe_text(exc.usage))
        emit(
            error_outcome(CliError.usage(exc.message)),
            as_json=_json_requested(argv_list),
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
        return 2

    outcome = _run(inv)
    try:
        emit(outcome, as_json=inv.as_json, stdout=sys.stdout, stderr=sys.stderr)
    except BrokenPipeError:
        pass
    if inv.result_file is not None and outcome.payload is not None:
        warning = write_result_file(inv.result_file, render_json(outcome.payload) + "\n")
        if warning is not None:
            _write(sys.stderr, safe_text(f"agent-spawn: warning: {warning}") + "\n")
    return outcome.exit_code


def _run(inv: Invocation) -> Outcome:
    try:
        result_file = getattr(inv.args, "result_file", None)
        if result_file is not None:
            inv.result_file = resolve_result_file(result_file)
        outcome = HANDLERS[inv.command](inv)
        if not isinstance(outcome, Outcome):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise TypeError(f"handler of {inv.command!r} returned {type(outcome).__name__}")
        return outcome
    except CliError as error:
        return error_outcome(error, inv.decision_ids)
    except Exception as exc:
        if inv.verbose:
            _write(sys.stderr, safe_text(traceback.format_exc()))
        error = CliError.of("internal", f"internal error ({type(exc).__name__})")
        return error_outcome(error, inv.decision_ids)
