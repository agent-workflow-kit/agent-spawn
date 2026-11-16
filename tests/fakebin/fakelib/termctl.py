"""Fake ``termctl`` (interface 0.1.0-design).

Implemented commands: ``open`` (also ``--dry-run``), ``show``, ``exists``, ``list``, ``rename``,
``mark``, ``unmark``, ``focus``, ``close``, ``label add|remove`` and ``placements`` (the built-in
placement only). Panes opened by ``open`` live in ``<FAKEBIN_LOG>.panes.json`` (keyed by socket
path), so ``show``, ``list``, ``close`` and ``exists`` see them and an unknown id exits 10.
Without ``FAKEBIN_TMUX`` no tmux is involved. With ``FAKEBIN_TMUX=<tmux binary>`` (a process
test's private server, chosen with ``TERMCTL_SOCKET=<socket path>``) ``open`` really starts its
argv in a new tmux session, ``list`` and ``show`` take the pane's state from tmux and ``close``
kills the pane. See ``tests/fakebin/FIDELITY.md``.

``open --placement`` must name a defined placement: ``default`` and the names in the environment
variable ``FAKEBIN_PLACEMENTS`` (comma separated); anything else is a usage error, exit 2, as in
termctl. A scripted ``open`` response with ``after_create: true`` applies after the pane exists.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

from . import grammars as g
from .core import (
    BOOL,
    MULTI,
    VALUE,
    Context,
    Exit,
    Parsed,
    base32_id,
    counter,
    first_positional,
    locked,
    parse_args,
    read_json,
    require_pattern,
    run_fake,
)

VERSION = "0.1.0"
CREATED = "2026-11-03T10:00:00.000Z"
GLOBAL = {
    "--json": BOOL,
    "--socket": VALUE,
    "--config": VALUE,
    "--verbose": BOOL,
    "--quiet": BOOL,
}
SPECS: dict[str, dict[str, str]] = {
    "open": {
        **GLOBAL,
        "--name": VALUE,
        "--cwd": VALUE,
        "--env": MULTI,
        "--label": MULTI,
        "--placement": VALUE,
        "--session": VALUE,
        "--target": VALUE,
        "--keep-on-exit": VALUE,
        "--focus": BOOL,
        "--dry-run": BOOL,
        "--actor": VALUE,
        "--decision-id": MULTI,
        "--override-escalation": BOOL,
        "--hook": MULTI,
    },
    "show": GLOBAL,
    "exists": GLOBAL,
    "list": {**GLOBAL, "--label": MULTI, "--session": VALUE, "--dead": BOOL},
    "rename": {**GLOBAL, "--dry-run": BOOL},
    "mark": {**GLOBAL, "--dry-run": BOOL},
    "unmark": {**GLOBAL, "--dry-run": BOOL},
    "focus": {**GLOBAL, "--client": VALUE, "--no-attach": BOOL, "--dry-run": BOOL},
    "close": {
        **GLOBAL,
        "--force": BOOL,
        "--grace": VALUE,
        "--actor": VALUE,
        "--decision-id": MULTI,
        "--override-escalation": BOOL,
        "--hook": MULTI,
        "--dry-run": BOOL,
    },
    "label": {**GLOBAL, "--dry-run": BOOL},
    "placements": {**GLOBAL, "--label": MULTI},
}
NAME_FLAGS = {"--json", "--socket", "--config", "--verbose", "--quiet"}


# --- state ---------------------------------------------------------------------------------


def _store_path() -> Path:
    return Path(os.environ["FAKEBIN_LOG"] + ".panes.json")


def load_panes() -> dict[str, dict[str, dict[str, Any]]]:
    return read_json(_store_path(), "panes file", dict, {})


def update_panes(change: Callable[[dict[str, dict[str, dict[str, Any]]]], Any]) -> Any:
    path = _store_path()
    with locked(path):
        data: dict[str, dict[str, dict[str, Any]]] = read_json(path, "panes file", dict, {})
        result = change(data)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data))
        os.replace(tmp, path)
        return result


def socket_path(ctx: Context, p: Parsed) -> str:
    """Server selection: ``--socket``, ``TERMCTL_SOCKET``, ``TMUX``, then tmux's default."""
    value = p.get("--socket") or os.environ.get("TERMCTL_SOCKET")
    if not value:
        tmux = os.environ.get("TMUX", "")
        first = tmux.split(",")[0]
        value = first if first else "default"
    if "/" in value:
        if not os.path.isabs(value):
            value = os.path.join(os.getcwd(), value)
        return canonical_socket(value)
    if not re.fullmatch(r"(?!\.\.?$)[A-Za-z0-9._-]{1,64}", value):
        ctx.usage(f"invalid socket name: {value!r}")
    tmpdir = next(
        (d for d in os.environ.get("TMUX_TMPDIR", "").split(":") if d and os.path.isdir(d)), "/tmp"
    )
    return canonical_socket(f"{tmpdir}/tmux-{os.getuid()}/{value}")


def canonical_socket(path: str) -> str:
    """Symlinked, ``//`` and ``..`` spellings of one socket share state: resolve the directory."""
    head, tail = os.path.split(os.path.normpath(path))
    return os.path.join(os.path.realpath(head), tail)


def tmux_bin() -> str | None:
    return os.environ.get("FAKEBIN_TMUX") or None


def tmux(sock: str, *args: str) -> subprocess.CompletedProcess[str]:
    """One call of the process test's private tmux server (``-f /dev/null -S <socket>``)."""
    env = {k: v for k, v in os.environ.items() if k not in ("TMUX", "TMUX_PANE")}
    env.setdefault("TERM", "xterm")
    binary = tmux_bin()
    assert binary is not None
    return subprocess.run(
        [binary, "-f", "/dev/null", "-S", sock, *args],
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        env=env,
        timeout=30,
        check=False,
    )


def sync_from_tmux(sock: str) -> None:
    """Take every pane's state from the private tmux server: a pane tmux no longer lists is
    gone, a dead one carries its exit status."""
    if tmux_bin() is None:
        return
    done = tmux(sock, "list-panes", "-a", "-F", "#{pane_id} #{pane_dead} #{pane_dead_status}")
    seen: dict[str, tuple[bool, int | None]] = {}
    if done.returncode == 0:
        for line in done.stdout.splitlines():
            parts = line.split()
            if len(parts) >= 2:
                status = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else None
                seen[parts[0]] = (parts[1] == "1", status)

    def change(data: dict[str, dict[str, dict[str, Any]]]) -> None:
        panes = data.get(sock, {})
        for pane_id in list(panes):
            info = seen.get(panes[pane_id]["pane_id"])
            if info is None:
                del panes[pane_id]
            elif info[0]:
                panes[pane_id]["state"] = "dead"
                panes[pane_id]["exit_status"] = info[1]

    update_panes(change)


def start_in_tmux(
    sock: str, number: int, name: str, cwd: str, argv: list[str], keep: str
) -> tuple[str, int]:
    """A new tmux session whose only pane runs ``argv`` directly; returns tmux's pane id and
    the pane's PID."""
    remain = {"never": "off", "failed": "failed", "always": "on"}[keep]
    tmux(
        sock,
        "start-server",
        ";",
        "set-option",
        "-g",
        "exit-empty",
        "off",
        ";",
        "set-option",
        "-g",
        "remain-on-exit",
        remain,
    )
    done = tmux(
        sock,
        "new-session",
        "-d",
        "-s",
        f"s{number}",
        "-n",
        name,
        "-c",
        cwd,
        "-P",
        "-F",
        "#{pane_id} #{pane_pid}",
        "--",
        *argv,
    )
    fields = done.stdout.split()
    if done.returncode != 0 or len(fields) != 2 or not fields[1].isdigit():
        raise Exit(1, f"fakebin: tmux new-session failed: {done.stderr.strip()}\n")
    return fields[0], int(fields[1])


def pane_view(pane: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "id name display_name marker marker_text labels placement session pane_id window_id "
        "socket_path created pid state exit_status exit_signal keep_on_exit"
    ).split()
    return {key: pane[key] for key in keys}


def resolve(ctx: Context, p: Parsed, raw: str, *, missing_ok: bool = False) -> tuple[str, str]:
    """(socket path, full id) for an id input: ``tc-`` optional, 4 to 8 characters, prefixes."""
    text = raw.lower()
    body = text[3:] if text.startswith("tc-") else text
    if not re.fullmatch(r"[a-z2-7]{4,8}", body):
        ctx.usage(f"invalid pane id: {raw!r}")
    sock = socket_path(ctx, p)
    sync_from_tmux(sock)
    panes = load_panes().get(sock, {})
    matches = sorted(i for i in panes if i.startswith(f"tc-{body}"))
    if len(body) == 8:
        matches = [i for i in matches if i == f"tc-{body}"]
    if not matches:
        if missing_ok:
            return sock, ""
        if sock not in load_panes():
            ctx.fail(10, "no_server", "no tmux server is running", details={"socket_path": sock})
        ctx.fail(10, "not_found", f"no managed pane matches {raw!r}")
    if len(matches) > 1:
        ctx.fail(
            21,
            "ambiguous",
            f"{raw!r} matches several panes",
            candidates=[
                {"id": i, "pane_id": panes[i]["pane_id"], "session": panes[i]["session"]}
                for i in matches
            ],
        )
    return sock, matches[0]


# --- dispatch ------------------------------------------------------------------------------


def find_command(ctx: Context, argv: list[str]) -> tuple[str, list[str]]:
    """Global flags may precede the subcommand; a ``--`` before it is a usage error."""
    i = 0
    while i < len(argv):
        token = argv[i]
        if token == "--":
            ctx.usage("a -- before the subcommand is not allowed")
        if token.startswith("-"):
            name, eq, _ = token.partition("=")
            if name not in NAME_FLAGS:
                ctx.usage(f"unrecognized argument: {token}")
            if GLOBAL[name] == VALUE and not eq:
                i += 1
        else:
            return token, argv[:i] + argv[i + 1 :]
        i += 1
    ctx.usage("a command is required")


def dispatch(ctx: Context) -> tuple[int, Any, str]:
    argv = ctx.argv
    before = argv[: argv.index("--")] if "--" in argv else argv
    # --version and --help are honored anywhere before --, ahead of every other check.
    if any(t == "--version" for t in _flag_tokens(before)):
        raise Exit(0, f"termctl {os.environ.get('FAKEBIN_VERSION', VERSION)}\n")
    ctx.json_mode = any(t == "--json" or t.startswith("--json=") for t in before)
    if any(t in ("-h", "--help") for t in _flag_tokens(before)):
        raise Exit(0, "usage: termctl [--json] [--socket S] <command> ...\n")
    command, rest = find_command(ctx, argv)
    if command not in SPECS:
        ctx.usage(f"unknown command: {command!r}")
    if command == "label":
        # The action is the first positional once the global value flags are skipped, so
        # `--socket PATH label add ...` and `label add --socket PATH ...` are the same call.
        at = first_positional(rest, {"--socket", "--config"})
        sub = "" if at is None else rest[at]
        if sub not in ("add", "remove"):
            ctx.usage("label needs add or remove")
        assert at is not None
        del rest[at]
        command = f"label {sub}"
    p = parse_args(rest, SPECS[command.split()[0]], dd_is_argv=command == "open")
    ctx.json_mode = p.has("--json")
    if p.has("--verbose") and p.has("--quiet"):
        ctx.usage("--verbose and --quiet are mutually exclusive")
    return HANDLERS[command](ctx, p)


def _flag_tokens(tokens: list[str]) -> list[str]:
    """Names of the flags among ``tokens`` (those before ``--``), skipping option values."""
    value_flags = {name for spec in SPECS.values() for name, kind in spec.items() if kind != BOOL}
    flags: list[str] = []
    skip = False
    for token in tokens:
        if skip:
            skip = False
            continue
        if not token.startswith("-") or token == "-":
            continue
        name, eq, _ = token.partition("=")
        if name in value_flags and not eq:
            skip = True
        if eq and name in ("--json", "--version", "--help", "-h"):
            continue  # a boolean flag with =VALUE is a usage error, not the flag
        flags.append(name)
    return flags


def check_actor_and_decisions(ctx: Context, p: Parsed) -> None:
    if (actor := p.get("--actor")) is not None:
        require_pattern(actor, g.ACTOR, "--actor")
    ids = p.multi("--decision-id")
    if len(ids) > 32:
        ctx.usage("at most 32 --decision-id values")
    for decision in ids:
        require_pattern(decision, g.DECISION_ID, "--decision-id")


def check_human(ctx: Context, p: Parsed, flag: str) -> None:
    """A human-only flag: ``--actor human:<user>``, or a terminal on stdin without an actor.

    An ``auto:`` actor is refused whatever the terminal (termctl 0.25.0, observed).
    """
    actor = p.get("--actor")
    if actor is not None:
        if not actor.startswith("human:"):
            ctx.usage(f"{flag} is refused for an auto: actor, whatever the terminal")
    elif not ctx.stdin_is_tty():
        ctx.usage(f"{flag} needs a terminal on stdin or --actor human:<user>")


def check_hooks(ctx: Context, p: Parsed, names: tuple[str, str]) -> None:
    for hook in p.multi("--hook"):
        name, eq, cmd = hook.partition("=")
        if name not in names or not eq or not cmd:
            ctx.usage(f"invalid --hook: {hook!r}")


def valid_name(name: str) -> bool:
    return (
        1 <= len(name) <= 64
        and "#[" not in name
        and not re.search(r"[\x00-\x1f\x7f-\x9f]", name)
        and not re.search(r"[\ud800-\udfff]", name)
    )


def find_in_path(program: str, path: str) -> bool:
    if program == "":
        return False
    return any(os.access(os.path.join(d or ".", program), os.X_OK) for d in path.split(":"))


def do_open(ctx: Context, p: Parsed) -> tuple[int, Any, str]:
    if p.positionals:
        ctx.usage(f"unexpected argument: {p.positionals[0]!r}")
    argv = p.after_dd
    if argv is None or len(argv) == 0:
        ctx.usage("open needs a command after --")
    name = p.get("--name")
    if name is not None and not valid_name(name):
        ctx.usage(f"invalid --name: {name!r}")
    labels = p.multi("--label")
    if len(labels) > 32:
        ctx.usage("at most 32 --label values")
    for label in labels:
        require_pattern(label, g.LABEL, "label")
    env = p.multi("--env")
    if len(env) > 32:
        ctx.usage("at most 32 --env values")
    env_names: list[str] = []
    for entry in env:
        var, eq, _ = entry.partition("=")
        if not eq:
            ctx.usage(f"--env needs VAR=value: {entry!r}")
        require_pattern(var, g.ENV_NAME, "environment variable name")
        if var in env_names:
            ctx.usage(f"--env {var} given twice")
        env_names.append(var)
    if (placement := p.get("--placement")) is not None:
        require_pattern(placement, g.PLACEMENT, "--placement")
        defined = {"default", *filter(None, os.environ.get("FAKEBIN_PLACEMENTS", "").split(","))}
        if placement not in defined:
            ctx.usage(f"--placement {placement!r} is not defined in config (placements)")
    if (session := p.get("--session")) is not None:
        require_pattern(session, g.SESSION_NAME, "--session")
    target = p.get("--target")
    if target is not None and target not in ("window", "pane"):
        ctx.usage(f"invalid --target: {target!r}")
    keep = p.get("--keep-on-exit", "never")
    if keep not in ("never", "failed", "always"):
        ctx.usage(f"invalid --keep-on-exit: {keep!r}")
    check_actor_and_decisions(ctx, p)
    check_hooks(ctx, p, ("pre-open", "post-open"))
    if p.has("--override-escalation"):
        check_human(ctx, p, "--override-escalation")
    if "=" in argv[0]:
        ctx.usage("argv[0] must not contain '=' (env would take it as an assignment)")
    if p.get("--cwd") == "":
        ctx.usage("--cwd must not be empty")
    cwd = os.path.join(os.getcwd(), p.get("--cwd") or ".")
    if not os.path.isdir(cwd):
        ctx.usage(f"--cwd {cwd!r} is not a directory")
    sock = socket_path(ctx, p)
    path_env = next((e[5:] for e in env if e.startswith("PATH=")), os.environ.get("PATH", ""))
    program = argv[0]
    found = (
        os.access(os.path.join(cwd, program), os.X_OK)
        and os.path.isfile(os.path.join(cwd, program))
        if "/" in program
        else find_in_path(program, path_env)
    )
    if not found:
        ctx.fail(12, "command_not_found", f"cannot find an executable {program!r}")
    pane_name = name or (
        os.path.basename(program) if valid_name(os.path.basename(program)) else "command"
    )
    if p.has("--dry-run"):
        ctx.check_scripted_error("open")  # a dry run exits as a real run would (tmux checks)
        out = {
            "dry_run": True,
            "id": "tc-aaaaaaaa",
            "name": pane_name,
            "socket_path": sock,
            "tmux_version": "3.4",
            "placement": placement or "default",
            "matched_rule": None,
            "session": session or "termctl",
            "session_exists": None if sock not in load_panes() else True,
            "target": target or "window",
            "cwd": cwd,
            "argv": argv,
            "env_names": sorted(env_names),
            "labels": sorted(set(labels)),
            "keep_on_exit": keep,
            "focus": p.has("--focus"),
            "actor": p.get("--actor", "auto:unknown"),
            "decision_ids": p.multi("--decision-id"),
            "override_escalation": p.has("--override-escalation"),
            "tmux_commands": [],
        }
        return 0, out, "open"
    if not ctx.response("open").get("after_create"):
        ctx.check_scripted_error("open")  # a refusal creates nothing
    number = counter("tc")
    pane_id = base32_id("tc-", number)
    real_pane, real_pid = (
        start_in_tmux(sock, number, pane_name, cwd, argv, keep)
        if tmux_bin() is not None
        else (None, None)
    )
    pane: dict[str, Any] = {
        "id": pane_id,
        "name": pane_name,
        "display_name": pane_name,
        "marker": None,
        "marker_text": None,
        "labels": sorted(set(labels)),
        "placement": placement or "default",
        "session": session or "termctl",
        "pane_id": real_pane if real_pane is not None else f"%{number}",
        "window_id": f"@{number}",
        "socket_path": sock,
        "created": CREATED,
        "pid": real_pid if real_pid is not None else 40000 + number,
        "state": "running",
        "exit_status": None,
        "exit_signal": None,
        "keep_on_exit": keep,
    }
    update_panes(lambda data: data.setdefault(sock, {}).__setitem__(pane_id, pane))
    # `after_create` in a scripted response: the call fails (or hangs) once the pane exists, as
    # when termctl is killed or fails after creating it; `apply_rule` applies the script.
    out = {**pane_view(pane), "focused": False, "decision_ids": p.multi("--decision-id")}
    return 0, out, "open"


def do_show(ctx: Context, p: Parsed) -> tuple[int, Any, str]:
    if len(p.positionals) != 1:
        ctx.usage("show takes exactly one pane id")
    sock, pane_id = resolve(ctx, p, p.positionals[0])
    return 0, pane_view(load_panes()[sock][pane_id]), "show"


def do_exists(ctx: Context, p: Parsed) -> tuple[int, Any, str]:
    if len(p.positionals) != 1:
        ctx.usage("exists takes exactly one pane id")
    sock, pane_id = resolve(ctx, p, p.positionals[0], missing_ok=True)
    out = {
        "input": p.positionals[0],
        "id": pane_id or None,
        "exists": bool(pane_id),
        "socket_path": sock,
    }
    return (0 if pane_id else 10), out, "exists"


def do_list(ctx: Context, p: Parsed) -> tuple[int, Any, str]:
    if p.positionals:
        ctx.usage("list takes no positional arguments")
    for label in p.multi("--label"):
        require_pattern(label, g.LABEL, "label")
    sock = socket_path(ctx, p)
    sync_from_tmux(sock)
    wanted = set(p.multi("--label"))
    panes = [
        pane_view(pane)
        for pane in load_panes().get(sock, {}).values()
        if wanted <= set(pane["labels"])
        and (p.get("--session") is None or pane["session"] == p.get("--session"))
        and (not p.has("--dead") or pane["state"] == "dead")
    ]
    panes.sort(key=lambda pane: (pane["created"], pane["id"]))
    return 0, {"socket_path": sock, "panes": panes}, "list"


def change_handler(command: str) -> Callable[[Context, Parsed], tuple[int, Any, str]]:
    def handler(ctx: Context, p: Parsed) -> tuple[int, Any, str]:
        arity = {"rename": 2, "mark": 2, "unmark": 1}
        if command in arity:
            if len(p.positionals) != arity[command]:
                ctx.usage(f"{command} takes {arity[command]} argument(s)")
        elif len(p.positionals) < 2:
            ctx.usage(f"{command} needs a pane id and at least one key=value")
        elif p.positional_indexes != list(
            range(p.positional_indexes[0], p.positional_indexes[0] + len(p.positionals))
        ):
            ctx.usage("the key=value pairs must be contiguous")
        if command == "rename" and not valid_name(p.positionals[1]):
            ctx.usage(f"invalid name: {p.positionals[1]!r}")
        markers = {"attention": "\U0001f514 ", **ctx.response(command).get("markers", {})}
        if command == "mark":
            require_pattern(p.positionals[1], r"[a-z0-9][a-z0-9_-]{0,31}", "marker name")
            if p.positionals[1] not in markers:
                ctx.usage(f"unknown marker {p.positionals[1]!r}: not defined in config (markers)")
        if command.startswith("label"):
            for pair in p.positionals[1:]:
                require_pattern(pair, g.LABEL, "label")
        sock, pane_id = resolve(ctx, p, p.positionals[0])
        dry = p.has("--dry-run")

        def change(data: dict[str, dict[str, dict[str, Any]]]) -> tuple[dict[str, Any], bool]:
            pane = dict(data[sock][pane_id])
            before = json.dumps(pane, sort_keys=True)
            if command == "rename":
                pane["name"] = p.positionals[1]
            elif command == "mark":
                pane["marker"] = p.positionals[1]
                pane["marker_text"] = markers[p.positionals[1]]
            elif command == "unmark":
                pane["marker"] = None
                pane["marker_text"] = None
            elif command == "label add":
                pane["labels"] = sorted(set(pane["labels"]) | set(p.positionals[1:]))
                if len(pane["labels"]) > 32:
                    ctx.usage("a pane carries at most 32 labels")
            else:
                pane["labels"] = sorted(set(pane["labels"]) - set(p.positionals[1:]))
            if pane["marker_text"]:
                pane["display_name"] = f"{pane['marker_text']}{pane['name']}"
            else:
                pane["display_name"] = pane["name"]
            changed = json.dumps(pane, sort_keys=True) != before
            if not dry:
                data[sock][pane_id] = pane
            return pane, changed

        ctx.check_scripted_error(command)
        pane, changed = update_panes(change)
        out = {
            **pane_view(pane),
            "changed": changed,
            "display_updated": changed and not command.startswith("label"),
            "dry_run": dry,
        }
        return 0, out, command

    return handler


def do_focus(ctx: Context, p: Parsed) -> tuple[int, Any, str]:
    if len(p.positionals) != 1:
        ctx.usage("focus takes exactly one pane id")
    if (client := p.get("--client")) is not None:
        require_pattern(client, r"/?[A-Za-z0-9_./-]{1,100}", "--client")
    sock, pane_id = resolve(ctx, p, p.positionals[0])
    pane = load_panes()[sock][pane_id]
    out = {
        "id": pane_id,
        "action": "switched",
        "client": "/dev/" + client.lstrip("/")
        if client and not client.startswith("/")
        else (client or "/dev/pts/0"),
        "dry_run": p.has("--dry-run"),
        "socket_path": sock,
        "session": pane["session"],
        "window_id": pane["window_id"],
        "pane_id": pane["pane_id"],
    }
    return 0, out, "focus"


def do_close(ctx: Context, p: Parsed) -> tuple[int, Any, str]:
    if len(p.positionals) != 1:
        ctx.usage("close takes exactly one pane id")
    check_actor_and_decisions(ctx, p)
    check_hooks(ctx, p, ("pre-close", "post-close"))
    if (grace := p.get("--grace")) is not None:
        require_pattern(grace, g.TC_DURATION, "--grace")
        if not p.has("--force"):
            ctx.usage("--grace needs --force")
    if p.has("--override-escalation"):
        check_human(ctx, p, "--override-escalation")
    if p.has("--force"):
        actor = p.get("--actor")
        human = actor.startswith("human:") if actor else ctx.stdin_is_tty()
        if not human:
            ctx.usage("--force is for humans (a terminal or --actor human:<user>)")
        if not re.fullmatch(r"tc-[a-z2-7]{8}|[a-z2-7]{8}", p.positionals[0].lower()):
            ctx.usage("--force needs a full pane id")
    sock, pane_id = resolve(ctx, p, p.positionals[0])
    pane = load_panes()[sock][pane_id]
    running = pane["state"] == "running"
    forced = p.has("--force")
    if running and not forced:
        ctx.fail(20, "refused", f"pane {pane_id} is running (use --force)", reason="running")
    dry = p.has("--dry-run")
    if dry:
        out: dict[str, Any] = {
            "dry_run": True,
            "id": pane_id,
            "state": pane["state"],
            "forced": forced,
            "pid": pane["pid"],
            "pid_mismatch": False,
            "signals_possible": ["SIGTERM", "SIGKILL"] if running and forced else [],
            "grace_ms": 10000,
            "socket_path": sock,
            "actor": p.get("--actor", "auto:unknown"),
            "decision_ids": p.multi("--decision-id"),
            "override_escalation": p.has("--override-escalation"),
        }
        return 0, out, "close"
    ctx.check_scripted_error("close")
    ctx.behave("close")
    if tmux_bin() is not None:
        tmux(sock, "kill-pane", "-t", pane["pane_id"])
    update_panes(lambda data: data[sock].pop(pane_id, None))
    out = {
        "id": pane_id,
        "pane_closed": True,
        "was_running": running and forced,
        "forced": forced,
        "signals_sent": [],
        "exited": True if running and forced else False,
        "pid_mismatch": False,
        "decision_ids": p.multi("--decision-id"),
        "duration_ms": 0,
        "socket_path": sock,
    }
    if not (running and forced):
        out["exited"] = False
    return 0, out, "close"


def do_placements(ctx: Context, p: Parsed) -> tuple[int, Any, str]:
    if p.positionals:
        ctx.usage("placements takes no positional arguments")
    labels = p.multi("--label")
    for label in labels:
        require_pattern(label, g.LABEL, "label")
    if len(labels) > 32:
        ctx.usage("at most 32 --label values")
    out: dict[str, Any] = {
        "placements": {
            "default": {
                "session": "termctl",
                "target": "window",
                "max_panes": 1,
                "layout": "tiled",
                "group_by": None,
                "origin": "built-in",
            }
        },
        "rules": [],
        "match": None,
    }
    if labels:
        out["match"] = {"labels": sorted(set(labels)), "rule": None, "placement": "default"}
    return 0, out, "placements"


HANDLERS: dict[str, Callable[[Context, Parsed], tuple[int, Any, str]]] = {
    "placements": do_placements,
    "open": do_open,
    "show": do_show,
    "exists": do_exists,
    "list": do_list,
    "rename": change_handler("rename"),
    "mark": change_handler("mark"),
    "unmark": change_handler("unmark"),
    "label add": change_handler("label add"),
    "label remove": change_handler("label remove"),
    "focus": do_focus,
    "close": do_close,
}


def main(argv: list[str]) -> int:
    return run_fake("termctl", argv, dispatch)
