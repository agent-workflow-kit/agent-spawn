"""Fake ``systemd-run`` (``systemd-run --user --scope``, the one form agent-spawn uses).

A real scope-mode ``systemd-run`` asks the user manager for a transient scope, moves itself into
it and ``exec``s the command in place, so the command runs with ``systemd-run``'s own PID. The
fake does the same without a manager: it records the unit under its PID in
``<FAKEBIN_LOG>.scopes.json`` (a unit whose recorded PID is still alive makes a second start
exit 1 ``Unit ... already exists``; a unit whose process is gone is free again, as a scope
vanishes when its last process exits), clears the signal mask, ignores SIGPIPE and SIGXFSZ, adds
``INVOCATION_ID`` to the environment and ``exec``s the command. The ``exec`` leaves no ``exit``
record in the log (there is no process left to write it).

Accepted: ``--user`` and ``--scope`` (both required), ``--quiet``/``-q``, ``--collect``/``-c``,
``--unit=NAME``/``--unit NAME``/``-u NAME`` (required: agent-spawn always names its units),
``--description=TEXT``/``--description TEXT``, ``--`` and the command. Everything else exits 1
with ``fake_unsupported`` on stderr (never a silent success). A unit name outside systemd's
character set exits 1 (the real program escapes it with a warning: a deliberate difference, so a
test cannot pass with an unvalidated name).

Scripting: a rule for ``scope`` answers instead of starting anything (``exit``, ``stderr``,
``sleep``, ``hang``...; the command is not run). See ``tests/fakebin/FIDELITY.md``.
"""

from __future__ import annotations

import errno
import json
import os
import re
import signal
import sys
from pathlib import Path

from .core import Context, Exit, _log_path, locked, run_fake

NAME = re.compile(r"[A-Za-z0-9:_.\\@-]{1,200}")
NO_BUS = (
    "Failed to connect to bus: $DBUS_SESSION_BUS_ADDRESS and $XDG_RUNTIME_DIR not defined "
    "(consider using --machine=<user>@.host --user to connect to bus of other user)\n"
)


def scopes_path() -> Path:
    return Path(str(_log_path()) + ".scopes.json")


def _alive(pid: int) -> bool:
    try:
        text = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return False
    return text.rpartition(")")[2].split()[0] != "Z"


def _unsupported(message: str) -> Exit:
    return Exit(1, "", f"systemd-run: fake_unsupported: {message}\n")


def parse(argv: list[str]) -> tuple[dict[str, object], list[str]]:
    flags: dict[str, object] = {}
    i = 0
    while i < len(argv):
        token = argv[i]
        if token == "--":
            i += 1
            break
        if token in ("--user", "--scope"):
            flags[token] = True
        elif token in ("--quiet", "-q", "--collect", "-c"):
            flags["--quiet" if token in ("--quiet", "-q") else "--collect"] = True
        elif token.split("=", 1)[0] in ("--unit", "--description", "-u"):
            name, eq, value = token.partition("=")
            if name == "-u" and eq:
                raise _unsupported(f"unsupported option {token}")
            if not eq:
                i += 1
                if i >= len(argv):
                    raise Exit(1, "", f"systemd-run: option '{name}' requires an argument\n")
                value = argv[i]
            flags["--unit" if name in ("--unit", "-u") else "--description"] = value
        elif token.startswith("-"):
            raise _unsupported(f"unsupported option {token}")
        else:
            break
        i += 1
    return flags, argv[i:]


def dispatch(ctx: Context) -> tuple[int, object, str]:
    flags, command = parse(ctx.argv)
    if "--user" not in flags or "--scope" not in flags:
        raise _unsupported("the fake models only `systemd-run --user --scope`")
    unit = flags.get("--unit")
    if not isinstance(unit, str):
        raise _unsupported("the fake needs --unit")
    if NAME.fullmatch(unit) is None:
        raise Exit(
            1, "", f"systemd-run: invalid unit name {unit!r} (the real program escapes it)\n"
        )
    if not command:
        raise Exit(1, "", "Command line to execute required.\n")
    if not os.environ.get("XDG_RUNTIME_DIR") and not os.environ.get("DBUS_SESSION_BUS_ADDRESS"):
        raise Exit(1, "", NO_BUS)
    if ctx.rule("scope") is not None:
        return 0, "", "scope"
    path = scopes_path()
    with locked(path):
        scopes: dict[str, int] = json.loads(path.read_text()) if path.exists() else {}
        pid = scopes.get(unit)
        if pid is not None and pid != os.getpid() and _alive(pid):
            raise Exit(
                1, "", f"Failed to start transient scope unit: Unit {unit}.scope already exists.\n"
            )
        scopes[unit] = os.getpid()
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(scopes))
        os.replace(tmp, path)
    sys.stdout.flush()
    sys.stderr.flush()
    # what the real program was observed to do before its exec (systemd 249)
    signal.pthread_sigmask(signal.SIG_SETMASK, set())
    signal.signal(signal.SIGPIPE, signal.SIG_IGN)
    signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
    env = dict(os.environ)
    env["INVOCATION_ID"] = "0123456789abcdef0123456789abcdef"
    try:
        os.execvpe(command[0], command, env)
    except OSError as exc:
        reason = os.strerror(exc.errno or errno.ENOENT)
        raise Exit(1, "", f"Failed to execute {command[0]}: {reason}\n") from None


def main(argv: list[str]) -> int:
    return run_fake("systemd-run", argv, dispatch)
