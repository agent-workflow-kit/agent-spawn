"""Fake ``tmux``.

Models only what a test of agent-spawn can rely on from tmux's own command line, as observed on
tmux 3.5a: the client options (``-2CDlNuVv``, ``-c``, ``-f``, ``-L``, ``-S``, ``-T``), ``-V``
(``tmux <version>``), ``;`` as a command separator (an escaped one is literal), and
the "no server" failure every other command gets without a script. Everything else is scripted
per test (``respond.stdout`` and ``respond.exit`` on a rule for the first command name). Process
tests that need the real behavior use a private real server instead (``private_tmux`` fixture).
"""

from __future__ import annotations

import os
from typing import Any

from .core import Context, Exit, run_fake

DEFAULT_VERSION = "3.4"
BOOL_OPTIONS = set("2CDlNuvV")
VALUE_OPTIONS = set("cfLST")
USAGE = (
    "usage: tmux [-2CDlNuVv] [-c shell-command] [-f file] [-L socket-name]\n"
    "            [-S socket-path] [-T features] [command [flags]]\n"
)


def split_commands(args: list[str]) -> list[list[str]]:
    """Split the command part of an argv at ``;`` separators, like tmux's argument parser."""
    commands: list[list[str]] = [[]]
    for token in args:
        if token == ";":
            commands.append([])
        elif token.endswith("\\;"):
            commands[-1].append(token[:-2] + ";")
        elif token.endswith(";"):
            commands[-1].append(token[:-1])
            commands.append([])
        else:
            commands[-1].append(token)
    return [c for c in commands if c]


def parse_client_options(argv: list[str]) -> tuple[dict[str, Any], list[str]]:
    options: dict[str, Any] = {}
    i = 0
    while i < len(argv):
        token = argv[i]
        if not token.startswith("-") or token == "-":
            break
        if token == "--":
            i += 1
            break
        cluster = token[1:]
        j = 0
        while j < len(cluster):
            letter = cluster[j]
            if letter in BOOL_OPTIONS:
                options[letter] = True
                j += 1
            elif letter in VALUE_OPTIONS:
                rest = cluster[j + 1 :]
                if rest:
                    options[letter] = rest
                else:
                    i += 1
                    if i >= len(argv):
                        raise Exit(1, "", f"tmux: option requires an argument -- {letter}\n{USAGE}")
                    options[letter] = argv[i]
                break
            else:
                raise Exit(1, "", f"tmux: unknown option -- {letter}\n{USAGE}")
        i += 1
    return options, argv[i:]


def socket_path(options: dict[str, Any]) -> str:
    if "S" in options:
        return str(options["S"])
    tmpdir = os.environ.get("TMUX_TMPDIR") or "/tmp"
    return f"{tmpdir}/tmux-{os.getuid()}/{options.get('L', 'default')}"


def dispatch(ctx: Context) -> tuple[int, Any, str]:
    options, rest = parse_client_options(ctx.argv)
    if options.get("V"):
        raise Exit(0, f"tmux {os.environ.get('FAKEBIN_TMUX_VERSION', DEFAULT_VERSION)}\n")
    commands = split_commands(rest)
    if not commands:
        # Real tmux with no command starts a session; the fake does not.
        ctx.fail(1, "fake_unsupported", "the fake does not start sessions")
    name = commands[0][0]
    response = ctx.response(name)
    if response:
        # A scripted command succeeds as the script says (exit/stdout/stderr applied after).
        return 0, "", name
    raise Exit(1, "", f"error connecting to {socket_path(options)} (No such file or directory)\n")


def main(argv: list[str]) -> int:
    return run_fake("tmux", argv, dispatch)
