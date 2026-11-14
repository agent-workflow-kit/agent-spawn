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
BOOL_OPTIONS = set("2ClNquvV")  # -q is accepted and ignored
VALUE_OPTIONS = set("cfLST")
#: -d, -D and -U print the usage and nothing else (tmux 3.5a): they belong to the real client's
#: attach/server modes, which the fake does not model.
USAGE_ONLY_OPTIONS = set("dDU")
#: Command aliases of tmux 3.5a (``tmux list-commands``): a scripted rule for either spelling
#: answers both.
ALIASES = {
    "attach": "attach-session", "bind": "bind-key", "breakp": "break-pane",
    "capturep": "capture-pane", "clearhist": "clear-history",
    "clearphist": "clear-prompt-history", "confirm": "confirm-before",
    "deleteb": "delete-buffer", "detach": "detach-client", "menu": "display-menu",
    "display": "display-message", "popup": "display-popup", "displayp": "display-panes",
    "findw": "find-window", "has": "has-session", "if": "if-shell", "joinp": "join-pane",
    "killp": "kill-pane", "killw": "kill-window", "lastp": "last-pane", "last": "last-window",
    "linkw": "link-window", "lsb": "list-buffers", "lsc": "list-clients",
    "lscm": "list-commands", "lsk": "list-keys", "lsp": "list-panes", "ls": "list-sessions",
    "lsw": "list-windows", "loadb": "load-buffer", "lockc": "lock-client",
    "lock": "lock-server", "locks": "lock-session", "movep": "move-pane",
    "movew": "move-window", "new": "new-session", "neww": "new-window",
    "nextl": "next-layout", "next": "next-window", "pasteb": "paste-buffer",
    "pipep": "pipe-pane", "prevl": "previous-layout", "prev": "previous-window",
    "refresh": "refresh-client", "rename": "rename-session", "renamew": "rename-window",
    "resizep": "resize-pane", "resizew": "resize-window", "respawnp": "respawn-pane",
    "respawnw": "respawn-window", "rotatew": "rotate-window", "run": "run-shell",
    "saveb": "save-buffer", "selectl": "select-layout", "selectp": "select-pane",
    "selectw": "select-window", "send": "send-keys", "setb": "set-buffer",
    "setenv": "set-environment", "set": "set-option", "setw": "set-window-option",
    "showb": "show-buffer", "showenv": "show-environment", "showmsgs": "show-messages",
    "show": "show-options", "showphist": "show-prompt-history", "showw": "show-window-options",
    "source": "source-file", "splitw": "split-window", "start": "start-server",
    "suspendc": "suspend-client", "swapp": "swap-pane", "swapw": "swap-window",
    "switchc": "switch-client", "unbind": "unbind-key", "unlinkw": "unlink-window",
    "wait": "wait-for",
}  # fmt: skip
USAGE = (
    "usage: tmux [-2CDlNuVv] [-c shell-command] [-f file] [-L socket-name]\n"
    "            [-S socket-path] [-T features] [command [flags]]\n"
)


def canonical(name: str) -> str:
    """The full command name for an alias (``ls`` -> ``list-sessions``)."""
    return ALIASES.get(name, name)


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
            if letter in USAGE_ONLY_OPTIONS:
                raise Exit(1, "", USAGE)
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
    ctx.canonical = canonical
    name = canonical(commands[0][0])
    if ctx.rule(name) is not None:
        # A scripted command succeeds as the script says (exit/stdout/stderr applied after);
        # an empty ``respond`` is still a rule.
        return 0, "", name
    if name == "start-server":
        return 0, "", name  # a server with no session exits at once; nothing is printed
    if name == "list-commands":
        ctx.fail(1, "fake_unsupported", "the fake does not print the command table")
    sock = socket_path(options)
    if os.path.lexists(sock):  # a socket file nobody listens on (or any other file)
        raise Exit(1, "", f"no server running on {sock}\n")
    raise Exit(1, "", f"error connecting to {sock} (No such file or directory)\n")


def main(argv: list[str]) -> int:
    return run_fake("tmux", argv, dispatch)
