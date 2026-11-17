"""``cron install`` and ``cron uninstall`` (interface.md, ``cron install`` / ``cron uninstall``).

The user's crontab is shared with every other tool, so the rules here are strict:

* the whole table is read with ``crontab -l`` and written back, whole, with ``crontab -`` on
  stdin (the one replacement the crontab program does atomically); the only lines that ever
  change are the ones between the two marker lines of the managed block, every other line stays
  byte for byte where it was;
* a read that fails (anything but exit 1 with ``no crontab for <user>``, which is an empty table)
  aborts before anything is written;
* a table with an unterminated or hand-edited block is refused rather than guessed at;
* values that go into the cron line (paths, the env file) are checked (absolute, no control
  characters, no ``:`` in a ``PATH`` entry), quoted for ``sh`` and have ``%`` escaped (cron turns
  an unescaped ``%`` into a newline); the env file is only named, never read;
* nothing here runs ``crontab`` except through the gateway, and ``--dry-run`` only reads.
"""

from __future__ import annotations

import os
import stat
import subprocess
from dataclasses import dataclass

from . import gateway
from .commands.base import Invocation
from .launch import resolve_self
from .output import CliError, Outcome, PayloadKind, safe_text

BEGIN = "# agent-spawn managed"
END = "# agent-spawn managed end"
DEFAULT_INTERVAL = 10
#: Vixie cron reads at most this many bytes of a line (the others allow more).
MAX_LINE = 1000
CRONTAB_TIMEOUT = 30.0
SYSTEM_PATH = ("/usr/local/bin", "/usr/bin", "/bin")
MACOS_PATH = "/opt/homebrew/bin"


def _failed(message: str) -> CliError:
    return CliError.of("failed", message)


# --- the cron line --------------------------------------------------------------------------


def _check_text(value: str, what: str) -> None:
    bad = any(ord(c) < 0x20 or ord(c) == 0x7F for c in value)
    if not value or bad:
        raise _failed(f"{what} is empty or contains a control character: {safe_text(value)!r}")


def _absolute(value: str, what: str) -> str:
    _check_text(value, what)
    if not os.path.isabs(value):
        raise _failed(f"{what} is not an absolute path: {safe_text(value)!r}")
    return value


def shell_quote(value: str) -> str:
    """``value`` as one ``sh`` word: single-quoted, so nothing in it is interpreted."""
    return "'" + value.replace("'", "'\\''") + "'"


def cron_escape(command: str) -> str:
    """Escape ``%`` (cron makes an unescaped one a newline; a backslash before it is dropped)."""
    return command.replace("%", "\\%")


def _search_path(self_path: str) -> str:
    home = os.path.expanduser("~")
    entries = [os.path.dirname(self_path), os.path.join(home, ".local", "bin"), *SYSTEM_PATH]
    if gateway.get_gateway().platform() == "darwin":
        entries.append(MACOS_PATH)
    seen: list[str] = []
    for entry in entries:
        _absolute(entry, "a PATH directory")
        if ":" in entry:
            raise _failed(f"a PATH directory contains ':': {safe_text(entry)!r}")
        if entry not in seen:
            seen.append(entry)
    return ":".join(seen)


def check_env_file(path: str, uid: int) -> str:
    """The env file of ``--env-file``: absolute, a regular file, owned by the user and not
    writable by group or others (else usage, exit 2). Only its metadata is read."""
    try:
        if not path or any(ord(c) < 0x20 or ord(c) == 0x7F for c in path):
            raise CliError.usage("--env-file contains a control character")
        if not os.path.isabs(path):
            raise CliError.usage(f"--env-file must be an absolute path: {safe_text(path)!r}")
        info = os.stat(path)
    except OSError as exc:
        raise CliError.usage(
            f"--env-file {safe_text(path)!r} cannot be used: {exc.strerror or 'error'}"
        ) from None
    if not stat.S_ISREG(info.st_mode):
        raise CliError.usage(f"--env-file {safe_text(path)!r} is not a regular file")
    if info.st_uid != uid:
        raise CliError.usage(f"--env-file {safe_text(path)!r} is not owned by you")
    if info.st_mode & 0o022:
        raise CliError.usage(
            f"--env-file {safe_text(path)!r} is writable by group or others (mode "
            f"{stat.S_IMODE(info.st_mode):04o})"
        )
    return path


def build_block(inv: Invocation, interval: int, env_file: str | None) -> list[str]:
    """The three lines of the managed block."""
    self_path = _absolute(resolve_self(), "the agent-spawn path")
    assignments = []
    for name, value in (
        ("XDG_STATE_HOME", os.environ.get("XDG_STATE_HOME", "")),
        ("XDG_CONFIG_HOME", os.environ.get("XDG_CONFIG_HOME", "")),
    ):
        # Where the state and config live for this user, as the commands here see them (cron
        # has no XDG variables); a relative value is ignored by the program, so it is here.
        if value and os.path.isabs(value):
            assignments.append(f"{name}={shell_quote(_absolute(value, name))}")
    assignments.append(f"PATH={shell_quote(_search_path(self_path))}")
    program = [shell_quote(self_path)]
    if inv.config_path is not None and inv.config_file is not None:
        program += ["--config", shell_quote(_absolute(inv.config_file, "the config path"))]
    program += ["reconcile", "--quiet"]
    command = " ".join([*assignments, *program])
    if env_file is not None:
        # allexport: a plain NAME=value file works like a .env file; the explicit PATH (and the
        # state variables) still come last and win.
        command = f"set -a && . {shell_quote(env_file)} && set +a && {command}"
    line = cron_escape(f"*/{interval} * * * * {command}")
    if len(line.encode("utf-8", "surrogateescape")) > MAX_LINE:
        raise _failed(f"the cron line would be longer than {MAX_LINE} bytes: use shorter paths")
    return [BEGIN, line, END]


# --- the table ------------------------------------------------------------------------------


def split_lines(table: str) -> list[str]:
    """Lines of ``table`` split on ``\\n`` only (never on the other Unicode line breaks)."""
    lines = table.split("\n")
    if lines[-1] == "":
        lines.pop()
    return lines


def join_lines(lines: list[str]) -> str:
    return "".join(f"{line}\n" for line in lines)


def _marker(line: str) -> str:
    return line.rstrip(" \t\r")


@dataclass(frozen=True)
class Span:
    """A managed block: ``lines[start:stop]``, markers included."""

    start: int
    stop: int


def find_blocks(lines: list[str]) -> list[Span]:
    """Every managed block. An unterminated one, or one holding a line that is neither a comment
    nor a reconcile job (a hand edit), is refused: guessing could swallow someone's job."""
    spans: list[Span] = []
    i = 0
    while i < len(lines):
        if _marker(lines[i]) != BEGIN:
            i += 1
            continue
        stop = next((j for j in range(i + 1, len(lines)) if _marker(lines[j]) == END), None)
        if stop is None:
            raise _failed(
                f"the crontab has `{BEGIN}` on line {i + 1} but no `{END}` line after it: "
                "fix it with `crontab -e`, then run this again"
            )
        for j in range(i + 1, stop):
            body = lines[j].strip()
            if body and not body.startswith("#") and "reconcile" not in body:
                raise _failed(
                    f"line {j + 1} of the crontab is inside the agent-spawn block but is not "
                    "an agent-spawn job: move it out of the block with `crontab -e`, then run "
                    "this again"
                )
        spans.append(Span(i, stop + 1))
        i = stop + 1
    return spans


def installed(lines: list[str], block: list[str]) -> tuple[str, list[str]]:
    """The action (``installed``, ``updated``, ``unchanged``) and the new table lines."""
    spans = find_blocks(lines)
    if not spans:
        return "installed", [*lines, *block]
    first = spans[0]
    out = lines[: first.start] + block
    cursor = first.stop
    for span in spans[1:]:
        out += lines[cursor : span.start]
        cursor = span.stop
    out += lines[cursor:]
    unchanged = len(spans) == 1 and lines[first.start : first.stop] == block
    return ("unchanged" if unchanged else "updated"), out


def uninstalled(lines: list[str]) -> tuple[list[str], list[str] | None]:
    """The table without any managed block, and the removed block (the first), if any."""
    spans = find_blocks(lines)
    if not spans:
        return lines, None
    out: list[str] = []
    cursor = 0
    for span in spans:
        out += lines[cursor : span.start]
        cursor = span.stop
    out += lines[cursor:]
    return out, lines[spans[0].start : spans[0].stop]


# --- crontab, through the gateway -----------------------------------------------------------


def _run(argv: list[str], *, stdin: str | None = None) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["LC_ALL"] = "C"  # the "no crontab for" message is localized by some implementations
    try:
        return gateway.get_gateway().run(argv, stdin=stdin, timeout=CRONTAB_TIMEOUT, env=env)
    except FileNotFoundError:
        raise _failed("the `crontab` command was not found") from None
    except (OSError, subprocess.SubprocessError, gateway.OutputTooLarge) as exc:
        raise _failed(f"`{' '.join(argv)}` failed: {safe_text(str(exc))}") from None


def _problem(done: subprocess.CompletedProcess[str]) -> str:
    text = (done.stderr or done.stdout).strip()
    return f"exit {done.returncode}" + (f": {safe_text(text[:500])}" if text else "")


def read_table() -> str:
    """The current crontab text ("" when the user has none); anything else aborts."""
    done = _run(["crontab", "-l"])
    if done.returncode == 0:
        return done.stdout
    if done.returncode == 1 and not done.stdout and "no crontab for" in done.stderr:
        return ""
    raise _failed(f"`crontab -l` failed ({_problem(done)}): the crontab was not changed")


def write_table(lines: list[str]) -> None:
    """Replace the whole crontab. An empty result removes it (an empty install is refused by
    some implementations)."""
    if not any(line.strip() for line in lines):
        done = _run(["crontab", "-r"])
        action = "crontab -r"
    else:
        done = _run(["crontab", "-"], stdin=join_lines(lines))
        action = "crontab -"
    if done.returncode != 0:
        raise _failed(f"`{action}` failed ({_problem(done)}): the crontab was not changed")


# --- the commands ---------------------------------------------------------------------------


def install_command(inv: Invocation) -> Outcome:
    dry_run = bool(inv.args.dry_run)
    interval: int = DEFAULT_INTERVAL if inv.args.interval is None else inv.args.interval
    env_file: str | None = inv.args.env_file
    if env_file is not None:
        env_file = check_env_file(env_file, gateway.get_gateway().getuid())
    block = build_block(inv, interval, env_file)
    lines = split_lines(read_table())
    action, new_lines = installed(lines, block)
    if action != "unchanged" and not dry_run:
        write_table(new_lines)
    payload = {
        "action": action,
        "block": "\n".join(block),
        "dry_run": dry_run,
        "interval_minutes": interval,
        "env_file": env_file,
    }
    if dry_run:
        verb = {"installed": "install", "updated": "update", "unchanged": "keep"}[action]
        head = f"would {verb} the agent-spawn cron block:"
    else:
        head = {
            "installed": "installed the agent-spawn cron block:",
            "updated": "updated the agent-spawn cron block:",
            "unchanged": "the agent-spawn cron block is already installed:",
        }[action]
    human = head + "\n" + safe_text(payload["block"], multiline=True)
    return Outcome(0, payload, PayloadKind.COMMAND, human)


def uninstall_command(inv: Invocation) -> Outcome:
    dry_run = bool(inv.args.dry_run)
    lines = split_lines(read_table())
    new_lines, removed = uninstalled(lines)
    if removed is not None and not dry_run:
        write_table(new_lines)
    payload = {
        "action": "unchanged" if removed is None else "removed",
        "block": None if removed is None else "\n".join(removed),
        "dry_run": dry_run,
    }
    if removed is None:
        human = "no agent-spawn cron block in the crontab"
    else:
        verb = "would remove" if dry_run else "removed"
        human = f"{verb} the agent-spawn cron block:\n" + safe_text(
            payload["block"] or "", multiline=True
        )
    return Outcome(0, payload, PayloadKind.COMMAND, human)
