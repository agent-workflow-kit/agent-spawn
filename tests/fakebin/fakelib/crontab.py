"""Fake ``crontab`` (a user's crontab, as vixie-cron and cronie print and install it).

Models only what agent-spawn uses: ``crontab -l`` (print the table, or ``no crontab for <user>``
with exit 1), ``crontab -`` (install the table read from stdin, replacing the old one as a
whole) and ``crontab -r`` (remove it). The table lives in ``<FAKEBIN_LOG>.crontab`` (bytes
preserved), never anywhere else: the real crontab of the user is never touched. An installed
table is checked the way the real ``crontab`` checks one, in a simplified form: it must end
with a newline, and every line must be blank, a comment, a ``NAME=value`` assignment, or have
a schedule (five fields or an ``@macro``) followed by a command; else exit 1 and nothing is
installed. Every other option exits 1 with ``fake_unsupported`` (never a silent success).

Scripting: a rule for ``list``, ``install`` or ``remove`` answers that call instead (the table is
not read or changed; ``respond.stdout``, ``respond.stderr`` and ``respond.exit`` apply). The
stdin of ``install`` is read and logged either way. See ``tests/fakebin/FIDELITY.md``.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from .core import Context, Exit, _log_path, locked, run_fake

USER = "fakeuser"
USAGE = (
    "usage:\tcrontab [-u user] file\n"
    "\tcrontab [-u user] [ -e | -l | -r ]\n"
    "\t\t(default operation is replace, per 1003.2)\n"
)
ASSIGNMENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*\s*=")
MACRO = re.compile(r"^@(reboot|yearly|annually|monthly|weekly|daily|midnight|hourly)\s+\S")


def table_path() -> Path:
    return Path(str(_log_path()) + ".crontab")


def check_table(text: str) -> str | None:
    """The reason the real crontab would refuse ``text``, or None."""
    if text and not text.endswith("\n"):
        return "premature EOF (missing newline)"
    for number, line in enumerate(text.split("\n")[:-1], 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if ASSIGNMENT.match(stripped) or MACRO.match(stripped):
            continue
        if len(stripped.split(None, 5)) >= 6:
            continue
        return f"bad command on line {number}"
    return None


def dispatch(ctx: Context) -> tuple[int, Any, str]:
    argv = ctx.argv
    if argv == ["-l"]:
        command = "list"
    elif argv == ["-"]:
        command = "install"
    elif argv == ["-r"]:
        command = "remove"
    elif not argv or argv[0].startswith("-") and argv[0] not in {"-e", "-i", "-u"}:
        raise Exit(1, "", USAGE)
    else:
        ctx.fail(1, "fake_unsupported", "the fake models only -l, - and -r")
    path = table_path()
    with locked(path):
        text = ""
        if command == "install":
            text = ctx.read_stdin()  # read (and logged) even when scripted
        if ctx.rule(command) is not None:
            return 0, "", command
        if command == "install":
            problem = check_table(text)
            if problem is not None:
                raise Exit(
                    1,
                    "",
                    f'"-":{problem}\ncrontab: errors in crontab file, can\'t install.\n',
                )
            tmp = path.with_suffix(".tmp")
            tmp.write_bytes(text.encode("utf-8", "surrogateescape"))
            os.replace(tmp, path)
            return 0, "", command
        if not path.exists():
            raise Exit(1, "", f"no crontab for {USER}\n")
        if command == "remove":
            path.unlink()
            return 0, "", command
        return 0, path.read_bytes().decode("utf-8", "surrogateescape"), command


def main(argv: list[str]) -> int:
    return run_fake("crontab", argv, dispatch)
