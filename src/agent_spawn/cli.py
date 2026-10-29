"""Command-line entry point for agent-spawn."""

from __future__ import annotations

import argparse
import sys

from . import __version__

PROG = "agent-spawn"
DESCRIPTION = (
    "Launch, resume, park and tear down Claude agents: worktree + terminal + Claude session."
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=PROG,
        description=DESCRIPTION,
    )
    parser.add_argument("--version", action="version", version=f"{PROG} {__version__}")
    parser.add_argument("--json", action="store_true", help="machine-readable JSON output")
    parser.add_subparsers(dest="command", metavar="<command>")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help(sys.stderr)
        return 2
    return 0
