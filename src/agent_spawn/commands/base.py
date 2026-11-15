"""What every handler receives and returns."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import NoReturn

from ..output import CliError, Outcome, ResultFileTarget


@dataclass
class Invocation:
    """One parsed and checked command line.

    ``command`` is the key into ``commands.table.HANDLERS`` (``launch``, ``label add``, ...).
    ``args`` is the argparse namespace. ``actor`` is the effective actor and ``decision_ids`` the
    merged ``--decision-id`` values; handlers pass both to every lower-tool call that accepts them.
    ``result_file`` is the checked ``--result-file`` (its parent already resolved), if any.
    """

    command: str
    args: argparse.Namespace
    as_json: bool
    actor: str
    actor_explicit: bool
    decision_ids: list[str]
    config_path: str | None = None
    state_dir: str | None = None
    verbose: bool = False
    quiet: bool = False
    result_file: ResultFileTarget | None = None
    warnings: list[str] = field(default_factory=lambda: [])


Handler = Callable[[Invocation], Outcome]


def not_implemented(command: str, issue: int) -> NoReturn:
    """The body of every handler that a later issue owns."""
    raise CliError.of("internal", f"`{command}` is not implemented yet (tracked in issue #{issue})")
