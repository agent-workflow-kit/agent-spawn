"""Commands that end agents: stop, teardown, prune."""

from __future__ import annotations

from typing import NoReturn

from .. import stop as stop_module
from ..output import Outcome
from .base import Invocation, not_implemented


def stop(inv: Invocation) -> Outcome:
    return stop_module.stop(inv)


def teardown(inv: Invocation) -> NoReturn:
    # TODO(#18): teardown without a worktree; TODO(#23): worktrees and named steps.
    not_implemented("teardown", 18)


def prune(inv: Invocation) -> NoReturn:
    # TODO(#24): prune.
    not_implemented("prune", 24)
