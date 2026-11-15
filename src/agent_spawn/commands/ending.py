"""Commands that end agents: stop, teardown, prune."""

from __future__ import annotations

from typing import NoReturn

from .. import teardown as teardown_module
from ..output import Outcome
from .base import Invocation, not_implemented


def stop(inv: Invocation) -> NoReturn:
    # TODO(#16): stop a run chain.
    not_implemented("stop", 16)


def teardown(inv: Invocation) -> Outcome:
    # TODO(#23): worktrees and windows; TODO(#32): the pre-teardown hook (see teardown.py).
    return teardown_module.teardown_command(inv)


def prune(inv: Invocation) -> NoReturn:
    # TODO(#24): prune.
    not_implemented("prune", 24)
