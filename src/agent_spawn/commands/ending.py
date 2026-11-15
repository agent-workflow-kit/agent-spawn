"""Commands that end agents: stop, teardown, prune."""

from __future__ import annotations

from typing import NoReturn

from .base import Invocation, not_implemented


def stop(inv: Invocation) -> NoReturn:
    # TODO(#16): stop a run chain.
    not_implemented("stop", 16)


def teardown(inv: Invocation) -> NoReturn:
    # TODO(#18): teardown without a worktree; TODO(#23): worktrees and named steps.
    not_implemented("teardown", 18)


def prune(inv: Invocation) -> NoReturn:
    # TODO(#24): prune.
    not_implemented("prune", 24)
