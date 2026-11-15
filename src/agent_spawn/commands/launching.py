"""``launch`` and ``resume``."""

from __future__ import annotations

from typing import NoReturn

from ..output import Outcome
from .base import Invocation, not_implemented


def launch(inv: Invocation) -> Outcome:
    # TODO(#11): launch order for headless agents (interface.md, Launch order).
    # TODO(#20): worktrees, #27: windows, #30: --placement here.
    not_implemented("launch", 11)


def resume(inv: Invocation) -> NoReturn:
    # TODO(#13): resume and fork (interface.md, Resume order).
    not_implemented("resume", 13)
