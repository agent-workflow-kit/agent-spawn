"""``launch`` and ``resume``."""

from __future__ import annotations

from typing import NoReturn

from .. import launch as launch_module
from ..output import Outcome
from .base import Invocation, not_implemented


def launch(inv: Invocation) -> Outcome:
    # TODO(#20): worktrees, #27: windows, #30: --placement here (see launch.Launch.usage_checks).
    return launch_module.launch(inv, restore_handlers=False)


def resume(inv: Invocation) -> NoReturn:
    # TODO(#13): resume and fork (interface.md, Resume order).
    not_implemented("resume", 13)
