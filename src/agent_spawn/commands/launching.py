"""``launch`` and ``resume``."""

from __future__ import annotations

from .. import launch as launch_module
from .. import resume as resume_module
from ..output import Outcome
from .base import Invocation


def launch(inv: Invocation) -> Outcome:
    # TODO(#27): windows, #30: --placement here (see launch.Launch.usage_checks).
    return launch_module.launch(inv, restore_handlers=False)


def resume(inv: Invocation) -> Outcome:
    # TODO(#27): windows, #30: --placement here (see resume.Resume.plan_run).
    return resume_module.resume(inv, restore_handlers=False)
