"""Internal commands (``supervise``); not for direct use."""

from __future__ import annotations

from .. import supervise as supervise_module
from ..output import Outcome
from .base import Invocation


def supervise(inv: Invocation) -> Outcome:
    return supervise_module.supervise(inv)
