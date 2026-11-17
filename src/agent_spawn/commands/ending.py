"""Commands that end agents: stop, teardown, prune."""

from __future__ import annotations

from .. import prune as prune_module
from .. import stop as stop_module
from .. import teardown as teardown_module
from ..output import Outcome
from .base import Invocation


def stop(inv: Invocation) -> Outcome:
    return stop_module.stop(inv)


def teardown(inv: Invocation) -> Outcome:
    return teardown_module.teardown_command(inv)


def prune(inv: Invocation) -> Outcome:
    return prune_module.prune_command(inv)
