"""Commands that end agents: stop, teardown, prune."""

from __future__ import annotations

from typing import NoReturn

from .. import stop as stop_module
from .. import teardown as teardown_module
from ..output import Outcome
from .base import Invocation, not_implemented


def stop(inv: Invocation) -> Outcome:
    return stop_module.stop(inv)


def teardown(inv: Invocation) -> Outcome:
    # TODO(#27): windows; TODO(#32): the pre-teardown hook (see teardown.py).
    return teardown_module.teardown_command(inv)


def prune(inv: Invocation) -> NoReturn:
    # TODO(#24): prune.
    not_implemented("prune", 24)
