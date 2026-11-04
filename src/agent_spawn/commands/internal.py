"""Internal commands (``supervise``, ``_session-event``); not for direct use."""

from __future__ import annotations

from typing import NoReturn

from .base import Invocation, not_implemented


def supervise(inv: Invocation) -> NoReturn:
    # TODO(#12): the supervisor; it refuses (exit 2) when the agent's record does not expect it.
    not_implemented("supervise", 12)


def session_event(inv: Invocation) -> NoReturn:
    # TODO(#14): the relay. It must exit 0 always once implemented (interface.md, State relay);
    # the usage checks and the config load (config_invalid, exit 2) of cli.py run before this
    # handler.
    not_implemented("_session-event", 14)
