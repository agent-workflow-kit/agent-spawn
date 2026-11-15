"""Internal commands (``supervise``, ``_session-event``); not for direct use."""

from __future__ import annotations

from .. import relay
from .. import supervise as supervise_module
from ..output import Outcome, PayloadKind
from .base import Invocation


def supervise(inv: Invocation) -> Outcome:
    return supervise_module.supervise(inv)


def session_event(inv: Invocation) -> Outcome:
    """Reached only through a handler call in tests: ``cli.main`` hands the real command to
    ``relay.run`` before parsing, so that it exits 0 whatever its argv or config."""
    relay.run(["--state-dir", inv.state_dir or "", "--config", inv.config_path or "", "_session-event"])
    return Outcome(0, {}, PayloadKind.COMMAND, quiet=True)
