"""``browse``: check the environment, run the screen, write the intent file."""

from __future__ import annotations

import signal
import sys

from .. import gateway as gw
from ..commands.base import Invocation
from ..output import CliError, Outcome, PayloadKind, resolve_result_file, safe_text
from .model import SubprocessBackend, write_intent

__all__ = ["INSTALL_HINT", "browse"]

INSTALL_HINT = (
    "browse needs Textual, which is not installed: install it with "
    "`pip install 'agent-spawn[tui]'` (or `uv tool install 'agent-spawn[tui]'`)"
)


def _is_tty(stream: object) -> bool:
    try:
        return bool(stream is not None and stream.isatty())  # pyright: ignore[reportAttributeAccessIssue]
    except (ValueError, OSError, AttributeError):
        return False


def browse(inv: Invocation) -> Outcome:
    try:
        import textual  # noqa: F401  (presence check only)
    except ImportError:
        raise CliError.usage(INSTALL_HINT) from None
    if not (_is_tty(sys.stdin) and _is_tty(sys.stdout)):
        raise CliError.usage("browse needs a terminal on stdin and stdout")
    target = None
    raw: str | None = inv.args.result_file
    if raw is not None:
        try:
            target = resolve_result_file(raw)
        except CliError:
            raise
        except OSError as exc:
            sys.stderr.write(
                safe_text(f"agent-spawn: warning: --result-file {raw} not resolved: {exc.strerror}")
                + "\n"
            )
    from .app import BrowseApp  # needs Textual

    backend = SubprocessBackend(
        config_path=inv.config_path, actor=inv.actor, decision_ids=inv.decision_ids
    )
    app = BrowseApp(backend, focus=inv.args.focus, record_intent=raw is not None)
    # SIGTERM and SIGHUP leave the screen cleanly (alt-screen and mouse mode restored, a `back`
    # intent written) instead of killing the process with the terminal still in TUI mode
    gateway = gw.get_gateway()
    previous = {
        sig: gateway.install_signal_handler(sig, lambda _signum, _frame: app.request_exit())
        for sig in (signal.SIGTERM, signal.SIGHUP)
    }
    try:
        app.run()
    finally:
        for sig, handler in previous.items():
            gateway.install_signal_handler(sig, handler)
        backend.close()
    if target is not None:
        intent, selected = app.intent if app.intent is not None else ("back", [])
        warning = write_intent(target, intent, selected)
        if warning is not None:
            sys.stderr.write(safe_text(f"agent-spawn: warning: {warning}") + "\n")
    return Outcome(0, {}, PayloadKind.COMMAND, quiet=True)
