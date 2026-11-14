"""Housekeeping commands: reconcile, cron, export, browse."""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from typing import NoReturn

from .. import records
from ..output import CliError, Outcome, PayloadKind, safe_text
from ..store import default_state_dir
from ..validate import GrammarError, parse_since
from .base import Invocation, not_implemented


def reconcile(inv: Invocation) -> NoReturn:
    # TODO(#33): reconcile.
    not_implemented("reconcile", 33)


def cron_install(inv: Invocation) -> NoReturn:
    # TODO(#34): cron install.
    not_implemented("cron install", 34)


def cron_uninstall(inv: Invocation) -> NoReturn:
    # TODO(#34): cron uninstall.
    not_implemented("cron uninstall", 34)


def export(inv: Invocation) -> Outcome:
    """Print ``audit.jsonl`` records, oldest first, filtered by ``--since`` and ``--action``."""
    actions: list[str] = inv.args.action or []
    for action in actions:
        if action not in records.AUDIT_ACTIONS:
            raise CliError.usage(
                f"unknown audit action {safe_text(action)!r}: expected one of "
                + ", ".join(sorted(records.AUDIT_ACTIONS))
            )
    since: datetime | None = None
    if inv.args.since is not None:
        try:
            since = parse_since(inv.args.since, datetime.now(UTC))
        except GrammarError as exc:
            raise CliError.usage(str(exc)) from None
    state_dir = inv.state_dir or str(default_state_dir())
    try:
        result = records.read_audit(state_dir)
    except records.RecordsError as exc:
        raise CliError.of("failed", f"cannot read the audit log: {exc}") from None
    selected = [
        record
        for record in result.records
        if (not actions or record["action"] in actions)
        and (since is None or records.parse_record_ts(record["ts"]) >= since)
    ]
    if result.skipped:
        message = (
            f"agent-spawn: warning: skipped {result.skipped} unreadable line(s) in audit.jsonl\n"
        )
        sys.stderr.write(message)
    rows = [
        f"{r['ts']}  {r['action']}  {r['target']}  {r['result']}  {r['actor']}" for r in selected
    ]
    human = "\n".join(rows) if rows else "no audit records"
    return Outcome(0, selected, PayloadKind.COMMAND, human)


def browse(inv: Invocation) -> NoReturn:
    # TODO(#38): the Textual TUI.
    # TODO(#38): browse writes its own intent file (schemas/cli/browse-result.json); the
    # generic --result-file handling of cli.py covers only launch, resume and wait.
    not_implemented("browse", 38)
