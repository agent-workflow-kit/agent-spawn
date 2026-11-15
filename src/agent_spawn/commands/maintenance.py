"""Housekeeping commands: reconcile, cron, export, browse."""

from __future__ import annotations

from typing import NoReturn

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


def export(inv: Invocation) -> NoReturn:
    # TODO(#7): export the audit log.
    not_implemented("export", 7)


def browse(inv: Invocation) -> NoReturn:
    # TODO(#38): the Textual TUI.
    # TODO(#38): browse writes its own intent file (schemas/cli/browse-result.json); the
    # generic --result-file handling of cli.py covers only launch, resume and wait.
    not_implemented("browse", 38)
