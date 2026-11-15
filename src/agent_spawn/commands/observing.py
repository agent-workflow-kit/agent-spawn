"""Commands that read or annotate agents: wait, list, show, attach, ack, label."""

from __future__ import annotations

from typing import NoReturn

from .. import annotate, listing, self_label, showing
from ..output import Outcome
from .base import Invocation, not_implemented


def wait(inv: Invocation) -> NoReturn:
    # TODO(#17): wait with exit codes, timeouts and result files.
    not_implemented("wait", 17)


def list_agents(inv: Invocation) -> Outcome:
    return listing.list_agents(inv)


def show(inv: Invocation) -> Outcome:
    return showing.show(inv)


def attach(inv: Invocation) -> NoReturn:
    # TODO(#28): attach to an agent's window.
    not_implemented("attach", 28)


def ack(inv: Invocation) -> Outcome:
    return annotate.ack(inv)


def label_add(inv: Invocation) -> Outcome:
    if inv.args.session is not None:
        return self_label.self_label(inv)
    return annotate.label_edit(inv, "add")


def label_remove(inv: Invocation) -> Outcome:
    return annotate.label_edit(inv, "remove")
