"""Commands that read or annotate agents: wait, list, show, attach, ack, label."""

from __future__ import annotations

from .. import annotate, attaching, listing, self_label, showing, wait_agent
from ..output import Outcome
from .base import Invocation


def wait(inv: Invocation) -> Outcome:
    return wait_agent.wait(inv)


def list_agents(inv: Invocation) -> Outcome:
    return listing.list_agents(inv)


def show(inv: Invocation) -> Outcome:
    return showing.show(inv)


def attach(inv: Invocation) -> Outcome:
    return attaching.attach(inv)


def ack(inv: Invocation) -> Outcome:
    return annotate.ack(inv)


def label_add(inv: Invocation) -> Outcome:
    if inv.args.session is not None:
        return self_label.self_label(inv)
    return annotate.label_edit(inv, "add")


def label_remove(inv: Invocation) -> Outcome:
    return annotate.label_edit(inv, "remove")
