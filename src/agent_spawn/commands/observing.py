"""Commands that read or annotate agents: wait, list, show, attach, ack, label."""

from __future__ import annotations

from typing import NoReturn

from .base import Invocation, not_implemented


def wait(inv: Invocation) -> NoReturn:
    # TODO(#17): wait with exit codes, timeouts and result files.
    not_implemented("wait", 17)


def list_agents(inv: Invocation) -> NoReturn:
    # TODO(#15): list.
    not_implemented("list", 15)


def show(inv: Invocation) -> NoReturn:
    # TODO(#15): show.
    not_implemented("show", 15)


def attach(inv: Invocation) -> NoReturn:
    # TODO(#28): attach to an agent's window.
    not_implemented("attach", 28)


def ack(inv: Invocation) -> NoReturn:
    # TODO(#15): ack.
    not_implemented("ack", 15)


def label_add(inv: Invocation) -> NoReturn:
    # TODO(#15): label add, including self-labeling (--session).
    not_implemented("label add", 15)


def label_remove(inv: Invocation) -> NoReturn:
    # TODO(#15): label remove.
    not_implemented("label remove", 15)
