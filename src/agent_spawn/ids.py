"""Agent ids: grammar, allocation and prefix resolution.

An id is ``ag-`` plus 8 base32 characters (``[a-z2-7]``, 40 random bits). Ids are never reused,
also after tombstone deletion: allocation asks the lookup, which must list tombstones.
Resolution accepts the optional ``ag-`` and a prefix of 4 to 7 characters; a prefix never matches a
torn-down agent, a full id always does (the caller decides what a torn-down agent means).

The state store (#6) provides the ``AgentIndex``; this module only needs its one lookup.
"""

from __future__ import annotations

import base64
import os
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

from .output import CliError
from .validate import AGENT_ID, GrammarError, arg_type, shown

__all__ = [
    "AgentIndex",
    "IdEntry",
    "ResolvedId",
    "agent_id_arg",
    "check_agent_id_text",
    "is_agent_id",
    "new_agent_id",
    "resolve_agent_id",
]

MIN_PREFIX = 4
ID_CHARS = 8
_PREFIX = re.compile(r"[a-z2-7]{1,8}")
_ALLOCATION_ATTEMPTS = 32


@dataclass(frozen=True)
class IdEntry:
    """One stored agent id; ``torn_down`` marks a tombstone."""

    agent_id: str
    torn_down: bool


class AgentIndex(Protocol):
    """What id allocation and resolution need from the state store.

    ``entries_with_prefix("ag-abcd")`` returns every stored id starting with the text, tombstones
    included; a full id gives its own entry or nothing.
    """

    def entries_with_prefix(self, prefix: str) -> Sequence[IdEntry]: ...


# TODO(#6): implement AgentIndex on the state store (one indexed prefix query over `agents`,
# tombstones included).


def is_agent_id(value: str) -> bool:
    return AGENT_ID.fullmatch(value) is not None


def new_agent_id(index: AgentIndex, *, random_bytes: Callable[[int], bytes] = os.urandom) -> str:
    """A fresh id no stored agent, live or tombstoned, has."""
    for _ in range(_ALLOCATION_ATTEMPTS):
        candidate = "ag-" + base64.b32encode(random_bytes(5)).decode("ascii").lower()
        if not index.entries_with_prefix(candidate):
            return candidate
    raise RuntimeError("could not allocate an unused agent id")


@dataclass(frozen=True)
class ResolvedId:
    agent_id: str
    torn_down: bool


def check_agent_id_text(text: str) -> str:
    """The part of ``text`` after the optional ``ag-``, or ``GrammarError``.

    At least 4 and at most 8 characters of ``[a-z2-7]``; fewer than 4 is exit 2, as is anything
    that cannot be an id prefix.
    """
    body = text[3:] if text.startswith("ag-") else text
    if len(text) > 16 or _PREFIX.fullmatch(body) is None:
        raise GrammarError(
            f"invalid agent id {shown(text)}: expected ag- and 8 characters of a-z and 2-7, "
            "or a prefix of at least 4 of them"
        )
    if len(body) < MIN_PREFIX:
        raise GrammarError(
            f"agent id prefix {shown(text)} is too short: give at least {MIN_PREFIX} "
            "characters after ag-"
        )
    return body


agent_id_arg = arg_type(check_agent_id_text)


def resolve_agent_id(text: str, index: AgentIndex) -> ResolvedId:
    """Resolve user input to one stored id.

    Malformed or too short: exit 2. A full id: its entry, tombstone or not, else exit 10. A prefix:
    among non-torn-down agents only; none exit 10, several exit 21 with ``candidates``.
    """
    try:
        body = check_agent_id_text(text)
    except GrammarError as exc:
        raise CliError.usage(str(exc)) from None
    prefix = "ag-" + body
    entries = [
        entry for entry in index.entries_with_prefix(prefix) if entry.agent_id.startswith(prefix)
    ]
    if len(body) == ID_CHARS:
        for entry in entries:
            if entry.agent_id == prefix:
                return ResolvedId(entry.agent_id, entry.torn_down)
        raise CliError.of("not_found", f"no agent {prefix}")
    live = sorted({entry.agent_id for entry in entries if not entry.torn_down})
    if not live:
        raise CliError.of("not_found", f"no agent matches {prefix}")
    if len(live) > 1:
        raise CliError.of("ambiguous", f"{prefix} matches {len(live)} agents", candidates=live)
    return ResolvedId(live[0], torn_down=False)
