"""Every shared input grammar: durations, names, actors, decision ids, shas and times.

Each ``validate_*`` / ``parse_*`` function returns the checked value or raises ``GrammarError``
(a ``ValueError``; the CLI turns it into a usage error, exit 2). Rules:

- Patterns are matched with ``fullmatch`` and spell out ASCII classes (``[0-9]``, never ``\\d``),
  so a trailing newline, a fullwidth digit or a lookalike letter never passes.
- A rejected value is shown in messages only through ``shown``: bounded and escaped, so an
  untrusted value never prints raw control characters or bidi marks.
- Ambiguity is resolved by rejecting (interface.md is normative; see each function).
"""

from __future__ import annotations

import argparse
import re
from collections.abc import Callable, Sequence
from datetime import UTC, datetime, timedelta, timezone
from decimal import Decimal

__all__ = [
    "AGENT_ID",
    "GrammarError",
    "MAX_DECISION_IDS",
    "MAX_DURATION_SECONDS",
    "arg_type",
    "default_name",
    "duration_arg",
    "parse_duration",
    "parse_since",
    "shown",
    "validate_actor",
    "validate_decision_id",
    "validate_decision_ids",
    "validate_merged_head",
    "validate_name",
]

AGENT_ID = re.compile(r"ag-[a-z2-7]{8}")
_ACTOR = re.compile(r"(?:human|auto):[A-Za-z0-9._@-]{1,64}")
_DECISION_ID = re.compile(r"[A-Za-z0-9._:@/-]{1,128}")
_MERGED_HEAD = re.compile(r"[0-9a-f]{40}")
_DURATION = re.compile(r"([0-9]+(?:\.[0-9]+)?)(ms|s|m|h|d)?")
_TIME = re.compile(
    r"([0-9]{4})-([0-9]{2})-([0-9]{2})"
    r"(?:[T ]([0-9]{2}):([0-9]{2})(?::([0-9]{2})(?:\.([0-9]{1,6}))?)?"
    r"(Z|[+-][0-9]{2}:[0-9]{2})?)?"
)

MAX_DECISION_IDS = 32
MAX_NAME_CODE_POINTS = 64
MAX_DURATION_TEXT = 32
# Ten years: far beyond any real timeout, small enough that no arithmetic on it can overflow.
MAX_DURATION_SECONDS = 10 * 365 * 86400
_UNIT_SECONDS = {
    None: Decimal(1),
    "ms": Decimal("0.001"),
    "s": Decimal(1),
    "m": Decimal(60),
    "h": Decimal(3600),
    "d": Decimal(86400),
}
_SHOWN_LIMIT = 40


class GrammarError(ValueError):
    """A value outside its grammar. The message is safe to print."""


def shown(value: object) -> str:
    """A bounded, escaped rendering of an untrusted value for messages."""
    text = value if isinstance(value, str) else repr(value)
    if len(text) > _SHOWN_LIMIT:
        text = text[:_SHOWN_LIMIT] + "..."
    return repr(text)  # repr escapes control, format and surrogate characters


# --- durations --------------------------------------------------------------------------------


def parse_duration(text: str) -> float:
    """Seconds in ``text``: ``<number>[ms|s|m|h|d]``, a bare number meaning seconds.

    The number is ``digits`` or ``digits.digits`` (no sign, exponent, separator, space or
    non-ASCII digit; leading zeros are fine). Zero exits 2, as does anything over ten years.
    """
    if len(text) > MAX_DURATION_TEXT:
        raise GrammarError(f"invalid duration {shown(text)}: too long")
    match = _DURATION.fullmatch(text)
    if match is None:
        raise GrammarError(
            f"invalid duration {shown(text)}: expected a number of seconds or <number>"
            "<unit> with unit ms, s, m, h or d"
        )
    seconds = Decimal(match.group(1)) * _UNIT_SECONDS[match.group(2)]
    if seconds <= 0:
        raise GrammarError(f"invalid duration {shown(text)}: must be greater than zero")
    if seconds > MAX_DURATION_SECONDS:
        raise GrammarError(f"invalid duration {shown(text)}: longer than ten years")
    return float(seconds)


# --- times ------------------------------------------------------------------------------------


def parse_since(text: str, now: datetime) -> datetime:
    """The instant ``--since`` names, as an aware UTC datetime.

    A duration means that long before ``now`` (tried first, so bare digits such as ``20240101``
    are seconds, as in claude-session). Otherwise ``YYYY-MM-DD`` or ``YYYY-MM-DD[T ]HH:MM[:SS
    [.f]]`` with an optional ``Z`` or ``+HH:MM`` offset; no offset means UTC.
    """
    if len(text) <= MAX_DURATION_TEXT and _DURATION.fullmatch(text):
        seconds = parse_duration(text)
        try:
            return now.astimezone(UTC) - timedelta(seconds=seconds)
        except OverflowError:
            raise GrammarError(f"invalid --since {shown(text)}: out of range") from None
    match = _TIME.fullmatch(text) if len(text) <= 64 else None
    if match is None:
        raise GrammarError(
            f"invalid time {shown(text)}: expected a duration or an ISO 8601 time "
            "such as 2026-11-01 or 2026-11-01T10:30:00Z"
        )
    year, month, day, hour, minute, second, fraction, zone = match.groups()
    try:
        moment = datetime(
            int(year),
            int(month),
            int(day),
            int(hour or 0),
            int(minute or 0),
            int(second or 0),
            int((fraction or "").ljust(6, "0") or 0),
            tzinfo=_zone(zone),
        )
        return moment.astimezone(UTC)
    except (ValueError, OverflowError):
        raise GrammarError(f"invalid time {shown(text)}: not a real date or time") from None


def _zone(text: str | None) -> timezone:
    if text is None or text == "Z":
        return UTC
    sign = -1 if text[0] == "-" else 1
    offset = timedelta(hours=int(text[1:3]), minutes=int(text[4:6]))
    if offset >= timedelta(hours=24) or int(text[4:6]) > 59:
        raise ValueError("offset out of range")
    return timezone(sign * offset)


# --- names ------------------------------------------------------------------------------------

_NAME_FORBIDDEN = re.compile("[\x00-\x1f\x7f-\x9f\ud800-\udfff]")


def validate_name(name: str) -> str:
    """A window name: 1 to 64 code points, no C0, DEL or C1 control character, no ``#[``.

    Lone surrogates (undecodable argv bytes) are rejected too: they cannot be stored or sent.
    """
    if not 1 <= len(name) <= MAX_NAME_CODE_POINTS:
        raise GrammarError(
            f"invalid name {shown(name)}: expected 1 to {MAX_NAME_CODE_POINTS} characters"
        )
    if _NAME_FORBIDDEN.search(name):
        raise GrammarError(f"invalid name {shown(name)}: control characters are not allowed")
    if "#[" in name:
        raise GrammarError(f"invalid name {shown(name)}: must not contain '#['")
    return name


def default_name(
    agent_id: str,
    *,
    name: str | None = None,
    worktree_name: str | None = None,
    kind: str | None = None,
) -> str:
    """``name``, else ``worktree_name``, else ``<kind or "agent"> <6 characters after ag->``.

    ``None`` means unset; an empty string is a (rejected) value. The chosen name is validated,
    so a ``kind`` or ``worktree_name`` outside the name rules is a usage error, never a fallback.
    """
    if AGENT_ID.fullmatch(agent_id) is None:
        raise GrammarError(f"invalid agent id {shown(agent_id)}")
    if name is not None:
        return validate_name(name)
    if worktree_name is not None:
        return validate_name(worktree_name)
    return validate_name(f"{'agent' if kind is None else kind} {agent_id[3:9]}")


# --- actors, decision ids, shas ---------------------------------------------------------------


def validate_actor(actor: str) -> str:
    if _ACTOR.fullmatch(actor) is None:
        raise GrammarError(
            f"invalid actor {shown(actor)}: expected human:<name> or auto:<name> "
            "(letters, digits and . _ @ -, at most 64 characters)"
        )
    return actor


def validate_decision_id(value: str) -> str:
    if _DECISION_ID.fullmatch(value) is None:
        raise GrammarError(
            f"invalid decision id {shown(value)}: expected 1 to 128 of letters, digits "
            "and . _ : @ / -"
        )
    return value


def validate_decision_ids(values: Sequence[str]) -> list[str]:
    if len(values) > MAX_DECISION_IDS:
        raise GrammarError(f"at most {MAX_DECISION_IDS} --decision-id values")
    return [validate_decision_id(value) for value in values]


def validate_merged_head(value: str) -> str:
    """A full 40-character lowercase hex sha (abbreviations, refs and uppercase exit 2)."""
    if _MERGED_HEAD.fullmatch(value) is None:
        raise GrammarError(
            f"invalid --merged-head {shown(value)}: expected a full 40-character lowercase hex sha"
        )
    return value


# --- argparse glue ----------------------------------------------------------------------------


def arg_type(check: Callable[[str], object]) -> Callable[[str], str]:
    """An argparse ``type`` that validates with ``check`` and keeps the raw text."""

    def convert(value: str) -> str:
        try:
            check(value)
        except GrammarError as exc:
            raise argparse.ArgumentTypeError(str(exc)) from None
        return value

    return convert


duration_arg = arg_type(parse_duration)
