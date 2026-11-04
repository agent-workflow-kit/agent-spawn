"""Labels: the ``key=value`` grammar, the multiset an agent stores and the reserved keys.

A label is split at the first ``=``. The agent's labels are a multiset in the design guide's sense:
a key may carry several values, an identical pair is stored once, so adding a pair the agent
has or removing one it lacks is a no-op. Stored order is insertion order (termctl mirrors only the
first 30). At most 64 pairs per agent; ``agent`` and ``kind`` belong to agent-spawn.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import NamedTuple

from .validate import GrammarError, arg_type, shown

__all__ = [
    "MAX_LABELS",
    "RESERVED_KEYS",
    "Label",
    "LabelChange",
    "LabelSet",
    "job_label",
    "label_arg",
    "parse_label",
    "parse_run_labels",
    "parse_selector",
    "parse_user_label",
    "parse_user_labels",
    "selector_arg",
]

MAX_LABELS = 64
RESERVED_KEYS = frozenset({"agent", "kind"})
JOB_KEY = "job"
_KEY = re.compile(r"[a-z0-9][a-z0-9_.-]{0,63}")
_VALUE = re.compile(r"[A-Za-z0-9._:/#@+=,-]{0,256}")
_MAX_TEXT = 64 + 1 + 256


class Label(NamedTuple):
    key: str
    value: str

    def __str__(self) -> str:
        return f"{self.key}={self.value}"


def parse_label(text: str) -> Label:
    """Check ``key=value`` against the grammar. Reserved keys pass: agent-spawn writes them."""
    if len(text) > _MAX_TEXT:
        raise GrammarError(f"invalid label {shown(text)}: too long")
    key, separator, value = text.partition("=")
    if not separator:
        raise GrammarError(f"invalid label {shown(text)}: expected key=value")
    if _KEY.fullmatch(key) is None:
        raise GrammarError(
            f"invalid label key in {shown(text)}: expected 1 to 64 of a-z, 0-9 and . _ -, "
            "starting with a letter or digit"
        )
    if _VALUE.fullmatch(value) is None:
        raise GrammarError(
            f"invalid label value in {shown(text)}: expected at most 256 of "
            "A-Z a-z 0-9 . _ : / # @ + = , -"
        )
    return Label(key, value)


def parse_user_label(text: str) -> Label:
    """A label a caller may add or remove: ``agent`` and ``kind`` are reserved (exit 2)."""
    label = parse_label(text)
    if label.key in RESERVED_KEYS:
        raise GrammarError(f"label key {label.key!r} is reserved for agent-spawn")
    return label


def parse_selector(text: str) -> Label:
    """A ``resume --select`` pair: the grammar only; a reserved key just matches nothing, because
    agent-spawn never stores it on the agent. (``list --label`` takes ``parse_user_label``.)"""
    return parse_label(text)


def parse_user_labels(texts: Iterable[str]) -> list[Label]:
    """Every user label of one command, deduplicated in first-seen order and capped at 64."""
    return list(LabelSet().add([parse_user_label(text) for text in texts]).labels.pairs)


def parse_run_labels(texts: Iterable[str]) -> list[Label]:
    """The ``--label`` pairs of launch or resume, deduplicated in first-seen order.

    Only the non-``job=`` pairs count toward the cap here: a ``job=`` pair beyond it is skipped
    with a warning when the labels are applied (``LabelSet.add_run``), never a usage error.
    """
    parsed = list(dict.fromkeys(parse_user_label(text) for text in texts))
    LabelSet().add([label for label in parsed if label.key != JOB_KEY])
    return parsed


def job_label(value: str) -> Label | None:
    """``job=<value>`` for the value of ``job_id_env``, or ``None`` if it does not fit."""
    try:
        return parse_label(f"{JOB_KEY}={value}")
    except GrammarError:
        return None


label_arg = arg_type(parse_user_label)
selector_arg = arg_type(parse_selector)


@dataclass(frozen=True)
class LabelChange:
    """The result of an edit: the new set, the pairs really changed and whether ``job=`` was
    skipped for the cap."""

    labels: LabelSet
    applied: tuple[Label, ...]
    skipped: bool = False

    @property
    def changed(self) -> bool:
        return bool(self.applied)


@dataclass(frozen=True)
class LabelSet:
    pairs: tuple[Label, ...] = ()

    @classmethod
    def from_texts(cls, texts: Iterable[str]) -> LabelSet:
        """Rebuild a stored set; duplicates collapse, bad grammar or too many raise."""
        return cls().add([parse_label(text) for text in texts]).labels

    def __len__(self) -> int:
        return len(self.pairs)

    def __contains__(self, item: object) -> bool:
        return item in self.pairs

    def texts(self) -> list[str]:
        return [str(pair) for pair in self.pairs]

    def add(self, new: Sequence[Label]) -> LabelChange:
        """Add the pairs not yet present (once each). Over 64 raises and changes nothing."""
        fresh = tuple(dict.fromkeys(pair for pair in new if pair not in self.pairs))
        if len(self.pairs) + len(fresh) > MAX_LABELS:
            raise GrammarError(f"at most {MAX_LABELS} labels per agent")
        return LabelChange(LabelSet(self.pairs + fresh), fresh)

    def add_job(self, label: Label) -> LabelChange:
        """Add a ``job=`` pair; beyond the cap it is skipped (the caller warns), never an error."""
        if label.key != JOB_KEY:
            raise ValueError("add_job only takes job= pairs")
        if label in self.pairs:
            return LabelChange(self, ())
        if len(self.pairs) >= MAX_LABELS:
            return LabelChange(self, (), skipped=True)
        return LabelChange(LabelSet(self.pairs + (label,)), (label,))

    def add_run(self, new: Sequence[Label]) -> LabelChange:
        """Add the labels of a launch or resume: the non-``job=`` pairs first (over 64 raises and
        changes nothing), then each ``job=`` pair as ``add_job`` does (skipped beyond the cap)."""
        change = self.add([pair for pair in new if pair.key != JOB_KEY])
        applied, skipped = list(change.applied), False
        for pair in dict.fromkeys(pair for pair in new if pair.key == JOB_KEY):
            change = change.labels.add_job(pair)
            applied.extend(change.applied)
            skipped = skipped or change.skipped
        return LabelChange(change.labels, tuple(applied), skipped)

    def remove(self, gone: Sequence[Label]) -> LabelChange:
        """Remove the pairs present; absent ones are ignored."""
        doomed = tuple(dict.fromkeys(pair for pair in gone if pair in self.pairs))
        kept = tuple(pair for pair in self.pairs if pair not in doomed)
        return LabelChange(LabelSet(kept), doomed)
