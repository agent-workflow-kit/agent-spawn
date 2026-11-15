"""The grammars the fakes enforce are the ones in the pinned sibling schemas."""

from __future__ import annotations

import re
from typing import Any

import pytest

from tests import sibling_schemas
from tests.fakebin.fakelib import grammars as g
from tests.sibling_schemas import PINNED_VERSION


def pattern_at(tool: str, schema_name: str, *path: str) -> str:
    node: Any = sibling_schemas._documents(tool, PINNED_VERSION)[f"{tool}/{schema_name}"]  # pyright: ignore[reportPrivateUsage]
    for key in path:
        node = node["items"] if key == "[]" else node[key]
    return node["pattern"]


def anchored(regex: str) -> str:
    return f"^{regex}$"


CHECKS = [
    ("claude-session", "cli/status.json", ("properties", "labels", "items"), g.LABEL),
    ("claude-session", "cli/status.json", ("properties", "launch_kind"), g.KIND),
    ("claude-session", "cli/start.json", ("properties", "csession_id"), g.CS_ID),
    ("termctl", "defs.json", ("$defs", "label"), g.LABEL),
    ("termctl", "defs.json", ("$defs", "actor"), g.ACTOR),
    ("termctl", "defs.json", ("$defs", "id"), g.TC_ID),
    ("termctl", "defs.json", ("$defs", "decision_ids", "[]"), g.DECISION_ID),
    ("termctl", "defs.json", ("$defs", "session_name"), g.SESSION_NAME),
    ("termctl", "defs.json", ("$defs", "placement_name"), g.PLACEMENT),
    ("termctl", "defs.json", ("$defs", "env_name"), g.ENV_NAME),
    ("termctl", "defs.json", ("$defs", "duration"), g.TC_DURATION),
    ("wtctl", "defs.json", ("$defs", "label"), g.LABEL),
    ("wtctl", "defs.json", ("$defs", "actor"), g.ACTOR),
    ("wtctl", "defs.json", ("$defs", "id"), g.WT_ID),
    ("wtctl", "defs.json", ("$defs", "decision_id"), g.DECISION_ID),
    ("wtctl", "defs.json", ("$defs", "name"), g.WT_NAME),
    ("wtctl", "defs.json", ("$defs", "holder"), g.HOLDER),
    ("wtctl", "defs.json", ("$defs", "duration"), g.WT_DURATION),
    ("wtctl", "defs.json", ("$defs", "sha"), g.SHA),
    ("wtctl", "defs.json", ("$defs", "repo_id"), g.REPO_ID),
]


@pytest.mark.parametrize(
    ("tool", "schema_name", "path", "regex"),
    [pytest.param(*row, id=f"{row[0]}:{'.'.join(row[2])}") for row in CHECKS],
)
def test_fake_grammar_equals_the_pinned_schema_pattern(
    tool: str, schema_name: str, path: tuple[str, ...], regex: str
) -> None:
    assert anchored(regex) == pattern_at(tool, schema_name, *path).replace(
        r"(human\|auto)", "(human|auto)"
    )


@pytest.mark.parametrize(
    ("regex", "good", "bad"),
    [
        (g.LABEL, ["a=b", "k=", "pr=1", "a.b_c-d=x:/#@+=,y"], ["A=b", "a", "a=b c", "=x", "a=\n"]),
        (g.ACTOR, ["human:me", "auto:cron.job@x"], ["robot", "human:", "auto:a b"]),
        (g.CS_ID, ["cs-abcdefgh"], ["cs-abcdef1h", "cs-abcdefg", "CS-abcdefgh"]),
        (
            g.CS_DURATION,
            ["5", "1.5", "90s", "500ms", "2h", "1d", "0.5m"],
            [".5", "5.", "1e3", "+5", "1_000", "5x"],
        ),
        (g.TC_DURATION, ["0", "5", "10s", "500ms", "10m"], ["11m", "-1", "1.5s", "5h", "601s"]),
        (g.WT_DURATION, ["1s", "30m", "7d", "2w"], ["0d", "5", "1.5h", "d"]),
    ],
)
def test_grammars_accept_and_reject_the_documented_examples(
    regex: str, good: list[str], bad: list[str]
) -> None:
    for value in good:
        assert re.fullmatch(regex, value), value
    for value in bad:
        assert not re.fullmatch(regex, value), value
