"""The shared grammars are enforced at the CLI's validation points (exit 2, envelope)."""

from __future__ import annotations

import json

import pytest

from agent_spawn import cli
from tests import schema_registry

ERROR_SCHEMA = "agent-spawn/cli/error.json"
ID = "ag-abcdefgh"
LAUNCH = ["launch", "--skill", "s", "--mode", "headless"]
HUGE = "x" * 5000

BAD = {
    "launch label without equals": [*LAUNCH, "--label", "nope"],
    "launch label uppercase key": [*LAUNCH, "--label", "K=v"],
    "launch label bad value": [*LAUNCH, "--label", "a=b c"],
    "launch label reserved agent": [*LAUNCH, "--label", "agent=x"],
    "launch label reserved kind": [*LAUNCH, "--label", "kind=x"],
    "launch label trailing newline": [*LAUNCH, "--label", "a=b\n"],
    "launch 65 labels": [*LAUNCH, *[x for i in range(65) for x in ("--label", f"k={i}")]],
    "resume label reserved": ["resume", ID, "--label", "kind=x"],
    "resume select bad": ["resume", "--select", "nope"],
    "list label bad": ["list", "--label", "A=b"],
    "label add reserved": ["label", "add", ID, "agent=x"],
    "label remove reserved": ["label", "remove", ID, "kind=x"],
    "label add bad pair": ["label", "add", ID, "pr=1", "bad pair"],
    "label add bad agent id": ["label", "add", "ag-ab", "pr=1"],
    "label add session reserved": ["label", "add", "--session", "cs-abcdefgh", "agent=x"],
    "label add session bad pair": ["label", "add", "--session", "cs-abcdefgh", "x"],
    "label add 65 pairs": ["label", "add", ID, *[f"k={i}" for i in range(65)]],
    "wait-timeout zero": [*LAUNCH, "--wait-timeout", "0"],
    "wait-timeout negative": [*LAUNCH, "--wait-timeout", "-5"],
    "wait-timeout unit": [*LAUNCH, "--wait-timeout", "5x"],
    "wait timeout zero": ["wait", ID, "--timeout", "0s"],
    "wait timeout float junk": ["wait", ID, "--timeout", "1e3"],
    "stop grace zero": ["stop", ID, "--grace", "0"],
    "prune older-than bad": ["prune", "--older-than", "forever"],
    "since bad": ["list", "--since", "yesterday"],
    "export since bad": ["export", "--since", "0"],
    "merged-head short": ["teardown", ID, "--merged-head", "abc123"],
    "merged-head uppercase": ["teardown", ID, "--merged-head", "A" * 40],
    "merged-head newline": ["teardown", ID, "--merged-head", "a" * 40 + "\n"],
    "name empty": [*LAUNCH, "--name", ""],
    "name too long": [*LAUNCH, "--name", "n" * 65],
    "name control": [*LAUNCH, "--name", "a\x85b"],
    "name tmux marker": [*LAUNCH, "--name", "a#[b"],
    "resume name bad": ["resume", ID, "--name", "a\nb"],
    "show short id": ["show", "ag-abc"],
    "show bad id": ["show", "ag-ABCDEFGH"],
    "show long id": ["show", "ag-abcdefghi"],
    "wait empty id": ["wait", ""],
    "resume short id": ["resume", "ab"],
    "forked-from bad": ["list", "--forked-from", "ag-1"],
    "browse focus bad": ["browse", "--focus", "zz"],
    "actor trailing newline": ["list", "--actor", "human:a\n"],
    "decision id lookalike": ["list", "--decision-id", "а"],
}


@pytest.mark.parametrize("argv", BAD.values(), ids=BAD.keys())
def test_invalid_values_exit_2_with_the_envelope(
    capsys: pytest.CaptureFixture[str], argv: list[str]
) -> None:
    code = cli.main(["--json", *argv])
    out = capsys.readouterr().out
    assert code == 2
    envelope = json.loads(out)
    schema_registry.validate(envelope, ERROR_SCHEMA)
    assert envelope["error"]["code"] == "usage"


GOOD = {
    "labels": [*LAUNCH, "--label", "pr=1", "--label", "a=b=c", "--label", "e="],
    "64 distinct labels": [*LAUNCH, *[x for i in range(64) for x in ("--label", f"k={i}")]],
    "duplicate labels do not count": [*LAUNCH, *["--label", "k=1"] * 100],
    "list filter on reserved key": ["list", "--label", "kind=review"],
    "select": ["resume", "--select", "pr=1", "--select", "pr=2"],
    "durations": [*LAUNCH, "--wait-timeout", "1.5h"],
    "bare duration": ["wait", ID, "--timeout", "90"],
    "grace": ["stop", ID, "--grace", "250ms"],
    "older-than": ["prune", "--older-than", "7d"],
    "since duration": ["list", "--since", "2h"],
    "since date": ["export", "--since", "2026-11-01T10:00:00Z"],
    "merged-head": ["teardown", ID, "--merged-head", "0123456789abcdef" * 2 + "01234567"],
    "name": [*LAUNCH, "--name", "café \U0001f600"],
    "id prefix": ["show", "ag-abcd"],
    "id prefix without ag": ["show", "abcd"],
    "label add": ["label", "add", ID, "pr=1", "pr=1"],
    "label add by prefix": ["label", "add", "abcd", "pr=1"],
    "label add session": ["label", "add", "--session", "cs-abcdefgh", "pr=1"],
}


@pytest.mark.parametrize("argv", GOOD.values(), ids=GOOD.keys())
def test_valid_values_pass_the_usage_checks(argv: list[str]) -> None:
    cli.parse_invocation(argv)


def test_label_add_splits_agent_id_and_pairs() -> None:
    inv = cli.parse_invocation(["label", "add", "abcd", "pr=1", "pr=2"])
    assert inv.args.agent_id == "abcd"
    assert inv.args.pairs == ["pr=1", "pr=2"]


def test_error_messages_do_not_echo_hostile_values_raw(
    capsys: pytest.CaptureFixture[str],
) -> None:
    hostile = "\x1b[31m‮" + HUGE
    for argv in (
        [*LAUNCH, "--label", "a=" + hostile],
        [*LAUNCH, "--name", hostile + "#["],
        ["wait", ID, "--timeout", hostile],
        ["show", hostile],
        ["teardown", ID, "--merged-head", hostile],
        ["list", "--since", hostile],
        ["label", "add", ID, hostile],
    ):
        code = cli.main(["--json", *argv])
        captured = capsys.readouterr()
        assert code == 2
        assert "\x1b" not in captured.out + captured.err
        assert "‮" not in captured.out + captured.err
        assert len(captured.out) < 2000
