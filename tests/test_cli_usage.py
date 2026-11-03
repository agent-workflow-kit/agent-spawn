"""Usage errors exit 2 with the envelope, also on argparse's own error path."""

import json
from typing import Any

import pytest

from agent_spawn import __version__, cli
from tests import schema_registry
from tests.cli_samples import MIN_ARGV

ERROR_SCHEMA = "agent-spawn/cli/error.json"


def _run(capsys: pytest.CaptureFixture[str], argv: list[str]) -> tuple[int, str, str]:
    code = cli.main(argv)
    captured = capsys.readouterr()
    return code, captured.out, captured.err


BAD_ARGV = {
    "unknown flag": ["list", "--nope"],
    "unknown command": ["frobnicate"],
    "missing positional": ["show"],
    "bad choice": ["list", "--mode", "sideways"],
    "bad choice for subcommand": ["launch", "--placement", "somewhere"],
    "missing flag value": ["wait", "ag-abcdefgh", "--timeout"],
    "mutually exclusive": ["launch", "--promote", "--no-promote"],
    "auto-exit exclusive": ["launch", "--auto-exit", "--no-auto-exit"],
    "label without subcommand": ["label"],
    "cron without subcommand": ["cron"],
    "label remove has no --session": ["label", "remove", "--session", "cs-abcdefgh", "pr=1"],
    "extra positional": ["show", "ag-abcdefgh", "ag-ijklmnop"],
    "interval not a number": ["cron", "install", "--interval", "soon"],
    "interval too small": ["cron", "install", "--interval", "0"],
    "interval too large": ["cron", "install", "--interval", "60"],
    "bad actor grammar": ["list", "--actor", "bob"],
    "bad actor kind": ["list", "--actor", "robot:bob"],
    "overlong actor": ["list", "--actor", "human:" + "a" * 65],
    "bad decision id": ["list", "--decision-id", "has space"],
    "empty decision id": ["list", "--decision-id", ""],
    "too many decision ids": ["list", *["--decision-id=d"] * 33],
    "resume without id or selector": ["resume"],
    "resume with both id and selector": ["resume", "ag-abcdefgh", "--select", "k=v"],
    "label add without pairs": ["label", "add", "ag-abcdefgh"],
    "label add --session without pairs": ["label", "add", "--session", "cs-abcdefgh"],
    "label add without agent": ["label", "add", "pr=1"],
    "state-dir on a user command": ["--state-dir", "/state", "list"],
    "relative state-dir": ["--state-dir", "state", "--config", "/c", "_session-event"],
    "relative internal config": ["--state-dir", "/s", "--config", "c", "_session-event"],
    "supervise without state-dir": ["--config", "/c", "supervise", "ag-abcdefgh"],
    "session-event without config": ["--state-dir", "/s", "_session-event"],
    "empty argv": [],
}


@pytest.mark.parametrize("argv", BAD_ARGV.values(), ids=BAD_ARGV.keys())
def test_usage_errors_exit_2_with_the_envelope_on_stdout_when_json(
    capsys: pytest.CaptureFixture[str], argv: list[str]
) -> None:
    code, out, err = _run(capsys, ["--json", *argv])
    assert code == 2
    envelope = json.loads(out)
    schema_registry.validate(envelope, ERROR_SCHEMA)
    assert envelope["error"]["code"] == "usage"
    assert envelope["error"]["tool"] == "agent-spawn"
    assert envelope["error"]["message"]
    assert "error" in err.lower()
    assert out.count("\n") == 1  # exactly one JSON line


@pytest.mark.parametrize("argv", BAD_ARGV.values(), ids=BAD_ARGV.keys())
def test_usage_errors_without_json_print_nothing_on_stdout(
    capsys: pytest.CaptureFixture[str], argv: list[str]
) -> None:
    code, out, err = _run(capsys, argv)
    assert code == 2
    assert out == ""
    assert "usage:" in err


def test_json_flag_after_the_failing_command_still_yields_the_envelope(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code, out, _ = _run(capsys, ["list", "--nope", "--json"])
    assert code == 2
    assert json.loads(out)["error"]["code"] == "usage"


def test_usage_error_with_json_after_double_dash_is_not_json_mode(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code, out, _ = _run(capsys, ["list", "--nope", "--", "--json"])
    assert code == 2
    assert out == ""


def test_usage_errors_echo_hostile_argv_safely(capsys: pytest.CaptureFixture[str]) -> None:
    hostile = "--x\udcff \x1b[31m"
    code, out, _ = _run(capsys, ["--json", "list", hostile])
    assert code == 2
    assert out.isascii()
    schema_registry.validate(json.loads(out), ERROR_SCHEMA)


def test_help_exits_0_and_prints_usage_on_stdout(capsys: pytest.CaptureFixture[str]) -> None:
    code, out, _ = _run(capsys, ["--help"])
    assert code == 0
    assert "usage: agent-spawn" in out
    assert "launch" in out
    assert "_session-event" not in out  # internal commands stay out of the help


def test_subcommand_help_exits_0(capsys: pytest.CaptureFixture[str]) -> None:
    code, out, _ = _run(capsys, ["label", "add", "--help"])
    assert code == 0
    assert "usage:" in out


def test_version_prints_the_version_and_exits_0(capsys: pytest.CaptureFixture[str]) -> None:
    code, out, _ = _run(capsys, ["--version"])
    assert code == 0
    assert out.strip() == f"agent-spawn {__version__}"


def test_every_minimal_argv_passes_the_usage_checks() -> None:
    for command, argv in MIN_ARGV.items():
        inv = cli.parse_invocation(argv)
        assert inv.command == command


def test_valid_edge_cases_pass() -> None:
    cases: list[list[str]] = [
        ["resume", "--select", "k=v", "--select", "a=b"],
        ["label", "add", "--session", "cs-abcdefgh", "pr=1", "pr=2"],
        ["label", "add", "ag-abcdefgh", "pr=1"],
        ["cron", "install", "--interval", "1"],
        ["cron", "install", "--interval", "59"],
        ["list", "--actor", "human:a.b_c@d-e"],
        ["list", "--decision-id", "a:b/c@d.e_f-g"],
        ["list", *["--decision-id=d"] * 32],
        ["launch", "--prompt-file", "-", "--skill", "x"],
    ]
    for argv in cases:
        cli.parse_invocation(argv)


def test_parsed_values_are_kept_raw_for_later_grammar_checks() -> None:
    inv: Any = cli.parse_invocation(
        ["launch", "--wait-timeout", "5m", "--label", "a=b", "--skill", "s"]
    )
    assert inv.args.wait_timeout == "5m"
    assert inv.args.label == ["a=b"]
