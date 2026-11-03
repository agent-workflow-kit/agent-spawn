"""Argparse coverage of every command and flag of docs/interface.md."""

import pytest

from agent_spawn import cli
from agent_spawn.commands.table import HANDLERS
from tests.cli_samples import FLAGS, GLOBAL_OPTIONS, MIN_ARGV


def _option_strings(parser: cli.SpawnParser) -> set[str]:
    return set(parser._option_string_actions) - GLOBAL_OPTIONS  # pyright: ignore[reportPrivateUsage]


def _argv_with(command: str, flag: str, value: str | None) -> list[str]:
    base = list(MIN_ARGV[command])
    if flag == "--select":
        base = ["resume"]  # a selector replaces the positional agent id
    return [*base, flag, *([] if value is None else [value])]


def test_the_parser_knows_exactly_the_documented_commands() -> None:
    parser = cli.build_parser()
    assert set(parser.command_parsers) == set(FLAGS)
    assert set(MIN_ARGV) == set(FLAGS)
    assert set(HANDLERS) == set(FLAGS)


@pytest.mark.parametrize("command", sorted(FLAGS))
def test_each_command_takes_exactly_its_documented_flags(command: str) -> None:
    parser = cli.build_parser().command_parsers[command]
    assert _option_strings(parser) == set(FLAGS[command])


def _every_flag() -> list[tuple[str, str, str | None]]:
    return [(c, f, v) for c, flags in sorted(FLAGS.items()) for f, v in sorted(flags.items())]


@pytest.mark.parametrize(("command", "flag", "value"), _every_flag())
def test_every_documented_flag_parses(command: str, flag: str, value: str | None) -> None:
    args = cli.build_parser().parse_args(_argv_with(command, flag, value))
    assert args.command == command.split()[0]


@pytest.mark.parametrize("command", sorted(MIN_ARGV))
def test_the_minimal_argv_of_every_command_parses(command: str) -> None:
    args = cli.build_parser().parse_args(MIN_ARGV[command])
    assert args.command == command.split()[0]


GLOBAL_FLAG_ARGV = [
    "--json",
    "--config",
    "/c.yaml",
    "--verbose",
    "--quiet",
    "--yes",
    "--actor",
    "auto:policy",
    "--decision-id",
    "d1",
]


@pytest.mark.parametrize("command", sorted(m for m in MIN_ARGV if not m.startswith(("super", "_"))))
def test_global_flags_are_accepted_before_and_after_the_command(command: str) -> None:
    argv = MIN_ARGV[command]
    before = cli.build_parser().parse_args([*GLOBAL_FLAG_ARGV, *argv])
    after = cli.build_parser().parse_args([*argv, *GLOBAL_FLAG_ARGV])
    for args in (before, after):
        assert args.json is True
        assert args.config == "/c.yaml"
        assert args.verbose is True
        assert args.quiet is True
        assert args.yes is True
        assert args.actor == "auto:policy"
    assert before.decision_id == ["d1"]


def test_decision_ids_given_before_and_after_the_command_are_merged() -> None:
    inv = cli.parse_invocation(
        ["--decision-id", "a", "show", "ag-abcdefgh", "--decision-id", "b", "--decision-id", "c"]
    )
    assert inv.decision_ids == ["a", "b", "c"]


def test_yes_is_accepted_and_changes_nothing() -> None:
    inv = cli.parse_invocation(["show", "ag-abcdefgh", "--yes"])
    plain = cli.parse_invocation(["show", "ag-abcdefgh"])
    assert inv.command == plain.command == "show"
    assert inv.actor == plain.actor


def test_dry_run_exists_exactly_where_the_interface_lists_it() -> None:
    with_dry_run = {c for c, flags in FLAGS.items() if "--dry-run" in flags}
    assert with_dry_run == {
        "launch",
        "resume",
        "label add",
        "label remove",
        "stop",
        "teardown",
        "prune",
        "reconcile",
        "cron install",
        "cron uninstall",
    }


def test_abbreviated_flags_are_refused() -> None:
    parser = cli.build_parser()
    with pytest.raises(cli.UsageError):
        parser.parse_args(["teardown", "ag-abcdefgh", "--forc"])
    with pytest.raises(cli.UsageError):
        parser.parse_args(["launch", "--over"])
