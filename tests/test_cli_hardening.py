"""Edge cases of the CLI found in review: usage errors and the result file, decision ids of
command groups, closed pipes, browse's own result file, intermixed label items, flag pairs and
an unusable cwd."""

import io
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from agent_spawn import cli
from agent_spawn.commands.base import Invocation
from agent_spawn.commands.table import HANDLERS
from agent_spawn.output import CliError, Outcome, PayloadKind
from tests import schema_registry
from tests.cli_samples import INTERNAL_GLOBALS

ERROR_SCHEMA = "agent-spawn/cli/error.json"


@pytest.fixture
def reached(monkeypatch: pytest.MonkeyPatch) -> list[Invocation]:
    seen: list[Invocation] = []

    def spy(inv: Invocation) -> Outcome:
        seen.append(inv)
        return Outcome(0, {"ok": True}, PayloadKind.COMMAND)

    for key in list(HANDLERS):
        monkeypatch.setitem(HANDLERS, key, spy)
    return seen


def _no_tty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "stdin_is_tty", lambda: False)


# --- usage errors write the result file and carry the parsed decision ids -------------------------


POST_PARSE_USAGE = {
    "human-only gate": ["resume", "ag-abcdefgh", "--allow-tainted"],
    "resume selector": ["resume"],
    "launch gate": ["launch", "--skill", "s", "--override-escalation"],
    "wait internal flag": ["--state-dir", "/s", "wait", "ag-abcdefgh"],
}


@pytest.mark.parametrize("argv", POST_PARSE_USAGE.values(), ids=POST_PARSE_USAGE.keys())
def test_a_usage_error_after_parsing_writes_the_result_file(
    argv: list[str],
    reached: list[Invocation],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    _no_tty(monkeypatch)
    path = tmp_path / "r.json"
    assert cli.main([*argv, "--result-file", str(path), "--json"]) == 2
    printed = capsys.readouterr().out
    assert reached == []
    assert path.read_text() == printed
    envelope = json.loads(printed)
    schema_registry.validate(envelope, ERROR_SCHEMA)
    assert envelope["error"]["code"] == "usage"


def test_a_usage_error_after_parsing_carries_the_parsed_decision_ids(
    reached: list[Invocation], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _no_tty(monkeypatch)
    argv = [
        "--decision-id",
        "d1",
        "teardown",
        "ag-abcd",
        "--force",
        "--json",
        "--decision-id",
        "d2",
    ]
    assert cli.main(argv) == 2
    envelope = json.loads(capsys.readouterr().out)
    assert envelope["error"]["decision_ids"] == ["d1", "d2"]


def test_too_many_decision_ids_carry_the_first_32(capsys: pytest.CaptureFixture[str]) -> None:
    ids = [f"--decision-id=d{n}" for n in range(40)]
    assert cli.main(["--json", "list", *ids]) == 2
    envelope = json.loads(capsys.readouterr().out)
    schema_registry.validate(envelope, ERROR_SCHEMA)
    assert envelope["error"]["decision_ids"] == [f"d{n}" for n in range(32)]


@pytest.mark.parametrize(
    "argv",
    [
        ["launch", "--bogus", "--result-file", "{path}"],
        ["launch", "--result-file={path}", "--bogus"],
        ["--json", "resume", "ag-abcdefgh", "--nope", "--result-file", "{path}"],
        ["wait", "--result-file", "{path}"],
        ["--config", "/c", "wait", "ag-abcdefgh", "extra", "--result-file", "{path}"],
    ],
)
def test_an_argparse_usage_error_writes_the_pre_scanned_result_file(
    argv: list[str], capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    path = tmp_path / "r.json"
    assert cli.main([a.replace("{path}", str(path)) for a in argv]) == 2
    envelope = json.loads(path.read_text())
    schema_registry.validate(envelope, ERROR_SCHEMA)
    assert envelope["error"]["code"] == "usage"
    assert envelope["error"]["decision_ids"] == []


@pytest.mark.parametrize(
    "argv",
    [
        ["list", "--nope", "--result-file", "{path}"],
        ["teardown", "ag-abcdefgh", "--result-file", "{path}"],
        ["browse", "--nope", "--result-file", "{path}"],
        ["launch", "--bogus", "--", "--result-file", "{path}"],
        ["--result-file", "{path}", "launch", "--bogus"],
    ],
)
def test_an_argparse_usage_error_writes_no_file_outside_launch_resume_and_wait(
    argv: list[str], tmp_path: Path
) -> None:
    path = tmp_path / "r.json"
    assert cli.main([a.replace("{path}", str(path)) for a in argv]) == 2
    assert not path.exists()


def test_a_dash_prefixed_token_after_result_file_is_not_pre_scanned_as_its_path(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    assert cli.main(["wait", "ag-abcdefgh", "--result-file", "--json"]) == 2
    assert list(tmp_path.iterdir()) == []
    envelope = json.loads(capsys.readouterr().out)
    schema_registry.validate(envelope, ERROR_SCHEMA)
    assert envelope["error"]["code"] == "usage"


def test_an_unusable_pre_scanned_result_file_is_a_warning(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    for target in ("dir/", str(tmp_path / "missing" / "r.json")):
        assert cli.main(["launch", "--bogus", "--result-file", target]) == 2
        assert "warning" in capsys.readouterr().err


# --- decision ids of command groups ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (
            ["label", "--decision-id", "a", "add", "--decision-id", "b", "ag-abcd", "k=v"],
            ["a", "b"],
        ),
        (["label", "--decision-id", "a", "remove", "ag-abcd", "k=v"], ["a"]),
        (
            [
                "--decision-id",
                "z",
                "label",
                "--decision-id",
                "a",
                "add",
                "ag-abcd",
                "--decision-id=b",
                "k=v",
            ],
            ["z", "a", "b"],
        ),
        (["cron", "--decision-id", "a", "install", "--decision-id", "b"], ["a", "b"]),
        (["cron", "--decision-id", "a", "uninstall"], ["a"]),
    ],
)
def test_decision_ids_of_a_command_group_are_kept(argv: list[str], expected: list[str]) -> None:
    assert cli.parse_invocation(argv).decision_ids == expected


# --- closed stdout --------------------------------------------------------------------------------


def _scratch_env(tmp_path: Path) -> dict[str, str]:
    env = {"PATH": os.environ.get("PATH", ""), "HOME": str(tmp_path)}
    for var in ("CONFIG", "STATE", "DATA", "CACHE", "RUNTIME"):
        env[f"XDG_{var}_HOME" if var != "RUNTIME" else "XDG_RUNTIME_DIR"] = str(tmp_path / var)
    return env


@pytest.mark.parametrize(
    ("argv", "expected"),
    [(["list", "--json"], 1), (["--json", "launch", "--bogus"], 2), (["--help"], 0), (["list"], 1)],
)
def test_a_closed_stdout_keeps_the_exit_code(
    argv: list[str], expected: int, tmp_path: Path
) -> None:
    read_end, write_end = os.pipe()
    os.close(read_end)
    try:
        done = subprocess.run(
            [sys.executable, "-m", "agent_spawn", *argv],
            stdin=subprocess.DEVNULL,
            stdout=write_end,
            stderr=subprocess.PIPE,
            env=_scratch_env(tmp_path),
            check=False,
            timeout=60,
        )
    finally:
        os.close(write_end)
    assert done.returncode == expected
    assert b"Traceback" not in done.stderr


# --- browse's result file belongs to its handler --------------------------------------------------


def test_browse_errors_leave_its_result_file_absent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "r.json"
    assert cli.main(["browse", "--result-file", str(path)]) == 1  # the stub handler
    assert not path.exists()

    def failing(inv: Invocation) -> Outcome:
        assert inv.result_file is None
        raise CliError.of("not_found", "no agent")

    monkeypatch.setitem(HANDLERS, "browse", failing)
    assert cli.main(["browse", "--focus", "ag-abcd", "--result-file", str(path)]) == 10
    assert not path.exists()


# --- human output with an ASCII-only stdout -------------------------------------------------------


def test_human_output_survives_an_ascii_stdout(monkeypatch: pytest.MonkeyPatch) -> None:
    raw = io.BytesIO()
    monkeypatch.setattr("sys.stdout", io.TextIOWrapper(raw, encoding="ascii"))
    monkeypatch.setitem(
        HANDLERS, "list", lambda inv: Outcome(0, {"a": 1}, PayloadKind.COMMAND, human="café")
    )
    assert cli.main(["list"]) == 0
    sys.stdout.flush()
    assert raw.getvalue() == b"caf\\xe9\n"


# --- label items intermixed with flags ------------------------------------------------------------


@pytest.mark.parametrize(
    ("argv", "agent_id", "pairs"),
    [
        (["label", "add", "ag-abcd", "--json", "k=v"], "ag-abcd", ["k=v"]),
        (["label", "remove", "ag-abcd", "k=v", "--dry-run", "j=w"], "ag-abcd", ["k=v", "j=w"]),
        (
            ["label", "add", "ag-abcd", "--quiet", "a=1", "--json", "b=2", "c=3"],
            "ag-abcd",
            ["a=1", "b=2", "c=3"],
        ),
        (["label", "add", "--session", "cs-1", "--json", "pr=1"], None, ["pr=1"]),
    ],
)
def test_label_items_may_be_intermixed_with_flags(
    argv: list[str], agent_id: str | None, pairs: list[str]
) -> None:
    args: Any = cli.parse_invocation(argv).args
    assert (args.agent_id, args.pairs) == (agent_id, pairs)


@pytest.mark.parametrize(
    "argv",
    [
        ["label", "add", "ag-abcd", "k=v", "--nope"],
        ["label", "add", "ag-abcd", "--json", "-k=v"],
        ["show", "ag-abcd", "--json", "extra"],
    ],
)
def test_unknown_flags_and_extra_positionals_are_still_usage_errors(argv: list[str]) -> None:
    with pytest.raises(cli.UsageError, match="unrecognized arguments"):
        cli.parse_invocation(argv)


# --- flag pairs and combinations ------------------------------------------------------------------


@pytest.mark.parametrize(
    "extra", [["--handoff", "/run/s.sock"], ["--handoff-timeout", "5"]], ids=["handoff", "timeout"]
)
def test_supervise_handoff_flags_come_as_a_pair(extra: list[str]) -> None:
    with pytest.raises(cli.UsageError, match="--handoff"):
        cli.parse_invocation([*INTERNAL_GLOBALS, "supervise", "ag-abcdefgh", *extra])


def test_supervise_with_both_handoff_flags_parses() -> None:
    argv = [*INTERNAL_GLOBALS, "supervise", "ag-abcd", "--handoff", "/s", "--handoff-timeout", "5"]
    assert cli.parse_invocation(argv).command == "supervise"


def test_self_labeling_has_no_dry_run() -> None:
    with pytest.raises(cli.UsageError, match="--dry-run"):
        cli.parse_invocation(["label", "add", "--session", "cs-1", "pr=1", "--dry-run"])


# --- an unusable cwd ------------------------------------------------------------------------------


def test_a_relative_result_file_with_a_deleted_cwd_is_a_warning(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    gone = tmp_path / "gone"
    gone.mkdir()
    monkeypatch.chdir(gone)
    gone.rmdir()
    monkeypatch.setitem(HANDLERS, "wait", lambda inv: Outcome(0, {"a": 1}, PayloadKind.RESULT))
    assert cli.main(["wait", "ag-abcdefgh", "--result-file", "r.json"]) == 0
    assert "warning" in capsys.readouterr().err
