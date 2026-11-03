"""Dispatch, stubs, the output table and --result-file as seen through ``cli.main``."""

import json
import os
import stat
from pathlib import Path
from typing import Any

import pytest

from agent_spawn import cli, output
from agent_spawn.commands.base import Invocation
from agent_spawn.commands.table import HANDLERS
from agent_spawn.output import CliError, Outcome, PayloadKind
from tests import schema_registry
from tests.cli_samples import MIN_ARGV
from tests.fixtures import TS, sample_agent, sample_outcome

GOLDEN = Path(__file__).parent / "golden"

# Which issue owns each command's handler (the seams siblings fill in).
OWNERS = {
    "launch": 11,
    "resume": 13,
    "wait": 17,
    "list": 15,
    "show": 15,
    "attach": 28,
    "ack": 15,
    "label add": 15,
    "label remove": 15,
    "stop": 16,
    "teardown": 18,
    "prune": 24,
    "reconcile": 33,
    "cron install": 34,
    "cron uninstall": 34,
    "export": 7,
    "browse": 38,
    "supervise": 12,
    "_session-event": 14,
}


@pytest.mark.parametrize("command", sorted(MIN_ARGV))
def test_every_command_is_a_stub_that_fails_through_the_envelope(
    command: str, capsys: pytest.CaptureFixture[str]
) -> None:
    code = cli.main(["--json", *MIN_ARGV[command]])
    captured = capsys.readouterr()
    assert code == 1
    envelope = json.loads(captured.out)
    schema_registry.validate(envelope, "agent-spawn/cli/error.json")
    error = envelope["error"]
    assert error["code"] == "internal"
    assert error["tool"] == "agent-spawn"
    assert f"#{OWNERS[command]}" in error["message"]
    assert "not implemented" in captured.err


def test_the_command_table_has_one_distinct_handler_per_command() -> None:
    assert set(HANDLERS) == set(OWNERS)
    assert len({id(h) for h in HANDLERS.values()}) == len(HANDLERS)


# --- output by exit code ------------------------------------------------------------------------


def _handler(outcome: Outcome | Exception) -> Any:
    def handler(inv: Invocation) -> Outcome:
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    return handler


def test_json_prints_the_command_payload_on_stdout(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    payload = {"agents": []}
    monkeypatch.setitem(HANDLERS, "list", _handler(Outcome(0, payload, PayloadKind.COMMAND)))
    assert cli.main(["--json", "list"]) == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out) == payload
    assert captured.err == ""


def test_without_json_a_result_prints_its_human_text(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    outcome = Outcome(0, {"agents": []}, PayloadKind.COMMAND, human="no agents")
    monkeypatch.setitem(HANDLERS, "list", _handler(outcome))
    assert cli.main(["list"]) == 0
    assert capsys.readouterr().out == "no agents\n"


def test_a_cli_error_prints_the_message_on_stderr_and_the_envelope_with_json(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    error = CliError.of("not_found", "no agent matches", details={"hint": "try list"})
    monkeypatch.setitem(HANDLERS, "show", _handler(error))
    assert cli.main(["show", "ag-abcdefgh"]) == 10
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "no agent matches" in captured.err
    assert "try list" in captured.err
    assert cli.main(["--json", "--decision-id", "d1", "show", "ag-abcdefgh"]) == 10
    envelope = json.loads(capsys.readouterr().out)
    assert envelope["error"]["decision_ids"] == ["d1"]  # filled from the invocation
    schema_registry.validate(envelope, "agent-spawn/cli/error.json")


def test_a_result_object_exits_with_the_runs_outcome_code(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    report = {**sample_outcome()["report"], "status": "failed"}
    run = sample_outcome(outcome="failed", exit_code=18, end_reason="normal", report=report)
    result = output.result_object(agent=sample_agent(), run=run, first_working_at=TS)
    monkeypatch.setitem(HANDLERS, "wait", _handler(Outcome(18, result, PayloadKind.RESULT)))
    assert cli.main(["--json", "wait", "ag-abcdefgh"]) == 18
    printed = json.loads(capsys.readouterr().out)
    assert printed["exit_code"] == 18
    schema_registry.validate(printed, "agent-spawn/cli/wait.json")


def test_a_dry_run_that_would_refuse_exits_20_with_the_plan(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    plan = {"dry_run": True, "would_refuse": {"reason": "live"}}
    monkeypatch.setitem(HANDLERS, "teardown", _handler(Outcome(20, plan, PayloadKind.DRY_RUN)))
    assert cli.main(["--json", "teardown", "ag-abcdefgh", "--dry-run"]) == 20
    assert json.loads(capsys.readouterr().out) == plan


def test_an_interrupted_waiter_prints_nothing(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setitem(HANDLERS, "wait", _handler(Outcome(143, None, PayloadKind.NONE)))
    assert cli.main(["--json", "wait", "ag-abcdefgh"]) == 143
    captured = capsys.readouterr()
    assert (captured.out, captured.err) == ("", "")


def test_an_unexpected_exception_becomes_an_internal_envelope_without_its_text(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setitem(HANDLERS, "list", _handler(RuntimeError("secret token abc123")))
    assert cli.main(["--json", "list"]) == 1
    captured = capsys.readouterr()
    envelope = json.loads(captured.out)
    schema_registry.validate(envelope, "agent-spawn/cli/error.json")
    assert envelope["error"]["code"] == "internal"
    assert "abc123" not in captured.out
    assert "abc123" not in captured.err
    assert "RuntimeError" in captured.err


def test_verbose_shows_the_traceback_of_an_unexpected_exception(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setitem(HANDLERS, "list", _handler(RuntimeError("detail-xyz")))
    assert cli.main(["--verbose", "list"]) == 1
    assert "detail-xyz" in capsys.readouterr().err


def test_a_handler_returning_a_disallowed_payload_is_an_internal_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setitem(HANDLERS, "list", lambda inv: (0, {}))
    assert cli.main(["--json", "list"]) == 1
    assert json.loads(capsys.readouterr().out)["error"]["code"] == "internal"


def test_hostile_text_in_the_message_cannot_break_stderr_or_stdout(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    error = CliError.of("failed", "bad \udcff \x1b[2J")
    monkeypatch.setitem(HANDLERS, "list", _handler(error))
    assert cli.main(["--json", "list"]) == 12
    captured = capsys.readouterr()
    assert captured.out.isascii()
    json.loads(captured.out)


# --- --result-file ------------------------------------------------------------------------------


def test_result_file_holds_exactly_what_json_prints_even_without_json(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    result = output.result_object(agent=sample_agent(), run=sample_outcome(), first_working_at=TS)
    monkeypatch.setitem(HANDLERS, "launch", _handler(Outcome(0, result, PayloadKind.RESULT)))
    path = tmp_path / "result.json"
    argv = ["launch", "--skill", "s", "--result-file", str(path)]
    assert cli.main(["--json", *argv]) == 0
    printed = capsys.readouterr().out
    assert path.read_text() == printed
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    path.unlink()
    assert cli.main(argv) == 0
    assert capsys.readouterr().out != printed  # human text on stdout...
    assert path.read_text() == printed  # ...but the file is the JSON


def test_result_file_golden_matches_and_validates(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    result = output.result_object(agent=sample_agent(), run=sample_outcome(), first_working_at=TS)
    monkeypatch.setitem(HANDLERS, "launch", _handler(Outcome(0, result, PayloadKind.RESULT)))
    path = tmp_path / "result.json"
    assert cli.main(["launch", "--skill", "s", "--result-file", str(path)]) == 0
    written = json.loads(path.read_text())
    assert written == json.loads((GOLDEN / "result_launch_done.json").read_text())
    schema_registry.validate(written, "agent-spawn/cli/launch.json")


@pytest.mark.parametrize(
    ("exit_code", "run_overrides"),
    [
        (0, {}),
        (11, {"outcome": "vetoed", "exit_code": 11}),
        (13, {"outcome": "blocked", "exit_code": 13}),
        (14, {"outcome": "exited-unreported", "exit_code": 14}),
        (18, {"outcome": "failed", "exit_code": 18}),
    ],
)
def test_result_file_is_written_for_every_result_exit_code(
    exit_code: int,
    run_overrides: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run = sample_outcome(**run_overrides)
    result = output.result_object(agent=sample_agent(), run=run)
    monkeypatch.setitem(HANDLERS, "wait", _handler(Outcome(exit_code, result, PayloadKind.RESULT)))
    path = tmp_path / "r.json"
    assert cli.main(["wait", "ag-abcdefgh", "--result-file", str(path)]) == exit_code
    assert json.loads(path.read_text())["exit_code"] == run["exit_code"]


def test_result_file_for_a_timed_out_wait_has_run_null(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    result = output.result_object(agent=sample_agent(), run=None, timed_out=True)
    monkeypatch.setitem(HANDLERS, "wait", _handler(Outcome(22, result, PayloadKind.RESULT)))
    path = tmp_path / "r.json"
    assert cli.main(["wait", "ag-abcdefgh", "--result-file", str(path)]) == 22
    written = json.loads(path.read_text())
    assert written["timed_out"] is True and written["run"] is None
    schema_registry.validate(written, "agent-spawn/cli/wait.json")


@pytest.mark.parametrize(
    ("code", "exit_code"), [("usage", 2), ("not_found", 10), ("refused", 20), ("failed", 12)]
)
def test_result_file_gets_the_envelope_for_every_error_exit_code(
    code: str, exit_code: int, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    error = CliError.of(code, "m", reason="live" if code == "refused" else None)
    monkeypatch.setitem(HANDLERS, "wait", _handler(error))
    path = tmp_path / "r.json"
    assert cli.main(["wait", "ag-abcdefgh", "--result-file", str(path)]) == exit_code
    written = json.loads(path.read_text())
    schema_registry.validate(written, "agent-spawn/cli/error.json")
    assert written["error"]["code"] == code


def test_result_file_envelope_golden_with_lower_error_passthrough(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    lower = json.loads((GOLDEN / "error_lower_refused.json").read_text())["error"]["lower_error"]
    error = output.from_lower_tool("wtctl", 20, lower, agent_id="ag-abcdefgh")
    monkeypatch.setitem(HANDLERS, "teardown", _handler(error))
    path = tmp_path / "r.json"
    # teardown has no --result-file, so write through a command that does
    monkeypatch.setitem(HANDLERS, "wait", _handler(error))
    assert cli.main(["wait", "ag-abcdefgh", "--result-file", str(path)]) == 20
    written = json.loads(path.read_text())
    assert written == json.loads((GOLDEN / "error_lower_refused.json").read_text())
    schema_registry.validate(written, "agent-spawn/cli/error.json")


def test_result_file_is_not_written_when_nothing_is_printed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setitem(HANDLERS, "wait", _handler(Outcome(130, None, PayloadKind.NONE)))
    path = tmp_path / "r.json"
    assert cli.main(["wait", "ag-abcdefgh", "--result-file", str(path)]) == 130
    assert not path.exists()


def test_result_file_failure_is_a_warning_and_never_changes_the_exit_code(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    monkeypatch.setitem(HANDLERS, "wait", _handler(Outcome(0, {"a": 1}, PayloadKind.RESULT)))
    path = tmp_path / "missing-dir" / "r.json"
    assert cli.main(["--json", "wait", "ag-abcdefgh", "--result-file", str(path)]) == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out) == {"a": 1}
    assert "warning" in captured.err.lower()
    assert "not written" in captured.err


def test_result_file_through_a_symlinked_parent_swapped_in_after_the_check_warns(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    victim = tmp_path / "victim"
    victim.mkdir()
    parent = tmp_path / "out"
    parent.mkdir()

    def handler(inv: Invocation) -> Outcome:
        # the handler runs after the path was checked; swap the checked parent for a link
        parent.rmdir()
        parent.symlink_to(victim)
        return Outcome(0, {"a": 1}, PayloadKind.RESULT)

    monkeypatch.setitem(HANDLERS, "wait", handler)
    argv = ["wait", "ag-abcdefgh", "--result-file", str(parent / "r.json")]
    assert cli.main(argv) == 0
    assert "path component 'out' is a symbolic link" in capsys.readouterr().err
    assert list(victim.iterdir()) == []


def test_result_file_at_a_symlink_is_refused_with_a_warning(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    victim = tmp_path / "victim.txt"
    victim.write_text("precious")
    link = tmp_path / "r.json"
    link.symlink_to(victim)
    monkeypatch.setitem(HANDLERS, "wait", _handler(Outcome(0, {"a": 1}, PayloadKind.RESULT)))
    assert cli.main(["wait", "ag-abcdefgh", "--result-file", str(link)]) == 0
    assert "symbolic link" in capsys.readouterr().err
    assert victim.read_text() == "precious"


def test_an_unusable_result_file_path_is_a_usage_error(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert cli.main(["--json", "wait", "ag-abcdefgh", "--result-file", "dir/"]) == 2
    assert json.loads(capsys.readouterr().out)["error"]["code"] == "usage"


def test_the_checked_target_is_handed_to_the_handler(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    seen: list[Invocation] = []

    def handler(inv: Invocation) -> Outcome:
        seen.append(inv)
        return Outcome(0, {"a": 1}, PayloadKind.RESULT)

    monkeypatch.setitem(HANDLERS, "wait", handler)
    real = tmp_path / "real"
    real.mkdir()
    (tmp_path / "link").symlink_to(real)
    cli.main(["wait", "ag-abcdefgh", "--result-file", str(tmp_path / "link" / "r.json")])
    assert seen[0].result_file is not None
    assert seen[0].result_file.parent == str(real)
    assert os.path.exists(real / "r.json")
