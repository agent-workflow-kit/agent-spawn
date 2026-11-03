"""The CLI loads the config: fail fast with config_invalid, me.name, the stored absolute path."""

import json
from pathlib import Path

import pytest

from agent_spawn import cli
from agent_spawn.commands.base import Invocation
from agent_spawn.commands.table import HANDLERS
from agent_spawn.output import Outcome, PayloadKind
from tests import schema_registry


@pytest.fixture
def reached(monkeypatch: pytest.MonkeyPatch) -> list[Invocation]:
    seen: list[Invocation] = []

    def spy(inv: Invocation) -> Outcome:
        seen.append(inv)
        return Outcome(0, {"ok": True}, PayloadKind.COMMAND)

    for key in list(HANDLERS):
        monkeypatch.setitem(HANDLERS, key, spy)
    return seen


def _cfg(tmp_path: Path, text: str) -> str:
    path = tmp_path / "c.yaml"
    path.write_text(text)
    return str(path)


def test_invalid_config_exits_2_naming_the_key(
    tmp_path: Path, reached: list[Invocation], capsys: pytest.CaptureFixture[str]
) -> None:
    code = cli.main(["--json", "--config", _cfg(tmp_path, "tombstone_days: -3\n"), "list"])
    envelope = json.loads(capsys.readouterr().out)
    assert code == 2
    assert reached == []
    schema_registry.validate(envelope, "agent-spawn/cli/error.json")
    assert envelope["error"]["code"] == "config_invalid"
    assert "tombstone_days" in envelope["error"]["message"]


def test_invalid_config_carries_decision_ids_and_writes_the_result_file(
    tmp_path: Path, reached: list[Invocation], capsys: pytest.CaptureFixture[str]
) -> None:
    result = tmp_path / "out.json"
    argv = ["--config", _cfg(tmp_path, "nope: 1\n"), "--decision-id", "d-1", "wait", "ag-abcdefgh"]
    code = cli.main([*argv, "--result-file", str(result), "--json"])
    printed = json.loads(capsys.readouterr().out)
    assert code == 2
    assert reached == []
    assert printed["error"]["code"] == "config_invalid"
    assert printed["error"]["decision_ids"] == ["d-1"]
    assert json.loads(result.read_text()) == printed


def test_invalid_default_location_config_fails_every_command(
    tmp_path: Path,
    reached: list[Invocation],
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    (tmp_path / "agent-spawn").mkdir()
    (tmp_path / "agent-spawn" / "config.yaml").write_text("nope: 1\n")
    assert cli.main(["list"]) == 2
    assert "nope" in capsys.readouterr().err


def test_invalid_environment_override_fails_fast(
    reached: list[Invocation],
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AGENT_SPAWN_HANDOFF_TIMEOUT", "0")
    assert cli.main(["list"]) == 2
    assert "AGENT_SPAWN_HANDOFF_TIMEOUT" in capsys.readouterr().err


def test_usage_error_wins_over_config_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cli.main(["--json", "--config", _cfg(tmp_path, "nope: 1\n"), "resume"])
    assert json.loads(capsys.readouterr().out)["error"]["code"] == "usage"


def test_me_name_from_config_is_the_default_actor_user(
    tmp_path: Path, reached: list[Invocation], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "stdin_is_tty", lambda: True)
    monkeypatch.setattr(cli, "me_name", lambda: "osuser")
    cli.main(["--config", _cfg(tmp_path, "me: {name: jo.e}\n"), "show", "ag-abcdefgh"])
    assert reached[-1].actor == "human:jo.e"
    cli.main(["--config", _cfg(tmp_path, "me: {name: null}\n"), "show", "ag-abcdefgh"])
    assert reached[-1].actor == "human:osuser"


def test_the_invocation_carries_the_config_and_its_absolute_path(
    tmp_path: Path, reached: list[Invocation], monkeypatch: pytest.MonkeyPatch
) -> None:
    _cfg(tmp_path, "tombstone_days: 5\n")
    monkeypatch.chdir(tmp_path)
    cli.main(["--config", "c.yaml", "show", "ag-abcdefgh"])
    inv = reached[-1]
    assert inv.config is not None and inv.config.tombstone_days == 5
    assert inv.config_file == str(tmp_path / "c.yaml")


def test_default_path_is_stored_when_no_file_exists(
    tmp_path: Path, reached: list[Invocation], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    cli.main(["show", "ag-abcdefgh"])
    assert reached[-1].config_file == str(tmp_path / "xdg" / "agent-spawn" / "config.yaml")
    assert reached[-1].warnings == []


def test_missing_named_config_warns_but_internal_commands_do_not(
    tmp_path: Path, reached: list[Invocation]
) -> None:
    cli.main(["--config", str(tmp_path / "gone.yaml"), "show", "ag-abcdefgh"])
    assert any("gone.yaml" in w for w in reached[-1].warnings)
    cli.main(["--state-dir", "/s", "--config", str(tmp_path / "gone.yaml"), "_session-event"])
    assert reached[-1].warnings == []
    assert reached[-1].config is not None
