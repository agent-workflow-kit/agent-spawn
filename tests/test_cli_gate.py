"""The human-only gate: --force, --override-escalation and --allow-tainted."""

import json
import os
import pwd
import re
from types import SimpleNamespace
from typing import Any

import pytest

from agent_spawn import cli
from agent_spawn.commands.base import Invocation
from agent_spawn.commands.table import HANDLERS
from agent_spawn.output import Outcome, PayloadKind
from tests import schema_registry

GATED = {
    "teardown --force": ["teardown", "ag-abcdefgh", "--force"],
    "teardown --override-escalation": ["teardown", "ag-abcdefgh", "--override-escalation"],
    "launch --override-escalation": ["launch", "--skill", "s", "--override-escalation"],
    "resume --override-escalation": ["resume", "ag-abcdefgh", "--override-escalation"],
    "resume --allow-tainted": ["resume", "ag-abcdefgh", "--allow-tainted"],
    "resume --fork --allow-tainted": ["resume", "ag-abcdefgh", "--fork", "--allow-tainted"],
}


@pytest.fixture
def reached(monkeypatch: pytest.MonkeyPatch) -> list[Invocation]:
    """Replace every handler with one that records that it was reached."""
    seen: list[Invocation] = []

    def spy(inv: Invocation) -> Outcome:
        seen.append(inv)
        return Outcome(0, {"ok": True}, PayloadKind.COMMAND)

    for key in list(HANDLERS):
        monkeypatch.setitem(HANDLERS, key, spy)
    return seen


def _tty(monkeypatch: pytest.MonkeyPatch, value: bool) -> None:
    monkeypatch.setattr(cli, "stdin_is_tty", lambda: value)


@pytest.mark.parametrize("argv", GATED.values(), ids=GATED.keys())
def test_refused_without_a_tty_and_without_a_human_actor(
    argv: list[str],
    reached: list[Invocation],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _tty(monkeypatch, False)
    code = cli.main(["--json", *argv])
    out = capsys.readouterr().out
    assert code == 2
    assert reached == []
    envelope = json.loads(out)
    schema_registry.validate(envelope, "agent-spawn/cli/error.json")
    assert envelope["error"]["code"] == "usage"
    assert "human" in envelope["error"]["message"]


@pytest.mark.parametrize("actor", ["auto:policy", "auto:unknown", "auto:human"])
@pytest.mark.parametrize("argv", GATED.values(), ids=GATED.keys())
def test_refused_without_a_tty_with_an_automatic_actor(
    argv: list[str], actor: str, reached: list[Invocation], monkeypatch: pytest.MonkeyPatch
) -> None:
    _tty(monkeypatch, False)
    assert cli.main([*argv, "--actor", actor]) == 2
    assert reached == []


@pytest.mark.parametrize("actor", ["auto:policy", "auto:unknown", "auto:human"])
@pytest.mark.parametrize("argv", GATED.values(), ids=GATED.keys())
def test_refused_with_a_tty_with_an_explicit_automatic_actor(
    argv: list[str],
    actor: str,
    reached: list[Invocation],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _tty(monkeypatch, True)
    assert cli.main(["--json", *argv, "--actor", actor]) == 2
    assert reached == []
    envelope = json.loads(capsys.readouterr().out)
    assert envelope["error"]["code"] == "usage"
    assert "human" in envelope["error"]["message"]


@pytest.mark.parametrize("argv", GATED.values(), ids=GATED.keys())
def test_accepted_with_a_tty_and_an_explicit_human_actor(
    argv: list[str], reached: list[Invocation], monkeypatch: pytest.MonkeyPatch
) -> None:
    _tty(monkeypatch, True)
    assert cli.main([*argv, "--actor", "human:alice"]) == 0
    assert reached[-1].actor == "human:alice"


@pytest.mark.parametrize("argv", GATED.values(), ids=GATED.keys())
def test_accepted_with_a_tty(
    argv: list[str], reached: list[Invocation], monkeypatch: pytest.MonkeyPatch
) -> None:
    _tty(monkeypatch, True)
    assert cli.main(argv) == 0
    assert len(reached) == 1


@pytest.mark.parametrize("argv", GATED.values(), ids=GATED.keys())
def test_accepted_without_a_tty_with_an_explicit_human_actor(
    argv: list[str], reached: list[Invocation], monkeypatch: pytest.MonkeyPatch
) -> None:
    _tty(monkeypatch, False)
    assert cli.main([*argv, "--actor", "human:alice"]) == 0
    assert reached[-1].actor == "human:alice"
    assert cli.main(["--actor=human:bob", *argv]) == 0
    assert reached[-1].actor == "human:bob"


@pytest.mark.parametrize("argv", GATED.values(), ids=GATED.keys())
def test_ungated_commands_never_need_a_human(
    argv: list[str], reached: list[Invocation], monkeypatch: pytest.MonkeyPatch
) -> None:
    _tty(monkeypatch, False)
    plain = [a for a in argv if a not in ("--force", "--override-escalation", "--allow-tainted")]
    assert cli.main(plain) == 0
    assert len(reached) == 1


def test_the_gate_is_decided_before_the_handler_and_its_envelope_goes_to_the_result_file(
    reached: list[Invocation],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    _tty(monkeypatch, False)
    path = tmp_path_factory.mktemp("rf") / "r.json"
    code = cli.main(["launch", "--skill", "s", "--override-escalation", "--result-file", str(path)])
    assert code == 2
    assert reached == []
    envelope = json.loads(path.read_text())
    schema_registry.validate(envelope, "agent-spawn/cli/error.json")
    assert "human" in envelope["error"]["message"]


def test_an_unreadable_stdin_counts_as_no_tty(monkeypatch: pytest.MonkeyPatch) -> None:
    class Broken:
        def isatty(self) -> bool:
            raise ValueError("I/O operation on closed file")

    monkeypatch.setattr("sys.stdin", Broken())
    assert cli.stdin_is_tty() is False
    monkeypatch.setattr("sys.stdin", None)
    assert cli.stdin_is_tty() is False


# --- the effective actor ------------------------------------------------------------------------


def test_default_actor_is_human_with_a_tty_and_auto_unknown_without(
    reached: list[Invocation], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "me_name", lambda: "alice")
    _tty(monkeypatch, True)
    cli.main(["show", "ag-abcdefgh"])
    assert reached[-1].actor == "human:alice"
    assert reached[-1].actor_explicit is False
    _tty(monkeypatch, False)
    cli.main(["show", "ag-abcdefgh"])
    assert reached[-1].actor == "auto:unknown"


def test_explicit_actor_wins_over_the_default(
    reached: list[Invocation], monkeypatch: pytest.MonkeyPatch
) -> None:
    _tty(monkeypatch, True)
    cli.main(["show", "ag-abcdefgh", "--actor", "auto:cron"])
    assert reached[-1].actor == "auto:cron"
    assert reached[-1].actor_explicit is True


@pytest.mark.parametrize(
    ("name", "expected"),
    [("alice", "alice"), ("a b", "a_b"), ("x" * 100, "x" * 64), ("", "unknown"), ("é", "_")],
)
def test_me_name_is_sanitised_to_fit_the_actor_grammar(
    name: str, expected: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli.pwd, "getpwuid", lambda uid: SimpleNamespace(pw_name=name))
    assert cli.me_name() == expected
    assert re.fullmatch(r"[A-Za-z0-9._@-]{1,64}", cli.me_name())


def test_me_name_survives_a_missing_user(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(uid: int) -> Any:
        raise KeyError(uid)

    monkeypatch.setattr(cli.pwd, "getpwuid", boom)
    assert cli.me_name() == "unknown"


def test_me_name_ignores_logname_and_user(monkeypatch: pytest.MonkeyPatch) -> None:
    real = pwd.getpwuid(os.geteuid()).pw_name
    for var in ("LOGNAME", "USER", "LNAME", "USERNAME"):
        monkeypatch.setenv(var, "someone-else")
    assert real != "someone-else"
    assert cli.me_name() == re.sub(r"[^A-Za-z0-9._@-]", "_", real)[:64]
