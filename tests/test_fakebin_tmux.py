"""The fake tmux: client option parsing, ``-V``, ``;`` splitting and scripted replies."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.fakebin.fakelib.tmux import split_commands
from tests.fakebin.harness import Fakebin


def test_version_prints_tmux_and_a_version_and_ignores_the_rest(fakebin: Fakebin) -> None:
    done = fakebin.run("tmux", "-V")
    assert (done.returncode, done.stdout) == (0, "tmux 3.4\n")
    assert fakebin.run("tmux", "-f", "/dev/null", "-S", "/x/sock", "-V").stdout == "tmux 3.4\n"
    assert fakebin.run("tmux", "-V", env={"FAKEBIN_TMUX_VERSION": "3.2a"}).stdout == "tmux 3.2a\n"


def test_without_a_script_every_command_finds_no_server(fakebin: Fakebin) -> None:
    done = fakebin.run("tmux", "-S", "/some/where/sock", "list-sessions")
    assert done.returncode == 1 and done.stdout == ""
    assert done.stderr == "error connecting to /some/where/sock (No such file or directory)\n"
    default = fakebin.run("tmux", "list-sessions")
    uid = __import__("os").getuid()
    assert default.stderr.startswith("error connecting to ")
    assert f"/tmux-{uid}/default" in default.stderr


def test_unknown_option_prints_usage_and_exits_1(fakebin: Fakebin) -> None:
    done = fakebin.run("tmux", "-x", "list-sessions")
    assert done.returncode == 1
    assert done.stderr.startswith("tmux: unknown option -- x\nusage: tmux [-2CDlNuVv]")


def test_clustered_flags_and_attached_values(fakebin: Fakebin) -> None:
    fakebin.script("tmux", "display-message", stdout="42\n")
    done = fakebin.run("tmux", "-uf/dev/null", "-S/s/sock", "display-message", "-p", "#{pid}")
    assert (done.returncode, done.stdout) == (0, "42\n")
    assert fakebin.run("tmux", "-S").returncode == 1  # option requires an argument


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        (["a", "b", ";", "c"], [["a", "b"], ["c"]]),
        (["a", "b;", "c"], [["a", "b"], ["c"]]),
        (["a", "b\\;", "c"], [["a", "b;", "c"]]),
        ([";", ";"], []),
        (["set", "-t", "=s:", "--", "name;"], [["set", "-t", "=s:", "--", "name"]]),
    ],
)
def test_semicolons_separate_commands_unless_escaped(
    args: list[str], expected: list[list[str]]
) -> None:
    assert split_commands(args) == expected


def test_scripted_command_replies_and_logs_exact_argv(fakebin: Fakebin) -> None:
    fakebin.script("tmux", "new-session", stdout="%0 @0\n")
    argv = [
        "-f",
        "/dev/null",
        "-S",
        "/s/sock",
        "new-session",
        "-d",
        "-P",
        "-F",
        "#{pane_id} #{window_id}",
        ";",
        "set",
        "x",
    ]
    done = fakebin.run("tmux", *argv)
    assert (done.returncode, done.stdout) == (0, "%0 @0\n")
    fakebin.assert_argv("tmux", argv)
    assert fakebin.invocations("tmux")[0].exit_code == 0


def test_scripted_failure_and_stderr(fakebin: Fakebin) -> None:
    fakebin.script("tmux", "has-session", exit=1, stderr="can't find session: nope\n")
    done = fakebin.run("tmux", "has-session", "-t", "nope")
    assert (done.returncode, done.stderr) == (1, "can't find session: nope\n")


def test_no_command_is_unsupported_not_a_session(fakebin: Fakebin) -> None:
    assert fakebin.run("tmux", "-f", "/dev/null").returncode == 1


# --- no server (R23), client options (R24), aliases and empty rules (R28) --------------------


def test_a_socket_file_nobody_listens_on_is_no_server_running(
    fakebin: Fakebin, tmp_path: Path
) -> None:
    import socket as socket_module

    stale = tmp_path / "stale.sock"
    listener = socket_module.socket(socket_module.AF_UNIX)
    listener.bind(str(stale))
    listener.close()  # the file stays, nothing listens
    regular = tmp_path / "regular"
    regular.touch()
    for path in (stale, regular):
        done = fakebin.run("tmux", "-S", str(path), "list-sessions")
        assert (done.returncode, done.stderr) == (1, f"no server running on {path}\n")
    missing = fakebin.run("tmux", "-S", str(tmp_path / "none"), "list-sessions")
    assert (
        missing.stderr == f"error connecting to {tmp_path / 'none'} (No such file or directory)\n"
    )


@pytest.mark.parametrize("option", ["-d", "-U", "-D"])
def test_attach_and_server_options_print_only_the_usage(fakebin: Fakebin, option: str) -> None:
    done = fakebin.run("tmux", "-S", "/s/sock", option, "list-sessions")
    assert done.returncode == 1 and done.stdout == ""
    assert done.stderr.startswith("usage: tmux [-2CDlNuVv]")


def test_the_q_option_is_ignored_and_start_server_succeeds_silently(fakebin: Fakebin) -> None:
    done = fakebin.run("tmux", "-q", "-S", "/s/sock", "list-sessions")
    assert done.stderr == "error connecting to /s/sock (No such file or directory)\n"
    started = fakebin.run("tmux", "-S", "/s/sock", "start-server")
    assert (started.returncode, started.stdout, started.stderr) == (0, "", "")
    assert fakebin.run("tmux", "-S", "/s/sock", "kill-server").returncode == 1


def test_an_empty_scripted_response_is_a_rule_not_a_missing_server(fakebin: Fakebin) -> None:
    fakebin.script("tmux", "has-session")
    done = fakebin.run("tmux", "-S", "/s/sock", "has-session", "-t", "x")
    assert (done.returncode, done.stdout, done.stderr) == (0, "", "")


def test_a_rule_for_a_command_answers_its_alias_and_the_other_way_round(fakebin: Fakebin) -> None:
    fakebin.script("tmux", "list-sessions", stdout="one\n")
    fakebin.script("tmux", "neww", stdout="%1 @1\n")
    assert fakebin.run("tmux", "-S", "/s", "ls").stdout == "one\n"
    assert fakebin.run("tmux", "-S", "/s", "list-sessions").stdout == "one\n"
    assert fakebin.run("tmux", "-S", "/s", "new-window").stdout == "%1 @1\n"
    assert fakebin.run("tmux", "-S", "/s", "neww").stdout == "%1 @1\n"
