"""The fake systemd-run: ``--user --scope`` execs the command in place, as the real one does."""

from __future__ import annotations

import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from tests.fakebin.harness import Fakebin

BASE = ["--user", "--scope", "--quiet", "--collect"]
BUS = {"XDG_RUNTIME_DIR": "/run/user/1000"}


def start(
    fakebin: Fakebin,
    unit: str,
    *command: str,
    extra: list[str] | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return fakebin.run(
        "systemd-run",
        *BASE,
        f"--unit={unit}",
        *(extra or []),
        "--",
        *command,
        env=BUS if env is None else env,
    )


def test_the_command_runs_in_the_fakes_own_process(fakebin: Fakebin) -> None:
    done = start(fakebin, "agent-spawn-x", "sh", "-c", "echo $$")
    assert done.returncode == 0
    [call] = fakebin.invocations("systemd-run")
    assert done.stdout.strip() == str(call.pid)  # exec in place: same PID, as a real scope
    assert call.exit_code is None  # an exec leaves no exit record


def test_the_command_exit_code_is_the_calls_exit_code(fakebin: Fakebin) -> None:
    assert start(fakebin, "agent-spawn-x", "sh", "-c", "exit 7").returncode == 7


def test_the_environment_is_the_callers_plus_invocation_id(fakebin: Fakebin) -> None:
    done = start(
        fakebin,
        "agent-spawn-x",
        "sh",
        "-c",
        "echo $AGENT_SPAWN_X $INVOCATION_ID",
        env={**BUS, "AGENT_SPAWN_X": "1"},
    )
    assert done.stdout.split() == ["1", "0123456789abcdef0123456789abcdef"]


def status_lines(
    fakebin: Fakebin, tmp_path: Path, blocked: set[int], ignored: set[int]
) -> dict[str, int]:
    """SigBlk and SigIgn (as integers) of ``grep`` exec'd directly through the fake, whose caller
    set ``blocked`` and ``ignored`` first."""
    script = tmp_path / "caller.py"
    script.write_text(
        "import os, signal, sys\n"
        f"signal.pthread_sigmask(signal.SIG_SETMASK, {sorted(int(n) for n in blocked)!r})\n"
        f"for n in {sorted(int(n) for n in ignored)!r}: signal.signal(n, signal.SIG_IGN)\n"
        "os.execv(sys.argv[1], sys.argv[1:])\n"
    )
    done = subprocess.run(
        [
            sys.executable,
            str(script),
            str(fakebin.bin_dir / "systemd-run"),
            *BASE,
            "--unit=agent-spawn-x",
            "--",
            "grep",
            "-E",
            "Sig(Blk|Ign)",
            "/proc/self/status",
        ],
        env=fakebin.env(BUS),
        capture_output=True,
        text=True,
        check=False,
    )
    pairs = [line.split() for line in done.stdout.splitlines()]
    return {name.rstrip(":"): int(value, 16) for name, value in pairs}


def bits(*signals: int) -> int:
    return sum(1 << (int(n) - 1) for n in signals)


def test_the_callers_blocked_mask_is_preserved_across_the_exec(
    fakebin: Fakebin, tmp_path: Path
) -> None:
    # observed on systemd 249 (SigBlk 0x4003 for TERM, HUP and INT): the early-signal protection
    # of spawn_detached(keep_blocked=...) carries through systemd-run
    got = status_lines(fakebin, tmp_path, {signal.SIGTERM, signal.SIGHUP, signal.SIGINT}, set())
    assert got["SigBlk"] == bits(signal.SIGTERM, signal.SIGHUP, signal.SIGINT) == 0x4003


def test_the_callers_dispositions_are_preserved_and_nothing_is_ignored_on_its_own(
    fakebin: Fakebin, tmp_path: Path
) -> None:
    # the baseline is what the caller passes on (a test runner may ignore a signal of its own)
    baseline = status_lines(fakebin, tmp_path, set(), set())["SigIgn"]
    assert baseline & bits(signal.SIGPIPE, signal.SIGXFSZ) == 0  # not Python's own startup habit
    got = status_lines(fakebin, tmp_path, set(), {signal.SIGUSR1})
    assert got["SigIgn"] == baseline | bits(signal.SIGUSR1)


def test_without_a_user_manager_address_it_fails_like_the_real_one(fakebin: Fakebin) -> None:
    done = start(fakebin, "agent-spawn-x", "true", env={})
    assert done.returncode == 1
    assert done.stderr.startswith("Failed to connect to bus: $DBUS_SESSION_BUS_ADDRESS")
    assert fakebin.invocations("systemd-run")[0].exit_code == 1


def test_a_live_unit_name_cannot_be_started_twice(fakebin: Fakebin, tmp_path: Path) -> None:
    ready = tmp_path / "ready"
    proc = subprocess.Popen(
        [
            str(fakebin.bin_dir / "systemd-run"),
            *BASE,
            "--unit=agent-spawn-x",
            "--",
            "sh",
            "-c",
            f"touch {ready}; read line",
        ],
        env=fakebin.env(BUS),
        stdin=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 20
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert ready.exists()
        again = start(fakebin, "agent-spawn-x", "true")
        assert again.returncode == 1
        assert "Unit agent-spawn-x.scope already exists." in again.stderr
        assert start(fakebin, "agent-spawn-y", "true").returncode == 0
    finally:
        assert proc.stdin is not None
        proc.stdin.write("\n")
        proc.stdin.close()
        proc.wait(timeout=20)
    assert start(fakebin, "agent-spawn-x", "true").returncode == 0  # the scope vanished


@pytest.mark.parametrize("unit", ["bad name", "a/b", "x!", "", "é", "x\n", "u" * 201])
def test_an_invalid_unit_name_is_refused_not_escaped(fakebin: Fakebin, unit: str) -> None:
    done = start(fakebin, unit, "true")
    assert done.returncode == 1 and "invalid unit name" in done.stderr


@pytest.mark.parametrize(
    "argv",
    [
        ["--scope", "--unit=a", "--", "true"],  # no --user
        ["--user", "--unit=a", "--", "true"],  # no --scope
        ["--user", "--scope", "--", "true"],  # no --unit
        ["--user", "--scope", "--unit=a", "--property=Foo=bar", "--", "true"],
        ["--user", "--scope", "--unit=a", "-E", "A=b", "--", "true"],
    ],
)
def test_anything_else_is_unsupported_never_a_silent_success(
    fakebin: Fakebin, argv: list[str]
) -> None:
    done = fakebin.run("systemd-run", *argv, env=BUS)
    assert done.returncode == 1 and "fake_unsupported" in done.stderr


def test_a_missing_command_and_an_unknown_command_exit_1(fakebin: Fakebin) -> None:
    assert start(fakebin, "agent-spawn-x").returncode == 1
    done = start(fakebin, "agent-spawn-x", "/nonexistent/prog")
    assert done.returncode == 1 and "Failed to execute" in done.stderr


def test_a_scripted_rule_answers_instead_of_starting_anything(fakebin: Fakebin) -> None:
    fakebin.script("systemd-run", "scope", exit=1, stderr="Failed to connect to bus\n")
    done = start(fakebin, "agent-spawn-x", "sh", "-c", "echo ran")
    assert (done.returncode, done.stdout) == (1, "")
    assert "Failed to connect to bus" in done.stderr
    assert not (fakebin.root / "invocations.jsonl.scopes.json").exists()


def test_the_option_forms_of_the_real_program_are_accepted(fakebin: Fakebin) -> None:
    done = fakebin.run(
        "systemd-run",
        "--user",
        "--scope",
        "-q",
        "-c",
        "-u",
        "agent-spawn-x",
        "--description",
        "agent-spawn supervisor",
        "echo",
        "hi",
        env=BUS,
    )
    assert (done.returncode, done.stdout) == (0, "hi\n")
