"""Compare the fakes with the real programs. `live`: never run in CI.

Run with ``uv run pytest -m live tests/live`` after tmux, Claude Code or sibling-tool upgrades.
Real programs are found on ``PATH`` (or ``LIVE_CLAUDE_SESSION``, ``LIVE_TERMCTL``,
``LIVE_WTCTL``, ``LIVE_TMUX``); a missing one skips its tests. Every call runs with the
isolated ``HOME``/``XDG_*``/``TMUX_TMPDIR`` of the ``isolation_env`` fixture (the temporary
tmux directory exists, so a tool can never fall back to the real default socket).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from collections.abc import Mapping
from pathlib import Path

import pytest

from tests.conftest import Isolation
from tests.fakebin.fakelib.tmux import split_commands
from tests.fakebin.fidelity_cases import CASES
from tests.fakebin.harness import Fakebin
from tests.private_tmux import PrivateTmux

pytestmark = pytest.mark.live

ENV_NAMES = {
    "claude-session": "LIVE_CLAUDE_SESSION",
    "termctl": "LIVE_TERMCTL",
    "wtctl": "LIVE_WTCTL",
    "tmux": "LIVE_TMUX",
}


def real_program(tool: str) -> str:
    found = os.environ.get(ENV_NAMES[tool]) or shutil.which(tool)
    if not found:
        pytest.skip(f"no real {tool} (set {ENV_NAMES[tool]} or put it on PATH)")
    return found


def run_real(
    program: str, argv: list[str], env: Mapping[str, str]
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [program, *argv],
        env={**env, "PATH": "/usr/local/bin:/usr/bin:/bin"},
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        timeout=60,
        check=False,
    )


def error_code(stdout: str) -> str | None:
    try:
        return json.loads(stdout)["error"]["code"]
    except (ValueError, KeyError, TypeError):
        return None


REAL_CASES = [
    pytest.param(tool, argv, exit_code, code, id=f"{tool}:{' '.join(argv)}")
    for tool, rows in CASES.items()
    if tool != "wtctl"  # the real wtctl is not implemented yet
    for argv, exit_code, code, verified in rows
    if verified
]


@pytest.mark.parametrize(("tool", "argv", "exit_code", "code"), REAL_CASES)
def test_the_real_program_gives_the_exit_and_error_code_the_fake_gives(
    isolation_env: Isolation, tool: str, argv: list[str], exit_code: int, code: str | None
) -> None:
    done = run_real(real_program(tool), argv, isolation_env.env)
    assert done.returncode == exit_code, (done.stdout, done.stderr)
    assert error_code(done.stdout) == code


def test_real_wtctl_when_implemented_matches_the_fake(isolation_env: Isolation) -> None:
    program = real_program("wtctl")
    probe = run_real(program, ["create", "--json"], isolation_env.env)
    if error_code(probe.stdout) is None:
        pytest.skip("the real wtctl does not print the error envelope yet (scaffold)")
    for argv, exit_code, code, _ in CASES["wtctl"]:
        if argv[0] != "--version":
            done = run_real(program, argv, isolation_env.env)
            assert (done.returncode, error_code(done.stdout)) == (exit_code, code), argv


def test_tmux_version_and_no_server_behave_as_the_fake(
    fakebin: Fakebin, isolation_env: Isolation
) -> None:
    program = real_program("tmux")
    version = run_real(program, ["-V"], isolation_env.env)
    assert re.fullmatch(r"tmux (next-)?\d+\.\d+[a-z]?\n", version.stdout)
    assert re.fullmatch(r"tmux \d+\.\d+[a-z]?\n", fakebin.run("tmux", "-V").stdout)
    socket = Path(isolation_env.tmux_tmpdir) / "sock"
    real = run_real(
        program, ["-f", "/dev/null", "-S", str(socket), "list-sessions"], isolation_env.env
    )
    fake = fakebin.run("tmux", "-f", "/dev/null", "-S", str(socket), "list-sessions")
    assert (real.returncode, real.stdout, real.stderr) == (
        fake.returncode,
        fake.stdout,
        fake.stderr,
    )
    bad = run_real(program, ["-x", "list-sessions"], isolation_env.env)
    fake_bad = fakebin.run("tmux", "-x", "list-sessions")
    assert (bad.returncode, bad.stderr) == (fake_bad.returncode, fake_bad.stderr)


def test_tmux_semicolon_splitting_matches_the_fake(private_tmux: PrivateTmux) -> None:
    private_tmux.start("one")
    cases = [["display-message", "-p", "a;"], ["display-message", "-p", "b\\;"]]
    expected = [["a"], ["b;"]]
    for argv, lines in zip(cases, expected, strict=True):
        done = private_tmux.run(*argv)
        assert done.stdout.splitlines() == lines
        assert split_commands(argv) == [["display-message", "-p", lines[0]]]
    # `;` between commands runs both
    both = private_tmux.run("display-message", "-p", "x", ";", "display-message", "-p", "y")
    assert both.stdout.splitlines() == ["x", "y"]
    assert (
        len(split_commands(["display-message", "-p", "x", ";", "display-message", "-p", "y"])) == 2
    )


def test_tmux_new_window_prints_pane_and_window_ids_like_the_scripted_fake_replies(
    private_tmux: PrivateTmux,
) -> None:
    """The ``new-window -P -F`` format agent-spawn's callers script is ``%N @N``."""
    private_tmux.start("one")
    done = private_tmux.run(
        "new-window", "-d", "-P", "-F", "#{pane_id} #{window_id}", "-t", "=one:", "sleep", "30"
    )
    assert re.fullmatch(r"%\d+ @\d+\n", done.stdout), done
