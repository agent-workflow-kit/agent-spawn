"""The installed entry point, run as a real process with a private HOME."""

import json
import os
import subprocess
import sys
from pathlib import Path

from agent_spawn import __version__


def _run(tmp_path: Path, *argv: str) -> subprocess.CompletedProcess[str]:
    env = {
        "HOME": str(tmp_path),
        "XDG_CONFIG_HOME": str(tmp_path / "config"),
        "XDG_STATE_HOME": str(tmp_path / "state"),
        "PATH": os.environ.get("PATH", ""),
    }
    return subprocess.run(
        [sys.executable, "-m", "agent_spawn", *argv],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="surrogateescape",
        env=env,
        stdin=subprocess.DEVNULL,
        check=False,
        timeout=60,
    )


def test_version_prints_version_and_exits_zero(tmp_path: Path) -> None:
    done = _run(tmp_path, "--version")
    assert done.returncode == 0
    assert done.stdout.strip() == f"agent-spawn {__version__}"


def test_usage_error_exits_2_with_the_envelope_on_stdout(tmp_path: Path) -> None:
    done = _run(tmp_path, "--json", "list", "--nope")
    assert done.returncode == 2
    assert json.loads(done.stdout)["error"]["code"] == "usage"
    assert "usage:" in done.stderr


def test_no_arguments_is_a_usage_error(tmp_path: Path) -> None:
    done = _run(tmp_path)
    assert done.returncode == 2
    assert "usage:" in done.stderr


def test_a_stub_command_fails_through_the_envelope_and_creates_no_state(tmp_path: Path) -> None:
    done = _run(tmp_path, "--json", "list")
    assert done.returncode == 1
    assert json.loads(done.stdout)["error"]["code"] == "internal"
    assert list(tmp_path.iterdir()) == []


def test_human_only_flag_without_a_tty_exits_2(tmp_path: Path) -> None:
    done = _run(tmp_path, "teardown", "ag-abcdefgh", "--force")
    assert done.returncode == 2
    done = _run(tmp_path, "teardown", "ag-abcdefgh", "--force", "--actor", "human:alice")
    assert done.returncode == 1  # reached the (stub) handler
