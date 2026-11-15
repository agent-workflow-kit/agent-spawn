"""The autouse environment guard keeps every test off the real home, XDG dirs and tmux."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from tests import isolation
from tests.conftest import Isolation

REPO_ROOT = Path(__file__).resolve().parent.parent
REAL = {
    "HOME": "/home/real",
    "XDG_CONFIG_HOME": "/home/real/.config",
    "XDG_STATE_HOME": "/home/real/.local/state",
}


def good_env(tmp_path: Path) -> dict[str, str]:
    return {
        "HOME": str(tmp_path / "home"),
        "XDG_CONFIG_HOME": str(tmp_path / "config"),
        "XDG_STATE_HOME": str(tmp_path / "state"),
        "XDG_DATA_HOME": str(tmp_path / "data"),
        "XDG_CACHE_HOME": str(tmp_path / "cache"),
        "TMUX_TMPDIR": str(tmp_path / "tmux"),
    }


def test_every_test_runs_with_temporary_home_and_xdg_directories(
    isolation_env: Isolation, tmp_path_factory: pytest.TempPathFactory
) -> None:
    assert os.environ["HOME"] == str(isolation_env.home) == str(Path.home())
    for name, directory in {
        "XDG_CONFIG_HOME": isolation_env.config,
        "XDG_STATE_HOME": isolation_env.state,
        "XDG_DATA_HOME": isolation_env.data,
        "XDG_CACHE_HOME": isolation_env.cache,
        "TMUX_TMPDIR": isolation_env.tmux_tmpdir,
    }.items():
        assert os.environ[name] == str(directory) and directory.is_dir()
    assert isolation.environment_problems(os.environ) == []
    assert isolation.environment_problems(isolation_env.env) == []


def test_a_clean_environment_has_no_problems(tmp_path: Path) -> None:
    assert isolation.environment_problems(good_env(tmp_path), REAL) == []


@pytest.mark.parametrize(
    ("name", "value", "fragment"),
    [
        ("HOME", "/home/real", "real home"),
        ("HOME", "relative", "absolute"),
        ("XDG_CONFIG_HOME", "/home/real/.config", "real directory"),
        ("XDG_STATE_HOME", "/home/real/.local/state/agent-spawn", "real directory"),
        ("XDG_DATA_HOME", "/home/real/.local/share", "real directory"),
        ("XDG_CACHE_HOME", "/home/real/.cache", "real directory"),
        ("TMUX_TMPDIR", "/tmp/tmux-" + str(isolation.REAL_UID), "real tmux"),
        ("TMUX", "/tmp/tmux-1000/default,1,0", "TMUX is set"),
        ("TMUX_PANE", "%1", "TMUX_PANE is set"),
        ("AGENT_SPAWN_STATE_DIR", "/x", "AGENT_SPAWN_STATE_DIR is set"),
        ("TERMCTL_SOCKET", "work", "TERMCTL_SOCKET is set"),
        ("CSESSION_ID", "cs-aaaaaaaa", "CSESSION_ID is set"),
    ],
)
def test_the_guard_flags_anything_that_reaches_the_real_user(
    tmp_path: Path, name: str, value: str, fragment: str
) -> None:
    env = {**good_env(tmp_path), name: value}
    original = {**REAL, "HOME": "/home/real"}
    problems = isolation.environment_problems(env, original)
    assert any(fragment in problem for problem in problems), problems


def test_tmux_tmpdir_that_reaches_the_real_server_is_flagged(tmp_path: Path) -> None:
    """tmux uses $TMUX_TMPDIR/tmux-UID/default: /tmp and the original value reach the real one."""
    original = {**REAL, "TMUX_TMPDIR": str(tmp_path / "their-tmux")}
    (tmp_path / "their-tmux").mkdir()
    link = tmp_path / "to-tmp"
    link.symlink_to("/tmp")
    inside = tmp_path / "x" / f"tmux-{isolation.REAL_UID}" / "sub"
    inside.mkdir(parents=True)
    for bad in ("/tmp", str(tmp_path / "their-tmux"), str(link), str(inside)):
        env = {**good_env(tmp_path), "TMUX_TMPDIR": bad}
        problems = isolation.environment_problems(env, original)
        assert any("TMUX_TMPDIR" in problem for problem in problems), bad
    assert isolation.environment_problems(good_env(tmp_path), original) == []


def test_unset_variables_are_problems_too(tmp_path: Path) -> None:
    env = good_env(tmp_path)
    del env["XDG_STATE_HOME"]
    del env["TMUX_TMPDIR"]
    problems = isolation.environment_problems(env, REAL)
    assert len(problems) == 2


def test_a_symlink_into_the_real_home_is_caught(tmp_path: Path) -> None:
    real = tmp_path / "realhome"
    (real / ".config").mkdir(parents=True)
    link = tmp_path / "link"
    link.symlink_to(real / ".config")
    env = {**good_env(tmp_path), "XDG_CONFIG_HOME": str(link)}
    problems = isolation.environment_problems(env, {"HOME": str(real)})
    assert any("XDG_CONFIG_HOME" in problem for problem in problems)


def test_pytest_started_inside_tmux_with_real_variables_is_isolated(tmp_path: Path) -> None:
    """Run a probe test through the real conftest with the dangerous variables set."""
    probe = tmp_path / "test_probe.py"
    probe.write_text(
        textwrap.dedent(
            """
            import os
            from pathlib import Path

            def test_probe(isolation_env):
                leaky = ("TMUX", "TMUX_PANE", "AGENT_SPAWN_X", "TERMCTL_SOCKET", "XDG_RUNTIME_DIR")
                for name in leaky:
                    assert name not in os.environ, name
                assert Path.home() == isolation_env.home
                assert os.environ["HOME"] != os.environ["ORIGINAL_HOME"]
                assert os.environ["TMUX_TMPDIR"] == str(isolation_env.tmux_tmpdir)
            """
        )
    )
    env = {
        **os.environ,
        "PYTHONPATH": str(REPO_ROOT),
        "TMUX": "/tmp/tmux-1000/default,1,0",
        "TMUX_PANE": "%1",
        "AGENT_SPAWN_X": "1",
        "TERMCTL_SOCKET": "work",
        "XDG_RUNTIME_DIR": "/run/user/1000",
        # the passwd home, not $HOME: pytest itself may run with a private HOME
        "ORIGINAL_HOME": isolation.REAL_PASSWD_HOME,
    }
    done = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "tests.conftest",
            "-p",
            "no:cacheprovider",
            "-q",
            "--rootdir",
            str(tmp_path),
            "-c",
            "/dev/null",
            str(probe),
        ],
        env=env,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
        stdin=subprocess.DEVNULL,
    )
    assert done.returncode == 0, done.stdout + done.stderr


def test_a_test_that_restores_a_real_directory_fails_at_teardown(tmp_path: Path) -> None:
    probe = tmp_path / "test_leak.py"
    probe.write_text(
        textwrap.dedent(
            """
            import os

            def test_leak():
                os.environ["HOME"] = os.environ["ORIGINAL_HOME"]
            """
        )
    )
    env = {
        **os.environ,
        "PYTHONPATH": str(REPO_ROOT),
        # the passwd home, not $HOME: pytest itself may run with a private HOME
        "ORIGINAL_HOME": isolation.REAL_PASSWD_HOME,
    }
    done = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "tests.conftest",
            "-p",
            "no:cacheprovider",
            "-q",
            "--rootdir",
            str(tmp_path),
            "-c",
            "/dev/null",
            str(probe),
        ],
        env=env,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
        stdin=subprocess.DEVNULL,
    )
    assert done.returncode != 0
    assert "left a non-isolated environment" in done.stdout
