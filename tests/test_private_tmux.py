"""The private tmux fixture: a real server on a private socket that is stopped exactly."""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

import pytest

from tests.conftest import REAL_TMUX, Isolation
from tests.private_tmux import PrivateTmux, process_start_time


def test_the_server_uses_a_private_socket_config_free_and_a_private_home(
    private_tmux: PrivateTmux, isolation_env: Isolation
) -> None:
    private_tmux.start("one")
    assert private_tmux.socket.exists() and private_tmux.socket.parent == private_tmux.directory
    assert private_tmux.run("list-sessions", "-F", "#{session_name}").stdout == "one\n"
    pid = private_tmux.server_pid
    assert pid is not None
    cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
    assert (
        b"-f" in cmdline
        and b"/dev/null" in cmdline
        and str(private_tmux.socket).encode() in cmdline
    )
    environ = Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
    assert f"HOME={isolation_env.home}".encode() in environ
    assert not any(item.startswith(b"TMUX=") for item in environ)
    assert f"TMUX_TMPDIR={private_tmux.directory}".encode() in environ


def test_stop_ends_the_server_and_removes_the_directory() -> None:
    if REAL_TMUX is None:
        pytest.skip("tmux is not installed")
    env = {
        "HOME": "/nonexistent",
        "XDG_CONFIG_HOME": "/nonexistent",
        "XDG_STATE_HOME": "/nonexistent",
    }
    server = PrivateTmux(REAL_TMUX, env)
    server.env["HOME"] = str(server.directory)  # keep it inside its own private directory
    server.start()
    pid = server.server_pid
    assert pid is not None and process_start_time(pid) is not None
    server.stop()
    assert not Path(f"/proc/{pid}").exists() or _zombie(pid)
    assert not server.directory.exists()


def _zombie(pid: int) -> bool:
    return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] == "Z"


def test_stop_signals_only_the_recorded_pid_and_only_if_the_start_time_matches() -> None:
    """A lingering server is SIGKILLed by its exact PID; a reused PID is left alone."""
    server = PrivateTmux(
        "/bin/true", {"HOME": "/x", "XDG_CONFIG_HOME": "/x", "XDG_STATE_HOME": "/x"}
    )
    server.kill_wait = 0.2
    lingering = subprocess.Popen(["sleep", "30"])
    bystander = subprocess.Popen(["sleep", "30"])
    try:
        server.server_pid = lingering.pid
        server._server_start = process_start_time(lingering.pid)  # pyright: ignore[reportPrivateUsage]
        server.stop()
        assert lingering.wait(timeout=5) == -9
        assert bystander.poll() is None  # not touched
    finally:
        bystander.kill()
        bystander.wait()
        if lingering.poll() is None:
            lingering.kill()
            lingering.wait()


def test_a_wrong_start_time_means_the_pid_was_reused_and_nothing_is_signaled() -> None:
    server = PrivateTmux(
        "/bin/true", {"HOME": "/x", "XDG_CONFIG_HOME": "/x", "XDG_STATE_HOME": "/x"}
    )
    server.kill_wait = 0.2
    other = subprocess.Popen(["sleep", "30"])
    try:
        server.server_pid = other.pid
        server._server_start = "1"  # pyright: ignore[reportPrivateUsage]
        started = time.monotonic()
        server.stop()
        assert time.monotonic() - started < 3
        assert other.poll() is None
    finally:
        other.kill()
        other.wait()


def test_the_real_tmux_default_socket_is_never_used(private_tmux: PrivateTmux) -> None:
    assert private_tmux.argv("x")[1:5] == ["-f", "/dev/null", "-S", str(private_tmux.socket)]
    assert "TMUX" not in private_tmux.env and os.environ.get("TMUX") is None
