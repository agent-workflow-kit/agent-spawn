"""The handoff against a real, private tmux server (design.md spikes A1-A8).

Every tmux call goes through ``PrivateTmux`` (``-S <tmp>/sock -f /dev/null``, private HOME,
XDG and TMUX_TMPDIR), never the user's server. The stand-in window process is
``handoff_child.py``; the listener is the test process, as the launcher would be.
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from agent_spawn import procs
from agent_spawn.handoff import MAX_PAYLOAD, HandoffFailed, HandoffListener
from tests.private_tmux import PrivateTmux
from tests.test_handoff import (
    CHILD,
    SECRET,
    hang_guard,  # noqa: F401  (autouse fixture)
    log_records,
    make_short_state,
    read_result,
    wait_for_file,
)


def open_window(tmux: PrivateTmux, socket_path: Path, result: Path, timeout: float = 20) -> int:
    """``new-window`` running the child directly (no shell); returns ``#{pane_pid}``."""
    done = tmux.run(
        "new-window", "-d", "-P", "-F", "#{pane_pid}", "--",
        sys.executable, CHILD, str(socket_path), str(timeout), str(result),
    )  # fmt: skip
    assert done.returncode == 0, done.stderr
    return int(done.stdout.strip())


@pytest.fixture
def state() -> Iterator[Path]:
    base = make_short_state()
    try:
        yield base
    finally:
        shutil.rmtree(base, ignore_errors=True)


def test_the_window_process_gets_the_environment_and_pane_pid_is_its_pid(
    private_tmux: PrivateTmux, state: Path, tmp_path: Path
) -> None:
    private_tmux.start()
    listener = HandoffListener.create(state)
    result = tmp_path / "r.json"
    pane_pid = open_window(private_tmux, listener.socket_path, result)
    env = {"LAUNCHER_ONLY": SECRET, "TERM": "launcher-term", "TMUX_PANE": "%launcher"}
    listener.serve(pane_pid=pane_pid, pane_state="running", timeout=20, env=env)
    got = read_result(result)
    assert got["pid"] == pane_pid  # A8
    assert got["env"]["LAUNCHER_ONLY"] == SECRET  # A1-A4
    assert "LAUNCHER_ONLY" not in got["before"]  # the window did not inherit it from tmux
    assert not listener.directory.exists()


def test_terminal_variables_come_from_the_pane_not_the_launcher(
    private_tmux: PrivateTmux, state: Path, tmp_path: Path
) -> None:
    private_tmux.start()
    listener = HandoffListener.create(state)
    result = tmp_path / "r.json"
    pane_pid = open_window(private_tmux, listener.socket_path, result)
    env = {"TERM": "launcher-term", "TMUX": "launcher", "TMUX_PANE": "%launcher", "COLORTERM": "x"}
    listener.serve(pane_pid=pane_pid, pane_state="running", timeout=20, env=env)
    got = read_result(result)
    for name in ("TERM", "TMUX", "TMUX_PANE"):  # A5: tmux sets these in the pane
        assert got["env"][name] == got["before"][name] != env[name]
    assert got["env"]["TMUX"].startswith(str(private_tmux.socket))
    assert "COLORTERM" not in got["env"] or got["env"]["COLORTERM"] == got["before"]["COLORTERM"]


THIEF = (
    "import json,socket,sys\n"
    "s=socket.socket(socket.AF_UNIX);s.connect(sys.argv[1]);open(sys.argv[2],'w').close()\n"
    "d=b''\n"
    "while True:\n c=s.recv(65536)\n if not c: break\n d+=c\n"
    "out={'failed':'empty_payload'} if d==b'' else {'data':d.decode('latin-1')}\n"
    "open(sys.argv[3],'w').write(json.dumps(out))"
)


def test_a_thief_window_never_gets_the_environment_and_the_pane_window_does(
    private_tmux: PrivateTmux, state: Path, tmp_path: Path
) -> None:
    """A6-A7: another process of the same user connecting to the socket is refused."""
    private_tmux.start()
    listener = HandoffListener.create(state, agent_id="ag-aaaaaaaa")
    connected, thief_result, real_result = tmp_path / "c", tmp_path / "thief.json", tmp_path / "r"
    done = private_tmux.run(
        "new-window", "-d", "-P", "-F", "#{pane_pid}", "--",
        sys.executable, "-c", THIEF, str(listener.socket_path), str(connected), str(thief_result),
    )  # fmt: skip
    assert done.returncode == 0, done.stderr
    thief_pid = int(done.stdout.strip())
    assert wait_for_file(connected)  # the thief is queued ahead of the real window
    real_pid = open_window(private_tmux, listener.socket_path, real_result)
    assert thief_pid != real_pid
    listener.serve(pane_pid=real_pid, pane_state="running", timeout=20, env={"K": SECRET})
    assert read_result(real_result)["env"]["K"] == SECRET
    assert read_result(thief_result) == {"failed": "empty_payload"}
    records: list[dict[str, Any]] = log_records(state)
    causes = [r["details"]["cause"] for r in records if r["event"] == "handoff_rejected"]
    assert causes == ["wrong_pid"]
    assert SECRET not in json.dumps(records)


def test_a_window_that_exits_before_serve_is_gone(
    private_tmux: PrivateTmux, state: Path, tmp_path: Path
) -> None:
    private_tmux.start()
    listener = HandoffListener.create(state)
    done = private_tmux.run(
        "new-window", "-d", "-P", "-F", "#{pane_pid}", "--", sys.executable, "-c", "pass"
    )
    pid = int(done.stdout.strip())
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:  # a zombie of tmux counts as not alive
        info = procs.inspect(pid)
        if info is None or not info.alive:
            break
        time.sleep(0.02)
    with pytest.raises(HandoffFailed) as info_:
        listener.serve(pane_pid=pid, pane_state="running", timeout=2, env={"A": "1"})
    assert info_.value.cause == "pane_gone"
    assert not listener.directory.exists()


def test_a_payload_over_the_cap_never_reaches_the_window(
    private_tmux: PrivateTmux, state: Path, tmp_path: Path
) -> None:
    private_tmux.start()
    listener = HandoffListener.create(state)
    result = tmp_path / "r.json"
    pane_pid = open_window(private_tmux, listener.socket_path, result, timeout=20)
    with pytest.raises(HandoffFailed) as info:
        listener.serve(
            pane_pid=pane_pid,
            pane_state="running",
            timeout=20,
            env={"BIG": "x" * (MAX_PAYLOAD + 1)},
        )
    assert info.value.cause == "payload_too_large"
    assert read_result(result)["failed"] in {"connect_failed", "empty_payload"}


def test_the_window_process_gives_up_after_its_own_deadline(
    private_tmux: PrivateTmux, state: Path, tmp_path: Path
) -> None:
    private_tmux.start()
    listener = HandoffListener.create(state)
    result = tmp_path / "r.json"
    open_window(private_tmux, listener.socket_path, result, timeout=0.5)  # listener never serves
    assert read_result(result)["failed"] == "timeout"
    listener.close()
