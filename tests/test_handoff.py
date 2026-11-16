"""The environment handoff (issue #26): payload rules, the listener's safety and the peer check.

Process tests with real sockets and real child processes (recorded and killed by exact identity
through ``proc_tracker``). The real-tmux spikes A1-A8 are in ``test_handoff_tmux.py``.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import socket
import stat
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from agent_spawn import handoff, procs
from agent_spawn.gateway import Gateway, PeerCredentials, get_gateway
from agent_spawn.handoff import (
    MAX_PAYLOAD,
    TERMINAL_VARIABLES,
    HandoffFailed,
    HandoffListener,
    build_environment,
    encode_payload,
    parse_payload,
    receive_environment,
)
from tests import schema_registry
from tests.conftest import Isolation
from tests.procguard import ProcTracker

CHILD = str(Path(__file__).parent / "handoff_child.py")
SECRET = "s3cr3t-token-value-7f3a"


def wait_for_file(path: Path, timeout: float = 20.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return True
        time.sleep(0.02)
    return path.exists()


def start_child(
    tracker: ProcTracker,
    iso: Isolation,
    socket_path: Path,
    result: Path,
    timeout: float = 20.0,
    started: Path | None = None,
) -> subprocess.Popen[bytes]:
    env = dict(iso.env)
    env["TERM"] = "xterm-pane"
    env["TMUX_PANE"] = "%pane"
    return tracker.popen(
        [sys.executable, CHILD, str(socket_path), str(timeout), str(result)]
        + ([str(started)] if started is not None else []),
        env=env,
        stdin=subprocess.DEVNULL,
    )


def read_result(path: Path) -> dict[str, Any]:
    assert wait_for_file(path), "the child wrote no result"
    return json.loads(path.read_text(encoding="utf-8"))


def log_records(state: Path) -> list[dict[str, Any]]:
    path = state / "log.jsonl"
    if not path.exists():
        return []
    records = [json.loads(line) for line in path.read_text().splitlines()]
    for record in records:
        schema_registry.validate(record, "agent-spawn/log.json")
    return records


def make_short_state() -> Path:
    """A state directory under a short base, so socket paths fit macOS's 104-byte limit."""
    return Path(tempfile.mkdtemp(prefix="hs", dir="/tmp")).resolve()


@pytest.fixture
def state() -> Iterator[Path]:
    base = make_short_state()
    try:
        yield base
    finally:
        shutil.rmtree(base, ignore_errors=True)


@pytest.fixture(autouse=True)
def hang_guard() -> Iterator[None]:
    """Fail (rather than hang) a test that runs over 120 s: SIGALRM raises in the main thread."""

    def on_alarm(signum: int, frame: object) -> None:
        raise TimeoutError("test exceeded the 120 s hang guard")

    previous = signal.signal(signal.SIGALRM, on_alarm)
    signal.alarm(120)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


class WrapGateway:
    """The real gateway with the peer credentials and/or the uid replaced."""

    def __init__(
        self,
        *,
        creds: PeerCredentials | None = None,
        peer_error: BaseException | None = None,
        uid: int | None = None,
    ) -> None:
        self._inner = get_gateway()
        self._creds = creds
        self._peer_error = peer_error
        self._uid = uid

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def getuid(self) -> int:
        return self._uid if self._uid is not None else self._inner.getuid()

    def peer_credentials(self, sock: socket.socket) -> PeerCredentials:
        if self._peer_error is not None:
            raise self._peer_error
        return self._creds if self._creds is not None else self._inner.peer_credentials(sock)


def gw(wrapper: WrapGateway) -> Gateway:
    return wrapper  # type: ignore[return-value]


# --- payload -------------------------------------------------------------------------------------


def test_the_payload_drops_the_terminal_variables_and_round_trips_unicode_and_bytes() -> None:
    env = {name: "x" for name in TERMINAL_VARIABLES}
    env.update({"A": "café ☃ \U0001f600", "B": "", "C": "line\nbreak", "D": "\udcff"})
    data = encode_payload(env)
    assert data.isascii()
    assert parse_payload(data) == {k: v for k, v in env.items() if k not in TERMINAL_VARIABLES}


@pytest.mark.parametrize(
    "env",
    [{"A": 1}, {1: "x"}],  # type: ignore[dict-item]
)
def test_the_writer_refuses_a_non_string_variable_without_naming_a_value(
    env: dict[str, Any],
) -> None:
    with pytest.raises(HandoffFailed) as info:
        encode_payload(env)
    assert info.value.cause == "invalid_env"


def test_the_writer_skips_entries_that_cannot_be_exported_and_delivers_the_rest() -> None:
    env = {
        "": "x",
        "A=B": "1",
        "N\0": "x",
        "V": "v\0",
        "S": "\ud800",
        "\udd00": "x",
        "OK": "1",
        "BYTES": "\udcff",
    }
    assert parse_payload(encode_payload(env)) == {"OK": "1", "BYTES": "\udcff"}


def test_the_cap_is_4_mib_for_the_writer_and_the_reader() -> None:
    overhead = len(encode_payload({"A": ""}))
    exact = encode_payload({"A": "x" * (MAX_PAYLOAD - overhead)})
    assert len(exact) == MAX_PAYLOAD
    assert parse_payload(exact)["A"] == "x" * (MAX_PAYLOAD - overhead)
    with pytest.raises(HandoffFailed) as writer:
        encode_payload({"A": "x" * (MAX_PAYLOAD - overhead + 1)})
    assert writer.value.cause == "payload_too_large"
    with pytest.raises(HandoffFailed) as reader:
        parse_payload(exact + b" ")
    assert reader.value.cause == "payload_too_large"


@pytest.mark.parametrize(
    ("data", "cause"),
    [
        (b"", "empty_payload"),
        (b"not json", "malformed_payload"),
        (b"\xff\xfe", "malformed_payload"),
        (b"[]", "malformed_payload"),
        (b"{}", "malformed_payload"),
        (b'{"env": []}', "malformed_payload"),
        (b'{"env": {}, "extra": 1}', "malformed_payload"),
        (b'{"env": {"A": 1}}', "malformed_payload"),
        (b'{"env": {"": "x"}}', "malformed_payload"),
        (b'{"env": {"A=B": "x"}}', "malformed_payload"),
        (b'{"env": {"A": "v\\u0000"}}', "malformed_payload"),
        (b'{"env": {"A": "1", "A": "2"}}', "malformed_payload"),
        (b'{"env": {"A": "\\ud800"}}', "malformed_payload"),
        (b'{"env": {"\\udd00": "x"}}', "malformed_payload"),
        (b'{"env": ' + b"[" * 100000, "malformed_payload"),
    ],
)
def test_the_reader_refuses_a_bad_payload(data: bytes, cause: str) -> None:
    with pytest.raises(HandoffFailed) as info:
        parse_payload(data)
    assert info.value.cause == cause


def test_the_pane_supplies_the_terminal_variables() -> None:
    payload = {"A": "1", "TERM": "from-launcher", "TMUX": "from-launcher"}
    pane = {"TERM": "xterm", "TMUX_PANE": "%3", "OTHER": "ignored"}
    assert build_environment(payload, pane) == {"A": "1", "TERM": "xterm", "TMUX_PANE": "%3"}


# --- the directory and the socket ----------------------------------------------------------------


def test_create_makes_a_private_directory_with_an_owner_file_and_a_socket(state: Path) -> None:
    with HandoffListener.create(state) as listener:
        root = state / "handoff"
        assert stat.S_IMODE(root.stat().st_mode) == 0o700
        assert listener.directory.parent == root and listener.directory.is_dir()
        assert stat.S_IMODE(listener.directory.stat().st_mode) == 0o700
        assert listener.socket_path == listener.directory / "s"
        assert stat.S_ISSOCK(listener.socket_path.lstat().st_mode)
        owner_file = listener.directory / "owner"
        assert stat.S_IMODE(owner_file.stat().st_mode) == 0o600
        owner = handoff.read_owner(listener.directory)
        assert owner == procs.read_identity(os.getpid())
        assert json.loads(owner_file.read_text()).keys() == {"pid", "start_time", "boot_id"}
    assert not listener.directory.exists()


def test_the_modes_do_not_depend_on_the_umask(state: Path) -> None:
    old = os.umask(0)
    try:
        listener = HandoffListener.create(state)
    finally:
        os.umask(old)
    with listener:
        assert stat.S_IMODE((state / "handoff").lstat().st_mode) == 0o700
        assert stat.S_IMODE(listener.directory.lstat().st_mode) == 0o700
        assert stat.S_IMODE(listener.socket_path.lstat().st_mode) == 0o600


def test_an_unreadable_listener_process_fails_cleanly(
    state: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*args: Any, **kwargs: Any) -> Any:
        raise procs.ProcessIdentityError("unreadable")

    monkeypatch.setattr(handoff.procs, "read_identity", boom)
    with pytest.raises(HandoffFailed) as info:
        HandoffListener.create(state)
    assert info.value.cause == "listen_failed"
    assert list((state / "handoff").iterdir()) == []


def test_two_listeners_get_distinct_directories(state: Path) -> None:
    with HandoffListener.create(state) as one, HandoffListener.create(state) as two:
        assert one.directory != two.directory


def test_close_is_idempotent_and_tolerates_missing_files(state: Path) -> None:
    listener = HandoffListener.create(state)
    os.unlink(listener.socket_path)
    listener.close()
    listener.close()
    assert not listener.directory.exists()


def test_a_symlinked_handoff_root_is_refused(state: Path, tmp_path: Path) -> None:
    target = tmp_path / "elsewhere"
    target.mkdir()
    (state / "handoff").symlink_to(target)
    with pytest.raises(HandoffFailed) as info:
        HandoffListener.create(state)
    assert info.value.cause == "unsafe_directory"
    assert list(target.iterdir()) == []


def test_a_loose_handoff_root_is_tightened(state: Path) -> None:
    root = state / "handoff"
    root.mkdir(mode=0o755)
    root.chmod(0o755)
    with HandoffListener.create(state):
        assert stat.S_IMODE(root.stat().st_mode) == 0o700


def test_a_relative_state_directory_is_refused() -> None:
    with pytest.raises(HandoffFailed) as info:
        HandoffListener.create("relative/state")
    assert info.value.cause == "unsafe_directory"


def test_a_socket_path_over_the_platform_limit_fails_cleanly(tmp_path: Path) -> None:
    long_state = tmp_path / ("d" * 90)
    long_state.mkdir()
    with pytest.raises(HandoffFailed) as info:
        HandoffListener.create(long_state)
    assert info.value.cause == "socket_path_too_long"
    assert list((long_state / "handoff").iterdir()) == []


# --- serving -------------------------------------------------------------------------------------


def test_success_hands_the_environment_to_the_pane_process_once(
    state: Path, tmp_path: Path, proc_tracker: ProcTracker, isolation_env: Isolation
) -> None:
    listener = HandoffListener.create(state, agent_id="ag-aaaaaaaa")
    result = tmp_path / "result.json"
    child = start_child(proc_tracker, isolation_env, listener.socket_path, result)
    env = {"KEEP": SECRET, "TERM": "launcher-term", "TMUX_PANE": "%launcher", "N": "café"}
    listener.serve(pane_pid=child.pid, pane_state="running", timeout=20, env=env)
    got = read_result(result)
    assert got["env"]["KEEP"] == SECRET and got["env"]["N"] == "café"
    assert got["env"]["TERM"] == "xterm-pane" and got["env"]["TMUX_PANE"] == "%pane"
    assert child.wait(timeout=20) == 0
    assert not listener.directory.exists()  # socket, owner file and directory are gone
    assert not any(r["event"].startswith("handoff") for r in log_records(state))
    with pytest.raises(OSError):  # one-shot: nobody can connect any more
        probe = socket.socket(socket.AF_UNIX)
        try:
            probe.connect(str(listener.socket_path))
        finally:
            probe.close()


def test_a_thief_is_rejected_and_logged_and_the_pane_process_still_gets_the_environment(
    state: Path, tmp_path: Path, proc_tracker: ProcTracker, isolation_env: Isolation
) -> None:
    listener = HandoffListener.create(state, agent_id="ag-aaaaaaaa")
    thief = socket.socket(socket.AF_UNIX)
    thief.connect(str(listener.socket_path))  # queued ahead of the pane process
    result = tmp_path / "result.json"
    child = start_child(proc_tracker, isolation_env, listener.socket_path, result)
    listener.serve(pane_pid=child.pid, pane_state="running", timeout=20, env={"KEEP": SECRET})
    thief.settimeout(10)
    assert thief.recv(65536) == b""  # closed without data
    thief.close()
    assert read_result(result)["env"]["KEEP"] == SECRET
    rejected = [r for r in log_records(state) if r["event"] == "handoff_rejected"]
    assert [r["details"]["cause"] for r in rejected] == ["wrong_pid"]
    assert SECRET not in (state / "log.jsonl").read_text()
    assert child.wait(timeout=20) == 0


def test_a_peer_with_the_right_pid_and_a_different_start_time_is_rejected(
    state: Path,
    tmp_path: Path,
    proc_tracker: ProcTracker,
    isolation_env: Isolation,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PID reuse: the connecting process has the pane PID but not the recorded start time."""
    listener = HandoffListener.create(state, agent_id="ag-aaaaaaaa")
    result, started = tmp_path / "result.json", tmp_path / "started"
    child = start_child(
        proc_tracker, isolation_env, listener.socket_path, result, timeout=20, started=started
    )
    assert wait_for_file(started)  # the child connects right after; the backlog queues it
    real_inspect = procs.inspect
    calls = {"n": 0}

    def inspect(pid: int, **kwargs: Any) -> procs.ProcInfo | None:
        info = real_inspect(pid, **kwargs)
        calls["n"] += 1
        if calls["n"] == 1 or info is None:  # the first read records the pane
            return info
        ident = info.identity
        forged = procs.ProcessIdentity(ident.pid, ident.start_time + 1, ident.boot_id)
        return procs.ProcInfo(forged, info.state)

    monkeypatch.setattr(handoff.procs, "inspect", inspect)
    with pytest.raises(HandoffFailed) as info:
        listener.serve(pane_pid=child.pid, pane_state="running", timeout=2, env={"KEEP": SECRET})
    assert info.value.cause == "timeout"
    got = read_result(result)
    assert got["failed"] == "empty_payload" and SECRET not in json.dumps(got)
    causes = [r["details"]["cause"] for r in log_records(state) if r["event"] == "handoff_rejected"]
    assert causes == ["wrong_start_time"]
    assert not listener.directory.exists()


def test_an_early_connection_waits_in_the_backlog_until_serve(
    state: Path, tmp_path: Path, proc_tracker: ProcTracker
) -> None:
    """The pane process has connected (and said so) before serve runs; it is still answered."""
    listener = HandoffListener.create(state)
    script = (
        "import socket,sys;s=socket.socket(socket.AF_UNIX);s.connect(sys.argv[1]);"
        "open(sys.argv[2],'w').close();d=b''\n"
        "while True:\n c=s.recv(65536)\n if not c: break\n d+=c\n"
        "open(sys.argv[3],'wb').write(d)"
    )
    connected, out = tmp_path / "connected", tmp_path / "out"
    child = proc_tracker.popen(
        [sys.executable, "-c", script, str(listener.socket_path), str(connected), str(out)],
        stdin=subprocess.DEVNULL,
    )
    assert wait_for_file(connected)  # connect() returned before serve was entered
    assert not out.exists()
    listener.serve(pane_pid=child.pid, pane_state="running", timeout=20, env={"A": "1"})
    assert wait_for_file(out)
    assert child.wait(timeout=20) == 0
    assert parse_payload(out.read_bytes()) == {"A": "1"}


@pytest.mark.parametrize(
    ("pid", "state_name", "cause"),
    [(None, "running", "no_pane_pid"), (0, "running", "no_pane_pid"), (4242, "gone", "pane_gone")],
)
def test_a_missing_or_gone_pane_fails_at_once_and_removes_the_socket(
    state: Path, pid: int | None, state_name: str, cause: str
) -> None:
    listener = HandoffListener.create(state, agent_id="ag-aaaaaaaa")
    with pytest.raises(HandoffFailed) as info:
        listener.serve(pane_pid=pid, pane_state=state_name, timeout=3600, env={"A": SECRET})
    assert info.value.cause == cause and SECRET not in str(info.value)
    assert not listener.directory.exists()
    [record] = log_records(state)
    assert record["event"] == "handoff_failed" and record["details"] == {"cause": cause}
    assert record["agent_id"] == "ag-aaaaaaaa"


def test_a_pane_pid_that_does_not_exist_is_gone(state: Path) -> None:
    listener = HandoffListener.create(state)
    with pytest.raises(HandoffFailed) as info:
        listener.serve(pane_pid=2**22 + 777, pane_state="running", timeout=3600, env={})
    assert info.value.cause == "pane_gone" and not listener.directory.exists()


def test_an_oversized_environment_fails_before_accepting_and_removes_the_socket(
    state: Path,
) -> None:
    listener = HandoffListener.create(state)
    with pytest.raises(HandoffFailed) as info:
        listener.serve(
            pane_pid=os.getpid(),
            pane_state="running",
            timeout=3600,
            env={"BIG": "x" * (MAX_PAYLOAD + 1)},
        )
    assert info.value.cause == "payload_too_large" and not listener.directory.exists()


def test_a_non_string_environment_is_refused_and_removes_the_socket(state: Path) -> None:
    listener = HandoffListener.create(state)
    with pytest.raises(HandoffFailed) as info:
        listener.serve(
            pane_pid=os.getpid(),
            pane_state="running",
            timeout=3600,
            env={"A": SECRET, 1: "x"},  # type: ignore[dict-item]
        )
    assert info.value.cause == "invalid_env" and SECRET not in str(info.value)
    assert not listener.directory.exists()
    assert SECRET not in (state / "log.jsonl").read_text()


def test_the_server_deadline_expires_and_removes_the_socket(
    state: Path, proc_tracker: ProcTracker
) -> None:
    child = proc_tracker.popen(
        [sys.executable, "-c", "import time; time.sleep(120)"], stdin=subprocess.DEVNULL
    )
    listener = HandoffListener.create(state, agent_id="ag-aaaaaaaa")
    with pytest.raises(HandoffFailed) as info:
        listener.serve(pane_pid=child.pid, pane_state="running", timeout=0.2, env={"A": "1"})
    assert info.value.cause == "timeout"
    assert not listener.directory.exists()
    assert [r["event"] for r in log_records(state)] == ["handoff_failed"]


def test_a_listener_serves_once(state: Path) -> None:
    listener = HandoffListener.create(state)
    with pytest.raises(HandoffFailed):
        listener.serve(pane_pid=None, pane_state=None, timeout=1, env={})
    with pytest.raises(HandoffFailed) as info:
        listener.serve(pane_pid=os.getpid(), pane_state="running", timeout=1, env={})
    assert info.value.cause == "listen_failed"


def test_an_exception_while_serving_removes_the_socket(
    state: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    listener = HandoffListener.create(state)

    def boom(*args: Any, **kwargs: Any) -> Any:
        raise KeyboardInterrupt

    monkeypatch.setattr(handoff.procs, "inspect", boom)
    with pytest.raises(KeyboardInterrupt):
        listener.serve(pane_pid=os.getpid(), pane_state="running", timeout=5, env={})
    assert not listener.directory.exists()


def test_a_client_that_never_reads_hits_the_server_deadline(
    state: Path, proc_tracker: ProcTracker, tmp_path: Path
) -> None:
    """The pane process connects but never reads a large payload: the send is bounded."""
    listener = HandoffListener.create(state)
    script = (
        "import socket,sys,time;s=socket.socket(socket.AF_UNIX);s.connect(sys.argv[1]);"
        "open(sys.argv[2],'w').close();time.sleep(120)"
    )
    ready = tmp_path / "connected"
    child = proc_tracker.popen(
        [sys.executable, "-c", script, str(listener.socket_path), str(ready)],
        stdin=subprocess.DEVNULL,
    )
    assert wait_for_file(ready)
    big = {"BIG": "x" * (MAX_PAYLOAD - 100)}
    with pytest.raises(HandoffFailed) as info:
        listener.serve(pane_pid=child.pid, pane_state="running", timeout=0.5, env=big)
    assert info.value.cause == "timeout" and not listener.directory.exists()


def test_a_rejection_flood_is_logged_only_up_to_a_cap(
    state: Path, proc_tracker: ProcTracker
) -> None:
    child = proc_tracker.popen(
        [sys.executable, "-c", "import time; time.sleep(120)"], stdin=subprocess.DEVNULL
    )
    listener = HandoffListener.create(state)
    socks = []
    for _ in range(40):
        s = socket.socket(socket.AF_UNIX)
        s.connect(str(listener.socket_path))
        socks.append(s)
    with pytest.raises(HandoffFailed):
        listener.serve(pane_pid=child.pid, pane_state="running", timeout=3, env={})
    for s in socks:
        s.close()
    events = [r["event"] for r in log_records(state)]
    assert events.count("handoff_rejected") == 20 and events[-1] == "handoff_failed"


def test_an_unreadable_pane_fails_with_a_fixed_cause_and_is_logged(
    state: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    listener = HandoffListener.create(state, agent_id="ag-aaaaaaaa")

    def boom(*args: Any, **kwargs: Any) -> Any:
        raise procs.ProcessIdentityError("garbled /proc")

    monkeypatch.setattr(handoff.procs, "inspect", boom)
    with pytest.raises(HandoffFailed) as info:
        listener.serve(pane_pid=os.getpid(), pane_state="running", timeout=5, env={})
    assert info.value.cause == "pane_unreadable" and not listener.directory.exists()
    [record] = log_records(state)
    assert record["event"] == "handoff_failed" and record["details"] == {"cause": "pane_unreadable"}


def _rejected_causes(state: Path) -> list[str]:
    return [r["details"]["cause"] for r in log_records(state) if r["event"] == "handoff_rejected"]


def _serve_against_one_connection(
    state: Path, proc_tracker: ProcTracker, wrapper: WrapGateway
) -> HandoffFailed:
    child = proc_tracker.popen(
        [sys.executable, "-c", "import time; time.sleep(120)"], stdin=subprocess.DEVNULL
    )
    listener = HandoffListener.create(state, gateway=gw(wrapper))
    conn = socket.socket(socket.AF_UNIX)
    try:
        conn.connect(str(listener.socket_path))
        with pytest.raises(HandoffFailed) as info:
            listener.serve(pane_pid=child.pid, pane_state="running", timeout=0.5, env={"K": SECRET})
        conn.settimeout(10)
        assert conn.recv(65536) == b""  # closed without data
    finally:
        conn.close()
    return info.value


def test_a_peer_of_another_user_is_rejected(state: Path, proc_tracker: ProcTracker) -> None:
    me = os.getuid()

    class Other(WrapGateway):
        def peer_credentials(self, sock: socket.socket) -> PeerCredentials:
            real = super().peer_credentials(sock)
            return PeerCredentials(pid=real.pid, uid=me + 1, gid=real.gid)

    failure = _serve_against_one_connection(state, proc_tracker, Other())
    assert failure.cause == "timeout" and _rejected_causes(state) == ["wrong_uid"]


def test_an_unreadable_peer_is_rejected(state: Path, proc_tracker: ProcTracker) -> None:
    failure = _serve_against_one_connection(
        state, proc_tracker, WrapGateway(peer_error=OSError("no creds"))
    )
    assert failure.cause == "timeout" and _rejected_causes(state) == ["peer_unreadable"]


def test_a_peer_that_died_is_rejected(
    state: Path, proc_tracker: ProcTracker, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_inspect = procs.inspect
    calls = {"n": 0}

    def inspect(pid: int, **kwargs: Any) -> procs.ProcInfo | None:
        calls["n"] += 1
        return real_inspect(pid, **kwargs) if calls["n"] == 1 else None  # gone after the pane read

    child_holder: dict[str, int] = {}

    class SamePid(WrapGateway):
        def peer_credentials(self, sock: socket.socket) -> PeerCredentials:
            real = super().peer_credentials(sock)
            return PeerCredentials(pid=child_holder["pid"], uid=real.uid, gid=real.gid)

    child = proc_tracker.popen(
        [sys.executable, "-c", "import time; time.sleep(120)"], stdin=subprocess.DEVNULL
    )
    child_holder["pid"] = child.pid
    listener = HandoffListener.create(state, gateway=gw(SamePid()))
    monkeypatch.setattr(handoff.procs, "inspect", inspect)  # after create's own identity read
    conn = socket.socket(socket.AF_UNIX)
    try:
        conn.connect(str(listener.socket_path))
        with pytest.raises(HandoffFailed):
            listener.serve(pane_pid=child.pid, pane_state="running", timeout=0.5, env={})
    finally:
        conn.close()
    assert _rejected_causes(state) == ["peer_gone"]


def test_an_already_expired_server_deadline_is_a_timeout(
    state: Path, proc_tracker: ProcTracker
) -> None:
    child = proc_tracker.popen(
        [sys.executable, "-c", "import time; time.sleep(120)"], stdin=subprocess.DEVNULL
    )
    listener = HandoffListener.create(state)
    with pytest.raises(HandoffFailed) as info:
        listener.serve(pane_pid=child.pid, pane_state="running", timeout=0, env={})
    assert info.value.cause == "timeout" and not listener.directory.exists()


# --- the client ----------------------------------------------------------------------------------


class RawServer:
    """A stand-in listener that sends fixed bytes (or nothing) to whoever connects."""

    def __init__(self, directory: Path, send: Callable[[socket.socket], None]) -> None:
        self.path = directory / "raw.sock"
        self.sock = socket.socket(socket.AF_UNIX)
        self.sock.bind(str(self.path))
        self.sock.listen(1)
        self.thread = threading.Thread(target=self._run, args=(send,), daemon=True)
        self.thread.start()

    def _run(self, send: Callable[[socket.socket], None]) -> None:
        try:
            conn, _ = self.sock.accept()
        except OSError:
            return
        with conn:
            try:
                send(conn)
            except OSError:
                pass

    def close(self) -> None:
        self.sock.close()
        self.thread.join(timeout=10)


def _serve_bytes(data: bytes) -> Callable[[socket.socket], None]:
    def send(conn: socket.socket) -> None:
        conn.sendall(data)
        conn.shutdown(socket.SHUT_WR)

    return send


@pytest.mark.parametrize(
    ("data", "cause"),
    [
        (b"", "empty_payload"),
        (b"{broken", "malformed_payload"),
        (b'{"env": {"A": 1}}', "malformed_payload"),
        (b"x" * (MAX_PAYLOAD + 1), "payload_too_large"),
    ],
)
def test_the_client_refuses_bad_payloads(tmp_path: Path, data: bytes, cause: str) -> None:
    server = RawServer(tmp_path, _serve_bytes(data))
    try:
        with pytest.raises(HandoffFailed) as info:
            receive_environment(server.path, timeout=20, pane_env={})
    finally:
        server.close()
    assert info.value.cause == cause


def test_the_client_deadline_expires_when_the_server_stays_silent(tmp_path: Path) -> None:
    release = threading.Event()

    def silent(conn: socket.socket) -> None:
        release.wait(30)

    server = RawServer(tmp_path, silent)
    try:
        with pytest.raises(HandoffFailed) as info:
            receive_environment(server.path, timeout=0.3, pane_env={})
    finally:
        release.set()
        server.close()
    assert info.value.cause == "timeout"


def test_the_client_logs_a_failure_with_the_cause_only(state: Path, tmp_path: Path) -> None:
    server = RawServer(tmp_path, _serve_bytes(b""))
    try:
        with pytest.raises(HandoffFailed):
            receive_environment(
                server.path, timeout=20, pane_env={}, state_dir=state, agent_id="ag-bbbbbbbb"
            )
    finally:
        server.close()
    [record] = log_records(state)
    assert record["event"] == "handoff_failed" and record["details"] == {"cause": "empty_payload"}


def test_the_client_fails_at_once_when_the_socket_is_gone(tmp_path: Path) -> None:
    with pytest.raises(HandoffFailed) as info:
        receive_environment(tmp_path / "missing", timeout=3600, pane_env={})
    assert info.value.cause == "connect_failed"


def test_the_client_refuses_a_path_that_is_not_a_socket(tmp_path: Path) -> None:
    plain = tmp_path / "file"
    plain.write_text("x")
    with pytest.raises(HandoffFailed) as info:
        receive_environment(plain, timeout=3600, pane_env={})
    assert info.value.cause == "connect_failed"


def test_the_client_deadline_is_the_handoff_timeout_plus_60_seconds() -> None:
    assert handoff.client_deadline(30.0) == 90.0


def test_the_client_stops_reading_at_the_cap_of_an_endless_stream(tmp_path: Path) -> None:
    def endless(conn: socket.socket) -> None:
        chunk = b"x" * 65536
        while True:
            conn.sendall(chunk)  # ends with an OSError once the client closes

    server = RawServer(tmp_path, endless)
    started = time.monotonic()
    try:
        with pytest.raises(HandoffFailed) as info:
            receive_environment(server.path, timeout=60, pane_env={})
    finally:
        server.close()
    assert info.value.cause == "payload_too_large" and time.monotonic() - started < 30


def test_the_client_with_an_expired_deadline_fails_with_a_timeout(tmp_path: Path) -> None:
    server = RawServer(tmp_path, _serve_bytes(b""))
    try:
        with pytest.raises(HandoffFailed) as info:
            receive_environment(server.path, timeout=0, pane_env={})
    finally:
        server.close()
    assert info.value.cause == "timeout"


def test_the_client_refuses_a_socket_owned_by_another_user(tmp_path: Path) -> None:
    server = RawServer(tmp_path, _serve_bytes(encode_payload({"A": "1"})))
    try:
        with pytest.raises(HandoffFailed) as info:
            receive_environment(
                server.path, timeout=20, pane_env={}, gateway=gw(WrapGateway(uid=os.getuid() + 1))
            )
    finally:
        server.close()
    assert info.value.cause == "connect_failed"
