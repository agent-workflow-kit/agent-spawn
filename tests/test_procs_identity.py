"""Process identity parsing and verified signaling, against the fake gateway."""

from __future__ import annotations

import signal
import socket

import pytest

from agent_spawn import procs
from agent_spawn.gateway import Gateway, SystemGateway, decode
from agent_spawn.procs import ProcessIdentity, ProcessIdentityError, parse_stat
from tests.fake_gateway import BOOT, FakeGateway, stat_line


def _gw() -> Gateway:
    return FakeGateway()


@pytest.mark.parametrize(
    "comm",
    ["bash", "a b c", "evil) S 1 2 3 (x", ")", "(", "a)b(c", "ünï ✓", " ", "x\ny"],
)
def test_parse_stat_handles_hostile_comm(comm: str) -> None:
    info = parse_stat(stat_line(4242, comm, "S", 987654, ppid=7))
    assert (info.pid, info.comm, info.state, info.ppid, info.start_time) == (
        4242,
        comm,
        "S",
        7,
        987654,
    )


@pytest.mark.parametrize(
    "text",
    [
        "",
        "garbage",
        "12 (a) S",
        "12 a) S 1" + " 0" * 30,
        "x (a) S 1" + " 0" * 30,
        "12 (a)S 1" + " 0" * 30,
        stat_line(12, "a", "S", 5).replace(" 5 ", " -5 "),
        stat_line(12, "a", "SS", 5),
    ],
)
def test_parse_stat_rejects_malformed_lines(text: str) -> None:
    with pytest.raises(ProcessIdentityError):
        parse_stat(text)


def test_inspect_reads_identity_and_boot_id() -> None:
    gw = FakeGateway()
    gw.add(50, comm="a (b) c", start=777)
    info = procs.inspect(50, gateway=gw)
    assert info is not None
    assert info.identity == ProcessIdentity(50, 777, BOOT)
    assert info.alive


def test_missing_process_is_none_and_not_alive() -> None:
    gw = FakeGateway()
    assert procs.inspect(51, gateway=gw) is None
    assert not procs.is_alive(ProcessIdentity(51, 1, BOOT), gateway=gw)


def test_unparseable_stat_raises_instead_of_reading_as_dead() -> None:
    gw = FakeGateway()
    gw.add(52)
    gw.read_file("/proc/52/stat")
    # Override the synced line with garbage after sync by replacing the proc's comm path.
    original = gw.read_file

    def broken(path: str) -> str:
        return "junk" if path == "/proc/52/stat" else original(path)

    gw.read_file = broken  # type: ignore[method-assign]
    with pytest.raises(ProcessIdentityError):
        procs.is_alive(ProcessIdentity(52, 1000, BOOT), gateway=gw)


def test_reused_pid_with_another_start_time_is_dead() -> None:
    gw = FakeGateway()
    gw.add(60, start=2000)
    assert not procs.is_alive(ProcessIdentity(60, 1000, BOOT), gateway=gw)
    assert procs.open_verified(ProcessIdentity(60, 1000, BOOT), gateway=gw) is None
    assert gw.open_fds() == 0  # the probe pidfd is closed again


def test_other_boot_id_is_dead() -> None:
    gw = FakeGateway()
    gw.add(61, start=1000)
    assert not procs.is_alive(ProcessIdentity(61, 1000, "another-boot"), gateway=gw)


@pytest.mark.parametrize("state", ["Z", "X", "x"])
def test_zombie_and_dead_states_are_dead(state: str) -> None:
    gw = FakeGateway()
    gw.add(62, start=1000, state=state)
    identity = ProcessIdentity(62, 1000, BOOT)
    assert not procs.is_alive(identity, gateway=gw)
    assert procs.open_verified(identity, gateway=gw) is None
    outcome = procs.forward_signal(identity, signal.SIGTERM, gateway=gw)
    assert not outcome.sent
    assert gw.signals == []


def test_forward_sends_signal_through_the_pidfd_never_by_pid() -> None:
    gw = FakeGateway()
    gw.add(63, start=1000)
    outcome = procs.forward_signal(ProcessIdentity(63, 1000, BOOT), signal.SIGHUP, gateway=gw)
    assert outcome.sent
    assert gw.signals == [(63, signal.SIGHUP)]
    assert gw.killed_by_pid == []
    assert gw.open_fds() == 0


def test_sigterm_is_followed_by_sigcont() -> None:
    gw = FakeGateway()
    gw.add(64, start=1000)
    procs.forward_signal(ProcessIdentity(64, 1000, BOOT), signal.SIGTERM, gateway=gw)
    assert gw.signals == [(64, signal.SIGTERM), (64, signal.SIGCONT)]


def test_pid_reused_between_open_and_check_receives_nothing() -> None:
    """The recorded process dies and the PID is reused right after the pidfd was opened."""
    gw = FakeGateway()
    gw.add(65, start=1000)

    def swap(pid: int) -> None:
        pass

    identity = ProcessIdentity(65, 1000, BOOT)
    handle = procs.open_verified(identity, gateway=gw)
    assert handle is not None
    gw.add(65, start=5000)  # reuse: the held pidfd stays bound to the old process
    assert not handle.send(signal.SIGTERM)
    handle.close()
    assert gw.signals == []


def test_reuse_before_pidfd_open_is_detected_by_the_start_time() -> None:
    gw = FakeGateway()
    gw.add(66, start=9999)  # already the new process when the pidfd is opened
    assert procs.open_verified(ProcessIdentity(66, 1000, BOOT), gateway=gw) is None
    assert gw.signals == []


def test_pid_reused_after_pidfd_open_but_before_the_read_is_detected() -> None:
    gw = FakeGateway()
    gw.add(67, start=1000)

    def reuse_after_open(pid: int) -> None:
        pass

    # Simulate: pidfd opened for the original, then the PID is reused before we read /proc.
    original_open = gw.pidfd_open

    def open_then_reuse(pid: int) -> int:
        fd = original_open(pid)
        gw.add(67, start=4242)
        return fd

    gw.pidfd_open = open_then_reuse  # type: ignore[method-assign]
    assert procs.open_verified(ProcessIdentity(67, 1000, BOOT), gateway=gw) is None
    assert gw.open_fds() == 0
    assert gw.signals == []


def test_pidfd_failure_other_than_missing_process_propagates() -> None:
    gw = FakeGateway(pidfd_error=PermissionError(1, "denied"))
    gw.add(68)
    with pytest.raises(PermissionError):
        procs.open_verified(ProcessIdentity(68, 1000, BOOT), gateway=gw)


def test_forwarder_is_idempotent_per_signal() -> None:
    gw = FakeGateway()
    gw.add(69, start=1000)
    forwarder = procs.SignalForwarder(ProcessIdentity(69, 1000, BOOT), gateway=gw)
    first = forwarder.forward(signal.SIGTERM)
    assert first is not None and first.sent
    assert forwarder.forward(signal.SIGTERM) is None
    assert forwarder.forward(signal.SIGHUP) is not None
    assert gw.signals == [(69, signal.SIGTERM), (69, signal.SIGCONT), (69, signal.SIGHUP)]


def test_install_signal_handlers_and_restore() -> None:
    gw = FakeGateway()
    restore = procs.install_signal_handlers(
        {signal.SIGHUP: signal.SIG_IGN, signal.SIGTERM: lambda s, f: None}, gateway=gw
    )
    assert gw.handlers[signal.SIGHUP] == signal.SIG_IGN
    restore()
    assert gw.handlers[signal.SIGHUP] == 0


def test_darwin_identity_uses_ps_and_sysctl_through_the_gateway() -> None:
    import subprocess

    gw = FakeGateway(platform_name="darwin")

    def fake(argv: object) -> subprocess.CompletedProcess[str]:
        args = list(argv)  # type: ignore[call-overload]
        if args[0] == "sysctl":
            return subprocess.CompletedProcess(args, 0, "{ sec = 1700000000, usec = 0 }\n", "")
        if args[0] == "ps" and args[-1] == "321":
            return subprocess.CompletedProcess(args, 0, "Mon Nov  3 10:00:00 2026 Ss\n", "")
        if args[0] == "ps" and args[-1] == "322":
            return subprocess.CompletedProcess(args, 0, "Mon Nov  3 10:00:00 2026 Z+\n", "")
        return subprocess.CompletedProcess(args, 1, "", "")

    gw.run_results = {"sysctl": fake, "ps": fake}
    info = procs.inspect(321, gateway=gw)
    assert info is not None and info.alive and info.identity.boot_id.startswith("{ sec")
    zombie = procs.inspect(322, gateway=gw)
    assert zombie is not None and not zombie.alive
    assert procs.inspect(323, gateway=gw) is None
    handle = procs.open_verified(info.identity, gateway=gw)
    assert handle is not None  # no pidfd on macOS
    assert handle.send(signal.SIGTERM)
    assert gw.killed_by_pid == [(321, signal.SIGTERM)]


def test_identity_dict_roundtrip_and_rejects_garbage() -> None:
    identity = ProcessIdentity(5, 6, "b")
    assert ProcessIdentity.from_dict(identity.to_dict()) == identity
    for bad in (
        {},
        {"pid": 0, "start_time": 1, "boot_id": "b"},
        {"pid": True, "start_time": 1, "boot_id": "b"},
        {"pid": 1, "start_time": "1", "boot_id": "b"},
    ):
        with pytest.raises(ValueError):
            ProcessIdentity.from_dict(bad)


def test_decode_is_utf8_surrogateescape_not_locale() -> None:
    assert decode(b"caf\xc3\xa9 \xff") == "café \udcff"


def test_system_run_decodes_explicitly(tmp_path) -> None:  # type: ignore[no-untyped-def]
    done = SystemGateway().run(
        ["python3", "-c", "import sys; sys.stdout.buffer.write(b'\\xc3\\xa9\\xff')"],
        env={"LC_ALL": "C", "PATH": "/usr/bin:/bin:/usr/local/bin"},
    )
    assert done.stdout == "é\udcff"


def test_system_run_passes_stdin_as_utf8() -> None:
    done = SystemGateway().run(["cat"], stdin="é✓")
    assert done.stdout == "é✓"


def test_prctl_goes_through_the_gateway() -> None:
    fake = FakeGateway()
    fake.prctl(36, 1)
    assert fake.prctl_calls == [(36, 1, 0, 0, 0)]


def test_peer_credentials_real_socketpair() -> None:
    import os

    a, b = socket.socketpair()
    with a, b:
        creds = SystemGateway().peer_credentials(a)
    assert (creds.pid, creds.uid) == (os.getpid(), os.getuid())
