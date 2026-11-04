"""Process identity parsing and verified signaling, against the fake gateway."""

from __future__ import annotations

import calendar
import errno
import os
import signal
import socket
import subprocess
import time
from collections.abc import Iterator, Sequence
from pathlib import Path

import pytest

from agent_spawn import procs
from agent_spawn.gateway import Gateway, SystemGateway, decode
from agent_spawn.procs import ProcessIdentity, ProcessIdentityError, parse_stat
from tests.fake_gateway import BOOT, DARWIN_BOOTTIME, FakeGateway, stat_line


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


def test_pid_reused_between_check_and_send_receives_nothing() -> None:
    """The recorded process dies and the PID is reused after the last check, before the send."""
    gw = FakeGateway()
    gw.add(65, start=1000)

    def swap(pid: int) -> None:
        gw.add(pid, start=5000)  # reuse: the held pidfd stays bound to the old process

    identity = ProcessIdentity(65, 1000, BOOT)
    handle = procs.open_verified(identity, gateway=gw)
    assert handle is not None
    gw.before_send_signal = swap
    assert not handle.send(signal.SIGTERM)
    handle.close()
    assert gw.signals == []
    assert gw.procs[65].start == 5000  # the swap really happened


def test_reuse_before_pidfd_open_is_detected_by_the_start_time() -> None:
    gw = FakeGateway()
    gw.add(66, start=9999)  # already the new process when the pidfd is opened
    assert procs.open_verified(ProcessIdentity(66, 1000, BOOT), gateway=gw) is None
    assert gw.signals == []


def test_pid_reused_after_pidfd_open_but_before_the_read_is_detected() -> None:
    gw = FakeGateway()
    gw.add(67, start=1000)

    def reuse_after_open(pid: int) -> None:
        gw.add(pid, start=4242)  # the pidfd is bound to the original, /proc shows the new one

    gw.after_pidfd_open = reuse_after_open
    assert procs.open_verified(ProcessIdentity(67, 1000, BOOT), gateway=gw) is None
    assert gw.procs[67].start == 4242
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
            return subprocess.CompletedProcess(args, 0, DARWIN_BOOTTIME, "")
        if args[0] == "ps" and args[-1] == "321":
            return subprocess.CompletedProcess(args, 0, "Mon Nov  3 10:00:00 2026 Ss\n", "")
        if args[0] == "ps" and args[-1] == "322":
            return subprocess.CompletedProcess(args, 0, "Mon Nov  3 10:00:00 2026 Z+\n", "")
        return subprocess.CompletedProcess(args, 1, "", "")

    gw.run_results = {"sysctl": fake, "ps": fake}
    gw.add(321)  # the fake kill knows this PID; 323 is unknown, so ps failing means gone
    info = procs.inspect(321, gateway=gw)
    assert info is not None and info.alive and info.identity.boot_id == "1700000000"
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
        {"pid": 2**31, "start_time": 1, "boot_id": "b"},
        {"pid": -3, "start_time": 1, "boot_id": "b"},
        {"pid": 1, "start_time": -1, "boot_id": "b"},
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


def test_peer_credentials_real_socketpair() -> None:
    import os

    a, b = socket.socketpair()
    with a, b:
        creds = SystemGateway().peer_credentials(a)
    assert (creds.pid, creds.uid) == (os.getpid(), os.getuid())


# ---- review round 1 ----------------------------------------------------------------------


def test_pidfd_einval_for_a_pid_that_now_names_a_thread_reads_as_gone() -> None:
    gw = FakeGateway(pidfd_error=OSError(errno.EINVAL, "Invalid argument"))
    gw.add(70, tgid=69, start=1000)  # PID 70 is now a non-leader thread of 69
    identity = ProcessIdentity(70, 1000, BOOT)
    assert procs.open_verified(identity, gateway=gw) is None
    assert procs.forward_signal(identity, signal.SIGTERM, gateway=gw).reason == "gone"
    assert procs.wait_for_exit(identity, 1.0, gateway=gw)
    assert gw.signals == [] and gw.killed_by_pid == []


def test_pidfd_einval_for_a_vanished_pid_reads_as_gone() -> None:
    gw = FakeGateway(pidfd_error=OSError(errno.EINVAL, "Invalid argument"))
    assert procs.open_verified(ProcessIdentity(71, 1000, BOOT), gateway=gw) is None


def test_pidfd_einval_for_a_thread_group_leader_still_raises() -> None:
    gw = FakeGateway(pidfd_error=OSError(errno.EINVAL, "Invalid argument"))
    gw.add(72, start=1000)
    with pytest.raises(OSError) as caught:
        procs.forward_signal(ProcessIdentity(72, 1000, BOOT), signal.SIGTERM, gateway=gw)
    assert caught.value.errno == errno.EINVAL
    assert gw.signals == []


def test_forwarder_retries_a_signal_whose_forward_failed() -> None:
    gw = FakeGateway(pidfd_error=PermissionError(errno.EPERM, "seccomp"))
    gw.add(73, start=1000)
    forwarder = procs.SignalForwarder(ProcessIdentity(73, 1000, BOOT), gateway=gw)
    with pytest.raises(PermissionError):
        forwarder.forward(signal.SIGHUP)
    gw.pidfd_error = None
    second = forwarder.forward(signal.SIGHUP)
    assert second is not None and second.sent
    assert forwarder.forward(signal.SIGHUP) is None  # a completed forward is not repeated
    assert gw.signals == [(73, signal.SIGHUP)]


def test_forwarder_does_not_repeat_a_forward_that_found_the_process_gone() -> None:
    gw = FakeGateway()
    forwarder = procs.SignalForwarder(ProcessIdentity(74, 1000, BOOT), gateway=gw)
    first = forwarder.forward(signal.SIGTERM)
    assert first is not None and first.reason == "gone"
    assert forwarder.forward(signal.SIGTERM) is None


def _darwin(ps_out: str, ps_rc: int = 0, boottime: str = DARWIN_BOOTTIME) -> FakeGateway:
    gw = FakeGateway(platform_name="darwin")

    def fake(argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
        args = list(argv)
        if args[0] == "sysctl":
            return subprocess.CompletedProcess(args, 0, boottime, "")
        return subprocess.CompletedProcess(args, ps_rc, ps_out, "")

    gw.run_results = {"sysctl": fake, "ps": fake}
    return gw


@pytest.fixture
def restore_tz() -> Iterator[None]:
    old = os.environ.get("TZ")
    try:
        yield
    finally:
        if old is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = old
        time.tzset()


def test_darwin_start_time_is_utc_whatever_the_callers_tz(restore_tz: None) -> None:
    seen = []
    for tz in ("UTC0", "EST5EDT", "Asia/Kolkata"):
        os.environ["TZ"] = tz
        time.tzset()
        gw = _darwin("Mon Nov  3 10:00:00 2026 Ss\n")
        info = procs.inspect(321, gateway=gw)
        assert info is not None
        seen.append(info.identity.start_time)
        ps_env = gw.run_envs[gw.run_calls.index(["ps", "-o", "lstart=,stat=", "-p", "321"])]
        assert ps_env is not None and ps_env["TZ"] == "UTC0"
    assert seen == [calendar.timegm((2026, 11, 3, 10, 0, 0, 0, 0, 0))] * 3


def test_darwin_boot_id_is_the_boottime_seconds_without_the_local_date() -> None:
    gw = _darwin("")
    assert procs.read_boot_id(gateway=gw) == "1700000000"


@pytest.mark.parametrize("out", ["", "garbage\n", "{ usec = 5 }\n", "{ sec = x, usec = 1 }\n"])
def test_darwin_unparseable_boottime_raises(out: str) -> None:
    gw = _darwin("", boottime=out)
    with pytest.raises(ProcessIdentityError):
        procs.read_boot_id(gateway=gw)


def test_darwin_ps_and_sysctl_run_with_a_timeout() -> None:
    gw = _darwin("Mon Nov  3 10:00:00 2026 Ss\n")
    procs.inspect(321, gateway=gw)
    assert gw.run_timeouts and all(t is not None and t > 0 for t in gw.run_timeouts)


def test_darwin_ps_failure_for_a_missing_pid_is_none() -> None:
    gw = _darwin("", ps_rc=1)  # the fake kill raises ProcessLookupError for unknown PIDs
    assert procs.inspect(330, gateway=gw) is None


def test_darwin_ps_failure_for_an_existing_pid_raises() -> None:
    gw = _darwin("", ps_rc=1)
    gw.add(331)
    with pytest.raises(ProcessIdentityError):
        procs.inspect(331, gateway=gw)


def test_darwin_ps_timeout_raises_an_identity_error() -> None:
    gw = _darwin("")

    def hang(argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(list(argv), 10)

    gw.run_results["ps"] = hang
    with pytest.raises(ProcessIdentityError):
        procs.inspect(332, gateway=gw)


@pytest.mark.parametrize(("threads", "alive"), [(1, False), (2, True), (5, True)])
def test_a_zombie_leader_is_dead_only_once_every_thread_exited(threads: int, alive: bool) -> None:
    gw = FakeGateway()
    gw.add(75, start=1000, state="Z", threads=threads)
    identity = ProcessIdentity(75, 1000, BOOT)
    assert procs.is_alive(identity, gateway=gw) is alive
    info = procs.inspect(75, gateway=gw)
    assert info is not None and info.alive is alive
    handle = procs.open_verified(identity, gateway=gw)
    assert (handle is not None) is alive
    if handle is not None:
        handle.close()
    assert procs.forward_signal(identity, signal.SIGHUP, gateway=gw).sent is alive
    assert procs.wait_for_exit(identity, 0.2, gateway=gw) is not alive
    assert gw.open_fds() == 0


def test_a_zombie_whose_status_vanished_is_dead() -> None:
    gw = FakeGateway()
    gw.add(76, start=1000, state="Z", threads=3, has_status=False)
    assert not procs.is_alive(ProcessIdentity(76, 1000, BOOT), gateway=gw)


def test_a_pid_out_of_range_is_gone_without_opening_a_pidfd() -> None:
    gw = FakeGateway()
    opened: list[int] = []
    gw.on_pidfd_open = opened.append
    for pid in (0, -1):
        assert (
            procs.forward_signal(ProcessIdentity(pid, 1, BOOT), signal.SIGTERM, gateway=gw).reason
            == "gone"
        )
    assert opened == []


@pytest.mark.parametrize(
    ("argv", "env"),
    [
        (["/bin/echo", "a\x00b"], None),
        (["/bin/ec\x00ho"], None),
        (["/bin/echo"], {"A=B": "1"}),
        (["/bin/echo"], {"": "1"}),
        (["/bin/echo"], {"A\x00": "1"}),
        (["/bin/echo"], {"A": "1\x002"}),
    ],
)
def test_spawn_detached_rejects_nul_and_bad_env_keys_before_forking(
    argv: list[str], env: dict[str, str] | None
) -> None:
    gw = FakeGateway()  # pipe/fork are not configured: reaching them would raise otherwise
    with pytest.raises(ValueError):
        procs.spawn_detached(argv, env=env, gateway=gw)


def _open_fd_count() -> int:
    return len(os.listdir("/proc/self/fd"))


@pytest.mark.skipif(not Path("/proc/self/fd").is_dir(), reason="needs /proc")
def test_a_failing_second_pipe_closes_the_first_pipe_and_the_log(tmp_path: Path) -> None:
    gw = FakeGateway(pipe_plan=[None, OSError(errno.EMFILE, "Too many open files")])
    before = _open_fd_count()
    with pytest.raises(OSError) as caught:
        procs.spawn_detached(["/bin/echo"], log_path=str(tmp_path / "log"), gateway=gw)
    assert caught.value.errno == errno.EMFILE
    assert gw.open_fds() == 0  # both ends of the first pipe were closed
    assert _open_fd_count() == before  # and the log descriptor
    assert gw.sigmask_calls == []  # nothing was blocked: the fork was never reached


def test_wait_for_exit_on_a_live_fake_times_out() -> None:
    gw = FakeGateway()
    gw.add(77, start=1000)
    started = gw.now
    assert not procs.wait_for_exit(ProcessIdentity(77, 1000, BOOT), 1.0, gateway=gw)
    assert 1.0 <= gw.now - started < 2.0
    assert gw.open_fds() == 0


def test_fake_kill_and_close_behave_like_the_real_ones() -> None:
    gw = FakeGateway()
    with pytest.raises(ProcessLookupError):
        gw.kill(99, signal.SIGTERM)
    with pytest.raises(OSError) as caught:
        gw.close(12345)
    assert caught.value.errno == errno.EBADF
