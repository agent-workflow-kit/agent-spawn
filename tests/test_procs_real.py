"""Process tests: real processes, real pidfds, recorded and killed by exact identity."""

from __future__ import annotations

import json
import os
import pty
import signal
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

import pytest

from agent_spawn import procs
from agent_spawn.gateway import SystemGateway
from agent_spawn.procs import ProcessIdentity
from tests import procguard
from tests.procguard import ProcTracker

SLEEP = [sys.executable, "-c", "import time; time.sleep(120)"]
IGNORE_TERM = [
    sys.executable,
    "-c",
    "import signal,time,sys; signal.signal(signal.SIGTERM, signal.SIG_IGN);"
    " print('ready', flush=True); time.sleep(120)",
]

GATEWAYS = [
    pytest.param(SystemGateway(), id="pidfd"),
    pytest.param(SystemGateway(force_syscall=True), id="syscall-fallback"),
]


def wait_until(check: Callable[[], bool], timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if check():
            return True
        time.sleep(0.02)
    return check()


def private_env(tmp_path: Path) -> dict[str, str]:
    return {
        "HOME": str(tmp_path / "home"),
        "XDG_CONFIG_HOME": str(tmp_path / "config"),
        "XDG_STATE_HOME": str(tmp_path / "state"),
        "PATH": os.environ.get("PATH", ""),
    }


def start(tracker: ProcTracker, argv: list[str], **kwargs: object) -> subprocess.Popen[bytes]:
    kwargs.setdefault("stdin", subprocess.DEVNULL)
    kwargs.setdefault("stdout", subprocess.DEVNULL)
    return tracker.popen(argv, **kwargs)


@pytest.mark.parametrize("gw", GATEWAYS)
def test_identity_pidfd_and_signal_of_a_real_child(
    gw: SystemGateway, proc_tracker: ProcTracker
) -> None:
    child = start(proc_tracker, SLEEP)
    identity = procs.read_identity(child.pid, gateway=gw)
    assert identity is not None and identity.pid == child.pid
    assert procs.is_alive(identity, gateway=gw)
    outcome = procs.forward_signal(identity, signal.SIGTERM, gateway=gw)
    assert outcome.sent
    assert child.wait(timeout=10) == -signal.SIGTERM


@pytest.mark.parametrize("gw", GATEWAYS)
def test_pidfd_is_cloexec_and_bound(gw: SystemGateway, proc_tracker: ProcTracker) -> None:
    child = start(proc_tracker, SLEEP)
    fd = gw.pidfd_open(child.pid)
    try:
        assert not os.get_inheritable(fd)
        gw.pidfd_send_signal(fd, 0)  # existence check through the pidfd
        gw.pidfd_send_signal(fd, signal.SIGKILL)
        child.wait(timeout=10)
        assert gw.poll_readable(fd, 5)
        with pytest.raises(ProcessLookupError):
            gw.pidfd_send_signal(fd, signal.SIGTERM)
    finally:
        gw.close(fd)


@pytest.mark.parametrize("gw", GATEWAYS)
def test_pidfd_open_of_a_missing_pid_raises_lookup_error(gw: SystemGateway) -> None:
    with pytest.raises(ProcessLookupError):
        gw.pidfd_open(2**22 + 12345)


def test_reused_pid_identity_is_dead_and_the_real_process_is_untouched(
    proc_tracker: ProcTracker,
) -> None:
    child = start(proc_tracker, SLEEP)
    real = procs.read_identity(child.pid)
    assert real is not None
    forged = ProcessIdentity(real.pid, real.start_time + 1, real.boot_id)
    assert not procs.is_alive(forged)
    assert not procs.forward_signal(forged, signal.SIGKILL).sent
    other_boot = ProcessIdentity(real.pid, real.start_time, "not-this-boot")
    assert not procs.forward_signal(other_boot, signal.SIGKILL).sent
    assert child.poll() is None and procs.is_alive(real)


def test_an_unreaped_zombie_is_dead(proc_tracker: ProcTracker) -> None:
    child = start(proc_tracker, [sys.executable, "-c", "pass"])
    identity = procs.read_identity(child.pid)
    assert identity is not None
    assert wait_until(lambda: not procs.is_alive(identity))  # exited, not yet reaped
    info = procs.inspect(child.pid)
    assert info is not None and info.state == "Z"  # still in the table, and dead
    assert procs.open_verified(identity) is None
    assert not procs.forward_signal(identity, signal.SIGTERM).sent


def test_a_comm_with_spaces_and_parens_in_a_real_process(proc_tracker: ProcTracker) -> None:
    child = start(
        proc_tracker,
        [
            sys.executable,
            "-c",
            "import ctypes,time; ctypes.CDLL(None).prctl(15, b'a b) (c', 0,0,0); time.sleep(120)",
        ],
    )
    assert wait_until(
        lambda: (
            (info := procs.inspect(child.pid)) is not None
            and "a b) (c" in Path(f"/proc/{child.pid}/stat").read_text()
            and info.alive
        )
    )
    identity = procs.read_identity(child.pid)
    assert identity is not None and procs.is_alive(identity)


def test_forwarding_by_exact_pid_reaches_a_child_in_its_own_group(
    tmp_path: Path, proc_tracker: ProcTracker
) -> None:
    """Spike C4: a group signal misses a child in its own process group; the PID forward hits it."""
    script = (
        "import subprocess,sys,time;"
        "c=subprocess.Popen([sys.executable,'-c','import time; time.sleep(120)'],"
        "start_new_session=True);"
        "print(c.pid,flush=True); time.sleep(120)"
    )
    parent = start(
        proc_tracker, [sys.executable, "-c", script], stdout=subprocess.PIPE, start_new_session=True
    )
    assert parent.stdout is not None
    child_pid = int(parent.stdout.readline())
    handle = procs.identify(child_pid)
    assert handle is not None
    with handle:
        proc_tracker.record(handle.identity)
        os.killpg(parent.pid, signal.SIGKILL)  # the group the parent created
        parent.wait(timeout=10)
        assert handle.is_alive()  # the group signal did not reach the child
        assert procs.forward_signal(handle.identity, signal.SIGTERM).sent
        assert handle.wait_exit(10)


def test_a_term_ignoring_child_needs_sigkill_and_the_tracker_delivers_it() -> None:
    tracker = ProcTracker()
    child = tracker.popen(IGNORE_TERM, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE)
    assert child.stdout is not None
    assert child.stdout.readline().strip() == b"ready"
    identity = tracker.identities[0]
    assert procs.forward_signal(identity, signal.SIGTERM).sent
    handle = procs.open_verified(identity)
    assert handle is not None
    with handle:
        assert not handle.wait_exit(0.3)  # ignored SIGTERM
    old = procguard.GRACE
    procguard.GRACE = 0.3
    try:
        tracker.cleanup()
    finally:
        procguard.GRACE = old
    assert not tracker.survivors
    assert child.poll() == -signal.SIGKILL


def test_wait_exit_on_a_non_child(tmp_path: Path, proc_tracker: ProcTracker) -> None:
    parent = start(
        proc_tracker,
        [
            sys.executable,
            "-c",
            "import subprocess,sys;"
            "c=subprocess.Popen([sys.executable,'-c','import time; time.sleep(1)']);"
            "print(c.pid,flush=True)",
        ],
        stdout=subprocess.PIPE,
    )
    assert parent.stdout is not None
    pid = int(parent.stdout.readline())
    handle = procs.identify(pid)
    assert handle is not None
    with handle:
        proc_tracker.record(handle.identity)
        assert not handle.wait_exit(0.05)
        assert procs.wait_for_exit(handle.identity, 10)
        assert not handle.is_alive()


# ---- double fork -------------------------------------------------------------------------

MARKER_SCRIPT = "import os,sys,time; open(sys.argv[1],'w').write(str(os.getpid())); time.sleep(120)"


def test_detached_start_reports_identity_and_execs_after_the_ack(
    tmp_path: Path, proc_tracker: ProcTracker
) -> None:
    marker = tmp_path / "marker"
    started = procs.spawn_detached([sys.executable, "-c", MARKER_SCRIPT, str(marker)])
    proc_tracker.record(started.identity)
    assert procs.is_alive(started.identity)
    time.sleep(0.3)
    assert not marker.exists()  # no exec before the acknowledgement
    started.acknowledge()
    assert wait_until(marker.exists)
    assert wait_until(lambda: marker.read_text() != "")
    assert int(marker.read_text()) == started.identity.pid  # exec keeps the PID ...
    assert procs.is_alive(started.identity)  # ... and the start time
    with pytest.raises(procs.DetachedForkError):
        started.acknowledge()


def test_detached_abort_without_ack_never_execs(tmp_path: Path, proc_tracker: ProcTracker) -> None:
    marker = tmp_path / "marker"
    started = procs.spawn_detached([sys.executable, "-c", MARKER_SCRIPT, str(marker)])
    proc_tracker.record(started.identity)
    started.abort()
    assert wait_until(lambda: not procs.is_alive(started.identity))
    assert not marker.exists()


def test_detached_launcher_death_before_ack_never_execs(
    tmp_path: Path, proc_tracker: ProcTracker
) -> None:
    marker = tmp_path / "marker"
    launcher = start(
        proc_tracker,
        [
            sys.executable,
            "-c",
            "import sys,json; from agent_spawn import procs;"
            f"s=procs.spawn_detached([sys.executable,'-c',{MARKER_SCRIPT!r},{str(marker)!r}]);"
            "print(json.dumps(s.identity.to_dict()),flush=True)",
        ],
        stdout=subprocess.PIPE,
        env=private_env(tmp_path),
    )
    assert launcher.stdout is not None
    identity = ProcessIdentity.from_dict(json.loads(launcher.stdout.readline()))
    proc_tracker.record(identity)
    launcher.wait(timeout=10)  # exits without acknowledging: the ack pipe closes
    assert wait_until(lambda: not procs.is_alive(identity))
    assert not marker.exists()


def test_detached_closes_inherited_fds_and_redirects_stdio(
    tmp_path: Path, proc_tracker: ProcTracker
) -> None:
    log = tmp_path / "supervisor.log"
    script = (
        "import os,sys;"
        "print('out',flush=True); print('err',file=sys.stderr,flush=True);"
        "print('fds',sorted(int(x) for x in os.listdir('/proc/self/fd')),flush=True);"
        "print('stdin',repr(os.read(0,10)),flush=True)"
    )
    extra = [os.pipe() for _ in range(3)]  # fds the caller holds open
    try:
        started = procs.spawn_detached([sys.executable, "-c", script], log_path=str(log))
        proc_tracker.record(started.identity)
        started.acknowledge()
        assert wait_until(lambda: "stdin" in (log.read_text() if log.exists() else ""))
    finally:
        for r, w in extra:
            os.close(r)
            os.close(w)
    lines = log.read_text().splitlines()
    assert "out" in lines and "err" in lines
    fds = next(line for line in lines if line.startswith("fds"))
    assert set(json.loads(fds[4:])) <= {0, 1, 2, 3}  # 3 is the listdir handle itself
    assert "stdin b''" in lines  # /dev/null
    assert oct(log.stat().st_mode & 0o777) == "0o600"


def test_detached_rejects_a_relative_argv0_and_a_bad_log_path(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        procs.spawn_detached(["python3", "-c", "pass"])
    with pytest.raises(OSError):
        procs.spawn_detached([sys.executable, "-c", "pass"], log_path=str(tmp_path / "no" / "log"))


def test_detached_exec_failure_is_logged_not_raised(
    tmp_path: Path, proc_tracker: ProcTracker
) -> None:
    log = tmp_path / "log"
    started = procs.spawn_detached([str(tmp_path / "missing-binary")], log_path=str(log))
    proc_tracker.record(started.identity)
    started.acknowledge()
    assert wait_until(lambda: not procs.is_alive(started.identity))
    assert "exec of" in log.read_text()


def test_detached_resets_ignored_sigpipe_and_signal_mask(
    tmp_path: Path, proc_tracker: ProcTracker
) -> None:
    out = tmp_path / "out"
    # A shell, not Python: Python re-ignores SIGPIPE at startup and would hide the inheritance.
    script = f"grep -E '^(SigBlk|SigIgn):' /proc/self/status > {out}"
    assert signal.getsignal(signal.SIGPIPE) == signal.SIG_IGN  # the caller ignores SIGPIPE
    previous = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGUSR2})
    try:
        started = procs.spawn_detached(["/bin/sh", "-c", script])
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous)
    proc_tracker.record(started.identity)
    started.acknowledge()
    assert wait_until(lambda: out.exists() and out.read_text().count("\n") == 2)
    masks = {k: int(v, 16) for k, v in (line.split(":\t") for line in out.read_text().splitlines())}
    assert masks["SigBlk"] == 0
    assert masks["SigIgn"] & (1 << (signal.SIGPIPE - 1)) == 0


LAUNCHER = """
import fcntl, json, sys, termios, time
fcntl.ioctl(0, termios.TIOCSCTTY, 0)
from agent_spawn import procs
s = procs.spawn_detached([sys.executable, "-c", "import time; time.sleep(120)"])
s.acknowledge()
print(json.dumps(s.identity.to_dict()), flush=True)
time.sleep(120)
"""


def _tty_nr(pid: int) -> int:
    text = Path(f"/proc/{pid}/stat").read_text()
    return int(text[text.rfind(")") + 2 :].split()[4])


def test_grandchild_has_no_controlling_tty_and_survives_the_hangup(
    tmp_path: Path, proc_tracker: ProcTracker
) -> None:
    """Spikes C1-C2."""
    master, slave = pty.openpty()
    try:
        launcher = start(
            proc_tracker,
            [sys.executable, "-c", LAUNCHER],
            stdin=slave,
            stdout=subprocess.PIPE,
            start_new_session=True,
            env=private_env(tmp_path),
        )
    finally:
        os.close(slave)
    assert launcher.stdout is not None
    try:
        identity = ProcessIdentity.from_dict(json.loads(launcher.stdout.readline()))
        proc_tracker.record(identity)
        assert _tty_nr(launcher.pid) != 0  # the launcher has the terminal ...
        assert _tty_nr(identity.pid) == 0  # ... the grandchild does not
        assert os.getsid(identity.pid) != os.getsid(launcher.pid)
    finally:
        os.close(master)  # hang up the terminal
    assert launcher.wait(timeout=10) == -signal.SIGHUP
    time.sleep(0.3)
    assert procs.is_alive(identity)


def test_detached_never_holds_the_callers_pipe(tmp_path: Path, proc_tracker: ProcTracker) -> None:
    """Spike C3: the caller reading the launcher's pipes sees EOF while the grandchild lives."""
    script = (
        "import sys,json; from agent_spawn import procs;"
        "s=procs.spawn_detached([sys.executable,'-c','import time; time.sleep(120)']);"
        "s.acknowledge(); print(json.dumps(s.identity.to_dict()),flush=True)"
    )
    launcher = subprocess.Popen(
        [sys.executable, "-c", script],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=private_env(tmp_path),
    )
    proc_tracker.popens.append(launcher)
    out, _ = launcher.communicate(timeout=20)  # blocks until every writer closed its end
    identity = ProcessIdentity.from_dict(json.loads(out))
    proc_tracker.record(identity)
    assert procs.is_alive(identity)


def test_real_prctl_and_peer_credentials_go_through_the_gateway() -> None:
    import socket

    gw = SystemGateway()
    assert gw.prctl(3) == 1  # PR_GET_DUMPABLE
    a, b = socket.socketpair()
    with a, b:
        creds = gw.peer_credentials(b)
    assert (creds.pid, creds.uid) == (os.getpid(), os.getuid())
