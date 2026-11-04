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
        "XDG_DATA_HOME": str(tmp_path / "data"),
        "XDG_CACHE_HOME": str(tmp_path / "cache"),
        "ZDOTDIR": str(tmp_path / "zdot"),
        "SHELL": "/bin/sh",
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


def test_a_term_ignoring_child_needs_sigkill_and_the_tracker_delivers_it(
    proc_tracker: ProcTracker,
) -> None:
    tracker = ProcTracker(sweep=False)
    child = tracker.popen(IGNORE_TERM, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE)
    identity = tracker.identities[0]
    proc_tracker.record(identity)  # the autouse tracker kills it if this test fails early
    old = procguard.GRACE
    try:
        assert child.stdout is not None
        assert child.stdout.readline().strip() == b"ready"
        assert procs.forward_signal(identity, signal.SIGTERM).sent
        handle = procs.open_verified(identity)
        assert handle is not None
        with handle:
            assert not handle.wait_exit(0.3)  # ignored SIGTERM
        procguard.GRACE = 0.3
    finally:
        tracker.cleanup()
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


def _fd_targets(pid: int) -> dict[int, str]:
    targets: dict[int, str] = {}
    for name in os.listdir(f"/proc/{pid}/fd"):
        try:
            targets[int(name)] = os.readlink(f"/proc/{pid}/fd/{name}")
        except FileNotFoundError:
            pass  # closed while listing
    return targets


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
    extra = [os.pipe() for _ in range(3)]  # fds the caller holds open, inheritable
    for r, w in extra:
        os.set_inheritable(r, True)
        os.set_inheritable(w, True)
    extra_targets = {os.readlink(f"/proc/self/fd/{fd}") for pair in extra for fd in pair}
    try:
        started = procs.spawn_detached([sys.executable, "-c", script], log_path=str(log))
        proc_tracker.record(started.identity)
        # Before the acknowledgement: stdio plus the acknowledgement pipe (the identity pipe
        # closes right after the report), and none of the caller's descriptors.
        assert wait_until(lambda: len(_fd_targets(started.identity.pid)) <= 4)
        before_ack = _fd_targets(started.identity.pid)
        assert set(before_ack) - {0, 1, 2} and min(set(before_ack) - {0, 1, 2}) >= 3
        assert not extra_targets & set(before_ack.values())
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
    launcher = proc_tracker.popen(  # recorded first: a timeout below still kills it
        [sys.executable, "-c", script],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=private_env(tmp_path),
    )
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


# ---- review round 1 ----------------------------------------------------------------------


def _run_launcher(
    tracker: ProcTracker, tmp_path: Path, script: str, *args: str, shell_prefix: str = ""
) -> subprocess.Popen[bytes]:
    """Run ``script`` in a fresh Python; ``shell_prefix`` redirections apply to it (via sh)."""
    argv = [sys.executable, "-c", script, *args]
    if shell_prefix:
        argv = ["/bin/sh", "-c", f'exec "$@" {shell_prefix}', "sh", *argv]
    return tracker.popen(argv, stdin=subprocess.DEVNULL, env=private_env(tmp_path))


ESCAPE_LAUNCHER = r"""
import os, random, signal, sys, time
from agent_spawn import procs
ME = os.getpid()
marker = sys.argv[1]
if os.getpgid(0) != ME:
    sys.exit(3)  # the test starts this launcher as the leader of its own group

class Boom(Exception):
    pass

def check():
    if os.getpid() != ME:  # a forked child is running the caller's code
        with open(marker, "a") as handle:
            handle.write(f"{os.getpid()}\n")
        os._exit(0)

SPAWN = procs.spawn_detached.__code__

def in_spawn(frame):
    while frame is not None:
        if frame.f_code is SPAWN:
            return True
        frame = frame.f_back
    return False

def boom(sig, frame):
    check()
    # Raise only inside spawn_detached (a raise after it returned would drop the handle),
    # in some spawns, letting others complete.
    if in_spawn(frame) and random.random() < 0.02:
        raise Boom

FD_DIR = "/proc/self/fd" if os.path.isdir("/proc/self/fd") else "/dev/fd"

def open_fds():
    return len(os.listdir(FD_DIR))

def tick(sig, frame):
    check()
    os.killpg(ME, signal.SIGUSR1)  # our own group: us and any child not yet in its session
    boom(sig, frame)

signal.signal(signal.SIGUSR1, boom)
signal.signal(signal.SIGALRM, tick)
signal.setitimer(signal.ITIMER_REAL, 0.0007, 0.0007)
baseline = open_fds()
deadline = time.monotonic() + 2.0
spawns = booms = 0
while time.monotonic() < deadline:
    try:
        started = procs.spawn_detached([sys.executable, "-c", "pass"])
        started.abort()
        spawns += 1
    except Boom:
        booms += 1
    check()
signal.setitimer(signal.ITIMER_REAL, 0, 0)
signal.signal(signal.SIGALRM, signal.SIG_IGN)
signal.signal(signal.SIGUSR1, signal.SIG_IGN)
check()
leaked = open_fds() - baseline  # before opening the counts file
with open(marker + ".counts", "w") as handle:
    handle.write(f"{spawns} {booms} {leaked}\n")
"""


def test_a_raising_signal_handler_never_runs_in_a_forked_child(
    tmp_path: Path, proc_tracker: ProcTracker
) -> None:
    marker = tmp_path / "escaped"
    launcher = proc_tracker.popen(
        [sys.executable, "-c", ESCAPE_LAUNCHER, str(marker)],
        stdin=subprocess.DEVNULL,
        env=private_env(tmp_path),
        start_new_session=True,
    )
    assert launcher.wait(timeout=60) == 0
    proc_tracker.settle(30)  # unacknowledged grandchildren exit once the launcher's pipes close
    assert not marker.exists(), f"children ran caller code: {marker.read_text()}"
    spawns, booms, leaked = map(int, (tmp_path / "escaped.counts").read_text().split())
    assert spawns > 0 and booms > 0
    assert leaked == 0, f"{leaked} descriptors leaked across {booms} interrupted starts"


STDIO_LAUNCHER = r"""
import json, sys
from agent_spawn import procs
out, log = sys.argv[1], sys.argv[2]
# $(...) redirects only the subshell's stdout: /proc/$$ is the probe shell's own table.
probe = (
    'fds=$(readlink /proc/$$/fd/0 /proc/$$/fd/1 /proc/$$/fd/2); echo "$fds" > '
    + out + ".fds; echo out; echo err >&2"
)
started = procs.spawn_detached(["/bin/sh", "-c", probe], log_path=log)
started.acknowledge()
with open(out, "w") as handle:
    handle.write(json.dumps(started.identity.to_dict()))
"""


@pytest.mark.parametrize("closed", ["0<&-", "1>&- 2>&-", "0<&- 1>&- 2>&-"])
def test_detached_start_with_the_callers_stdio_closed(
    closed: str, tmp_path: Path, proc_tracker: ProcTracker
) -> None:
    out, log = tmp_path / "identity", tmp_path / "log"
    launcher = _run_launcher(
        proc_tracker, tmp_path, STDIO_LAUNCHER, str(out), str(log), shell_prefix=closed
    )
    assert launcher.wait(timeout=30) == 0
    identity = proc_tracker.record(ProcessIdentity.from_dict(json.loads(out.read_text())))
    fds_file = Path(str(out) + ".fds")
    assert wait_until(lambda: fds_file.exists() and fds_file.read_text().count("\n") == 3)
    assert wait_until(lambda: log.exists() and "err" in log.read_text())
    assert fds_file.read_text().splitlines() == ["/dev/null", str(log), str(log)]
    assert log.read_text().splitlines() == ["out", "err"]
    assert wait_until(lambda: not procs.is_alive(identity))


def test_a_pid_that_now_names_a_thread_reads_as_gone() -> None:
    import threading

    ready, stop = threading.Event(), threading.Event()
    tids: list[int] = []

    def run() -> None:
        tids.append(threading.get_native_id())
        ready.set()
        stop.wait(30)

    helper = threading.Thread(target=run)
    helper.start()
    try:
        assert ready.wait(10)
        tid = tids[0]
        assert tid != os.getpid()
        boot = procs.read_boot_id()
        forged = ProcessIdentity(tid, 1, boot)
        assert procs.forward_signal(forged, signal.SIGTERM).reason == "gone"
        assert procs.open_verified(forged) is None
        assert procs.wait_for_exit(forged, 1.0)
        for gw in (SystemGateway(), SystemGateway(force_syscall=True)):
            assert procs.open_verified(forged, gateway=gw) is None
    finally:
        stop.set()
        helper.join(10)


HIGH_FD_LAUNCHER = r"""
import json, os, resource, sys
from agent_spawn import procs
out = sys.argv[1]
soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
r, w = os.pipe()
os.dup2(w, 70000, inheritable=True)
os.close(w)
probe = "ls /proc/$$/fd > " + out + ".after"
started = procs.spawn_detached(["/bin/sh", "-c", probe])
pid = started.identity.pid
before = sorted(int(x) for x in os.listdir(f"/proc/{pid}/fd"))
os.close(70000)
import select
eof = bool(select.select([r], [], [], 5)[0]) and os.read(r, 1) == b""
started.acknowledge()
with open(out, "w") as handle:
    handle.write(json.dumps({"identity": started.identity.to_dict(), "before": before, "eof": eof}))
"""


def test_detached_closes_an_inherited_fd_above_65536(
    tmp_path: Path, proc_tracker: ProcTracker
) -> None:
    import resource

    if resource.getrlimit(resource.RLIMIT_NOFILE)[1] <= 70001:
        pytest.skip("the hard RLIMIT_NOFILE is too low for an fd above 65536")
    out = tmp_path / "result"
    launcher = _run_launcher(proc_tracker, tmp_path, HIGH_FD_LAUNCHER, str(out))
    assert launcher.wait(timeout=30) == 0
    result = json.loads(out.read_text())
    proc_tracker.record(ProcessIdentity.from_dict(result["identity"]))
    assert 70000 not in result["before"]
    assert result["eof"]  # the grandchild did not hold the caller's pipe end
    after = Path(str(out) + ".after")
    assert wait_until(lambda: after.exists() and after.read_text() != "")
    assert "70000" not in after.read_text().split()


IGNORED_LAUNCHER = r"""
import json, signal, sys
from agent_spawn import procs
out = sys.argv[1]
for sig in (signal.SIGCHLD, signal.SIGUSR1, signal.SIGALRM):
    signal.signal(sig, signal.SIG_IGN)
probe = "grep -E '^SigIgn:' /proc/$$/status > " + out + ".sig"
started = procs.spawn_detached(["/bin/sh", "-c", probe])
started.acknowledge()
with open(out, "w") as handle:
    handle.write(json.dumps(started.identity.to_dict()))
"""


def test_detached_start_with_ignored_sigchld_and_inherited_ignores_reset(
    tmp_path: Path, proc_tracker: ProcTracker
) -> None:
    out = tmp_path / "identity"
    launcher = _run_launcher(proc_tracker, tmp_path, IGNORED_LAUNCHER, str(out))
    assert launcher.wait(timeout=30) == 0
    proc_tracker.record(ProcessIdentity.from_dict(json.loads(out.read_text())))
    sig_file = Path(str(out) + ".sig")
    assert wait_until(lambda: sig_file.exists() and sig_file.read_text() != "")
    ignored = int(sig_file.read_text().split(":")[1], 16)
    for sig in (signal.SIGCHLD, signal.SIGUSR1, signal.SIGALRM, signal.SIGPIPE):
        assert ignored & (1 << (sig - 1)) == 0, signal.Signals(sig).name


SIGCHLD_CLI = r"""
import contextlib, io, signal, sys
signal.signal(signal.SIGCHLD, signal.SIG_IGN)
from agent_spawn import cli, gateway
before = gateway.run(["/bin/false"]).returncode
with contextlib.redirect_stdout(io.StringIO()):
    try:
        cli.main(["--version"])
    except SystemExit:
        pass
after = gateway.run(["/bin/false"]).returncode
print(before, after, signal.getsignal(signal.SIGCHLD) == signal.SIG_DFL)
"""


def test_the_cli_entry_resets_an_inherited_ignored_sigchld(tmp_path: Path) -> None:
    done = subprocess.run(
        [sys.executable, "-c", SIGCHLD_CLI],
        capture_output=True,
        text=True,
        env=private_env(tmp_path),
        stdin=subprocess.DEVNULL,
        timeout=60,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    before, after, reset = done.stdout.split()
    assert before == "0"  # the failure is lost while SIGCHLD is ignored ...
    assert after == "1" and reset == "True"  # ... and reported after main() reset it


ZOMBIE_LEADER_C = r"""
#include <pthread.h>
#include <unistd.h>
static void *run(void *arg) { (void)arg; sleep(120); return 0; }
int main(void) {
    pthread_t t;
    pthread_create(&t, 0, run, 0);
    write(1, "ready\n", 6);
    pthread_exit(0);
}
"""


def test_a_zombie_leader_with_a_live_thread_is_alive(
    tmp_path: Path, proc_tracker: ProcTracker
) -> None:
    import shutil

    cc = shutil.which("cc") or shutil.which("gcc")
    if cc is None:
        pytest.skip("no C compiler")
    source, binary = tmp_path / "leader.c", tmp_path / "leader"
    source.write_text(ZOMBIE_LEADER_C)
    subprocess.run(
        [cc, "-pthread", "-o", str(binary), str(source)],
        check=True,
        env=private_env(tmp_path),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        timeout=120,
    )
    child = start(proc_tracker, [str(binary)], stdout=subprocess.PIPE)
    assert child.stdout is not None
    assert child.stdout.readline() == b"ready\n"
    identity = proc_tracker.identities[-1]
    assert wait_until(lambda: (info := procs.inspect(child.pid)) is not None and info.state == "Z")
    assert procs.is_alive(identity)  # the leader exited, its thread runs on
    handle = procs.open_verified(identity)
    assert handle is not None
    with handle:
        assert handle.is_alive()
        assert not handle.wait_exit(0.2)
    assert procs.forward_signal(identity, signal.SIGTERM).sent
    assert procs.wait_for_exit(identity, 10)
    assert child.wait(timeout=10) == -signal.SIGTERM
