"""``SystemGateway.run``: stdin, the per-stream output cap and the timeout, with real children."""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from agent_spawn import gateway as gw
from agent_spawn import procs
from tests.procguard import ProcTracker


def run(tmp_path: Path, code: str, **kwargs: object) -> subprocess.CompletedProcess[str]:
    env = {"PATH": os.environ.get("PATH", ""), "HOME": str(tmp_path)}
    return gw.SystemGateway().run(
        [sys.executable, "-I", "-c", code],
        env=env,
        cwd=str(tmp_path),
        **kwargs,  # type: ignore[arg-type]
    )


def test_output_and_exit_code_are_returned(tmp_path: Path) -> None:
    done = run(tmp_path, "import sys; print('o'); print('e', file=sys.stderr); sys.exit(3)")
    assert (done.returncode, done.stdout, done.stderr) == (3, "o\n", "e\n")


def test_a_run_without_stdin_gets_dev_null_not_the_callers_stdin(tmp_path: Path) -> None:
    code = "import os; print(os.path.samestat(os.fstat(0), os.stat('/dev/null')))"
    assert run(tmp_path, code).stdout == "True\n"


def test_stdin_text_is_delivered_even_when_large(tmp_path: Path) -> None:
    text = "x" * (2 * 1024 * 1024)
    done = run(tmp_path, "import sys; print(len(sys.stdin.read()))", stdin=text, timeout=30)
    assert done.stdout == f"{len(text)}\n"


def test_a_child_that_ignores_stdin_does_not_block(tmp_path: Path) -> None:
    done = run(tmp_path, "pass", stdin="y" * (1024 * 1024), timeout=30)
    assert done.returncode == 0


@pytest.mark.parametrize("stream", ["stdout", "stderr"])
def test_a_flooding_child_is_killed_and_raises(tmp_path: Path, stream: str) -> None:
    pid_file = tmp_path / "pid"
    code = (
        "import os, sys\n"
        f"open({str(pid_file)!r}, 'w').write(str(os.getpid()))\n"
        f"out = sys.{stream}.buffer\n"
        "while True:\n"
        "    out.write(b'x' * 65536)\n"
        "    out.flush()\n"
    )
    start = time.monotonic()
    with pytest.raises(gw.OutputTooLarge):
        run(tmp_path, code, timeout=60)
    assert time.monotonic() - start < 30
    pid = int(pid_file.read_text())
    with pytest.raises(ProcessLookupError):  # killed and reaped by run itself
        os.kill(pid, 0)


def test_output_exactly_at_the_cap_is_kept(tmp_path: Path) -> None:
    code = f"import sys; sys.stdout.write('x' * {gw.MAX_RUN_OUTPUT})"
    assert len(run(tmp_path, code, timeout=60).stdout) == gw.MAX_RUN_OUTPUT


def test_a_timeout_returns_even_when_a_grandchild_holds_the_pipes(
    tmp_path: Path, proc_tracker: ProcTracker
) -> None:
    pid_file = tmp_path / "grandchild"
    code = (
        "import subprocess, time\n"
        f"p = subprocess.Popen(['sleep', '60'])\n"
        f"open({str(pid_file)!r}, 'w').write(str(p.pid))\n"
        "time.sleep(60)\n"
    )
    start = time.monotonic()
    with pytest.raises(gw.TimeoutExpired):
        run(tmp_path, code, timeout=1.5)
    # The grandchild sleeps 60 s; returning in under half of that shows run did not wait for it.
    assert time.monotonic() - start < 30
    handle = procs.identify(int(pid_file.read_text()))
    if handle is not None:  # record the grandchild so the guard kills exactly it
        with handle:
            proc_tracker.record(handle.identity)


def test_an_interrupted_run_sends_sigterm_first(tmp_path: Path) -> None:
    import signal

    seen = tmp_path / "seen"
    ready = tmp_path / "ready"
    code = (
        "import signal, sys, time\n"
        f"def term(*a):\n    open({str(seen)!r}, 'w').write('term'); sys.exit(0)\n"
        "signal.signal(signal.SIGTERM, term)\n"
        f"open({str(ready)!r}, 'w').write('x')\n"
        "time.sleep(60)\n"
    )

    class Interrupt(BaseException):
        pass

    def fire(signum: int, frame: object) -> None:
        raise Interrupt

    main_ident = threading.get_ident()
    stop = threading.Event()

    def deliver_once_ready() -> None:
        # Interrupt only once the child exists and traps SIGTERM, never earlier.
        deadline = time.monotonic() + 30
        while not stop.is_set() and time.monotonic() < deadline:
            if ready.exists():
                signal.pthread_kill(main_ident, signal.SIGALRM)
                return
            time.sleep(0.01)

    helper = threading.Thread(target=deliver_once_ready, daemon=True)
    old = signal.signal(signal.SIGALRM, fire)
    try:
        helper.start()
        with pytest.raises(Interrupt):
            gw.SystemGateway().run_on_terminal([sys.executable, "-I", "-c", code], timeout=60)
    finally:
        stop.set()
        helper.join(timeout=5)
        signal.signal(signal.SIGALRM, old)
    assert seen.read_text() == "term"


def test_run_on_terminal_captures_stdout_and_bounds_a_hung_child() -> None:
    import subprocess

    from agent_spawn.gateway import SystemGateway

    gateway = SystemGateway()
    done = gateway.run_on_terminal(["/bin/sh", "-c", "echo hi"], timeout=30)
    assert (done.returncode, done.stdout) == (0, "hi\n")
    with pytest.raises(subprocess.TimeoutExpired):
        gateway.run_on_terminal(["/bin/sleep", "30"], timeout=0.3)


def test_exec_replace_replaces_the_process_and_fails_with_oserror() -> None:
    import subprocess
    import sys

    code = (
        "from agent_spawn.gateway import SystemGateway;"
        "SystemGateway().exec_replace(['/bin/echo', 'replaced'])"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert (done.returncode, done.stdout) == (0, "replaced\n")
    from agent_spawn.gateway import SystemGateway

    with pytest.raises(OSError):
        SystemGateway().exec_replace(["/nonexistent/tmux"])


def test_run_on_terminal_passes_the_users_terminal_through() -> None:
    import pty

    code = (
        "from agent_spawn.gateway import SystemGateway;"
        "done = SystemGateway().run_on_terminal("
        "['/bin/sh', '-c', '[ -t 0 ] && [ -t 2 ] && echo tty || echo notty']);"
        "print(done.stdout.strip())"
    )
    master, slave = pty.openpty()
    try:
        done = subprocess.run(
            [sys.executable, "-c", code],
            stdin=slave,
            stderr=slave,
            stdout=subprocess.PIPE,
            text=True,
            timeout=60,
            check=False,
        )
    finally:
        os.close(slave)
        os.close(master)
    assert (done.returncode, done.stdout) == (0, "tty\n")


def test_run_on_terminal_bounds_the_output_while_reading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(gw, "MAX_RUN_OUTPUT", 256 * 1024)
    pid_file = tmp_path / "pid"
    code = (
        "import os, sys\n"
        f"open({str(pid_file)!r}, 'w').write(str(os.getpid()))\n"
        "while True:\n"
        "    sys.stdout.buffer.write(b'x' * 65536)\n"
        "    sys.stdout.buffer.flush()\n"
    )
    with pytest.raises(gw.OutputTooLarge):
        gw.SystemGateway().run_on_terminal([sys.executable, "-I", "-c", code], timeout=60)
    with pytest.raises(ProcessLookupError):  # killed and reaped
        os.kill(int(pid_file.read_text()), 0)


def test_an_interrupted_run_on_terminal_stops_the_child_politely(tmp_path: Path) -> None:
    import signal

    seen = tmp_path / "seen"
    ready = tmp_path / "ready"
    code = (
        "import signal, sys, time\n"
        f"def term(*a):\n    open({str(seen)!r}, 'w').write('term'); sys.exit(0)\n"
        "signal.signal(signal.SIGTERM, term)\n"
        f"open({str(ready)!r}, 'w').write('x')\n"
        "time.sleep(60)\n"
    )

    class Interrupt(BaseException):
        pass

    def fire(signum: int, frame: object) -> None:
        deadline = time.monotonic() + 30  # fire only once the child traps SIGTERM
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        raise Interrupt

    old = signal.signal(signal.SIGALRM, fire)
    try:
        signal.setitimer(signal.ITIMER_REAL, 0.2)
        with pytest.raises(Interrupt):
            gw.SystemGateway().run_on_terminal([sys.executable, "-I", "-c", code], timeout=60)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)
    assert seen.read_text() == "term"


def test_a_failed_exec_restores_the_signal_state(monkeypatch: pytest.MonkeyPatch) -> None:
    import signal

    def handler(signum: int, frame: object) -> None:
        pass

    def refuse(path: str, argv: list[str]) -> None:
        raise OSError("no exec")

    monkeypatch.setattr(os, "execv", refuse)
    old_int = signal.signal(signal.SIGINT, handler)
    old_term = signal.signal(signal.SIGTERM, handler)
    old_mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGUSR2})
    try:
        with pytest.raises(OSError):
            gw.SystemGateway().exec_replace(["/nonexistent/tmux"])
        assert signal.getsignal(signal.SIGINT) is handler
        assert signal.getsignal(signal.SIGTERM) is handler
        assert signal.SIGUSR2 in signal.pthread_sigmask(signal.SIG_BLOCK, ())
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, old_mask)
        signal.signal(signal.SIGINT, old_int)
        signal.signal(signal.SIGTERM, old_term)


def test_run_does_not_wait_for_a_grandchild_that_holds_the_pipes(tmp_path: Path) -> None:
    import os
    import signal

    from agent_spawn.gateway import SystemGateway

    pidfile = tmp_path / "grandchild.pid"
    script = f"sleep 60 & echo $! > {pidfile}; echo done"
    try:
        done = SystemGateway().run(["/bin/sh", "-c", script], timeout=30)
        assert (done.returncode, done.stdout) == (0, "done\n")
    finally:
        if pidfile.exists():
            try:
                os.kill(int(pidfile.read_text()), signal.SIGKILL)  # the exact sleep it started
            except ProcessLookupError:
                pass
