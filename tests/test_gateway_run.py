"""``SystemGateway.run``: stdin, the per-stream output cap and the timeout, with real children."""

from __future__ import annotations

import os
import subprocess
import sys
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
        f"p = subprocess.Popen(['sleep', '5'])\n"
        f"open({str(pid_file)!r}, 'w').write(str(p.pid))\n"
        "time.sleep(60)\n"
    )
    start = time.monotonic()
    with pytest.raises(gw.TimeoutExpired):
        run(tmp_path, code, timeout=1.5)
    assert time.monotonic() - start < 4
    handle = procs.identify(int(pid_file.read_text()))
    if handle is not None:  # record the grandchild so the guard kills exactly it
        with handle:
            proc_tracker.record(handle.identity)
