"""``launch|resume --placement here``: the supervisor runs in the launching process, in the
foreground, with the terminal as ``exec``'s stdio (process tests against the fake
claude-session; the terminal is a pty the test creates, never the user's)."""

from __future__ import annotations

import json
import os
import pty
import signal
import subprocess
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from agent_spawn import procs
from tests import schema_registry
from tests.conftest import Isolation
from tests.fakebin.harness import Fakebin
from tests.launch_support import Env, script_done_run, script_hung_exec, script_live_run
from tests.procguard import ProcTracker

LAUNCH = ("launch", "--mode", "interactive", "--skill", "do-it", "--placement", "here", "--json")


class Terminal:
    """A pty whose slave is the launcher's stdin and stdout; a thread drains the master so
    nothing blocks on a full buffer."""

    def __init__(self) -> None:
        self.master, self.slave = pty.openpty()
        self.chunks: list[bytes] = []
        self.thread = threading.Thread(target=self._drain, daemon=True)
        self.thread.start()

    def _drain(self) -> None:
        while True:
            try:
                data = os.read(self.master, 4096)
            except OSError:
                return
            if not data:
                return
            self.chunks.append(data)

    def text(self) -> str:
        return b"".join(self.chunks).decode(errors="replace")

    def close(self) -> None:
        for fd in (self.slave, self.master):
            try:
                os.close(fd)
            except OSError:
                pass
        self.thread.join(timeout=5)


class HereRun:
    """One launcher process on a pty, tracked so the test's cleanup kills exactly it."""

    def __init__(
        self,
        env: Env,
        tracker: ProcTracker,
        *args: str,
        extra_env: Mapping[str, str] | None = None,
    ) -> None:
        self.env = env
        self.terminal = Terminal()
        self.proc = tracker.popen(
            env.argv(*args),
            env=env.env(extra_env),
            stdin=self.terminal.slave,
            stdout=self.terminal.slave,
            stderr=subprocess.PIPE,
        )
        os.close(self.terminal.slave)  # the child has its own copy
        self.terminal.slave = os.open(os.devnull, os.O_RDONLY)  # a harmless fd for close()

    def agent_row(self) -> dict[str, Any] | None:
        agents = self.env.agents()
        return agents[0] if agents else None

    def wait_for_exec(self, fakebin: Fakebin) -> dict[str, Any]:
        """Block until the supervisor recorded ``exec`` and the fake reached its hang (its
        handlers are installed). Returns the agent row."""
        holder: dict[str, Any] = {}

        def recorded() -> bool:
            row = self.agent_row()
            if row is not None and row.get("exec_proc") is not None:
                holder["row"] = row
                return True
            return False

        self.env.wait_for(recorded, "exec to be recorded")
        row: dict[str, Any] = holder["row"]
        fakebin.wait_for_hang(row["exec_proc"].pid)
        return row

    def signal(self, signum: int) -> None:
        """Signal exactly the launcher (the PID this test started)."""
        assert self.proc.poll() is None, "the launcher already exited"
        os.kill(self.proc.pid, signum)

    def finish(self, timeout: float = 60.0) -> tuple[int, str, str]:
        _out, err = self.proc.communicate(timeout=timeout)
        self.terminal.close()
        return self.proc.returncode, self.terminal.text(), err.decode()


@pytest.fixture
def env(fakebin: Fakebin, isolation_env: Isolation) -> Env:
    return Env(fakebin, isolation_env)


def last_json(text: str) -> Any:
    """The last JSON document on the terminal (``exec``'s own output comes first)."""
    decoder = json.JSONDecoder()
    found = None
    index = 0
    while index < len(text):
        if text[index] == "{":
            try:
                found, end = decoder.raw_decode(text, index)
                index = end
                continue
            except ValueError:
                pass
        index += 1
    assert found is not None, text
    return found


def test_here_runs_the_supervisor_in_the_launcher_and_writes_the_result_file(
    env: Env, fakebin: Fakebin, proc_tracker: ProcTracker, tmp_path: Path
) -> None:
    script_done_run(fakebin)
    result_file = tmp_path / "result" / "r.json"
    result_file.parent.mkdir()
    run = HereRun(env, proc_tracker, *LAUNCH, "--result-file", str(result_file))
    code, terminal, _err = run.finish()
    assert code == 0, terminal
    body = json.loads(result_file.read_text())
    schema_registry.validate(body, "agent-spawn/cli/launch.json")
    assert oct(result_file.stat().st_mode & 0o777) == "0o600"
    assert body["outcome"] == "done" and body["exit_code"] == 0
    assert last_json(terminal) == body  # the normal output follows once `exec` ended
    row = env.agent(body["agent"]["agent_id"])
    assert row["status"] == "parked" and row["park_reason"] == "done"
    assert row["placement"] == "here" and row["mode"] == "interactive"
    # The supervisor is the launcher's own process; there is no window.
    assert row["supervisor"].pid == run.proc.pid
    assert row["supervisor"] == row["launcher"]
    assert body["agent"]["window"] is None
    assert body["agent"]["placement"] == "here"
    # `exec` ran on the launcher's terminal (a pty: the interactive fake insists on a tty), and
    # no detached supervisor and no window were started.
    execs = [i for i in fakebin.invocations("claude-session") if i.argv[0] == "exec"]
    assert len(execs) == 1 and execs[0].stdin_is_tty
    assert not (env.state_dir / "supervise").exists() or not list(
        (env.state_dir / "supervise").glob("*.log")
    )
    assert fakebin.invocations("termctl") == [] and fakebin.invocations("tmux") == []


def test_here_stores_the_implied_wait(
    env: Env, fakebin: Fakebin, proc_tracker: ProcTracker
) -> None:
    script_done_run(fakebin)
    run = HereRun(env, proc_tracker, *LAUNCH)
    code, terminal, _err = run.finish()
    assert code == 0, terminal
    row = env.agents()[0]
    assert row["settings"]["settings"]["wait"] is True


def test_here_wait_timeout_exits_2_and_writes_the_error_to_the_result_file(
    env: Env, fakebin: Fakebin, proc_tracker: ProcTracker, tmp_path: Path
) -> None:
    result_file = tmp_path / "r.json"
    run = HereRun(
        env, proc_tracker, *LAUNCH, "--wait-timeout", "5s", "--result-file", str(result_file)
    )
    code, terminal, _err = run.finish()
    assert code == 2, terminal
    body = json.loads(result_file.read_text())
    schema_registry.validate(body, "agent-spawn/cli/error.json")
    assert body["error"]["code"] == "usage" and "wait-timeout" in body["error"]["message"]
    assert env.agents() == []
    assert [i for i in fakebin.invocations("claude-session") if i.argv[0] == "start"] == []


def test_here_without_a_terminal_exits_2(env: Env) -> None:
    done = env.run(*LAUNCH)  # stdin is /dev/null
    assert done.returncode == 2
    assert json.loads(done.stdout)["error"]["code"] == "usage"
    assert env.agents() == []


def _hang_and_wait(
    env: Env, fakebin: Fakebin, proc_tracker: ProcTracker, *extra: str
) -> tuple[HereRun, dict[str, Any]]:
    script_live_run(fakebin)
    script_hung_exec(fakebin)
    run = HereRun(env, proc_tracker, *LAUNCH, *extra)
    return run, run.wait_for_exec(fakebin)


@pytest.mark.parametrize(
    ("signum", "end_reason"),
    [(signal.SIGTERM, "forwarded_term"), (signal.SIGHUP, "forwarded_hup")],
)
def test_here_forwards_sigterm_and_sighup_to_exec(
    env: Env,
    fakebin: Fakebin,
    proc_tracker: ProcTracker,
    tmp_path: Path,
    signum: signal.Signals,
    end_reason: str,
) -> None:
    result_file = tmp_path / "r.json"
    run, row = _hang_and_wait(
        env, fakebin, proc_tracker, "--result-file", str(result_file)
    )
    assert row["supervisor"].pid == run.proc.pid
    run.signal(signum)
    code, terminal, _err = run.finish()
    assert code == 15, terminal  # the outcome code of the killed run, never 143 or 130
    body = json.loads(result_file.read_text())
    schema_registry.validate(body, "agent-spawn/cli/launch.json")
    assert body["outcome"] == "killed" and body["end_reason"] == end_reason
    agent = env.agent(body["agent"]["agent_id"])
    assert agent["status"] == "parked" and agent["park_reason"] == "killed"
    assert agent["needs_attention"] is None
    # `exec` received exactly the forwarded signal, once, from the launcher.
    received = [
        s for i in fakebin.invocations("claude-session") if i.argv[0] == "exec" for s in i.signals
    ]
    assert [s for s in received if s != signal.SIGCONT] == [int(signum)]


def test_here_ignores_sigint_and_sigquit_while_exec_runs(
    env: Env, fakebin: Fakebin, proc_tracker: ProcTracker, tmp_path: Path
) -> None:
    result_file = tmp_path / "r.json"
    run, _row = _hang_and_wait(env, fakebin, proc_tracker, "--result-file", str(result_file))
    run.signal(signal.SIGINT)  # the terminal delivers it to `claude` itself
    run.signal(signal.SIGQUIT)
    assert run.proc.poll() is None  # signals to a process are handled in order: still running
    run.signal(signal.SIGTERM)
    code, terminal, _err = run.finish()
    assert code == 15, terminal
    body = json.loads(result_file.read_text())
    # Had SIGINT been forwarded, the first signal `exec` saw would have ended the run
    # `forwarded_int`: only the first signal counts.
    assert body["end_reason"] == "forwarded_term"


def test_here_exec_that_ends_not_started_writes_the_error_to_the_result_file(
    env: Env, fakebin: Fakebin, proc_tracker: ProcTracker, tmp_path: Path
) -> None:
    fakebin.script("claude-session", "exec", outcome="not-started", end_reason="canceled")
    fakebin.script("claude-session", "status", outcome="not-started", end_reason="canceled")
    result_file = tmp_path / "r.json"
    run = HereRun(env, proc_tracker, *LAUNCH, "--result-file", str(result_file))
    code, terminal, _err = run.finish()
    assert code == 12, terminal
    body = json.loads(result_file.read_text())
    schema_registry.validate(body, "agent-spawn/cli/error.json")
    assert body["error"]["code"] == "launch_failed"
    agent = env.agents()[0]
    assert agent["status"] == "parked" and agent["park_reason"] == "killed"


def test_attach_reports_no_window_for_a_here_agent(
    env: Env, fakebin: Fakebin, proc_tracker: ProcTracker
) -> None:
    run, row = _hang_and_wait(env, fakebin, proc_tracker)
    agent_id = row["agent_id"]
    done = env.run("attach", agent_id, "--json")
    assert done.returncode == 12
    assert json.loads(done.stdout)["error"]["code"] == "no_window"
    run.signal(signal.SIGTERM)
    code, _terminal, _err = run.finish()
    assert code == 15


def test_the_supervisor_identity_is_verified_before_a_signal_is_forwarded(
    env: Env, fakebin: Fakebin, proc_tracker: ProcTracker, tmp_path: Path
) -> None:
    """``exec``'s recorded identity (PID and start time) is what the launcher signals."""
    run, row = _hang_and_wait(env, fakebin, proc_tracker, "--result-file", str(tmp_path / "r"))
    exec_proc = row["exec_proc"]
    assert procs.read_identity(exec_proc.pid) == exec_proc
    run.signal(signal.SIGTERM)
    run.finish()


def test_here_resume_runs_in_process_and_never_inherits_here(
    env: Env, fakebin: Fakebin, proc_tracker: ProcTracker, tmp_path: Path
) -> None:
    script_done_run(fakebin)
    first = HereRun(env, proc_tracker, *LAUNCH)
    assert first.finish()[0] == 0
    agent_id = env.agents()[0]["agent_id"]
    result_file = tmp_path / "r.json"
    second = HereRun(
        env, proc_tracker, "resume", agent_id, "--placement", "here", "--json",
        "--result-file", str(result_file),
    )  # fmt: skip
    code, terminal, _err = second.finish()
    assert code == 0, terminal
    body = json.loads(result_file.read_text())
    schema_registry.validate(body, "agent-spawn/cli/resume.json")
    row = env.agent(agent_id)
    assert row["supervisor"].pid == second.proc.pid and row["placement"] == "here"
    # A resume that does not say `here` wants a window (termctl), never the stored placement.
    dry = env.run("resume", agent_id, "--mode", "interactive", "--dry-run", "--json")
    plan = json.loads(dry.stdout)
    assert plan["agent_settings"]["placement"] == "window", dry.stdout


def test_an_interrupt_before_exec_is_spawned_follows_the_launcher_rows(
    env: Env, fakebin: Fakebin, monkeypatch: pytest.MonkeyPatch
) -> None:
    """In process, with the terminal faked: SIGTERM between the start and the spawn of ``exec``
    cancels the pending run, parks ``killed`` without attention and exits 143 (one process
    parks, no supervisor is left recorded)."""
    from agent_spawn import cli, launch  # noqa: PLC0415
    from agent_spawn.supervise import Supervisor  # noqa: PLC0415

    def interrupted(self: Supervisor, row: dict[str, Any]) -> Any:
        signal.raise_signal(signal.SIGTERM)  # the launcher's handler raises its interrupt
        raise AssertionError("the interrupt must propagate")

    monkeypatch.setattr(Supervisor, "supervise", interrupted)
    monkeypatch.setattr(launch, "_tty_pair", lambda: True)
    from tests.session_backend_support import FakebinGateway  # noqa: PLC0415

    monkeypatch.setattr("agent_spawn.gateway._default", FakebinGateway(fakebin))
    monkeypatch.setenv("PATH", fakebin.env()["PATH"])
    saved = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT)}
    try:
        inv = cli.parse_invocation([*LAUNCH])
        outcome = launch.launch(inv)
    finally:
        for sig, handler in saved.items():
            signal.signal(sig, handler)
    assert outcome.exit_code == 143
    row = env.agents()[0]
    assert row["status"] == "parked" and row["park_reason"] == "killed"
    assert row["needs_attention"] is None and row["supervisor"] is None
    assert [i for i in fakebin.invocations("claude-session") if i.argv[0] == "exec"] == []
    assert [i for i in fakebin.invocations("claude-session") if i.argv[0] == "stop"]
