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
from collections.abc import Iterator, Mapping
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
        """Close the slave, let the drain thread read everything up to the hangup (EIO), and
        close the master last, so no output is lost."""
        try:
            os.close(self.slave)
        except OSError:
            pass
        self.thread.join(timeout=10)
        try:
            os.close(self.master)
        except OSError:
            pass


class HereRun:
    """One launcher process on a pty, tracked so the test's cleanup kills exactly it."""

    def __init__(
        self,
        env: Env,
        tracker: ProcTracker,
        *args: str,
        extra_env: Mapping[str, str] | None = None,
        tty_stdin: bool = True,
        tty_stdout: bool = True,
    ) -> None:
        self.env = env
        self.terminal = Terminal()
        self.proc = tracker.popen(
            env.argv(*args),
            env=env.env(extra_env),
            stdin=self.terminal.slave if tty_stdin else subprocess.DEVNULL,
            stdout=self.terminal.slave if tty_stdout else subprocess.PIPE,
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
        out, err = self.proc.communicate(timeout=timeout)
        self.terminal.close()
        return self.proc.returncode, self.terminal.text() + (out or b"").decode(), err.decode()


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
    with env.store() as store:  # the run row records it too
        run_row = store.get_run(row["agent_id"], row["run"])
    assert run_row is not None and run_row["placement"] == "here"
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
    run, row = _hang_and_wait(env, fakebin, proc_tracker, "--result-file", str(result_file))
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


def ignored_signals(pid: int) -> set[int]:
    """The signals a process ignores (``SigIgn`` of ``/proc/<pid>/status``)."""
    for line in Path(f"/proc/{pid}/status").read_text().splitlines():
        if line.startswith("SigIgn:"):
            mask = int(line.split()[1], 16)
            return {n for n in range(1, 65) if mask & (1 << (n - 1))}
    raise AssertionError("no SigIgn line")


def test_here_ignores_sigint_and_sigquit_while_exec_runs(
    env: Env, fakebin: Fakebin, proc_tracker: ProcTracker, tmp_path: Path
) -> None:
    result_file = tmp_path / "r.json"
    run, _row = _hang_and_wait(env, fakebin, proc_tracker, "--result-file", str(result_file))
    both = {int(signal.SIGINT), int(signal.SIGQUIT)}
    # Wait for the ignore to be in effect (never an immediate poll): a signal sent before it
    # would kill (SIGQUIT) or interrupt the launcher.
    env.wait_for(lambda: both <= ignored_signals(run.proc.pid), "SIGINT and SIGQUIT ignored")
    run.signal(signal.SIGINT)  # the terminal delivers it to `claude` itself, never forwarded
    run.signal(signal.SIGQUIT)
    run.signal(signal.SIGTERM)
    code, terminal, _err = run.finish()
    assert code == 15, terminal  # still running after both, then the outcome of the SIGTERM
    body = json.loads(result_file.read_text())
    assert body["end_reason"] == "forwarded_term"
    received = [
        s for i in fakebin.invocations("claude-session") if i.argv[0] == "exec" for s in i.signals
    ]
    assert signal.SIGINT not in received and signal.SIGQUIT not in received
    assert [s for s in received if s != signal.SIGCONT] == [int(signal.SIGTERM)]


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


def test_the_forwarded_signal_goes_to_the_recorded_exec_identity(
    env: Env, fakebin: Fakebin, proc_tracker: ProcTracker, tmp_path: Path
) -> None:
    """``exec`` is recorded with the identity (PID and start time) read from ``/proc``, and the
    forwarded signal is logged for exactly that PID. (The forwarder holds the identity read when
    ``exec`` was spawned, so a corrupted stored copy cannot be made to matter deterministically.)"""
    run, row = _hang_and_wait(env, fakebin, proc_tracker, "--result-file", str(tmp_path / "r"))
    exec_proc = row["exec_proc"]
    assert procs.read_identity(exec_proc.pid) == exec_proc
    run.signal(signal.SIGTERM)
    run.finish()
    log = [json.loads(line) for line in (env.state_dir / "log.jsonl").read_text().splitlines()]
    sent = [r for r in log if r["event"] == "signal_sent"]
    assert [r["pid"] for r in sent] == [exec_proc.pid]


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


# --- the implied wait of a fork and the dry run (R2) -------------------------------------------


def _done_agent(env: Env, fakebin: Fakebin, proc_tracker: ProcTracker) -> str:
    script_done_run(fakebin)
    first = HereRun(env, proc_tracker, *LAUNCH)
    assert first.finish()[0] == 0
    return str(env.agents()[0]["agent_id"])


def test_a_fork_with_here_stores_the_implied_wait(
    env: Env, fakebin: Fakebin, proc_tracker: ProcTracker
) -> None:
    agent_id = _done_agent(env, fakebin, proc_tracker)
    fork = HereRun(env, proc_tracker, "resume", agent_id, "--fork", "--placement", "here", "--json")
    code, terminal, _err = fork.finish()
    assert code == 0, terminal
    forks = [a for a in env.agents() if a["agent_id"] != agent_id]
    assert len(forks) == 1
    assert forks[0]["settings"]["settings"]["wait"] is True
    assert forks[0]["placement"] == "here"


def test_the_dry_runs_plan_the_implied_wait_and_agree(
    env: Env, fakebin: Fakebin, proc_tracker: ProcTracker
) -> None:
    """Neither dry run needs a terminal, and launch and resume plan the same wait."""
    launch_plan = json.loads(env.run(*LAUNCH, "--dry-run").stdout)
    assert launch_plan["agent_settings"]["placement"] == "here"
    assert launch_plan["agent_settings"]["wait"] is True
    agent_id = _done_agent(env, fakebin, proc_tracker)
    for extra in ((), ("--fork",)):
        done = env.run(
            "resume", agent_id, *extra, "--mode", "interactive", "--placement", "here",
            "--dry-run", "--json",
        )  # fmt: skip
        assert done.returncode == 0, done.stdout
        settings = json.loads(done.stdout)["agent_settings"]
        assert settings["placement"] == "here" and settings["wait"] is True


# --- headless ignores --placement here (R10) ----------------------------------------------------


def test_headless_with_here_is_an_ordinary_headless_launch(env: Env, fakebin: Fakebin) -> None:
    script_done_run(fakebin)
    done = env.run(
        "launch", "--mode", "headless", "--skill", "do-it", "--placement", "here",
        "--wait", "--wait-timeout", "30s", "--json",
    )  # fmt: skip
    assert done.returncode == 0, (done.stdout, done.stderr)  # no TTY, a wait-timeout: fine
    body = json.loads(done.stdout)
    assert any("ignored" in w for w in body["warnings"])
    assert env.agents()[0]["placement"] is None


def test_a_headless_resume_with_here_ignores_the_here_checks(
    env: Env, fakebin: Fakebin, proc_tracker: ProcTracker
) -> None:
    agent_id = _done_agent(env, fakebin, proc_tracker)
    done = env.run(
        "resume", agent_id, "--mode", "headless", "--placement", "here",
        "--wait", "--wait-timeout", "30s", "--json",
    )  # fmt: skip
    assert done.returncode == 0, (done.stdout, done.stderr)


# --- tty checks on each stream (R3e) ------------------------------------------------------------


@pytest.mark.parametrize(("tty_stdin", "tty_stdout"), [(True, False), (False, True)])
def test_here_needs_a_terminal_on_both_stdin_and_stdout(
    env: Env,
    fakebin: Fakebin,
    proc_tracker: ProcTracker,
    tty_stdin: bool,
    tty_stdout: bool,
) -> None:
    run = HereRun(env, proc_tracker, *LAUNCH, tty_stdin=tty_stdin, tty_stdout=tty_stdout)
    code, terminal, _err = run.finish()
    assert code == 2, terminal
    assert env.agents() == []


@pytest.mark.parametrize(("tty_stdin", "tty_stdout"), [(True, False), (False, True)])
def test_a_here_resume_needs_a_terminal_on_both_stdin_and_stdout(
    env: Env,
    fakebin: Fakebin,
    proc_tracker: ProcTracker,
    tty_stdin: bool,
    tty_stdout: bool,
) -> None:
    agent_id = _done_agent(env, fakebin, proc_tracker)
    before = env.agent(agent_id)["status"]
    run = HereRun(
        env, proc_tracker, "resume", agent_id, "--mode", "interactive", "--placement", "here",
        "--json", tty_stdin=tty_stdin, tty_stdout=tty_stdout,
    )  # fmt: skip
    code, terminal, _err = run.finish()
    assert code == 2, terminal
    assert env.agent(agent_id)["status"] == before


# --- a refused --result-file (R3d) --------------------------------------------------------------


def test_here_with_a_result_file_inside_a_throwaway_root_exits_2_and_writes_nothing(
    env: Env, fakebin: Fakebin, proc_tracker: ProcTracker
) -> None:
    from tests.test_worktrees_launch import fake_throwaway, real_starts  # noqa: PLC0415

    throwaway = fake_throwaway(fakebin)
    inside = Path(throwaway["path"]) / "out" / "r.json"
    run = HereRun(
        env, proc_tracker, *LAUNCH, "--cwd", throwaway["path"], "--reads-third-party",
        "--result-file", str(inside),
    )  # fmt: skip
    code, terminal, _err = run.finish()
    assert code == 2, terminal
    assert env.agents() == [] and real_starts(fakebin) == []
    assert not inside.exists()


def test_here_with_a_post_start_classification_mismatch_never_writes_into_the_session_dir(
    env: Env, fakebin: Fakebin, proc_tracker: ProcTracker, tmp_path: Path
) -> None:
    csd = tmp_path / "csd"
    csd.mkdir()
    fakebin.script(
        "claude-session", "status", times=1,
        merge={"run": {"reads_third_party": True, "cwd": "/tmp/not-the-session-dir"}},
    )  # fmt: skip
    fakebin.script("claude-session", "start", merge={"csession_dir": str(csd)})
    fakebin.script("claude-session", "status", merge={"csession_dir": str(csd)})
    target = csd / "r.json"
    run = HereRun(env, proc_tracker, *LAUNCH, "--result-file", str(target))
    code, terminal, _err = run.finish()
    assert code == 12, terminal
    assert not target.exists()
    assert [i for i in fakebin.invocations("claude-session") if i.argv[0] == "exec"] == []


# --- a signal before exec, as a process (R3a) ---------------------------------------------------


@pytest.mark.parametrize(
    ("signum", "exit_code"),
    [(signal.SIGTERM, 143), (signal.SIGINT, 130), (signal.SIGHUP, 130)],
)
def test_a_signal_before_exec_cancels_the_run_and_parks_without_attention(
    env: Env,
    fakebin: Fakebin,
    proc_tracker: ProcTracker,
    tmp_path: Path,
    signum: signal.Signals,
    exit_code: int,
) -> None:
    gate = tmp_path / "gate"
    # The first `status` (after `start`, before the supervisor exists) waits at the gate; the
    # script is added before `script_done_run` so that it matches first.
    fakebin.script("claude-session", "status", times=1, gate=str(gate), outcome="done")
    script_done_run(fakebin)
    result_file = tmp_path / "r.json"
    run = HereRun(env, proc_tracker, *LAUNCH, "--result-file", str(result_file))
    fakebin.wait_for_gate(gate)
    run.signal(signum)
    gate.write_text("")
    code, terminal, _err = run.finish()
    assert code == exit_code, terminal
    body = json.loads(result_file.read_text())
    schema_registry.validate(body, "agent-spawn/cli/error.json")
    assert body["error"]["code"] == "launch_failed"
    assert body["error"]["details"] == {"cause": "interrupted"}
    (row,) = env.agents()
    assert row["status"] == "parked" and row["park_reason"] == "killed"
    assert row["needs_attention"] is None and row["supervisor"] is None
    calls = [i.argv[0] for i in fakebin.invocations("claude-session")]
    assert "exec" not in calls
    assert calls == ["start", "start", "status", "stop"]


# --- in-process: the spawn window, a mid-run store fault and the supervisor's failures ----------

SAFE_SIGNALS = (signal.SIGTERM, signal.SIGHUP, signal.SIGINT, signal.SIGQUIT)


class _Hooks:
    before_exec: Any = None
    after_exec: Any = None


@pytest.fixture
def in_process(
    env: Env, fakebin: Fakebin, monkeypatch: pytest.MonkeyPatch
) -> Iterator[tuple[Terminal, _Hooks]]:
    """Run ``launch.launch`` in the test process with a pty as the terminal; every handler the
    launcher may replace is saved first, and a harmless one is in place while it runs, so a
    signal that finds no handler of the launcher's can never kill the test worker."""
    from agent_spawn import gateway, launch  # noqa: PLC0415
    from tests.session_backend_support import FakebinGateway  # noqa: PLC0415

    terminal = Terminal()
    hooks = _Hooks()

    class TerminalGateway(FakebinGateway):
        def spawn(self, argv: Any, **kwargs: Any) -> Any:  # exec gets the pty and the fake env
            if "exec" in argv:
                kwargs.update(stdin=terminal.slave, stdout=terminal.slave, stderr=terminal.slave)
                kwargs["env"] = {**os.environ, **env.env()}
                if hooks.before_exec is not None:
                    hooks.before_exec()
            child = super().spawn(argv, **kwargs)
            if "exec" in argv and hooks.after_exec is not None:
                hooks.after_exec(child)
            return child

    monkeypatch.setattr(gateway, "_default", TerminalGateway(fakebin))
    monkeypatch.setattr(launch, "_tty_pair", lambda: True)
    monkeypatch.setenv("PATH", fakebin.env()["PATH"])
    saved = {s: signal.getsignal(s) for s in SAFE_SIGNALS}
    for sig in SAFE_SIGNALS:
        signal.signal(sig, lambda *_a: None)
    try:
        yield terminal, hooks
    finally:
        for sig, handler in saved.items():
            signal.signal(sig, handler)
        terminal.close()


def _launch_in_process() -> Any:
    from agent_spawn import cli, launch  # noqa: PLC0415

    return launch.launch(cli.parse_invocation([*LAUNCH]))


def _exec_signals(fakebin: Fakebin) -> list[int]:
    return [
        s
        for i in fakebin.invocations("claude-session")
        if i.argv[0] == "exec"
        for s in i.signals
        if s != signal.SIGCONT
    ]


def test_a_signal_in_the_spawn_window_is_forwarded_to_exec_not_raised_by_the_launcher(
    env: Env, fakebin: Fakebin, in_process: tuple[Terminal, _Hooks]
) -> None:
    """SIGTERM arrives just before ``exec`` is spawned: with the signals held it stays pending
    until the supervisor's handlers are installed, so it is forwarded to ``exec`` (the run ends
    ``killed``) and the launcher's own handler never raises its interrupt (exit 143)."""
    script_live_run(fakebin)
    script_hung_exec(fakebin)
    hooks = in_process[1]
    hooks.before_exec = lambda: signal.raise_signal(signal.SIGTERM)  # held: stays pending
    # Hold the spawn until the fake installed its handlers, so the forwarded signal is not lost.
    hooks.after_exec = lambda child: fakebin.wait_for_hang(child.pid)
    outcome = _launch_in_process()
    assert outcome.exit_code == 15
    assert _exec_signals(fakebin) == [int(signal.SIGTERM)]
    row = env.agents()[0]
    assert row["status"] == "parked" and row["park_reason"] == "killed"
    assert row["needs_attention"] is None


def test_a_store_failure_after_exec_started_reports_failed_and_stops_exec(
    env: Env,
    fakebin: Fakebin,
    in_process: tuple[Terminal, _Hooks],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mid-run ``StoreError`` (after ``exec`` was recorded): ``failed``, ``exec`` stopped by the
    supervisor's ``watch``; the parked state is left to the liveness check (TODO(#31))."""
    from agent_spawn.output import CliError  # noqa: PLC0415
    from agent_spawn.store import StoreError  # noqa: PLC0415
    from agent_spawn.supervise import Supervisor  # noqa: PLC0415

    script_live_run(fakebin)
    script_hung_exec(fakebin)

    def broken(self: Supervisor, *_args: Any) -> bool:
        raise StoreError("the database went away")

    monkeypatch.setattr(Supervisor, "look_started", broken)
    in_process[1].after_exec = lambda child: fakebin.wait_for_hang(child.pid)
    with pytest.raises(CliError) as excinfo:
        _launch_in_process()
    assert excinfo.value.code == "failed"
    assert _exec_signals(fakebin) == [int(signal.SIGTERM)]  # stopped, by the supervisor
    assert env.agents()[0]["status"] in ("launching", "live")  # not parked: see TODO(#31)


@pytest.mark.parametrize("failure", ["cli", "os"])
def test_a_supervisor_failure_before_exec_cancels_the_run_and_restores_sigquit(
    env: Env,
    fakebin: Fakebin,
    in_process: tuple[Terminal, _Hooks],
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    from agent_spawn.output import CliError  # noqa: PLC0415
    from agent_spawn.store import Store  # noqa: PLC0415
    from agent_spawn.supervise import Supervisor  # noqa: PLC0415

    seen: list[Any] = []
    original = Store.update_agent

    def recording(self: Store, lease: Any, agent_id: str, fields: Any) -> Any:
        if fields.get("supervisor") is not None:  # the moment this process becomes the supervisor
            seen.append(signal.getsignal(signal.SIGQUIT))
        return original(self, lease, agent_id, fields)

    def broken(self: Supervisor, store: Any, row: Any) -> Any:
        if failure == "cli":
            raise CliError("internal", "the supervisor broke")
        raise OSError("no fork")

    monkeypatch.setattr(Store, "update_agent", recording)
    monkeypatch.setattr(Supervisor, "run_in_process", broken)
    quit_handler = signal.getsignal(signal.SIGQUIT)
    with pytest.raises(CliError) as excinfo:
        _launch_in_process()
    assert seen[0] == signal.SIG_IGN  # ignored before the supervisor is even recorded (R8)
    assert signal.getsignal(signal.SIGQUIT) == quit_handler  # restored on the way out
    assert excinfo.value.code == ("internal" if failure == "cli" else "launch_failed")
    row = env.agents()[0]
    assert row["status"] == "parked" and row["park_reason"] == "killed"
    assert row["supervisor"] is None and row["needs_attention"] is None
    calls = [i.argv[0] for i in fakebin.invocations("claude-session")]
    assert "exec" not in calls and "stop" in calls
