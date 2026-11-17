"""Round 1 review of the headless launch: signals, cleanup on failure, usage errors, dry run."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from agent_spawn import cli, launch, procs
from agent_spawn.ids import new_agent_id
from agent_spawn.output import CliError
from agent_spawn.store import Store, StoreError
from agent_spawn.supervise import Signals, Supervisor
from tests import schema_registry
from tests.conftest import Isolation
from tests.fakebin.harness import Fakebin
from tests.launch_support import (
    Env,
    result_json,
    script_hung_exec,
    script_live_run,
    track_agent,
)
from tests.procguard import ProcTracker
from tests.session_backend_support import FakebinGateway

LAUNCH = ("launch", "--mode", "headless", "--skill", "do-it", "--json")
SESSION = "cs-abcdefgh"


class SpawningGateway(FakebinGateway):
    """The fakebin gateway that also starts its children with the fakes' environment."""

    def __init__(self, fakebin: Fakebin) -> None:
        super().__init__(fakebin)
        self._fakes = fakebin

    def spawn(self, argv: Any, **kwargs: Any) -> Any:
        kwargs["env"] = self._fakes.env()
        return super().spawn(argv, **kwargs)


@pytest.fixture
def env(fakebin: Fakebin, isolation_env: Isolation) -> Env:
    return Env(fakebin, isolation_env)


def non_dry(fakebin: Fakebin, command: str) -> list[list[str]]:
    return [a for a in fakebin.argvs("claude-session") if a[0] == command and "--dry-run" not in a]


def stops(fakebin: Fakebin) -> list[list[str]]:
    return [a for a in fakebin.argvs("claude-session") if a[0] == "stop"]


def script_pass_dry_run(fakebin: Fakebin) -> None:
    fakebin.script("claude-session", "start", argv_contains=["--dry-run"])


def popen_launcher(env: Env, proc_tracker: ProcTracker, *args: str) -> subprocess.Popen[bytes]:
    return proc_tracker.popen(
        env.argv(*args),
        env=env.env(),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


# --- H1: signals before the supervisor installed its handlers ----------------------------


def test_a_signal_before_the_exec_is_held_not_fatal(
    tmp_path: Path, proc_tracker: ProcTracker
) -> None:
    out = tmp_path / "out"
    code = (
        "import signal, sys, time\n"
        f"blocked = signal.SIGTERM in signal.pthread_sigmask(signal.SIG_BLOCK, [])\n"
        f"open({str(out)!r}, 'w').write('blocked' if blocked else 'open')\n"
        "def got(*a):\n"
        f"    open({str(out)!r}, 'a').write(' got')\n"
        "    sys.exit(0)\n"
        "signal.signal(signal.SIGTERM, got)\n"
        "signal.pthread_sigmask(signal.SIG_UNBLOCK, [signal.SIGTERM])\n"
        "time.sleep(30)\n"
    )
    started = procs.spawn_detached(
        [sys.executable, "-c", code], keep_blocked=[signal.SIGTERM], ack_timeout=30
    )
    proc_tracker.record(started.identity)
    os.kill(started.identity.pid, signal.SIGTERM)  # before the exec: pending, not fatal
    started.acknowledge()
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline and not (out.exists() and out.read_text().endswith("got")):
        time.sleep(0.05)
    assert out.read_text() == "blocked got"


def test_the_supervisor_unblocks_its_signals(
    env: Env, fakebin: Fakebin, proc_tracker: ProcTracker
) -> None:
    script_live_run(fakebin)
    script_hung_exec(fakebin)
    body = result_json(env.run(*LAUNCH))
    agent_id = body["agent"]["agent_id"]
    track_agent(env, proc_tracker, agent_id)
    pid = env.agent(agent_id)["supervisor"].pid
    blocked = next(
        int(line.split()[1], 16)
        for line in Path(f"/proc/{pid}/status").read_text().splitlines()
        if line.startswith("SigBlk:")
    )
    for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
        assert not blocked & (1 << (sig - 1)), sig.name


# --- M1: an interrupted lower call gets SIGTERM first --------------------------------------


def test_interrupt_sends_sigterm_to_the_in_flight_start(
    env: Env, fakebin: Fakebin, proc_tracker: ProcTracker
) -> None:
    script_pass_dry_run(fakebin)
    fakebin.script("claude-session", "start", hang=True, on_signal={"SIGTERM": {"exit": 143}})
    launcher = popen_launcher(env, proc_tracker, *LAUNCH)
    env.wait_for(lambda: bool(non_dry(fakebin, "start")), "the real start")
    time.sleep(1.0)  # the fake installs its handlers
    launcher.send_signal(signal.SIGTERM)
    launcher.communicate(timeout=30)
    (start,) = [i for i in fakebin.invocations("claude-session") if i.argv[0] == "start"][-1:]
    assert signal.SIGTERM in start.signals


# --- M2: exit 12 with a run is recorded with its not-started outcome -----------------------


def test_start_exit_12_with_a_run_records_the_not_started_outcome(
    env: Env, fakebin: Fakebin
) -> None:
    script_pass_dry_run(fakebin)
    fakebin.script(
        "claude-session",
        "start",
        exit=12,
        error={
            "code": "launch_failed",
            "message": "x",
            "csession_id": SESSION,
            "run": 1,
            "end_reason": "canceled",
        },
    )
    done = env.run(*LAUNCH)
    assert done.returncode == 12
    (row,) = env.agents()
    assert row["csession_id"] == SESSION and row["run"] == 1
    assert row["last_end_reason"] == "canceled"
    assert row["last_outcome"]["outcome"] == "not-started"
    assert row["last_outcome"]["phase"] == "launch"
    with env.store() as store:
        run = store.get_run(row["agent_id"], 1)
    assert run is not None and run["outcome"]["outcome"] == "not-started"
    assert run["end_reason"] == "canceled"


def test_start_failure_that_may_have_allocated_finds_and_stops_the_session(
    env: Env, fakebin: Fakebin
) -> None:
    script_pass_dry_run(fakebin)
    fakebin.script("claude-session", "start", exit=1, error={"code": "failed", "message": "x"})
    fakebin.script("claude-session", "list", stdout={"sessions": [{"csession_id": SESSION}]})
    done = env.run(*LAUNCH)
    assert done.returncode == 1
    (row,) = env.agents()
    assert row["csession_id"] == SESSION and row["status"] == "parked"
    assert [SESSION in a for a in stops(fakebin)] == [True]


# --- M5 and L14: an interrupt cancels the pending run and removes spent copies -------------


def test_interrupt_with_a_found_session_cancels_it_and_removes_the_copies(
    env: Env, fakebin: Fakebin, proc_tracker: ProcTracker
) -> None:
    script_pass_dry_run(fakebin)
    fakebin.script("claude-session", "start", sleep=30)
    fakebin.script("claude-session", "list", stdout={"sessions": [{"csession_id": SESSION}]})
    launcher = popen_launcher(env, proc_tracker, *LAUNCH, "--prompt", "secret words")
    env.wait_for(lambda: bool(non_dry(fakebin, "start")), "the real start")
    launcher.send_signal(signal.SIGTERM)
    launcher.communicate(timeout=30)
    assert launcher.returncode == 143
    (row,) = env.agents()
    assert row["csession_id"] == SESSION and row["park_reason"] == "killed"
    assert [SESSION in a for a in stops(fakebin)] == [True]
    assert not (env.state_dir / "prompts" / row["agent_id"]).exists()


def test_interrupt_without_a_session_keeps_the_copies(
    env: Env, fakebin: Fakebin, proc_tracker: ProcTracker
) -> None:
    script_pass_dry_run(fakebin)
    fakebin.script("claude-session", "start", sleep=30)
    fakebin.script("claude-session", "list", stdout={"sessions": []})
    launcher = popen_launcher(env, proc_tracker, *LAUNCH, "--prompt", "words")
    env.wait_for(lambda: bool(non_dry(fakebin, "start")), "the real start")
    launcher.send_signal(signal.SIGTERM)
    launcher.communicate(timeout=30)
    (row,) = env.agents()
    assert row["csession_id"] is None and not stops(fakebin)
    assert (env.state_dir / "prompts" / row["agent_id"] / "prompt").exists()


def test_interrupt_after_start_returned_cancels_the_known_run(
    env: Env, fakebin: Fakebin, proc_tracker: ProcTracker
) -> None:
    fakebin.script("claude-session", "status", sleep=30)  # the post-start check hangs
    launcher = popen_launcher(env, proc_tracker, *LAUNCH)
    env.wait_for(
        lambda: any(a["csession_id"] for a in _agents(env)) and bool(fakebin.invocations()),
        "the session",
    )
    env.wait_for(
        lambda: any(i.argv[0] == "status" for i in fakebin.invocations("claude-session")),
        "the status call",
    )
    launcher.send_signal(signal.SIGTERM)
    launcher.communicate(timeout=30)
    (row,) = env.agents()
    assert row["park_reason"] == "killed" and row["needs_attention"] is None
    assert [row["csession_id"] in a for a in stops(fakebin)] == [True]


def _agents(env: Env) -> list[dict[str, Any]]:
    try:
        return env.agents()
    except Exception:  # noqa: BLE001 - the database may not exist yet
        return []


# --- M7: the supervisor died while the agent was still launching ---------------------------

#: A run still in its launch phase: not final, no session started yet.
LIVE_LAUNCH_PHASE = {
    "phase": "launch",
    "state": "starting",
    "outcome": None,
    "end_reason": None,
    "exit_code": None,
    "report": None,
    "ended_at": None,
    "session_started_at": None,
}


def test_a_dead_supervisor_of_a_launching_agent_is_settled_by_the_launchers_liveness_check(
    env: Env, fakebin: Fakebin, proc_tracker: ProcTracker
) -> None:
    script_hung_exec(fakebin)
    # `status` reports the run live in its launch phase, so the supervisor's status poll never
    # moves the agent to `live` (the canned `status` reports a final, session-phase run).
    fakebin.script("claude-session", "status", merge={"run": LIVE_LAUNCH_PHASE})
    launcher = popen_launcher(env, proc_tracker, *LAUNCH)
    env.wait_for(lambda: any(a.get("exec_proc") for a in _agents(env)), "the supervisor's exec")
    row = env.agents()[0]
    track_agent(env, proc_tracker, row["agent_id"])
    assert row["status"] == "launching"
    procs.forward_signal(row["supervisor"], signal.SIGKILL)
    env.wait_for(lambda: not procs.is_alive(row["supervisor"]), "the supervisor to die")
    # The run is adopted and still in its launch phase: the agent stays `launching` and the
    # launcher keeps waiting (nothing is stopped, nothing is set).
    assert launcher.poll() is None
    assert env.agent(row["agent_id"])["status"] == "launching"
    # claude-session's own expiry ends the run `never_adopted`: the launcher's next poll applies
    # the outcome table (a launch failure), without stopping anything itself.
    fakebin.clear_script("claude-session", "status")
    fakebin.script(
        "claude-session",
        "status",
        outcome="not-started",
        merge={"run": {"end_reason": "never_adopted"}},
    )
    out, _err = launcher.communicate(timeout=60)
    body = json.loads(out)
    assert launcher.returncode == 12 and body["error"]["code"] == "launch_failed"
    parked = env.agent(row["agent_id"])
    assert parked["status"] == "parked" and parked["park_reason"] == "launch_failed"
    assert parked["needs_attention"] == "launch_failed"
    assert stops(fakebin) == []


# --- M9: a classification cancel that failed is reported -----------------------------------


def test_a_failed_cancel_after_a_classification_mismatch_is_reported(
    env: Env, fakebin: Fakebin
) -> None:
    fakebin.script(
        "claude-session",
        "status",
        times=1,
        merge={"run": {"reads_third_party": True, "cwd": "/tmp/not-the-session-dir"}},
    )
    fakebin.script("claude-session", "stop", exit=1, error={"code": "failed", "message": "x"})
    done = env.run(*LAUNCH)
    body = result_json(done)
    assert done.returncode == 12
    schema_registry.validate(body, "agent-spawn/cli/error.json")
    assert body["error"]["details"] == {
        "cause": "classification_mismatch",
        "cancel_failed": True,
        "cancel_code": "internal",
    }
    (row,) = env.agents()
    assert row["csession_id"] is not None and row["park_reason"] == "launch_failed"
    log = (env.state_dir / "log.jsonl").read_text()
    assert "could not be canceled" in log


# --- M10 and L3: user input errors are exit 2 ----------------------------------------------


def test_bad_input_file_names_are_usage_errors(env: Env, fakebin: Fakebin, tmp_path: Path) -> None:
    bad = tmp_path / "bad name.txt"
    bad.write_text("x")
    done = env.run(*LAUNCH, "--input-file", str(bad))
    assert done.returncode == 2 and result_json(done)["error"]["code"] == "usage"
    first, second = tmp_path / "a" / "same.txt", tmp_path / "b" / "same.txt"
    for path in (first, second):
        path.parent.mkdir()
        path.write_text("x")
    done = env.run(*LAUNCH, "--input-file", str(first), "--input-file", str(second))
    assert done.returncode == 2 and "same.txt" in result_json(done)["error"]["message"]
    assert env.agents() == [] and not non_dry(fakebin, "start")
    assert not list((env.state_dir / "prompts").glob("*"))


@pytest.mark.parametrize(
    "extra",
    [("--env-allow", "bad name"), ("--timeout", "0")],
)
def test_claude_session_rejecting_an_argument_is_exit_2(
    env: Env, fakebin: Fakebin, extra: tuple[str, ...]
) -> None:
    done = env.run(*LAUNCH, *extra)
    body = result_json(done)
    assert done.returncode == 2, (done.stdout, done.stderr)
    assert body["error"]["code"] == "usage"
    assert env.agents() == [] and not non_dry(fakebin, "start")


def test_claude_session_exit_2_of_the_classification_keeps_its_error(
    env: Env, fakebin: Fakebin
) -> None:
    fakebin.script(
        "claude-session",
        "start",
        exit=2,
        error={"code": "usage", "message": "bad flag", "extra": 1},
    )
    done = env.run(*LAUNCH)
    error = result_json(done)["error"]
    assert done.returncode == 2 and error["code"] == "usage"
    assert error["tool"] == "claude-session" and error["lower_error"]["message"] == "bad flag"


def test_a_reserved_label_is_exit_2_before_tool_resolution(env: Env) -> None:
    done = env.run(*LAUNCH, "--label", "agent=mine", extra_env={"PATH": "/usr/bin:/bin"})
    assert done.returncode == 2 and result_json(done)["error"]["code"] == "usage"


# --- M11: pre-hook decision ids reach the launcher's own calls ------------------------------


def test_pre_hook_decision_ids_are_passed_to_start(env: Env, fakebin: Fakebin) -> None:
    done = env.run(
        *LAUNCH, "--wait", "--actor", "auto:test",
        "--hook", """pre-launch=echo '{"decision_id": "d-77"}'""",
    )  # fmt: skip
    assert done.returncode == 0, (done.stdout, done.stderr)
    (start,) = non_dry(fakebin, "start")
    assert "--decision-id=d-77" in start


# --- M12: a failing state database never leaves the agent launching -------------------------


@pytest.fixture
def restore_signals() -> Any:
    saved = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT)}
    yield
    for sig, handler in saved.items():
        signal.signal(sig, handler)


def test_a_store_error_while_recording_the_supervisor_parks_the_agent(
    fakebin: Fakebin,
    isolation_env: Isolation,
    monkeypatch: pytest.MonkeyPatch,
    restore_signals: None,
) -> None:
    monkeypatch.setattr("agent_spawn.gateway._default", SpawningGateway(fakebin))
    monkeypatch.setenv("PATH", fakebin.env()["PATH"])
    real = Store.update_agent

    def failing(self: Store, lease: Any, agent_id: str, changes: Any, **kw: Any) -> Any:
        if changes.get("supervisor") is not None:
            raise StoreError("lease timed out")
        return real(self, lease, agent_id, changes, **kw)

    monkeypatch.setattr(Store, "update_agent", failing)
    inv = cli.parse_invocation(
        ["launch", "--mode", "headless", "--skill", "do-it", "--json", "--prompt", "p"]
    )
    with pytest.raises(CliError) as caught:
        launch.launch(inv)
    assert caught.value.code == "launch_failed"
    with Store.open(isolation_env.state / "agent-spawn", create=False) as store:
        (row,) = store.list_agents()
    assert row["status"] == "parked" and row["needs_attention"] == "launch_failed"
    assert stops(fakebin)


# --- L2: a dry run creates nothing --------------------------------------------------------


def test_dry_run_creates_no_state_and_calls_the_dry_run_once(
    env: Env, fakebin: Fakebin, tmp_path: Path
) -> None:
    note = tmp_path / "in.txt"
    note.write_text("x")
    done = env.run(*LAUNCH, "--dry-run", "--prompt", "hello", "--input-file", str(note))
    body = result_json(done)
    assert done.returncode == 0, (done.stdout, done.stderr)
    schema_registry.validate(body, "agent-spawn/cli/launch-dry-run.json")
    assert not env.state_dir.exists()
    assert len([a for a in fakebin.argvs("claude-session") if "--dry-run" in a]) == 1
    assert body["agent_settings"]["input_names"] == ["in.txt"]


# --- L4: the --wait-timeout result carries first_working_at -------------------------------


def test_wait_timeout_result_has_first_working_at(
    env: Env, fakebin: Fakebin, proc_tracker: ProcTracker
) -> None:
    script_live_run(fakebin)
    script_hung_exec(fakebin)
    launcher = popen_launcher(env, proc_tracker, *LAUNCH, "--wait", "--wait-timeout", "6s")
    env.wait_for(lambda: any(a["status"] == "live" for a in _agents(env)), "a live agent")
    agent_id = env.agents()[0]["agent_id"]
    track_agent(env, proc_tracker, agent_id)
    with env.store() as store:
        with store.lease(f"agent:{agent_id}") as lease:
            store.put_run(lease, agent_id, 1, {"first_working_at": 1700000000.0})
    out, _err = launcher.communicate(timeout=60)
    body = json.loads(out)
    assert launcher.returncode == 22 and body["timed_out"] is True
    assert body["first_working_at"].startswith("2023-11-14")


# --- L5, L11: the supervisor's terminal and a missing store --------------------------------


def test_supervisor_writes_nothing_to_its_log_on_success(env: Env, fakebin: Fakebin) -> None:
    body = result_json(env.run(*LAUNCH, "--wait"))
    agent_id = body["agent"]["agent_id"]
    env.wait_for(lambda: not procs.is_alive(env.agent(agent_id)["supervisor"]), "supervisor exit")
    assert (env.state_dir / "supervise" / f"{agent_id}.log").read_text() == ""


def test_supervise_without_a_store_is_usage_with_a_log_entry(env: Env) -> None:
    env.state_dir.mkdir(parents=True)
    done = env.run(
        "supervise", "ag-aaaaaaaa", "--state-dir", str(env.state_dir), "--config", str(env.config)
    )
    assert done.returncode == 2, (done.stdout, done.stderr)
    assert "no state database" in (env.state_dir / "log.jsonl").read_text()


# --- L6: relative PATH entries --------------------------------------------------------------


def test_resolve_tool_makes_a_which_result_absolute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "rel").mkdir()
    tool = tmp_path / "rel" / "some-tool"
    tool.write_text("#!/bin/sh\n")
    tool.chmod(0o755)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PATH", "rel")
    assert launch.resolve_tool(["some-tool", "x"], "some-tool") == [str(tool), "x"]


# --- L10: signals ---------------------------------------------------------------------------


def test_a_second_signal_never_raises_again() -> None:
    state = launch.SignalState()
    with pytest.raises(launch.LaunchInterrupted):
        state.handle(signal.SIGTERM, None)
    state.handle(signal.SIGINT, None)  # inside the interrupt handler: recorded only
    assert state.take() == [signal.SIGINT]


def test_signals_are_held_while_a_section_runs() -> None:
    seen: list[int] = []
    old = signal.signal(signal.SIGTERM, lambda signum, frame: seen.append(signum))
    try:
        with launch._signals_held():  # pyright: ignore[reportPrivateUsage]
            os.kill(os.getpid(), signal.SIGTERM)
            assert seen == []
        assert seen == [signal.SIGTERM]
    finally:
        signal.signal(signal.SIGTERM, old)


# --- supervisor, in process: M6, M8, L8 -------------------------------------------------------


def make_supervisor(
    fakebin: Fakebin,
    isolation_env: Isolation,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    **session: Any,
) -> tuple[Supervisor, Signals, Path, str]:
    monkeypatch.setattr("agent_spawn.gateway._default", SpawningGateway(fakebin))
    state = tmp_path / "state"
    session_id = fakebin.seed_session(SESSION, adopted=False, final=False, labels=[], **session)
    cs = str(fakebin.bin_dir / "claude-session")
    with Store.open(state) as store:
        agent_id = new_agent_id(store)
        store.insert_agent(
            agent_id,
            {
                "status": "launching",
                "name": "agent test",
                "mode": "headless",
                "supervisor": store.holder_identity(),
                "csession_id": session_id,
                "run": 1,
                "claude_session_path": cs,
                "state_dir": str(state),
            },
        )
        with store.lease(f"agent:{agent_id}") as lease:
            store.put_run(lease, agent_id, 1, {"outcome_token": "tok", "mode": "headless"})
    inv = cli.parse_invocation(
        ["supervise", agent_id, "--state-dir", str(state), "--config", str(tmp_path / "c.yaml")]
    )
    signals = Signals()
    return Supervisor(inv, signals), signals, state, agent_id


def test_a_store_error_after_exec_started_stops_exec_and_reraises(
    fakebin: Fakebin,
    isolation_env: Isolation,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    script_hung_exec(fakebin)
    supervisor, _signals, state, agent_id = make_supervisor(
        fakebin, isolation_env, monkeypatch, tmp_path
    )
    real = Store.update_agent

    def failing(self: Store, lease: Any, agent_id: str, changes: Any, **kw: Any) -> Any:
        if "exec_proc" in changes:
            deadline = time.monotonic() + 20
            while time.monotonic() < deadline:  # let the fake `exec` install its handlers
                calls = [i for i in fakebin.invocations("claude-session") if i.argv[0] == "exec"]
                if calls:
                    fakebin.wait_for_hang(calls[0].pid)
                    break
                time.sleep(0.05)
            raise StoreError("lease timed out")
        return real(self, lease, agent_id, changes, **kw)

    monkeypatch.setattr(Store, "update_agent", failing)
    with pytest.raises(StoreError):
        supervisor.run()
    (call,) = [i for i in fakebin.invocations("claude-session") if i.argv[0] == "exec"]
    assert signal.SIGTERM in call.signals


def test_a_missing_outcome_is_logged_with_a_valid_event_and_retried(
    fakebin: Fakebin,
    isolation_env: Isolation,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fakebin.script("claude-session", "status", error={"code": "failed", "message": "x"}, exit=1)
    supervisor, signals, state, agent_id = make_supervisor(
        fakebin, isolation_env, monkeypatch, tmp_path
    )
    signals.received.append(signal.SIGTERM)  # cancel path: stop, then read the outcome
    supervisor.run()
    statuses = [i for i in fakebin.invocations("claude-session") if i.argv[0] == "status"]
    assert len(statuses) >= 3  # retried
    log = [json.loads(line) for line in (state / "log.jsonl").read_text().splitlines()]
    assert [r for r in log if r["event"] == "warning" and "no outcome" in r["message"]]
    with Store.open(state, create=False) as store:
        parked = store.get_agent(agent_id)
    assert parked is not None and parked["park_reason"] == "killed"


def test_a_transient_outcome_failure_is_retried(
    fakebin: Fakebin,
    isolation_env: Isolation,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fakebin.script(
        "claude-session", "status", times=2, error={"code": "failed", "message": "x"}, exit=1
    )
    supervisor, signals, state, agent_id = make_supervisor(
        fakebin, isolation_env, monkeypatch, tmp_path
    )
    signals.received.append(signal.SIGTERM)
    supervisor.run()
    with Store.open(state, create=False) as store:
        parked = store.get_agent(agent_id)
    assert parked is not None and parked["last_outcome"]["outcome"] == "not-started"
