"""Promotion (issue #29): a headless run that ended blocked or needs-input continues in a window
hosted by ``supervise --host``. Process tests against the fake binaries; the fake termctl starts
its panes in a real tmux server that is private to the test (``-S <tmp>/sock -f /dev/null``,
private HOME, XDG and TMUX_TMPDIR); nothing here touches the user's server."""

from __future__ import annotations

import json
import os
import signal
import sqlite3
import subprocess
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from agent_spawn import chain, handoff, lifecycle, procs, promotion
from agent_spawn.store import StoreError
from agent_spawn.supervise import Supervisor
from tests import schema_registry
from tests.conftest import Isolation
from tests.fakebin.harness import Fakebin
from tests.launch_support import result_json, script_hung_exec, script_live_run, track_agent
from tests.private_tmux import PrivateTmux
from tests.procguard import ProcTracker
from tests.test_worktrees_launch import fake_throwaway
from tests.window_support import Capture, WindowEnv, send_signal

LAUNCH = ("launch", "--mode", "headless", "--skill", "do-it", "--json")


@pytest.fixture
def wenv(fakebin: Fakebin, isolation_env: Isolation, private_tmux: PrivateTmux) -> WindowEnv:
    private_tmux.start()
    return WindowEnv(fakebin, isolation_env, private_tmux)


def events(env: WindowEnv, agent_id: str) -> list[tuple[str, dict[str, Any]]]:
    db = sqlite3.connect(f"file:{env.state_dir / 'state.db'}?mode=ro", uri=True)
    try:
        rows = db.execute(
            "SELECT event, payload FROM events WHERE agent_id = ? ORDER BY event_id", (agent_id,)
        ).fetchall()
    finally:
        db.close()
    return [(name, json.loads(payload)) for name, payload in rows]


def log_events(env: WindowEnv) -> list[dict[str, Any]]:
    path = env.state_dir / "log.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def runs(env: WindowEnv, agent_id: str) -> list[dict[str, Any]]:
    with env.store() as store:
        return store.get_runs(agent_id)


def real_starts(fakebin: Fakebin) -> list[list[str]]:
    return [a for a in fakebin.argvs("claude-session") if a[0] == "start" and "--dry-run" not in a]


def script_first_run(fakebin: Fakebin, outcome: str = "blocked") -> None:
    """The headless run ends ``outcome`` (the rule answers one call); what the next `exec`
    answers is scripted by the test (``script_promoted_done`` or its own rule)."""
    fakebin.script("claude-session", "exec", times=1, outcome=outcome)


def script_promoted_done(fakebin: Fakebin) -> None:
    """The promoted run's `exec` (the second call) ends `done`, scripted, not the fake's default."""
    fakebin.script("claude-session", "exec", times=1, outcome="done")


def host_exited(capture: Capture, code: int | None = None) -> bool:
    """The ``supervise --host`` process exited (with ``code``, when given); False while the
    wrapper has not written its exit code yet."""
    try:
        done = capture.host_exit_code()
    except FileNotFoundError:
        return False
    return code is None or done == code


def stops(fakebin: Fakebin) -> list[list[str]]:
    return [a for a in fakebin.argvs("claude-session") if "stop" in a]


def execs(fakebin: Fakebin) -> list[Any]:
    return [i for i in fakebin.invocations("claude-session") if "exec" in i.argv]


def script_gated_open(fakebin: Fakebin, gate: Path) -> None:
    """The window preflight passes; the real `termctl open` creates its pane and then waits at
    ``gate`` (the supervisor is blocked inside it until the test creates the file)."""
    fakebin.script("termctl", "open", argv_contains=["--dry-run"])
    fakebin.script("termctl", "open", gate=str(gate))


def test_blocked_run_is_promoted_and_the_promoted_run_ends_done(
    wenv: WindowEnv, fakebin: Fakebin, tmp_path: Path
) -> None:
    wenv.capture_supervisor(tmp_path / "cap")
    script_first_run(fakebin)
    script_promoted_done(fakebin)
    done = wenv.run(*LAUNCH, "--promote", "--wait")
    body = result_json(done)
    assert done.returncode == 0, (done.stdout, done.stderr)
    schema_registry.validate(body, "agent-spawn/cli/launch.json")
    agent_id = body["agent"]["agent_id"]
    row = wenv.agent(agent_id)
    assert row["status"] == "parked" and row["park_reason"] == "done"
    assert row["promotion_pending"] is False and row["run"] == 2
    first, second = runs(wenv, agent_id)
    assert first["outcome"]["outcome"] == "blocked" and not first["promoted"]
    assert second["promoted"] is True and second["mode"] == "interactive"
    assert second["placement"] == "window" and second["term_id"].startswith("tc-")
    assert second["outcome"]["outcome"] == "done" and second["outcome"]["promoted"] is True
    start = real_starts(fakebin)[1]
    assert f"--resume={row['csession_id']}" in start and "--mode=interactive" in start
    launches = [p for name, p in events(wenv, agent_id) if name == "post-launch"]
    assert [(p["run"], p["promoted"]) for p in launches] == [(1, False), (2, True)]


def open_calls(fakebin: Fakebin, *, dry_run: bool) -> list[list[str]]:
    return [
        argv
        for argv in fakebin.argvs("termctl")
        if "open" in argv and (("--dry-run" in argv) == dry_run)
    ]


def post_park(env: WindowEnv, agent_id: str) -> dict[str, Any]:
    parks = [p for name, p in events(env, agent_id) if name == "post-park"]
    assert len(parks) == 1  # exactly one process parked, once
    schema_registry.validate(
        {
            **parks[0],
            "tool": "agent-spawn",
            "hook": "post-park",
            "object_type": "agent",
            "object_id": agent_id,
            "event_id": 1,
            "ts": "2026-11-14T00:00:00.000Z",
        },
        "agent-spawn/hooks/post-park.json",
    )
    return parks[0]


# --- the decision ----


def test_a_run_is_not_promoted_unless_promote_resolves_true(
    wenv: WindowEnv, fakebin: Fakebin
) -> None:
    script_first_run(fakebin)
    done = wenv.run(*LAUNCH, "--wait")
    body = result_json(done)
    assert done.returncode == 13, (done.stdout, done.stderr)
    row = wenv.agent(body["agent"]["agent_id"])
    assert row["park_reason"] == "blocked" and row["needs_attention"] == "blocked"
    assert len(real_starts(fakebin)) == 1 and not open_calls(fakebin, dry_run=False)
    assert not open_calls(fakebin, dry_run=True)  # no preflight either


def test_the_kind_config_promotes_and_no_promote_overrides_it(
    fakebin: Fakebin, isolation_env: Isolation, private_tmux: PrivateTmux, tmp_path: Path
) -> None:
    private_tmux.start()
    env = WindowEnv(
        fakebin, isolation_env, private_tmux, "kinds:\n  implement:\n    promote: true\n"
    )
    env.capture_supervisor(tmp_path / "cap")
    script_first_run(fakebin, "needs-input")
    script_promoted_done(fakebin)
    body = result_json(env.run(*LAUNCH, "--wait"))
    assert body["outcome"] == "done"  # the promoted run's outcome
    assert len(real_starts(fakebin)) == 2
    # --no-promote wins over the config
    fakebin.script("claude-session", "exec", times=1, outcome="blocked")  # the third run
    done = env.run(*LAUNCH, "--no-promote", "--wait")
    assert done.returncode == 13
    assert len(real_starts(fakebin)) == 3  # one more start, no promotion


def test_a_promoted_run_is_never_promoted_again(
    wenv: WindowEnv, fakebin: Fakebin, tmp_path: Path
) -> None:
    wenv.capture_supervisor(tmp_path / "cap")
    fakebin.script("claude-session", "exec", times=2, outcome="blocked")
    done = wenv.run(*LAUNCH, "--promote", "--wait")
    assert done.returncode == 13, (done.stdout, done.stderr)
    agent_id = result_json(done)["agent"]["agent_id"]
    row = wenv.agent(agent_id)
    assert row["park_reason"] == "blocked" and row["needs_attention"] == "blocked"
    assert row["last_outcome"]["run"] == 2 and row["last_outcome"]["promoted"] is True
    assert len(real_starts(fakebin)) == 2 and len(open_calls(fakebin, dry_run=False)) == 1
    assert post_park(wenv, agent_id)["chain_failure"] is None


@pytest.mark.parametrize(
    "end_reason", ["stopped", "forwarded_term", "forwarded_hup", "forwarded_int"]
)
def test_a_run_stopped_or_canceled_after_needs_input_is_not_promoted(
    wenv: WindowEnv, fakebin: Fakebin, end_reason: str
) -> None:
    fakebin.script("claude-session", "exec", times=1, outcome="needs-input", end_reason=end_reason)
    done = wenv.run(*LAUNCH, "--promote", "--wait")
    assert done.returncode == 13, (done.stdout, done.stderr)
    assert len(real_starts(fakebin)) == 1 and not open_calls(fakebin, dry_run=False)
    row = wenv.agent(result_json(done)["agent"]["agent_id"])
    assert row["status"] == "parked" and row["promotion_pending"] is False


def test_an_interactive_agent_is_never_promoted(wenv: WindowEnv, fakebin: Fakebin) -> None:
    fakebin.script("claude-session", "exec", times=1, outcome="blocked")
    done = wenv.run("launch", "--skill", "do-it", "--promote", "--wait", "--json")
    assert done.returncode == 13, (done.stdout, done.stderr)
    assert len(real_starts(fakebin)) == 1


# --- the chain rule ----


def test_a_failed_window_preflight_creates_no_pending_run(
    wenv: WindowEnv, fakebin: Fakebin
) -> None:
    script_first_run(fakebin, "needs-input")
    fakebin.script("termctl", "open", error={"code": "no_tmux", "message": "no tmux"}, exit=12)
    done = wenv.run(*LAUNCH, "--promote", "--wait")
    body = result_json(done)
    assert done.returncode == 13, (done.stdout, done.stderr)
    agent_id = body["agent"]["agent_id"]
    assert len(real_starts(fakebin)) == 1  # no promoted start
    assert not open_calls(fakebin, dry_run=False)
    assert not [a for a in fakebin.argvs("claude-session") if "stop" in a]  # nothing to cancel
    row = wenv.agent(agent_id)
    assert row["status"] == "parked" and row["park_reason"] == "escalated"
    assert row["needs_attention"] == "escalated" and row["run"] == 1
    assert row["promotion_pending"] is False
    failure = post_park(wenv, agent_id)["chain_failure"]
    assert failure == {"tool": "termctl", "reason": "no_tmux", "run": None, "end_reason": None}
    assert len(runs(wenv, agent_id)) == 1


def test_a_window_that_fails_to_open_cancels_the_pending_run_and_parks_per_the_trigger(
    wenv: WindowEnv, fakebin: Fakebin
) -> None:
    script_first_run(fakebin)
    fakebin.script("termctl", "open", argv_contains=["--dry-run"])  # the preflight passes
    fakebin.script("termctl", "open", error={"code": "failed", "message": "no"}, exit=12)
    done = wenv.run(*LAUNCH, "--promote", "--wait")
    body = result_json(done)
    assert done.returncode == 13, (done.stdout, done.stderr)
    agent_id = body["agent"]["agent_id"]
    row = wenv.agent(agent_id)
    assert row["status"] == "parked" and row["park_reason"] == "blocked"
    assert row["needs_attention"] == "blocked" and row["run"] == 2
    assert row["last_outcome"]["run"] == 1 and row["last_outcome"]["outcome"] == "blocked"
    failure = post_park(wenv, agent_id)["chain_failure"]
    assert failure["tool"] == "termctl" and failure["run"] == 2
    stops = [a for a in fakebin.argvs("claude-session") if "stop" in a]
    assert len(stops) == 1 and row["csession_id"] in stops[0]
    first, second = runs(wenv, agent_id)
    assert second["promoted"] is True and second["outcome"]["end_reason"] == "canceled"
    assert not [i for i in fakebin.invocations("claude-session") if "exec" in i.argv][1:]


def test_a_promoted_run_that_is_never_adopted_parks_per_the_trigger(
    wenv: WindowEnv, fakebin: Fakebin, tmp_path: Path
) -> None:
    wenv.capture_supervisor(tmp_path / "cap")
    script_first_run(fakebin, "needs-input")
    fakebin.script(
        "claude-session", "exec", times=1, outcome="not-started", end_reason="never_adopted"
    )
    done = wenv.run(*LAUNCH, "--promote", "--wait")
    body = result_json(done)
    assert done.returncode == 13, (done.stdout, done.stderr)
    agent_id = body["agent"]["agent_id"]
    row = wenv.agent(agent_id)
    assert row["status"] == "parked" and row["park_reason"] == "escalated"
    assert row["needs_attention"] == "escalated" and row["escalated_by"] is None
    assert row["run"] == 2 and row["last_outcome"]["run"] == 1
    failure = post_park(wenv, agent_id)["chain_failure"]
    assert failure == {
        "tool": "claude-session",
        "reason": "not_started",
        "run": 2,
        "end_reason": "never_adopted",
    }
    second = runs(wenv, agent_id)[1]
    assert second["outcome"]["end_reason"] == "never_adopted" and second["placement"] == "window"


def test_a_pane_without_a_pid_fails_the_handoff_and_parks_per_the_trigger(
    wenv: WindowEnv, fakebin: Fakebin
) -> None:
    script_first_run(fakebin)
    fakebin.script("termctl", "open", argv_contains=["--dry-run"])
    fakebin.script("termctl", "open", merge={"pid": None})
    done = wenv.run(*LAUNCH, "--promote", "--wait")
    body = result_json(done)
    assert done.returncode == 13, (done.stdout, done.stderr)
    agent_id = body["agent"]["agent_id"]
    row = wenv.agent(agent_id)
    assert row["park_reason"] == "blocked" and row["needs_attention"] == "blocked"
    failure = post_park(wenv, agent_id)["chain_failure"]
    assert failure == {
        "tool": "agent-spawn",
        "reason": "handoff_failed",
        "run": 2,
        "end_reason": None,
    }
    assert len(stops(fakebin)) == 1
    second = runs(wenv, agent_id)[1]  # the window is recorded first, the identity is not
    assert second["term_id"].startswith("tc-") and second["pane_proc"] is None
    # the host got no environment: it exits 1, tmux keeps the dead pane and the supervisor closes it
    wenv.wait_for(lambda: wenv.panes() == [], "the dead promotion pane to be closed")
    assert not [i for i in fakebin.invocations("claude-session") if "exec" in i.argv][1:]


# --- the host and the signals ----


def start_hung_promotion(
    wenv: WindowEnv, fakebin: Fakebin, proc_tracker: ProcTracker, tmp_path: Path
) -> tuple[str, Capture]:
    """A headless agent whose promoted run's `exec` (in the host window) runs until signaled.
    Returns the agent id once the host's `exec` hangs."""
    capture = wenv.capture_supervisor(tmp_path / "cap")
    script_first_run(fakebin)
    script_hung_exec(fakebin)
    script_live_run(fakebin)
    body = result_json(wenv.run(*LAUNCH, "--promote"))
    agent_id = body["agent"]["agent_id"]
    track_agent(wenv, proc_tracker, agent_id)
    wenv.wait_for(lambda: len(fakebin.records("hang_ready")) >= 1, "the host's exec to hang")
    return agent_id, capture


def host_identity(wenv: WindowEnv, agent_id: str) -> procs.ProcessIdentity:
    run = wenv.agent(agent_id)["run"]
    with wenv.store() as store:
        row = store.get_run(agent_id, run)
    assert row is not None and row["pane_proc"] is not None
    return row["pane_proc"]


@pytest.mark.parametrize("sent", [signal.SIGTERM, signal.SIGHUP])
def test_signals_during_a_promotion_go_to_the_host_pane_process_only(
    wenv: WindowEnv,
    fakebin: Fakebin,
    proc_tracker: ProcTracker,
    tmp_path: Path,
    sent: signal.Signals,
) -> None:
    agent_id, capture = start_hung_promotion(wenv, fakebin, proc_tracker, tmp_path)
    row = wenv.agent(agent_id)
    assert row["promotion_pending"] is True and row["run"] == 2 and row["exec_proc"] is None
    host = host_identity(wenv, agent_id)
    assert host != row["supervisor"]  # the host is the pane process, not the supervisor
    # SIGINT and SIGQUIT are ignored; nothing reaches exec
    send_signal(row["supervisor"], signal.SIGINT)
    send_signal(row["supervisor"], signal.SIGQUIT)
    send_signal(row["supervisor"], sent)
    wenv.wait_for(lambda: wenv.agent(agent_id)["status"] == "parked", "the agent to park")
    done = wenv.agent(agent_id)
    assert done["park_reason"] == "killed" and done["needs_attention"] is None
    assert done["promotion_pending"] is False
    logged = [e for e in log_events(wenv) if e["event"] == "signal_sent"]
    # exactly two forwards of the one signal: supervisor -> the host pane process, host -> exec
    assert [e["signal"] for e in logged] == [sent.name, sent.name]
    assert [e["pid"] for e in logged].count(host.pid) == 1  # the supervisor's: only the host
    exec_calls = execs(fakebin)
    assert exec_calls[1].signals == [int(sent)]  # the host forwarded it once
    assert exec_calls[0].signals == []  # the first (headless) exec never got the signal
    wenv.wait_for(lambda: host_exited(capture, 0), "the host to exit")
    assert capture.out.read_text() == "" and capture.err.read_text() == ""


def test_signals_sent_to_the_host_pane_process_itself_are_handled_like_a_terminal_would(
    wenv: WindowEnv, fakebin: Fakebin, proc_tracker: ProcTracker, tmp_path: Path
) -> None:
    agent_id, capture = start_hung_promotion(wenv, fakebin, proc_tracker, tmp_path)
    host = host_identity(wenv, agent_id)
    send_signal(host, signal.SIGINT)  # the terminal's: the host ignores it
    send_signal(host, signal.SIGQUIT)
    send_signal(host, signal.SIGHUP)  # a closed window: forwarded to exec
    wenv.wait_for(lambda: host_exited(capture, 0), "the host to exit")
    exec_calls = execs(fakebin)
    assert exec_calls[1].signals == [int(signal.SIGHUP)]  # neither SIGINT nor SIGQUIT got through
    wenv.wait_for(lambda: wenv.agent(agent_id)["status"] == "parked", "the agent to park")
    done = wenv.agent(agent_id)
    assert done["park_reason"] == "killed" and done["needs_attention"] is None
    assert capture.out.read_text() == "" and capture.err.read_text() == ""


def test_supervise_host_exits_0_after_exec_exits_whatever_its_outcome(
    wenv: WindowEnv, fakebin: Fakebin, tmp_path: Path
) -> None:
    capture = wenv.capture_supervisor(tmp_path / "cap")
    script_first_run(fakebin)
    fakebin.script("claude-session", "exec", times=1, outcome="failed")  # the promoted run fails
    done = wenv.run(*LAUNCH, "--promote", "--wait")
    assert done.returncode == 18, (done.stdout, done.stderr)
    wenv.wait_for(lambda: host_exited(capture, 0), "the host to exit")
    assert capture.out.read_text() == "" and capture.err.read_text() == ""
    wenv.wait_for(lambda: wenv.panes() == [], "the window to close")


def test_a_host_without_a_handoff_socket_exits_1_and_runs_nothing(
    wenv: WindowEnv, fakebin: Fakebin, proc_tracker: ProcTracker, tmp_path: Path
) -> None:
    done = wenv.run(*LAUNCH)
    agent_id = result_json(done)["agent"]["agent_id"]
    wenv.wait_for(lambda: wenv.agent(agent_id)["status"] == "parked", "the agent to park")
    before = len([i for i in fakebin.invocations("claude-session") if "exec" in i.argv])
    refused = wenv.run(
        "supervise", agent_id, "--host", "--handoff", str(tmp_path / "nowhere"),
        "--handoff-timeout", "1", "--state-dir", str(wenv.state_dir),
        "--config", str(wenv.config),
    )  # fmt: skip
    assert refused.returncode == 1  # no handoff socket: nothing was started
    assert refused.stdout == "" and refused.stderr == ""
    after = len([i for i in fakebin.invocations("claude-session") if "exec" in i.argv])
    assert after == before


def test_supervise_host_needs_a_handoff(wenv: WindowEnv) -> None:
    done = wenv.run(
        "supervise", "ag-aaaaaaaa", "--host", "--state-dir", str(wenv.state_dir),
        "--config", str(wenv.config),
    )  # fmt: skip
    assert done.returncode == 2


# --- the pre-start checks fail closed for a promotion ----


def test_a_trusted_promotion_in_a_throwaway_is_refused_before_the_window_is_touched(
    wenv: WindowEnv, fakebin: Fakebin
) -> None:
    throwaway = fake_throwaway(fakebin)
    script_first_run(fakebin)
    done = wenv.run(
        *LAUNCH, "--cwd", throwaway["path"], "--reads-third-party", "--promote", "--wait"
    )
    agent_id = result_json(done)["agent"]["agent_id"]
    failure = post_park(wenv, agent_id)["chain_failure"]
    assert failure is not None
    assert (failure["tool"], failure["reason"]) == ("agent-spawn", "third_party_run")
    assert len(real_starts(fakebin)) == 1  # nothing was started
    assert not open_calls(fakebin, dry_run=True) and not open_calls(fakebin, dry_run=False)
    row = wenv.agent(agent_id)
    assert row["park_reason"] == "blocked" and row["promotion_pending"] is False
    assert any(e["event"] == "third_party_refused" for e in log_events(wenv))


def test_a_promotion_whose_records_lookup_fails_creates_no_run_and_no_window(
    wenv: WindowEnv, fakebin: Fakebin, tmp_path: Path
) -> None:
    script_first_run(fakebin)
    fakebin.script("wtctl", "list", times=2)  # the two lookups of the launch
    fakebin.script("wtctl", "list", error={"code": "failed", "message": "broken"}, exit=1)
    done = wenv.run(*LAUNCH, "--cwd", str(tmp_path), "--promote", "--wait")
    agent_id = result_json(done)["agent"]["agent_id"]
    failure = post_park(wenv, agent_id)["chain_failure"]
    assert failure is not None and failure["tool"] == "wtctl" and failure["run"] is None
    assert len(real_starts(fakebin)) == 1
    assert not open_calls(fakebin, dry_run=False)
    assert wenv.agent(agent_id)["park_reason"] == "blocked"


# --- signals, stops and failures while the window opens ----


def signal_delivered(pid: int, signum: int) -> bool:
    """Whether the process took ``signum`` (it is no longer pending for it). Where ``/proc`` does
    not exist the signal is assumed delivered (``kill`` returns after the kernel queued it)."""
    try:
        text = Path(f"/proc/{pid}/status").read_text()
    except OSError:
        return True
    for line in text.splitlines():
        if line.startswith(("SigPnd:", "ShdPnd:")) and int(line.split()[1], 16) >> (signum - 1) & 1:
            return False
    return True


def start_gated_promotion(
    wenv: WindowEnv, fakebin: Fakebin, proc_tracker: ProcTracker, tmp_path: Path
) -> tuple[str, Path, Capture]:
    """A headless agent that ended `blocked` and is promoted: its `termctl open` has created the
    pane and waits at the returned gate, the supervisor is blocked inside it."""
    capture = wenv.capture_supervisor(tmp_path / "cap")
    script_first_run(fakebin)
    gate = tmp_path / "open-gate"
    script_gated_open(fakebin, gate)
    body = result_json(wenv.run(*LAUNCH, "--promote"))
    agent_id = body["agent"]["agent_id"]
    track_agent(wenv, proc_tracker, agent_id)
    fakebin.wait_for_gate(gate)
    return agent_id, gate, capture


@pytest.mark.parametrize("sent", [signal.SIGTERM, signal.SIGHUP, signal.SIGINT])
def test_a_signal_while_the_window_opens_cancels_the_promotion_and_parks_per_the_trigger(
    wenv: WindowEnv, fakebin: Fakebin, proc_tracker: ProcTracker, tmp_path: Path, sent: int
) -> None:
    agent_id, gate, capture = start_gated_promotion(wenv, fakebin, proc_tracker, tmp_path)
    supervisor = wenv.agent(agent_id)["supervisor"]
    send_signal(supervisor, sent)
    wenv.wait_for(lambda: signal_delivered(supervisor.pid, sent), "the supervisor to take it")
    gate.write_text("")  # `termctl open` returns now
    wenv.wait_for(lambda: wenv.agent(agent_id)["status"] == "parked", "the agent to park")
    row = wenv.agent(agent_id)
    assert row["park_reason"] == "blocked" and row["needs_attention"] == "blocked"
    assert row["promotion_pending"] is False
    assert post_park(wenv, agent_id)["chain_failure"] is None  # a stop, not a failure
    assert len(execs(fakebin)) == 1  # no `exec` ran in a host
    assert len(stops(fakebin)) == 1  # the pending run was canceled exactly once
    second = runs(wenv, agent_id)[1]
    assert second["promoted"] is True and second["outcome"]["end_reason"] == "canceled"
    host = second["pane_proc"]
    assert host is not None and not wenv.alive(host)  # the host was stopped, first
    assert not [e for e in log_events(wenv) if e["event"] == "signal_sent"]  # nothing forwarded
    wenv.wait_for(lambda: wenv.panes() == [], "the dead promotion pane to be closed")
    assert capture.out.read_text() == "" and capture.err.read_text() == ""


def test_a_stop_during_the_window_open_parks_killed_without_a_chain_failure(
    wenv: WindowEnv, fakebin: Fakebin, proc_tracker: ProcTracker, tmp_path: Path
) -> None:
    agent_id, gate, capture = start_gated_promotion(wenv, fakebin, proc_tracker, tmp_path)
    csession = wenv.agent(agent_id)["csession_id"]
    # somebody else stops the pending promoted run (`claude-session stop`), the supervisor lives
    stopped = fakebin.run("claude-session", "stop", csession, "--json")
    assert stopped.returncode == 0, stopped.stderr
    gate.write_text("")
    wenv.wait_for(lambda: wenv.agent(agent_id)["status"] == "parked", "the agent to park")
    row = wenv.agent(agent_id)
    assert row["park_reason"] == "killed" and row["needs_attention"] is None
    assert row["promotion_pending"] is False
    assert post_park(wenv, agent_id)["chain_failure"] is None
    assert len(stops(fakebin)) == 1  # the supervisor canceled nothing itself
    assert execs(fakebin)[1].exit_code == 20  # the host's `exec` was refused: the run is canceled
    wenv.wait_for(lambda: host_exited(capture, 0), "the host to exit")  # and it still exits 0
    second = runs(wenv, agent_id)[1]
    assert second["outcome"]["outcome"] == "not-started"
    assert second["outcome"]["end_reason"] == "canceled"


def test_a_host_that_exits_before_its_exec_adopted_the_run_parks_per_the_trigger(
    wenv: WindowEnv, fakebin: Fakebin, tmp_path: Path
) -> None:
    capture = wenv.capture_supervisor(tmp_path / "cap")
    script_first_run(fakebin)
    # the host's `exec` is refused before it adopts anything; the run stays pending
    fakebin.script(
        "claude-session", "exec", times=1, error={"code": "refused", "message": "no"}, exit=20
    )
    fakebin.script("claude-session", "status", pending_unadopted=True, until_called="stop")
    done = wenv.run(*LAUNCH, "--promote", "--wait")
    assert done.returncode == 13, (done.stdout, done.stderr)
    agent_id = result_json(done)["agent"]["agent_id"]
    row = wenv.agent(agent_id)
    assert row["park_reason"] == "blocked" and row["needs_attention"] == "blocked"
    assert post_park(wenv, agent_id)["chain_failure"] == {
        "tool": "claude-session",
        "reason": "not_started",
        "run": 2,
        "end_reason": "canceled",
    }
    assert len(stops(fakebin)) == 1  # the supervisor's own cancel of the pending run
    wenv.wait_for(lambda: host_exited(capture, 0), "the host to exit")


def test_a_host_gone_with_its_run_still_live_has_the_run_stopped_before_the_outcome_is_read(
    wenv: WindowEnv, fakebin: Fakebin, tmp_path: Path
) -> None:
    wenv.capture_supervisor(tmp_path / "cap")
    script_first_run(fakebin)
    script_promoted_done(fakebin)
    script_live_run(fakebin, until_called="stop")  # status keeps saying "live" until a stop
    done = wenv.run(*LAUNCH, "--promote", "--wait")
    assert done.returncode == 0, (done.stdout, done.stderr)
    agent_id = result_json(done)["agent"]["agent_id"]
    assert len(stops(fakebin)) == 1
    row = wenv.agent(agent_id)
    assert row["park_reason"] == "done" and post_park(wenv, agent_id)["chain_failure"] is None


def test_a_handoff_that_times_out_with_a_live_host_stops_the_host_then_cancels_the_run(
    fakebin: Fakebin,
    isolation_env: Isolation,
    private_tmux: PrivateTmux,
    proc_tracker: ProcTracker,
    tmp_path: Path,
) -> None:
    private_tmux.start()
    env = WindowEnv(fakebin, isolation_env, private_tmux, "handoff_timeout: 1s\n")
    capture = env.capture_supervisor(tmp_path / "cap", host_gate=tmp_path / "never-opened")
    script_first_run(fakebin)
    fakebin.script("termctl", "open", argv_contains=["--dry-run"])
    body = result_json(env.run(*LAUNCH, "--promote"))
    agent_id = body["agent"]["agent_id"]
    track_agent(env, proc_tracker, agent_id)
    env.wait_for(lambda: env.agent(agent_id)["status"] == "parked", "the agent to park")
    row = env.agent(agent_id)
    assert row["park_reason"] == "blocked" and row["needs_attention"] == "blocked"
    assert post_park(env, agent_id)["chain_failure"] == {
        "tool": "agent-spawn",
        "reason": "handoff_failed",
        "run": 2,
        "end_reason": None,
    }
    second = runs(env, agent_id)[1]
    host = second["pane_proc"]  # live when the handoff timed out: it got a SIGTERM, then...
    assert host is not None and not env.alive(host)
    assert len(stops(fakebin)) == 1  # ...the pending run was canceled
    assert len(execs(fakebin)) == 1
    assert not host_exited(capture)  # the host never ran its work: it died on the signal
    env.wait_for(lambda: env.panes() == [], "the dead promotion pane to be closed")


def test_a_window_that_failed_to_open_but_created_its_pane_is_found_and_closed(
    wenv: WindowEnv, fakebin: Fakebin
) -> None:
    script_first_run(fakebin)
    fakebin.script("termctl", "open", argv_contains=["--dry-run"])
    fakebin.script(
        "termctl", "open", after_create=True, error={"code": "failed", "message": "no"}, exit=12
    )
    done = wenv.run(*LAUNCH, "--promote", "--wait")
    assert done.returncode == 13, (done.stdout, done.stderr)
    agent_id = result_json(done)["agent"]["agent_id"]
    failure = post_park(wenv, agent_id)["chain_failure"]
    assert failure["tool"] == "termctl" and failure["run"] == 2
    second = runs(wenv, agent_id)[1]
    assert second["term_id"].startswith("tc-") and second["pane_proc"] is None  # recovered
    # the panes were listed on the server the preflight named, not the default one
    sock = str(wenv.tmux.socket)
    lookups = [
        a
        for a in fakebin.argvs("termctl")
        if a[:1] == ["list"] and f"--label=agent={agent_id}" in a
    ]
    assert lookups and all(f"--socket={sock}" in a for a in lookups)
    wenv.wait_for(lambda: wenv.panes() == [], "the dead promotion pane to be closed")


def test_a_handoff_listener_that_cannot_be_created_fails_the_promotion_cleanly(
    wenv: WindowEnv, fakebin: Fakebin, proc_tracker: ProcTracker, tmp_path: Path
) -> None:
    wenv.capture_supervisor(tmp_path / "cap")
    gate = tmp_path / "first-exec"
    fakebin.script("claude-session", "exec", times=1, outcome="blocked", gate=str(gate))
    body = result_json(wenv.run(*LAUNCH, "--promote"))
    agent_id = body["agent"]["agent_id"]
    track_agent(wenv, proc_tracker, agent_id)
    fakebin.wait_for_gate(gate)
    handoff.handoff_root(wenv.state_dir).write_text("not a directory")  # nothing can listen
    gate.write_text("")
    wenv.wait_for(lambda: wenv.agent(agent_id)["status"] == "parked", "the agent to park")
    row = wenv.agent(agent_id)
    assert row["park_reason"] == "blocked" and row["needs_attention"] == "blocked"
    assert post_park(wenv, agent_id)["chain_failure"] == {
        "tool": "agent-spawn",
        "reason": "handoff_failed",
        "run": 2,
        "end_reason": None,
    }
    assert len(stops(fakebin)) == 1 and not open_calls(fakebin, dry_run=False)
    assert wenv.panes() == []


def test_window_labels_changed_while_the_window_opened_are_pushed_to_it(
    wenv: WindowEnv, fakebin: Fakebin, proc_tracker: ProcTracker, tmp_path: Path
) -> None:
    agent_id, gate, _ = start_gated_promotion(wenv, fakebin, proc_tracker, tmp_path)
    script_promoted_done(fakebin)
    # no window is recorded yet: the label is stored, not mirrored
    result_json(wenv.run("label", "add", agent_id, "extra=1", "--json"))
    assert not [a for a in fakebin.argvs("termctl") if a[:2] == ["label", "add"]]
    gate.write_text("")
    wenv.wait_for(lambda: wenv.agent(agent_id)["status"] == "parked", "the agent to park")
    pushed = [a for a in fakebin.argvs("termctl") if a[:2] == ["label", "add"]]
    assert len(pushed) == 1 and any("extra=1" in token for token in pushed[0])


def test_the_window_command_carries_absolute_paths_and_no_secret(
    wenv: WindowEnv, fakebin: Fakebin, tmp_path: Path
) -> None:
    wenv.capture_supervisor(tmp_path / "cap")
    script_first_run(fakebin)
    script_promoted_done(fakebin)
    done = wenv.run(*LAUNCH, "--promote", "--wait", extra_env={"SECRET_TOKEN": "s3cr3t-value"})
    agent_id = result_json(done)["agent"]["agent_id"]
    (argv,) = open_calls(fakebin, dry_run=False)
    cut = argv.index("--")
    options, command = argv[:cut], argv[cut + 1 :]
    assert "--keep-on-exit=failed" in options and "--json" in options
    assert not [o for o in options if o.startswith("--env")]
    assert os.path.isabs(command[0]) and command[1:4] == ["supervise", agent_id, "--host"]
    flags = dict(zip(command[4::2], command[5::2], strict=True))
    assert set(flags) == {"--handoff", "--handoff-timeout", "--state-dir", "--config"}
    for name in ("--handoff", "--state-dir", "--config"):
        assert os.path.isabs(flags[name])
    assert float(flags["--handoff-timeout"]) > 0
    assert "s3cr3t-value" not in " ".join(argv)


def test_the_agent_object_is_interactive_in_a_window_once_promoted(
    wenv: WindowEnv, fakebin: Fakebin, tmp_path: Path
) -> None:
    wenv.capture_supervisor(tmp_path / "cap")
    script_first_run(fakebin)
    script_promoted_done(fakebin)
    done = wenv.run(*LAUNCH, "--promote", "--wait")
    body = result_json(done)
    agent_id = body["agent"]["agent_id"]
    assert body["agent"]["mode"] == "interactive" and body["agent"]["placement"] == "window"
    shown = result_json(wenv.run("show", agent_id, "--json"))
    schema_registry.validate(shown, "agent-spawn/cli/show.json")
    assert shown["agent"]["mode"] == "interactive" and shown["agent"]["placement"] == "window"
    for name, payload in events(wenv, agent_id):
        if name == "post-launch" and payload["run"] == 2:
            assert payload["agent"]["mode"] == "interactive"
            assert payload["agent"]["placement"] == "window"
    parked = post_park(wenv, agent_id)
    assert parked["agent"]["mode"] == "interactive" and parked["agent"]["placement"] == "window"


# --- the host's start-up check (supervise --host) ----


def seed_host_state(env: WindowEnv, agent_id: str, **fields: Any) -> None:
    """Rewrite the agent row (status, promotion_pending, csession_id) directly: a state the
    supervisor would never produce but a stray ``--host`` process might meet."""
    columns = ", ".join(f"{name} = ?" for name in fields)
    db = sqlite3.connect(env.state_dir / "state.db")
    try:
        with db:
            db.execute(
                f"UPDATE agents SET {columns} WHERE agent_id = ?", [*fields.values(), agent_id]
            )
    finally:
        db.close()


def run_host_process(
    env: WindowEnv,
    proc_tracker: ProcTracker,
    agent_id: str,
    seed: Callable[[procs.ProcessIdentity], None],
) -> subprocess.CompletedProcess[str]:
    """Start ``supervise --host`` for the agent, call ``seed`` with its identity, then complete a
    real handoff (the test process is the listener and the host's peer is its pane PID)."""
    listener = handoff.HandoffListener.create(env.state_dir, agent_id=agent_id)
    host = subprocess.Popen(
        env.argv(
            "supervise",
            agent_id,
            "--host",
            "--handoff",
            str(listener.socket_path),
            "--handoff-timeout",
            "30",
            "--state-dir",
            str(env.state_dir),
            "--config",
            str(env.config),
        ),  # fmt: skip
        env=env.env(),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        identity = procs.read_identity(host.pid)
        assert identity is not None
        proc_tracker.record(identity)
        seed(identity)
        listener.serve(pane_pid=host.pid, pane_state="running", timeout=30, env=env.env())
        out, err = host.communicate(timeout=60)
    except BaseException:
        host.kill()  # our own child
        host.wait()
        raise
    finally:
        listener.close()
    return subprocess.CompletedProcess(host.args, host.returncode, out, err)


def parked_agent(wenv: WindowEnv) -> str:
    done = wenv.run(*LAUNCH, "--wait")  # blocked, never promoted: a parked agent with a session
    agent_id = result_json(done)["agent"]["agent_id"]
    wenv.wait_for(lambda: wenv.agent(agent_id)["status"] == "parked", "the agent to park")
    return agent_id


def set_promoted_run(env: WindowEnv, agent_id: str, **fields: Any) -> None:
    with env.store() as store:
        row = env.agent(agent_id)
        with store.lease(lifecycle.lease_key(agent_id), wait=5.0) as lease:
            store.put_run(lease, agent_id, row["run"], fields)


@pytest.mark.parametrize(
    ("case", "problem"),
    [
        ("not live", "no promotion is pending"),
        ("not pending", "no promotion is pending"),
        ("not promoted", "does not name this process as its host"),
        ("other pane process", "does not name this process as its host"),
        ("no session", "the agent has no session"),
    ],
)
def test_a_host_that_meets_a_state_without_its_promotion_exits_2_and_runs_nothing(
    wenv: WindowEnv, fakebin: Fakebin, proc_tracker: ProcTracker, case: str, problem: str
) -> None:
    script_first_run(fakebin)
    agent_id = parked_agent(wenv)
    before = len(execs(fakebin))
    calls = len(fakebin.argvs("termctl"))

    def seed(identity: procs.ProcessIdentity) -> None:
        if case == "not live":
            return  # parked, no promotion pending
        seed_host_state(wenv, agent_id, status="live", promotion_pending=int(case != "not pending"))
        if case == "not promoted":
            return
        set_promoted_run(wenv, agent_id, promoted=True)
        if case == "other pane process":
            return
        set_promoted_run(wenv, agent_id, pane_proc=identity)
        if case == "no session":
            seed_host_state(wenv, agent_id, csession_id=None)

    done = run_host_process(wenv, proc_tracker, agent_id, seed)
    assert done.returncode == 2, (done.stdout, done.stderr)
    assert done.stdout == "" and done.stderr == ""  # nothing is written to the terminal
    assert len(execs(fakebin)) == before  # nothing ran
    assert len(fakebin.argvs("termctl")) == calls
    refusals = [
        e["message"]
        for e in log_events(wenv)
        if e.get("message", "").startswith("supervise --host")
    ]
    assert refusals == [f"supervise --host refused: {problem}"] or any(
        problem in message for message in refusals
    )


def test_a_host_with_a_pending_promotion_naming_it_runs_exec_and_exits_0(
    wenv: WindowEnv, fakebin: Fakebin, proc_tracker: ProcTracker
) -> None:
    script_first_run(fakebin)
    agent_id = parked_agent(wenv)
    before = len(execs(fakebin))

    def seed(identity: procs.ProcessIdentity) -> None:
        seed_host_state(wenv, agent_id, status="live", promotion_pending=1)
        set_promoted_run(wenv, agent_id, promoted=True, pane_proc=identity)

    done = run_host_process(wenv, proc_tracker, agent_id, seed)
    assert done.returncode == 0, (done.stdout, done.stderr)  # whatever `exec` answered
    assert len(execs(fakebin)) == before + 1  # the check passed: `exec` ran, in the host


# --- units ----


def test_a_store_failure_after_promotion_pending_is_a_chain_failure() -> None:
    @contextmanager
    def lease(*_args: Any, **_kwargs: Any) -> Any:
        yield object()

    def broken(*_args: Any) -> Any:
        raise StoreError("the store is gone")

    store = SimpleNamespace(
        lease=lease, compare_and_set=lambda *_a, **_k: {"name": "n"}, get_run=broken
    )
    sup = SimpleNamespace(store=store, agent_id="ag-aaaaaaaa", promoted=False)
    trigger = SimpleNamespace(ctx=SimpleNamespace(run=1))
    with pytest.raises(chain.ChainFailure) as caught:
        promotion.start(sup, None, "cs-aaaaaaaa", trigger)  # type: ignore[arg-type]
    assert (caught.value.tool, caught.value.reason) == ("agent-spawn", "failed")


def test_a_run_is_not_promoted_after_the_supervisor_itself_was_terminated() -> None:
    result = SimpleNamespace(phase="session", exit_code=13, outcome="blocked", end_reason="normal")
    final = SimpleNamespace(result=result, ctx=SimpleNamespace(run=1))
    sup = SimpleNamespace(promoted=False, terminated=True, store=None)
    assert promotion.wants(sup, final) is False  # type: ignore[arg-type]


def test_a_supervisor_that_fails_while_watching_the_host_waits_for_the_host_to_stop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    host = SimpleNamespace()
    handle = SimpleNamespace(
        is_alive=lambda: True, wait_exit=lambda _t: True, close=lambda: calls.append("close")
    )
    monkeypatch.setattr(procs, "open_verified", lambda *_a, **_k: handle)
    monkeypatch.setattr(
        procs,
        "wait_for_exit",
        lambda *_a, **_k: calls.append("wait") or True,  # type: ignore[func-returns-value]
    )

    def failing(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("the supervisor failed")

    me = SimpleNamespace(
        signals=SimpleNamespace(forwarder="stale", take_sent=list),
        forwarder="stale",
        gateway=None,
        watch_until=failing,
        forward_host_stop=lambda _host: calls.append("signal"),
        log_sent=lambda _sent: None,
    )
    ctx = SimpleNamespace(host=host)
    with pytest.raises(RuntimeError):
        Supervisor.watch_host(me, None, "cs-aaaaaaaa", ctx)  # type: ignore[arg-type]
    assert calls == ["signal", "wait", "close"]  # told to stop, given its grace, then released
    assert me.forwarder is None and me.signals.forwarder is None
