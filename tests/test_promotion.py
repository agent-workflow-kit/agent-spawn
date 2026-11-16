"""Promotion (issue #29): a headless run that ended blocked or needs-input continues in a window
hosted by ``supervise --host``. Process tests against the fake binaries; the fake termctl starts
its panes in a real tmux server that is private to the test (``-S <tmp>/sock -f /dev/null``,
private HOME, XDG and TMUX_TMPDIR); nothing here touches the user's server."""

from __future__ import annotations

import json
import signal
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from agent_spawn import procs
from tests import schema_registry
from tests.conftest import Isolation
from tests.fakebin.harness import Fakebin
from tests.launch_support import result_json, script_hung_exec, script_live_run, track_agent
from tests.procguard import ProcTracker
from tests.private_tmux import PrivateTmux
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
    """The headless run ends ``outcome``; later `exec`s end ``done``."""
    fakebin.script("claude-session", "exec", times=1, outcome=outcome)


def test_blocked_run_is_promoted_and_the_promoted_run_ends_done(
    wenv: WindowEnv, fakebin: Fakebin, tmp_path: Path
) -> None:
    wenv.capture_supervisor(tmp_path / "cap")
    script_first_run(fakebin)
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


# --- the decision ---------------------------------------------------------------------------------


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
    body = result_json(env.run(*LAUNCH, "--wait"))
    assert body["outcome"] == "done"  # the promoted run's outcome
    assert len(real_starts(fakebin)) == 2
    # --no-promote wins over the config
    fakebin.script("claude-session", "exec", times=1, outcome="blocked")
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


@pytest.mark.parametrize("end_reason", ["stopped", "forwarded_term"])
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


# --- the chain rule ---------------------------------------------------------------------------------


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
    assert failure == {"tool": "termctl", "reason": failure["reason"], "run": None, "end_reason": None}
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
    assert failure == {"tool": "agent-spawn", "reason": "handoff_failed", "run": 2, "end_reason": None}
    assert len([a for a in fakebin.argvs("claude-session") if "stop" in a]) == 1
    # the host got no environment: it exits 1, tmux keeps the dead pane and the supervisor closes it
    wenv.wait_for(lambda: wenv.panes() == [], "the dead promotion pane to be closed")
    assert not [i for i in fakebin.invocations("claude-session") if "exec" in i.argv][1:]


# --- the host and the signals -------------------------------------------------------------------------


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


def test_signals_during_a_promotion_go_to_the_host_pane_process_only(
    wenv: WindowEnv, fakebin: Fakebin, proc_tracker: ProcTracker, tmp_path: Path
) -> None:
    agent_id, capture = start_hung_promotion(wenv, fakebin, proc_tracker, tmp_path)
    row = wenv.agent(agent_id)
    assert row["promotion_pending"] is True and row["run"] == 2 and row["exec_proc"] is None
    host = host_identity(wenv, agent_id)
    assert host != row["supervisor"]  # the host is the pane process, not the supervisor
    # SIGINT and SIGQUIT are ignored; nothing reaches exec
    send_signal(row["supervisor"], signal.SIGINT)
    send_signal(row["supervisor"], signal.SIGQUIT)
    send_signal(row["supervisor"], signal.SIGTERM)
    wenv.wait_for(lambda: wenv.agent(agent_id)["status"] == "parked", "the agent to park")
    done = wenv.agent(agent_id)
    assert done["park_reason"] == "killed" and done["needs_attention"] is None
    assert done["promotion_pending"] is False
    sent = [e for e in log_events(wenv) if e["event"] == "signal_sent"]
    # exactly two forwards, both SIGTERM: supervisor -> the host pane process, host -> exec
    assert [e["signal"] for e in sent] == ["SIGTERM", "SIGTERM"]
    assert sent[0]["pid"] == host.pid and sent[1]["pid"] != host.pid
    exec_calls = [i for i in fakebin.invocations("claude-session") if "exec" in i.argv]
    assert exec_calls[1].signals == [int(signal.SIGTERM)]  # the host forwarded SIGTERM once
    assert exec_calls[0].signals == []  # the first (headless) exec never got the signal
    wenv.wait_for(lambda: capture.host_exit_code() == 0, "the host to exit")
    assert capture.out.read_text() == "" and capture.err.read_text() == ""


def test_supervise_host_exits_0_after_exec_exits_whatever_its_outcome(
    wenv: WindowEnv, fakebin: Fakebin, tmp_path: Path
) -> None:
    capture = wenv.capture_supervisor(tmp_path / "cap")
    script_first_run(fakebin)
    fakebin.script("claude-session", "exec", times=1, outcome="failed")  # the promoted run fails
    done = wenv.run(*LAUNCH, "--promote", "--wait")
    assert done.returncode == 18, (done.stdout, done.stderr)
    assert capture.host_exit_code() == 0
    assert capture.out.read_text() == "" and capture.err.read_text() == ""
    wenv.wait_for(lambda: wenv.panes() == [], "the window to close")


def test_a_host_without_a_pending_promotion_exits_2_and_runs_nothing(
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
