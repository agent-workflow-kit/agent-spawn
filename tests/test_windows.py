"""Window-hosted agents (issue #27): launch step 7 with a window, the window-hosted supervisor and
the signals of a closed window, against the fake binaries. The fake termctl starts its panes in a
real tmux server that is private to the test (``-S <tmp>/sock -f /dev/null``, private HOME, XDG
and TMUX_TMPDIR); nothing here touches the user's server."""

from __future__ import annotations

import json
import os
import signal
from pathlib import Path
from typing import Any

import pytest

from agent_spawn import procs
from tests import schema_registry
from tests.conftest import Isolation
from tests.fakebin.harness import Fakebin
from tests.launch_support import result_json, script_hung_exec, script_live_run, track_agent
from tests.private_tmux import PrivateTmux
from tests.procguard import ProcTracker
from tests.window_support import WindowEnv

LAUNCH = ("launch", "--skill", "do-it", "--json")


@pytest.fixture
def wenv(fakebin: Fakebin, isolation_env: Isolation, private_tmux: PrivateTmux) -> WindowEnv:
    private_tmux.start()
    return WindowEnv(fakebin, isolation_env, private_tmux)


def log_events(env: WindowEnv) -> list[dict[str, Any]]:
    path = env.state_dir / "log.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def open_calls(fakebin: Fakebin, *, dry_run: bool) -> list[list[str]]:
    return [
        argv
        for argv in fakebin.argvs("termctl")
        if "open" in argv and (("--dry-run" in argv) == dry_run)
    ]


# --- launch: the happy path and its argv --------------------------------------------------------


def test_launch_in_a_window_and_the_golden_output(wenv: WindowEnv, fakebin: Fakebin) -> None:
    done = wenv.run(*LAUNCH, "--wait", "--label", "ticket=T-1", "--name", "my window")
    body = result_json(done)
    assert done.returncode == 0, (done.stdout, done.stderr)
    schema_registry.validate(body, "agent-spawn/cli/launch.json")
    agent = body["agent"]
    assert agent["mode"] == "interactive" and agent["placement"] == "window"
    assert agent["window"]["term_id"].startswith("tc-")
    assert agent["window"]["socket_path"] == str(wenv.tmux.socket)
    assert body["outcome"] == "done"
    # the preflight is the first termctl call and the real open follows it
    (dry,), (real,) = open_calls(fakebin, dry_run=True), open_calls(fakebin, dry_run=False)
    assert fakebin.argvs("termctl").index(dry) < fakebin.argvs("termctl").index(real)
    row = wenv.agent(agent["agent_id"])
    spawn = row["agent_spawn_path"]
    before = real[: real.index("--")]
    assert "--name=my window" in before
    assert f"--label=agent={agent['agent_id']}" in before
    assert "--label=ticket=T-1" in before
    assert "--label=kind=implement" in before
    assert before.index("--label=kind=implement") < before.index(
        f"--label=agent={agent['agent_id']}"
    )
    assert before.index(f"--label=agent={agent['agent_id']}") < before.index("--label=ticket=T-1")
    assert "--keep-on-exit=failed" in before and "--json" in before
    assert any(a.startswith("--actor=") for a in before)
    tail = real[real.index("--") + 1 :]
    assert tail[0] == spawn and os.path.isabs(spawn)
    assert tail[1:3] == ["supervise", agent["agent_id"]]
    assert tail[3] == "--handoff" and os.path.isabs(tail[4])
    assert tail[5] == "--handoff-timeout"
    assert tail[7:] == [
        "--state-dir", str(wenv.state_dir), "--config", row["config_path"]
    ]  # fmt: skip
    assert os.path.isabs(row["config_path"])
    # the start was interactive, and the handoff directory is gone
    (start,) = [a for a in fakebin.argvs("claude-session") if "start" in a and "--dry-run" not in a]
    assert fakebin.flag_values(
        "claude-session", "--mode", index=fakebin.argvs("claude-session").index(start)
    ) == ["interactive"]
    assert not (wenv.state_dir / "handoff").exists() or not list(
        (wenv.state_dir / "handoff").iterdir()
    )


def test_a_clean_exit_leaves_no_dead_pane(wenv: WindowEnv) -> None:
    body = result_json(wenv.run(*LAUNCH, "--wait"))
    assert body["outcome"] == "done"
    # supervise exits 0 after its bookkeeping: tmux closes the pane, none stays dead (spike B6)
    wenv.wait_for(lambda: wenv.panes() == [], "the pane to close")


def test_the_window_cwd_of_a_session_dir_run_is_its_work_directory(
    wenv: WindowEnv, fakebin: Fakebin
) -> None:
    body = result_json(wenv.run(*LAUNCH, "--wait", "--cwd", "session-dir"))
    agent_id = body["agent"]["agent_id"]
    (dry,), (real,) = open_calls(fakebin, dry_run=True), open_calls(fakebin, dry_run=False)
    assert not any(a.startswith("--cwd") for a in dry)  # the directory does not exist yet
    cwd = [a for a in real if a.startswith("--cwd=")]
    assert len(cwd) == 1 and cwd[0].endswith("/work")
    assert os.path.isabs(cwd[0][len("--cwd=") :])
    assert agent_id


def test_an_existing_cwd_goes_to_the_preflight_and_the_open(
    wenv: WindowEnv, fakebin: Fakebin, tmp_path: Path
) -> None:
    work = tmp_path / "real"
    work.mkdir()
    link = tmp_path / "link"
    link.symlink_to(work)
    result_json(wenv.run(*LAUNCH, "--wait", "--cwd", str(link)))
    for argv in (*open_calls(fakebin, dry_run=True), *open_calls(fakebin, dry_run=False)):
        assert f"--cwd={os.path.realpath(work)}" in argv  # a realpath, never the symlink
    (dry,) = open_calls(fakebin, dry_run=True)
    assert dry[dry.index("--") + 1 :][1:] == ["supervise"]  # `-- <abs path> supervise`


def test_term_placement_is_passed_and_stored(wenv: WindowEnv, fakebin: Fakebin) -> None:
    body = result_json(wenv.run(*LAUNCH, "--wait", "--term-placement", "default"))
    assert body["agent"]["term_placement"] == "default"
    for argv in (*open_calls(fakebin, dry_run=True), *open_calls(fakebin, dry_run=False)):
        assert "--placement=default" in argv


def test_labels_beyond_the_first_thirty_are_dropped_with_a_warning(
    wenv: WindowEnv, fakebin: Fakebin
) -> None:
    labels = [arg for i in range(34) for arg in ("--label", f"k{i:02d}=v")]
    body = result_json(wenv.run(*LAUNCH, "--wait", *labels))
    (real,) = open_calls(fakebin, dry_run=False)
    window = [a for a in real if a.startswith("--label=")]
    assert len(window) == 32  # kind=, agent= and the first 30
    assert window[2] == "--label=k00=v" and window[-1] == "--label=k29=v"
    assert any("left off the window" in w for w in body["warnings"])


def test_the_launchers_environment_reaches_the_run(wenv: WindowEnv, fakebin: Fakebin) -> None:
    result_json(wenv.run(*LAUNCH, "--wait", extra_env={"AGENT_SPAWN_HANDED_OVER": "yes"}))
    (exec_call,) = [i for i in fakebin.invocations("claude-session") if "exec" in i.argv]
    assert exec_call.env.get("AGENT_SPAWN_HANDED_OVER") == "yes"  # not in the tmux server's
    assert exec_call.stdin_is_tty  # the window's terminal is exec's stdin


# --- launch: failures before a supervisor exists ----------------------------------------------


def preflight_error(fakebin: Fakebin, code: str, exit_code: int) -> None:
    fakebin.script(
        "termctl", "open", argv_contains=["--dry-run"], error={"code": code, "message": "no"},
        exit=exit_code,
    )  # fmt: skip


def test_a_preflight_failure_exits_12_before_any_hook_or_allocation(
    wenv: WindowEnv, fakebin: Fakebin, tmp_path: Path
) -> None:
    sentinel = tmp_path / "hook-ran"
    preflight_error(fakebin, "tmux_missing", 12)
    done = wenv.run(*LAUNCH, "--hook", f"pre-launch=touch {sentinel}")
    error = result_json(done)["error"]
    assert done.returncode == 12
    assert (error["code"], error["tool"], error["lower_code"]) == (
        "failed",
        "termctl",
        "tmux_missing",
    )
    assert "--placement here" in error["details"]["hint"]
    assert not sentinel.exists() and wenv.agents() == []
    assert not [a for a in fakebin.argvs("claude-session") if "start" in a and "--dry-run" not in a]
    assert open_calls(fakebin, dry_run=False) == []
    assert not (wenv.state_dir / "prompts").exists() or not list(
        (wenv.state_dir / "prompts").iterdir()
    )


def test_a_preflight_usage_error_exits_2(wenv: WindowEnv, fakebin: Fakebin) -> None:
    preflight_error(fakebin, "not_supported", 2)
    done = wenv.run(*LAUNCH)
    error = result_json(done)["error"]
    assert done.returncode == 2
    assert (error["code"], error["tool"], error["lower_code"]) == (
        "usage",
        "termctl",
        "not_supported",
    )
    assert wenv.agents() == []


def test_termctl_missing_exits_12_before_anything_changes(wenv: WindowEnv) -> None:
    done = wenv.run(
        *LAUNCH, extra_env={"PATH": "/usr/bin:/bin"}
    )  # the fakes' directory is not on PATH
    assert done.returncode == 12 and result_json(done)["error"]["code"] == "tool_missing"
    assert wenv.agents() == []


def test_a_headless_launch_needs_no_termctl(wenv: WindowEnv, fakebin: Fakebin) -> None:
    result_json(wenv.run("launch", "--mode", "headless", "--skill", "do-it", "--json", "--wait"))
    fakebin.assert_not_called("termctl")


def fail_real_open(fakebin: Fakebin, exit_code: int, code: str = "x") -> None:
    """Let the preflight (the first `open`) pass and fail the real open."""
    fakebin.script("termctl", "open", times=1)
    fakebin.script("termctl", "open", error={"code": code, "message": "m"}, exit=exit_code)


@pytest.mark.parametrize(
    ("exit_code", "reason", "attention", "escalated_by", "code"),
    [
        (11, "vetoed", None, None, "vetoed"),
        (13, "escalated", "escalated", "termctl", "escalated"),
        (12, "launch_failed", "launch_failed", None, "failed"),
    ],
)
def test_a_failed_open_stops_the_pending_run_and_parks_once(
    wenv: WindowEnv,
    fakebin: Fakebin,
    exit_code: int,
    reason: str,
    attention: str | None,
    escalated_by: str | None,
    code: str,
) -> None:
    fail_real_open(fakebin, exit_code)
    done = wenv.run(*LAUNCH)
    error = result_json(done)["error"]
    assert done.returncode == exit_code
    assert (error["code"], error["tool"]) == (code, "termctl")
    (agent,) = wenv.agents()
    assert error["agent_id"] == agent["agent_id"]
    assert agent["status"] == "parked" and agent["park_reason"] == reason
    assert agent["needs_attention"] == attention and agent["escalated_by"] == escalated_by
    assert agent["supervisor"] is None
    # no supervisor exists: the launcher canceled the pending run, exactly once
    stops = [a for a in fakebin.argvs("claude-session") if "stop" in a]
    assert len(stops) == 1 and agent["csession_id"] in stops[0]
    assert not (wenv.state_dir / "handoff").exists() or not list(
        (wenv.state_dir / "handoff").iterdir()
    )


def test_a_pane_without_a_pid_fails_the_handoff_and_keeps_a_dead_pane(
    wenv: WindowEnv, fakebin: Fakebin
) -> None:
    fakebin.script("termctl", "open", times=1)
    fakebin.script("termctl", "open", merge={"pid": None})
    done = wenv.run(*LAUNCH)
    error = result_json(done)["error"]
    assert done.returncode == 12
    assert (error["code"], error["details"]["cause"]) == ("handoff_failed", "no_pane_pid")
    schema_registry.validate(result_json(done), "agent-spawn/cli/error.json")
    (agent,) = wenv.agents()
    assert agent["status"] == "parked" and agent["park_reason"] == "launch_failed"
    assert agent["needs_attention"] == "launch_failed" and agent["supervisor"] is None
    assert len([a for a in fakebin.argvs("claude-session") if "stop" in a]) == 1
    assert fakebin.records("exit")  # the fakes ran
    assert any(e["event"] == "handoff_failed" for e in log_events(wenv))
    # the window process never got an environment: it exits 1 and tmux keeps the dead pane (B6)
    wenv.wait_for(
        lambda: len(wenv.panes()) == 1 and wenv.panes()[0]["state"] == "dead", "dead pane"
    )
    assert wenv.panes()[0]["exit_status"] == 1
    # nothing ran: no exec, and the agent was not made live by that process
    assert not [i for i in fakebin.invocations("claude-session") if "exec" in i.argv]
    log = wenv.state_dir / "supervise" / f"{agent['agent_id']}.log"
    assert not log.exists() or "Traceback" not in log.read_text()


def test_a_dead_pane_is_closed_by_its_label_and_never_a_running_one(
    wenv: WindowEnv, fakebin: Fakebin
) -> None:
    fakebin.script("termctl", "open", times=1)
    fakebin.script("termctl", "open", merge={"pid": None})
    wenv.run(*LAUNCH)
    wenv.wait_for(lambda: bool(wenv.panes()) and wenv.panes()[0]["state"] == "dead", "dead pane")
    assert [p["labels"] for p in wenv.panes()][0][0].startswith("agent=")


# --- the window-hosted supervisor ---------------------------------------------------------------


def launch_live(wenv: WindowEnv, fakebin: Fakebin, proc_tracker: ProcTracker) -> dict[str, Any]:
    """A live window agent whose `exec` runs until it is signaled."""
    script_hung_exec(fakebin)
    script_live_run(fakebin)
    body = result_json(wenv.run(*LAUNCH))
    agent = body["agent"]
    assert agent["status"] == "live", body
    track_agent(wenv, proc_tracker, agent["agent_id"])
    return agent


def exec_signals(fakebin: Fakebin) -> list[int]:
    (call,) = [i for i in fakebin.invocations("claude-session") if "exec" in i.argv]
    return call.signals


def test_closing_the_window_parks_killed_without_attention(
    wenv: WindowEnv, fakebin: Fakebin, proc_tracker: ProcTracker
) -> None:
    agent = launch_live(wenv, fakebin, proc_tracker)
    agent_id = agent["agent_id"]
    (pane,) = wenv.panes()
    assert pane["pid"] == wenv.pane_pid(agent_id) and pane["state"] == "running"
    supervisor = wenv.agent(agent_id)["supervisor"]
    assert supervisor.pid == pane["pid"]  # A8: the pane's PID is the supervisor
    done = wenv.tmux.run("kill-pane", "-t", pane["pane_id"])
    assert done.returncode == 0, done.stderr
    wenv.wait_for(lambda: wenv.agent(agent_id)["status"] == "parked", "the agent to park")
    row = wenv.agent(agent_id)
    assert (row["park_reason"], row["needs_attention"]) == ("killed", None)
    wenv.wait_for(lambda: not procs.is_alive(supervisor), "the supervisor to exit")
    # SIGHUP reached `exec` only through the supervisor's forward (B1-B3), once
    assert exec_signals(fakebin).count(int(signal.SIGHUP)) == 1
    sent = [e for e in log_events(wenv) if e["event"] == "signal_sent"]
    assert [e["signal"] for e in sent] == ["SIGHUP"]
    # B2: the supervisor wrote nothing after the hangup and finished its bookkeeping
    log = wenv.state_dir / "supervise" / f"{agent_id}.log"
    assert not log.exists() or "Traceback" not in log.read_text()


def test_sigint_and_sigquit_are_ignored_by_a_window_supervisor(
    wenv: WindowEnv, fakebin: Fakebin, proc_tracker: ProcTracker
) -> None:
    agent = launch_live(wenv, fakebin, proc_tracker)
    agent_id = agent["agent_id"]
    supervisor = wenv.agent(agent_id)["supervisor"]
    with procs.open_verified(supervisor) as handle:
        assert handle is not None
        handle.send(signal.SIGINT)
        handle.send(signal.SIGQUIT)
    (pane,) = wenv.panes()
    # still the supervisor, still live: the ignored signals did nothing and reached nobody
    assert procs.is_alive(supervisor) and wenv.agent(agent_id)["status"] == "live"
    assert exec_signals(fakebin) == []
    wenv.tmux.run("kill-pane", "-t", pane["pane_id"])
    wenv.wait_for(lambda: wenv.agent(agent_id)["status"] == "parked", "the agent to park")
    assert exec_signals(fakebin).count(int(signal.SIGINT)) == 0
    assert exec_signals(fakebin).count(int(signal.SIGQUIT)) == 0


def test_sigterm_of_a_window_supervisor_is_forwarded_to_exec(
    wenv: WindowEnv, fakebin: Fakebin, proc_tracker: ProcTracker
) -> None:
    agent = launch_live(wenv, fakebin, proc_tracker)
    agent_id = agent["agent_id"]
    supervisor = wenv.agent(agent_id)["supervisor"]
    with procs.open_verified(supervisor) as handle:
        assert handle is not None
        handle.send(signal.SIGTERM)
    wenv.wait_for(lambda: wenv.agent(agent_id)["status"] == "parked", "the agent to park")
    assert int(signal.SIGTERM) in exec_signals(fakebin)
    assert wenv.agent(agent_id)["needs_attention"] is None


def test_a_sigkilled_window_supervisor_leaves_a_live_agent_for_liveness(
    wenv: WindowEnv, fakebin: Fakebin, proc_tracker: ProcTracker
) -> None:
    agent = launch_live(wenv, fakebin, proc_tracker)
    agent_id = agent["agent_id"]
    supervisor = wenv.agent(agent_id)["supervisor"]
    with procs.open_verified(supervisor) as handle:
        assert handle is not None
        handle.send(signal.SIGKILL)
    wenv.wait_for(lambda: not procs.is_alive(supervisor), "the supervisor to die")
    # Nothing else parks the agent: it stays `live` with a dead supervisor (spike B9). Setting
    # `supervisor_lost` is the liveness check's job.  TODO(#31): assert supervisor_lost here.
    row = wenv.agent(agent_id)
    assert row["status"] == "live" and row["park_reason"] is None


def test_a_supervisor_with_a_failing_start_writes_nothing_to_its_terminal(
    wenv: WindowEnv, fakebin: Fakebin
) -> None:
    """`supervise` refuses an unknown agent (exit 2): the pane shows nothing but the shell's echo."""
    from tests.launch_support import agent_spawn_path

    config = wenv.config
    config.parent.mkdir(parents=True, exist_ok=True)
    state = wenv.state_dir
    state.mkdir(parents=True, exist_ok=True)
    marker = state.parent / "rc"
    done = wenv.tmux.run(
        "new-window", "-d", "-P", "-F", "#{pane_id}", "--", "sh", "-c",
        f"{agent_spawn_path()} supervise ag-aaaaaaaa --state-dir {state} --config {config}; "
        f"echo rc=$? > {marker}; sleep 30",
    )  # fmt: skip
    pane = done.stdout.strip()
    wenv.wait_for(marker.exists, "the supervisor to exit")
    assert marker.read_text().strip() == "rc=2"
    screen = wenv.tmux.run("capture-pane", "-p", "-t", pane).stdout
    assert screen.strip() == ""  # no error message, no usage text, nothing
