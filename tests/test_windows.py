"""Window-hosted agents (issue #27): launch step 7 with a window, the window-hosted supervisor and
the signals of a closed window, against the fake binaries. The fake termctl starts its panes in a
real tmux server that is private to the test (``-S <tmp>/sock -f /dev/null``, private HOME, XDG
and TMUX_TMPDIR); nothing here touches the user's server."""

from __future__ import annotations

import json
import os
import shlex
import signal
import subprocess
from pathlib import Path
from typing import Any

import pytest

from agent_spawn import procs
from tests import schema_registry
from tests.conftest import Isolation
from tests.fakebin.harness import Fakebin
from tests.launch_support import (
    result_json,
    script_done_run,
    script_hung_exec,
    script_live_run,
    track_agent,
)
from tests.private_tmux import PrivateTmux
from tests.procguard import ProcTracker
from tests.window_support import WindowEnv, send_signal

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


def parked_records(env: WindowEnv) -> list[dict[str, Any]]:
    """The log's status changes to ``parked``."""
    return [
        e
        for e in log_events(env)
        if e["event"] == "status_change" and (e.get("details") or {}).get("to") == "parked"
    ]


def open_calls(fakebin: Fakebin, *, dry_run: bool) -> list[list[str]]:
    return [
        argv
        for argv in fakebin.argvs("termctl")
        if "open" in argv and (("--dry-run" in argv) == dry_run)
    ]


# --- launch: the happy path and its argv --------------------------------------------------------


def test_launch_in_a_window_and_the_golden_output(wenv: WindowEnv, fakebin: Fakebin) -> None:
    script_done_run(wenv.fakebin)
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
    script_done_run(wenv.fakebin)
    body = result_json(wenv.run(*LAUNCH, "--wait"))
    assert body["outcome"] == "done"
    # supervise exits 0 after its bookkeeping: tmux closes the pane, none stays dead (spike B6)
    wenv.wait_for(lambda: wenv.panes() == [], "the pane to close")


def test_the_window_cwd_of_a_session_dir_run_is_its_work_directory(
    wenv: WindowEnv, fakebin: Fakebin
) -> None:
    script_done_run(wenv.fakebin)
    body = result_json(wenv.run(*LAUNCH, "--wait", "--cwd", "session-dir"))
    agent_id = body["agent"]["agent_id"]
    (dry,), (real,) = open_calls(fakebin, dry_run=True), open_calls(fakebin, dry_run=False)
    assert not any(a.startswith("--cwd") for a in dry)  # the directory does not exist yet
    cwd = [a for a in real if a.startswith("--cwd=")]
    assert len(cwd) == 1 and cwd[0].endswith("/work")
    assert os.path.isabs(cwd[0][len("--cwd=") :])
    session = wenv.agent(agent_id)["csession_id"]
    status = fakebin.run("claude-session", "status", session, "--json", env=wenv.env())
    directory = json.loads(status.stdout)["csession_dir"]
    assert cwd[0] == f"--cwd={os.path.realpath(directory)}/work"


def test_an_existing_cwd_goes_to_the_preflight_and_the_open(
    wenv: WindowEnv, fakebin: Fakebin, tmp_path: Path
) -> None:
    script_done_run(wenv.fakebin)
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
    script_done_run(wenv.fakebin)
    body = result_json(wenv.run(*LAUNCH, "--wait", "--term-placement", "default"))
    assert body["agent"]["term_placement"] == "default"
    for argv in (*open_calls(fakebin, dry_run=True), *open_calls(fakebin, dry_run=False)):
        assert "--placement=default" in argv


def test_labels_beyond_the_first_thirty_are_dropped_with_a_warning(
    wenv: WindowEnv, fakebin: Fakebin
) -> None:
    script_done_run(wenv.fakebin)
    labels = [arg for i in range(34) for arg in ("--label", f"k{i:02d}=v")]
    body = result_json(wenv.run(*LAUNCH, "--wait", *labels))
    (real,) = open_calls(fakebin, dry_run=False)
    window = [a for a in real if a.startswith("--label=")]
    assert len(window) == 32  # kind=, agent= and the first 30
    assert window[2] == "--label=k00=v" and window[-1] == "--label=k29=v"
    assert any("left off the window" in w for w in body["warnings"])


def test_the_launchers_environment_reaches_the_run(wenv: WindowEnv, fakebin: Fakebin) -> None:
    script_done_run(wenv.fakebin)
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
    script_done_run(wenv.fakebin)
    result_json(wenv.run("launch", "--mode", "headless", "--skill", "do-it", "--json", "--wait"))
    fakebin.assert_not_called("termctl")


def fail_real_open(fakebin: Fakebin, exit_code: int, code: str = "x") -> None:
    """Let the preflight (the first `open`) pass and fail the real open."""
    fakebin.script("termctl", "open", times=1)
    fakebin.script("termctl", "open", times=1, error={"code": code, "message": "m"}, exit=exit_code)


@pytest.mark.parametrize(
    ("exit_code", "reason", "attention", "escalated_by", "code"),
    [
        (11, "vetoed", None, None, "vetoed"),
        (13, "escalated", "escalated", "termctl", "escalated"),
        (12, "launch_failed", "launch_failed", None, "failed"),
        (2, "launch_failed", "launch_failed", None, "failed"),  # the preflight passed: not usage
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
    assert done.returncode == (12 if exit_code == 2 else exit_code)
    assert (error["code"], error["tool"]) == (code, "termctl")
    (agent,) = wenv.agents()
    assert error["agent_id"] == agent["agent_id"]
    assert agent["status"] == "parked" and agent["park_reason"] == reason
    assert agent["needs_attention"] == attention and agent["escalated_by"] == escalated_by
    assert agent["supervisor"] is None
    # no supervisor exists: the launcher canceled the pending run, exactly once
    stops = [a for a in fakebin.argvs("claude-session") if "stop" in a]
    assert len(stops) == 1 and agent["csession_id"] in stops[0]
    assert len(parked_records(wenv)) == 1  # one process parked it, once
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
    # B2: the supervisor finished its bookkeeping after the hangup (that it wrote nothing and
    # exited 0 is asserted by test_a_supervisor_whose_terminal_hangs_up_exits_0_...)


def test_sigint_and_sigquit_are_ignored_by_a_window_supervisor(
    wenv: WindowEnv, fakebin: Fakebin, proc_tracker: ProcTracker
) -> None:
    agent = launch_live(wenv, fakebin, proc_tracker)
    agent_id = agent["agent_id"]
    supervisor = wenv.agent(agent_id)["supervisor"]
    send_signal(supervisor, signal.SIGINT)
    send_signal(supervisor, signal.SIGQUIT)
    (pane,) = wenv.panes()
    # still the supervisor, still live: the ignored signals did nothing and reached nobody
    assert procs.is_alive(supervisor) and wenv.agent(agent_id)["status"] == "live"
    wenv.tmux.run("kill-pane", "-t", pane["pane_id"])
    wenv.wait_for(lambda: wenv.agent(agent_id)["status"] == "parked", "the agent to park")
    wenv.wait_for(lambda: not procs.is_alive(supervisor), "the supervisor to exit")
    # the call's signals are complete once it ended: only the hangup was ever forwarded
    assert exec_signals(fakebin) == [int(signal.SIGHUP)]


def test_sigterm_of_a_window_supervisor_is_forwarded_to_exec(
    wenv: WindowEnv, fakebin: Fakebin, proc_tracker: ProcTracker
) -> None:
    agent = launch_live(wenv, fakebin, proc_tracker)
    agent_id = agent["agent_id"]
    supervisor = wenv.agent(agent_id)["supervisor"]
    send_signal(supervisor, signal.SIGTERM)
    wenv.wait_for(lambda: wenv.agent(agent_id)["status"] == "parked", "the agent to park")
    assert int(signal.SIGTERM) in exec_signals(fakebin)
    assert wenv.agent(agent_id)["needs_attention"] is None


def test_a_sigkilled_window_supervisor_leaves_a_live_agent_for_liveness(
    wenv: WindowEnv, fakebin: Fakebin, proc_tracker: ProcTracker
) -> None:
    agent = launch_live(wenv, fakebin, proc_tracker)
    agent_id = agent["agent_id"]
    supervisor = wenv.agent(agent_id)["supervisor"]
    send_signal(supervisor, signal.SIGKILL)
    wenv.wait_for(lambda: not procs.is_alive(supervisor), "the supervisor to die")
    # Nothing else parks the agent: it stays `live` with a dead supervisor (spike B9) until the
    # liveness check looks: the session is still live, so it is flagged, once, and not parked.
    row = wenv.agent(agent_id)
    assert row["status"] == "live" and row["park_reason"] is None
    assert row["supervisor_lost"] is False
    for command in ("show", "show"):  # the second look changes nothing and queues nothing more
        shown = result_json(wenv.run(command, agent_id, "--json"))
        assert shown["agent"]["status"] == "live"
        assert shown["agent"]["supervisor_lost"] is True
        assert shown["agent"]["needs_attention"] == "lost"
        assert shown["supervisor"]["alive"] is False
    row = wenv.agent(agent_id)
    assert row["status"] == "live" and row["supervisor_lost"] is True
    with wenv.store() as store:
        escalations = [e for e in store.pending_events(agent_id) if e["event"] == "post-escalate"]
    assert len(escalations) == 1
    # liveness never stops the run (the remedy is `agent-spawn stop`)
    assert [a for a in fakebin.argvs("claude-session") if a[0] == "stop"] == []


def test_a_window_supervisor_that_fails_writes_nothing_to_its_terminal(
    wenv: WindowEnv, tmp_path: Path
) -> None:
    """A failed handoff (exit 1) writes nothing: no message, no usage text, on either stream."""
    from tests.launch_support import agent_spawn_path

    state = wenv.state_dir
    state.mkdir(parents=True, exist_ok=True)
    out, err, marker = tmp_path / "out", tmp_path / "err", tmp_path / "rc"
    script = (
        f"{shlex.quote(agent_spawn_path())} supervise ag-aaaaaaaa "
        f"--handoff {shlex.quote(str(state))}/nothing --handoff-timeout 1 "
        f"--state-dir {shlex.quote(str(state))} --config {shlex.quote(str(wenv.config))} "
        f">{shlex.quote(str(out))} 2>{shlex.quote(str(err))}; "
        f"echo $? > {shlex.quote(str(marker))}.tmp; "
        f"mv {shlex.quote(str(marker))}.tmp {shlex.quote(str(marker))}"
    )  # the exit code lands in the marker atomically, after the supervisor has finished
    done = wenv.tmux.run("new-window", "-d", "--", "sh", "-c", script)
    assert done.returncode == 0, done.stderr
    wenv.wait_for(marker.exists, "the supervisor to exit")
    assert marker.read_text().strip() == "1"
    assert out.read_text() == "" and err.read_text() == ""
    assert any(e["event"] == "handoff_failed" for e in log_events(wenv))


def test_a_supervisor_whose_terminal_hangs_up_exits_0_having_written_nothing(
    wenv: WindowEnv, fakebin: Fakebin, proc_tracker: ProcTracker, tmp_path: Path
) -> None:
    capture = wenv.capture_supervisor(tmp_path / "capture")
    agent = launch_live(wenv, fakebin, proc_tracker)
    (pane,) = wenv.panes()
    assert wenv.tmux.run("kill-pane", "-t", pane["pane_id"]).returncode == 0
    wenv.wait_for(lambda: wenv.agent(agent["agent_id"])["status"] == "parked", "the agent to park")
    wenv.wait_for(capture.rc.exists, "the supervisor to finish")
    assert capture.exit_code() == 0  # it did not crash on the hung-up terminal
    assert capture.out.read_text() == "" and capture.err.read_text() == ""


def test_a_refused_window_supervisor_exits_2_and_prints_nothing(
    wenv: WindowEnv, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The record does not name this process: exit 2, logged, nothing on stdout or stderr."""
    from agent_spawn import cli, handoff

    monkeypatch.setattr(handoff, "receive_environment", lambda *a, **k: dict(os.environ))
    wenv.state_dir.mkdir(parents=True, exist_ok=True)
    with wenv.store_create() as store:
        assert store is not None
    code = cli.main(
        [
            "supervise", "ag-aaaaaaaa", "--handoff", "/x", "--handoff-timeout", "1",
            "--state-dir", str(wenv.state_dir), "--config", str(wenv.config),
        ]
    )  # fmt: skip
    captured = capsys.readouterr()
    assert code == 2 and captured.out == "" and captured.err == ""
    assert "supervise refused" in (wenv.state_dir / "log.jsonl").read_text()


def test_the_handed_environment_replaces_the_panes_and_the_launchers_overrides_apply(
    wenv: WindowEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    from agent_spawn import cli, handoff
    from agent_spawn.supervise import Signals, Supervisor

    handed = {"HOME": "/handed", "AGENT_SPAWN_TOMBSTONE_DAYS": "7", "KEPT": "yes"}
    argv = [
        "supervise", "ag-aaaaaaaa", "--handoff", "/x", "--handoff-timeout", "1",
        "--state-dir", str(wenv.state_dir), "--config", str(wenv.config),
    ]  # fmt: skip
    monkeypatch.setattr(handoff, "receive_environment", lambda *a, **k: dict(handed))
    with monkeypatch.context() as scoped:
        scoped.setattr(os, "environ", {"AGENT_SPAWN_TOMBSTONE_DAYS": "99", "PANE_ONLY": "1"})
        inv = cli.parse_invocation(argv)
        assert (
            inv.config is not None and inv.config.tombstone_days == 99
        )  # loaded from the pane's environment
        supervisor = Supervisor(inv, Signals(interactive=True))
        assert supervisor.adopt_handoff() is None
        assert dict(os.environ) == handed  # the pane's own variables are gone
        assert supervisor.config.tombstone_days == 7
        assert inv.config is not None and inv.config.tombstone_days == 7


def test_a_failed_handoff_exits_1_and_starts_nothing(
    wenv: WindowEnv, fakebin: Fakebin, monkeypatch: pytest.MonkeyPatch
) -> None:
    from agent_spawn import cli, handoff
    from agent_spawn.supervise import Signals, Supervisor

    def refuse(*args: object, **kwargs: object) -> dict[str, str]:
        raise handoff.HandoffFailed("timeout", "x")

    monkeypatch.setattr(handoff, "receive_environment", refuse)
    inv = cli.parse_invocation(
        [
            "supervise", "ag-aaaaaaaa", "--handoff", "/x", "--handoff-timeout", "1",
            "--state-dir", str(wenv.state_dir), "--config", str(wenv.config),
        ]
    )  # fmt: skip
    outcome = Supervisor(inv, Signals(interactive=True)).adopt_handoff()
    assert outcome is not None and outcome.exit_code == 1 and outcome.quiet
    fakebin.assert_not_called("claude-session")


# --- resume ------------------------------------------------------------------------------------


def launched_window(wenv: WindowEnv, *extra: str) -> str:
    body = result_json(wenv.run(*LAUNCH, "--wait", *extra))
    return body["agent"]["agent_id"]


def test_resume_opens_a_new_window_and_records_it_on_the_new_run(
    wenv: WindowEnv, fakebin: Fakebin
) -> None:
    script_done_run(wenv.fakebin)
    agent_id = launched_window(wenv)
    done = wenv.run("resume", agent_id, "--json", "--wait")
    body = result_json(done)
    assert done.returncode == 0, (done.stdout, done.stderr)
    schema_registry.validate(body, "agent-spawn/cli/launch.json")
    assert body["agent"]["mode"] == "interactive" and body["agent"]["placement"] == "window"
    assert body["agent"]["run"] == 2 and body["agent"]["window"] is not None
    assert (
        len(open_calls(fakebin, dry_run=False)) == 2 and len(open_calls(fakebin, dry_run=True)) == 2
    )
    with wenv.store() as store:
        first, second = store.get_run(agent_id, 1), store.get_run(agent_id, 2)
    assert first is not None and second is not None
    assert first["term_id"] != second["term_id"] and second["mode"] == "interactive"
    assert second["placement"] == "window" and second["pane_proc"] is not None


def test_resume_name_and_term_placement_replace_the_stored_values(
    wenv: WindowEnv, fakebin: Fakebin
) -> None:
    script_done_run(wenv.fakebin)
    agent_id = launched_window(wenv, "--name", "first")
    result_json(wenv.run("resume", agent_id, "--json", "--wait", "--name", "second",
                         "--term-placement", "default"))  # fmt: skip
    last = open_calls(fakebin, dry_run=False)[-1]
    assert "--name=second" in last and "--placement=default" in last
    row = wenv.agent(agent_id)
    assert row["name"] == "second" and row["term_placement"] == "default"


def test_a_headless_agent_resumed_interactively_gets_a_window(
    wenv: WindowEnv, fakebin: Fakebin
) -> None:
    script_done_run(wenv.fakebin)
    body = result_json(
        wenv.run("launch", "--mode", "headless", "--skill", "do-it", "--json", "--wait")
    )
    agent_id = body["agent"]["agent_id"]
    fakebin.assert_not_called("termctl")
    done = wenv.run("resume", agent_id, "--mode", "interactive", "--json", "--wait")
    assert done.returncode == 0, (done.stdout, done.stderr)
    assert result_json(done)["agent"]["window"] is not None
    assert len(open_calls(fakebin, dry_run=False)) == 1


def test_a_resume_preflight_failure_restores_the_pre_resume_row(
    wenv: WindowEnv, fakebin: Fakebin
) -> None:
    script_done_run(wenv.fakebin)
    agent_id = launched_window(wenv)
    before = wenv.agent(agent_id)
    preflight_error(fakebin, "tmux_too_old", 12)
    done = wenv.run("resume", agent_id, "--json")
    error = result_json(done)["error"]
    assert done.returncode == 12 and error["lower_code"] == "tmux_too_old"
    after = wenv.agent(agent_id)
    assert after["status"] == "parked" and after["updated"] == before["updated"]
    assert len(open_calls(fakebin, dry_run=False)) == 1  # only the launch's


def test_resume_override_escalation_goes_to_termctl_open_only(
    wenv: WindowEnv, fakebin: Fakebin
) -> None:
    script_done_run(wenv.fakebin)
    fail_real_open(fakebin, 13, "escalated")
    done = wenv.run(*LAUNCH)
    assert done.returncode == 13
    (agent,) = wenv.agents()
    assert agent["escalated_by"] == "termctl"
    result_json(
        wenv.run(
            "resume",
            agent["agent_id"],
            "--json",
            "--wait",
            "--override-escalation",
            "--actor",
            "human:tester",
        )  # fmt: skip
    )
    real = open_calls(fakebin, dry_run=False)
    assert "--override-escalation" not in real[0] and "--override-escalation" in real[-1]
    assert not [a for a in open_calls(fakebin, dry_run=True) if "--override-escalation" in a]
    starts = [
        a for a in fakebin.argvs("claude-session") if "start" in a and "--resume" in " ".join(a)
    ]
    assert starts and all("--override-escalation" not in a for a in starts)
    audit = (wenv.state_dir / "audit.jsonl").read_text()
    assert "resume_escalation_override" in audit


def test_a_fork_passes_the_override_to_termctl_open_only(wenv: WindowEnv, fakebin: Fakebin) -> None:
    script_done_run(wenv.fakebin)
    agent_id = launched_window(wenv)
    done = wenv.run("resume", agent_id, "--fork", "--json", "--wait", "--override-escalation",
                    "--actor", "human:tester")  # fmt: skip
    assert done.returncode == 0, (done.stdout, done.stderr)
    assert "--override-escalation" in open_calls(fakebin, dry_run=False)[-1]
    forks = [a for a in fakebin.argvs("claude-session") if "--fork" in a]
    assert forks and all("--override-escalation" not in a for a in forks)


def test_override_without_a_target_still_exits_2(wenv: WindowEnv, fakebin: Fakebin) -> None:
    script_done_run(wenv.fakebin)
    agent_id = launched_window(wenv)
    done = wenv.run(
        "resume", agent_id, "--json", "--override-escalation", "--actor", "human:tester"
    )
    assert done.returncode == 2
    assert wenv.agent(agent_id)["status"] == "parked"


def test_dry_runs_report_the_window_plan(wenv: WindowEnv, fakebin: Fakebin) -> None:
    script_done_run(wenv.fakebin)
    done = wenv.run(*LAUNCH, "--dry-run")
    plan = result_json(done)
    assert done.returncode == 0, (done.stdout, done.stderr)
    schema_registry.validate(plan, "agent-spawn/cli/launch-dry-run.json")
    assert plan["tools"]["termctl"] is not None and plan["term_plan"]["match"]["rule"] is None
    assert plan["term_plan"]["match"]["labels"][0].startswith("agent=")
    assert plan["agent_settings"]["mode"] == "interactive"
    assert plan["agent_settings"]["placement"] == "window"
    assert open_calls(fakebin, dry_run=False) == [] and wenv.agents() == []
    agent_id = launched_window(wenv)
    done = wenv.run("resume", agent_id, "--dry-run", "--json")
    plan = result_json(done)
    assert done.returncode == 0, (done.stdout, done.stderr)
    schema_registry.validate(plan, "agent-spawn/cli/launch-dry-run.json")
    assert plan["term_plan"] is not None and plan["tools"]["termctl"] is not None


def test_a_dry_run_fails_like_the_real_run_when_the_preflight_does(
    wenv: WindowEnv, fakebin: Fakebin
) -> None:
    preflight_error(fakebin, "tmux_missing", 12)
    done = wenv.run(*LAUNCH, "--dry-run")
    assert done.returncode == 12 and result_json(done)["error"]["lower_code"] == "tmux_missing"


# --- teardown: the window step -----------------------------------------------------------------


def dead_pane_agent(wenv: WindowEnv, fakebin: Fakebin) -> str:
    """A parked agent whose window process failed at the handoff: a dead pane stays (B6)."""
    fakebin.script("termctl", "open", times=1)
    fakebin.script("termctl", "open", times=1, merge={"pid": None})
    done = wenv.run(*LAUNCH)
    assert done.returncode == 12
    (agent,) = wenv.agents()
    wenv.wait_for(lambda: any(p["state"] == "dead" for p in wenv.panes()), "dead pane")
    return agent["agent_id"]


def term_calls(fakebin: Fakebin, command: str) -> list[list[str]]:
    """The calls of one command (the tests' own `list` views, without `--dead`, do not count)."""
    return [
        a
        for a in fakebin.argvs("termctl")
        if a and a[0] == command and (command != "list" or "--dead" in a)
    ]


def test_teardown_closes_the_dead_panes_of_the_agent_only(
    wenv: WindowEnv, fakebin: Fakebin
) -> None:
    agent_id = dead_pane_agent(wenv, fakebin)
    other = wenv.fakebin.run(
        "termctl", "open", "--name=other", "--label=agent=ag-zzzzzzzz", "--keep-on-exit=failed",
        "--json", "--", "sh", "-c", "exit 3", env=wenv.env(),
    )  # fmt: skip
    assert other.returncode == 0, other.stderr
    wenv.wait_for(
        lambda: len(wenv.panes()) == 2 and all(p["state"] == "dead" for p in wenv.panes()), "panes"
    )
    done = wenv.run("teardown", agent_id, "--json")
    body = result_json(done)
    assert done.returncode == 0, (done.stdout, done.stderr)
    schema_registry.validate(body, "agent-spawn/cli/teardown.json")
    (term_id,) = body["windows_closed"]
    assert [p["id"] for p in wenv.panes()] == [p["id"] for p in wenv.panes() if p["id"] != term_id]
    remaining = wenv.panes()
    assert len(remaining) == 1 and "agent=ag-zzzzzzzz" in remaining[0]["labels"]
    listing, default = term_calls(fakebin, "list")  # the recorded socket, then the default server
    assert f"--label=agent={agent_id}" in listing and "--dead" in listing
    assert f"--socket={wenv.tmux.socket}" in listing
    assert f"--label=agent={agent_id}" in default and not any(
        a.startswith("--socket") for a in default
    )
    assert wenv.agent(agent_id)["status"] == "torn_down"


def test_a_running_pane_is_never_closed_by_teardown(wenv: WindowEnv, fakebin: Fakebin) -> None:
    agent_id = dead_pane_agent(wenv, fakebin)
    (dead,) = [p["id"] for p in wenv.panes() if p["state"] == "dead"]
    running = wenv.fakebin.run(
        "termctl", "open", "--name=live", f"--label=agent={agent_id}", "--json", "--",
        "sleep", "300", env=wenv.env(),
    )  # fmt: skip
    assert running.returncode == 0
    done = wenv.run("teardown", agent_id, "--json")
    body = result_json(done)
    assert done.returncode == 0 and body["windows_closed"] == [dead]
    left = wenv.panes()
    assert [p["state"] for p in left] == ["running"]
    (close,) = term_calls(fakebin, "close")  # one close: the dead pane, never forced
    assert close[1] == dead and not any(a.startswith("--force") for a in close)


def test_no_window_no_termctl_call(wenv: WindowEnv, fakebin: Fakebin) -> None:
    script_done_run(wenv.fakebin)
    body = result_json(
        wenv.run("launch", "--mode", "headless", "--skill", "do-it", "--json", "--wait")
    )
    fakebin.assert_not_called("termctl")
    done = wenv.run("teardown", body["agent"]["agent_id"], "--json")
    assert done.returncode == 0 and result_json(done)["windows_closed"] == []
    fakebin.assert_not_called("termctl")


@pytest.mark.parametrize(
    ("error", "exit_code", "warned"),
    [
        ({"code": "not_found", "message": "gone"}, 10, False),
        ({"code": "refused", "message": "running", "reason": "running"}, 20, False),
        ({"code": "locked", "message": "busy"}, 12, True),
    ],
)
def test_close_exits_10_and_running_are_no_ops_and_other_failures_warn(
    wenv: WindowEnv, fakebin: Fakebin, error: dict[str, Any], exit_code: int, warned: bool
) -> None:
    agent_id = dead_pane_agent(wenv, fakebin)
    fakebin.script("termctl", "close", error=error, exit=exit_code)
    done = wenv.run("teardown", agent_id, "--json")
    body = result_json(done)
    assert done.returncode == 0 and body["windows_closed"] == []
    assert bool(body["warnings"]) is warned
    assert wenv.agent(agent_id)["status"] == "torn_down"  # never a refusal


def test_a_failing_list_is_a_warning_and_teardown_goes_on(
    wenv: WindowEnv, fakebin: Fakebin
) -> None:
    agent_id = dead_pane_agent(wenv, fakebin)
    fakebin.script("termctl", "list", error={"code": "tmux_missing", "message": "x"}, exit=12)
    done = wenv.run("teardown", agent_id, "--json")
    body = result_json(done)
    assert done.returncode == 0 and body["warnings"]
    assert wenv.agent(agent_id)["status"] == "torn_down"


def test_every_distinct_socket_of_the_runs_is_listed_once(
    wenv: WindowEnv, fakebin: Fakebin
) -> None:
    agent_id = dead_pane_agent(wenv, fakebin)
    from agent_spawn.lifecycle import lease_key

    with wenv.store() as store:
        with store.lease(lease_key(agent_id)) as lease:
            store.put_run(
                lease, agent_id, 2, {"term_id": "tc-bbbbbbbb", "socket_path": "/tmp/other-sock"}
            )
            store.put_run(
                lease, agent_id, 3, {"term_id": "tc-cccccccc", "socket_path": str(wenv.tmux.socket)}
            )
    result_json(wenv.run("teardown", agent_id, "--json"))
    sockets = [
        next((a for a in call if a.startswith("--socket=")), None)
        for call in term_calls(fakebin, "list")
    ]
    # each recorded socket once, in run order, then the default server (panes no run records)
    assert sockets == [f"--socket={wenv.tmux.socket}", "--socket=/tmp/other-sock", None]


def test_a_retry_after_the_session_step_does_not_close_windows_again(
    wenv: WindowEnv, fakebin: Fakebin
) -> None:
    agent_id = dead_pane_agent(wenv, fakebin)
    fakebin.script("claude-session", "forget", times=1, exit=20,
                   error={"code": "refused", "message": "live", "reason": "live"})  # fmt: skip
    first = wenv.run("teardown", agent_id, "--json")
    assert first.returncode == 20
    assert wenv.agent(agent_id)["status"] == "tearing_down"
    assert wenv.agent(agent_id)["teardown_step"] == "windows_done"
    lists = len(term_calls(fakebin, "list"))
    second = wenv.run("teardown", agent_id, "--json")
    assert second.returncode == 0
    assert len(term_calls(fakebin, "list")) == lists  # step 5 was done


# --- label mirroring to the pane ---------------------------------------------------------------


def pane_labels(wenv: WindowEnv) -> list[str]:
    (pane,) = wenv.panes()
    return pane["labels"]


def label_calls(fakebin: Fakebin, action: str) -> list[list[str]]:
    return [a for a in fakebin.argvs("termctl") if a[:2] == ["label", action]]


def test_label_changes_are_mirrored_to_a_live_window(
    wenv: WindowEnv, fakebin: Fakebin, proc_tracker: ProcTracker
) -> None:
    agent = launch_live(wenv, fakebin, proc_tracker)
    agent_id = agent["agent_id"]
    done = wenv.run("label", "add", agent_id, "pr=7", "ticket=T-1", "--json")
    assert done.returncode == 0 and result_json(done)["warnings"] == []
    (call,) = label_calls(fakebin, "add")
    assert call[2].startswith("tc-") and call[3:5] == ["pr=7", "ticket=T-1"]
    assert f"--socket={wenv.tmux.socket}" in call
    assert {"pr=7", "ticket=T-1"} <= set(pane_labels(wenv))
    with wenv.store() as store:
        assert store.mirror_pending(agent_id) == []  # session and pane both took it
    done = wenv.run("label", "remove", agent_id, "pr=7", "--json")
    assert done.returncode == 0
    assert "pr=7" not in pane_labels(wenv) and "ticket=T-1" in pane_labels(wenv)
    assert {"agent=" + agent_id} <= set(pane_labels(wenv))  # never touched
    with wenv.store() as store:
        assert store.mirror_pending(agent_id) == []


def test_labels_beyond_the_first_thirty_are_not_mirrored_to_the_pane(
    wenv: WindowEnv, fakebin: Fakebin, proc_tracker: ProcTracker
) -> None:
    script_hung_exec(fakebin)
    script_live_run(fakebin)
    labels = [arg for i in range(30) for arg in ("--label", f"k{i:02d}=v")]
    agent = result_json(wenv.run(*LAUNCH, *labels))["agent"]
    track_agent(wenv, proc_tracker, agent["agent_id"])
    agent_id = agent["agent_id"]
    done = wenv.run("label", "add", agent_id, "extra=1", "--json")
    body = result_json(done)
    assert done.returncode == 0 and "extra=1" in body["labels"]
    assert any("beyond the first 30" in w for w in body["warnings"])
    assert label_calls(fakebin, "add") == []  # nothing was pushed
    assert "extra=1" not in pane_labels(wenv)
    with wenv.store() as store:
        assert store.mirror_pending(agent_id) == []  # not queued: it never will be mirrored
    # a removal is pushed whatever its position, and so is a pair within the first 30
    result_json(wenv.run("label", "remove", agent_id, "k00=v", "--json"))
    assert label_calls(fakebin, "remove")
    assert "k00=v" not in pane_labels(wenv)


def test_a_failed_pane_mirror_stays_queued(
    wenv: WindowEnv, fakebin: Fakebin, proc_tracker: ProcTracker
) -> None:
    agent = launch_live(wenv, fakebin, proc_tracker)
    agent_id = agent["agent_id"]
    fakebin.script(
        "termctl", "label add", times=1, exit=12, error={"code": "locked", "message": "m"}
    )
    done = wenv.run("label", "add", agent_id, "pr=7", "--json")
    body = result_json(done)
    assert done.returncode == 0 and len(body["warnings"]) == 1
    with wenv.store() as store:
        assert [(str(p), op) for p, op in store.mirror_pending(agent_id)] == [("pr=7", "add")]
    # the session took it (claude-session label add ran), the pane did not
    assert any(a[:2] == ["label", "add"] for a in fakebin.argvs("claude-session"))
    assert "pr=7" not in pane_labels(wenv)


def test_a_pane_that_is_gone_counts_as_mirrored(
    wenv: WindowEnv, fakebin: Fakebin, proc_tracker: ProcTracker
) -> None:
    agent = launch_live(wenv, fakebin, proc_tracker)
    agent_id = agent["agent_id"]
    fakebin.script(
        "termctl", "label add", times=1, exit=10, error={"code": "not_found", "message": "m"}
    )
    done = wenv.run("label", "add", agent_id, "pr=7", "--json")
    assert done.returncode == 0 and result_json(done)["warnings"] == []
    with wenv.store() as store:
        assert store.mirror_pending(agent_id) == []


def test_no_pane_call_for_a_parked_agent(wenv: WindowEnv, fakebin: Fakebin) -> None:
    script_done_run(wenv.fakebin)
    agent_id = launched_window(wenv)
    done = wenv.run("label", "add", agent_id, "pr=7", "--json")
    assert done.returncode == 0
    assert label_calls(fakebin, "add") == []


def test_self_labeling_mirrors_to_the_live_pane_too(
    wenv: WindowEnv, fakebin: Fakebin, proc_tracker: ProcTracker
) -> None:
    agent = launch_live(wenv, fakebin, proc_tracker)
    agent_id = agent["agent_id"]
    session = wenv.agent(agent_id)["csession_id"]
    directory = wenv.iso.state / "claude-session" / "sessions" / session
    directory.mkdir(parents=True, exist_ok=True)
    secret = directory / "secret"
    secret.write_text("s3cr3t\n")
    secret.chmod(0o600)
    fakebin.script("claude-session", "verify-secret", secret="s3cr3t")
    done = wenv.run(
        "label", "add", "--session", session, "pr=h/o/r#7", "--json",
        extra_env={"CSESSION_ID": session, "CSESSION_DIR": str(directory)},
    )  # fmt: skip
    assert done.returncode == 0, (done.stdout, done.stderr)
    assert "pr=h/o/r#7" in pane_labels(wenv)


# --- stop --------------------------------------------------------------------------------------


def test_stop_ends_a_window_agent(
    wenv: WindowEnv, fakebin: Fakebin, proc_tracker: ProcTracker
) -> None:
    agent = launch_live(wenv, fakebin, proc_tracker)
    agent_id = agent["agent_id"]
    supervisor = wenv.agent(agent_id)["supervisor"]
    exec_proc = wenv.agent(agent_id)["exec_proc"]
    stopper = proc_tracker.popen(
        wenv.argv("stop", agent_id, "--json", "--grace", "1s"),
        env=wenv.env(), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )  # fmt: skip
    # claude-session's `stop` signals `exec` in the real tool; the fake only records the request
    wenv.wait_for(
        lambda: any(a[0] == "stop" for a in fakebin.argvs("claude-session")), "claude-session stop"
    )
    send_signal(exec_proc, signal.SIGTERM)
    out, err = stopper.communicate(timeout=60)
    body = json.loads(out)
    assert stopper.returncode == 0, (out, err)
    schema_registry.validate(body, "agent-spawn/cli/stop.json")
    assert body["action"] == "session_stop" and body["agent"]["status"] == "parked"
    assert body["agent"]["park_reason"] == "killed" and body["agent"]["needs_attention"] is None
    wenv.wait_for(lambda: not procs.is_alive(supervisor), "the supervisor to exit")
    wenv.wait_for(lambda: wenv.panes() == [], "the pane to close")  # exit 0: nothing stays dead


# --- launch: signals and the handoff peer check --------------------------------------------------


def test_a_launcher_interrupted_while_the_window_opens_parks_killed_once(
    wenv: WindowEnv, fakebin: Fakebin, proc_tracker: ProcTracker
) -> None:
    fakebin.script("termctl", "open", times=1)  # the preflight
    fakebin.script("termctl", "open", sleep=30)
    launcher = proc_tracker.popen(
        wenv.argv(*LAUNCH), env=wenv.env(), stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )  # fmt: skip
    wenv.wait_for(lambda: bool(open_calls(fakebin, dry_run=False)), "the real open")
    launcher.send_signal(signal.SIGTERM)
    out, _err = launcher.communicate(timeout=30)
    body = json.loads(out)
    assert launcher.returncode == 143
    assert body["error"]["details"] == {"cause": "interrupted"}
    (row,) = wenv.agents()
    assert row["status"] == "parked" and row["park_reason"] == "killed"
    assert row["needs_attention"] is None and row["supervisor"] is None
    assert len([a for a in fakebin.argvs("claude-session") if a[0] == "stop"]) == 1
    assert not (wenv.state_dir / "handoff").exists() or not list(
        (wenv.state_dir / "handoff").iterdir()
    )


def test_a_connection_from_another_process_is_rejected_and_the_launch_fails(
    wenv: WindowEnv, fakebin: Fakebin, proc_tracker: ProcTracker
) -> None:
    """termctl reports a PID that is not the window process: the window's own connection is a
    thief (wrong peer PID) and gets nothing (spike A6/A8). The handoff deadline is far away; the
    test waits for the rejection record, then ends the launch with SIGTERM (its own recorded
    launcher): the launch fails, no supervisor exists, no clock is involved."""
    other = int(wenv.tmux.run("display-message", "-p", "#{pane_pid}").stdout.strip())
    fakebin.script("termctl", "open", times=1)
    fakebin.script("termctl", "open", merge={"pid": other})
    launcher = proc_tracker.popen(
        wenv.argv(*LAUNCH), env=wenv.env({"AGENT_SPAWN_HANDOFF_TIMEOUT": "60"}),
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )  # fmt: skip
    wenv.wait_for(
        lambda: any(e["event"] == "handoff_rejected" for e in log_events(wenv)),
        "the thief's connection to be rejected",
        timeout=60,
    )
    launcher.send_signal(signal.SIGTERM)
    out, _err = launcher.communicate(timeout=60)
    body = json.loads(out)
    assert launcher.returncode == 143
    assert body["error"]["details"] == {"cause": "interrupted"}
    rejected = [e for e in log_events(wenv) if e["event"] == "handoff_rejected"]
    assert rejected[0]["details"]["cause"] == "wrong_pid"
    (agent,) = wenv.agents()
    assert agent["status"] == "parked" and agent["park_reason"] == "killed"
    assert agent["supervisor"] is None  # no supervisor exists after a failed handoff
    assert not [i for i in fakebin.invocations("claude-session") if "exec" in i.argv]
    # the window process was refused: it gets no environment, exits 1 and leaves a dead pane
    wenv.wait_for(lambda: any(p["state"] == "dead" for p in wenv.panes()), "dead pane")


def test_a_resume_whose_window_fails_to_open_parks_launch_failed(
    wenv: WindowEnv, fakebin: Fakebin
) -> None:
    script_done_run(wenv.fakebin)
    agent_id = launched_window(wenv)
    fail_real_open(fakebin, 12, "tmux_missing")
    done = wenv.run("resume", agent_id, "--json")
    assert done.returncode == 12 and result_json(done)["error"]["tool"] == "termctl"
    row = wenv.agent(agent_id)
    assert row["status"] == "parked" and row["park_reason"] == "launch_failed"
    assert row["needs_attention"] == "launch_failed" and row["supervisor"] is None
    stops = [a for a in fakebin.argvs("claude-session") if a[0] == "stop"]
    assert len(stops) == 1  # the pending run of this resume, once


def test_hook_decision_ids_reach_termctl_open(wenv: WindowEnv, fakebin: Fakebin) -> None:
    script_done_run(wenv.fakebin)
    done = wenv.run(
        *LAUNCH, "--wait", "--decision-id", "d-1",
        "--hook", """pre-launch=echo '{"decision_id": "d-77"}'""",
    )  # fmt: skip
    assert done.returncode == 0, (done.stdout, done.stderr)
    (real,) = open_calls(fakebin, dry_run=False)
    assert "--decision-id=d-1" in real and "--decision-id=d-77" in real
    (dry,) = open_calls(fakebin, dry_run=True)
    # the preflight carries the command's own ids; the hook's came later
    assert "--decision-id=d-1" in dry and "--decision-id=d-77" not in dry
    assert any(a.startswith("--actor=") for a in dry)


def test_a_waiting_launcher_forwards_sigterm_to_the_window_supervisor(
    wenv: WindowEnv, fakebin: Fakebin, proc_tracker: ProcTracker
) -> None:
    script_hung_exec(fakebin)
    script_live_run(fakebin)
    launcher = proc_tracker.popen(
        wenv.argv(*LAUNCH, "--wait"), env=wenv.env(), stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )  # fmt: skip
    wenv.wait_for(lambda: any(a["status"] == "live" for a in wenv.agents()), "the agent to be live")
    (row,) = wenv.agents()
    track_agent(wenv, proc_tracker, row["agent_id"])
    launcher.send_signal(signal.SIGTERM)
    out, _err = launcher.communicate(timeout=60)
    body = json.loads(out)
    # the launcher forwarded SIGTERM to the supervisor (cancellation chain) and kept waiting
    assert body["agent"]["status"] == "parked" and body["agent"]["park_reason"] == "killed"
    assert launcher.returncode != 143
    assert int(signal.SIGTERM) in exec_signals(fakebin)


def test_show_and_list_report_the_window(wenv: WindowEnv) -> None:
    script_done_run(wenv.fakebin)
    agent_id = launched_window(wenv, "--name", "my window")
    shown = result_json(wenv.run("show", agent_id, "--json"))
    schema_registry.validate(shown, "agent-spawn/cli/show.json")
    assert shown["agent"]["window"]["term_id"].startswith("tc-")
    assert shown["agent"]["name"] == "my window"
    listed = result_json(wenv.run("list", "--json"))
    schema_registry.validate(listed, "agent-spawn/cli/list.json")
    assert listed["agents"][0]["window"] == shown["agent"]["window"]


# --- review round 1 -----------------------------------------------------------------------------


def test_a_bad_config_in_the_panes_environment_does_not_stop_the_handoff(
    wenv: WindowEnv, fakebin: Fakebin
) -> None:
    """The window supervisor starts with the tmux server's environment; an invalid
    AGENT_SPAWN_* value there must not stop it before the handoff (the launcher's environment,
    which it adopts, is the authoritative one)."""
    script_done_run(fakebin)
    done = wenv.tmux.run("set-environment", "-g", "AGENT_SPAWN_TOMBSTONE_DAYS", "not-a-number")
    assert done.returncode == 0, done.stderr
    launched = wenv.run(*LAUNCH, "--wait")
    assert launched.returncode == 0, (launched.stdout, launched.stderr)
    assert result_json(launched)["outcome"] == "done"


def test_a_window_supervisor_with_a_bad_config_loads_defaults_until_the_handoff(
    wenv: WindowEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    from agent_spawn import cli

    tail = ["--state-dir", str(wenv.state_dir), "--config", str(wenv.config)]
    window = ["supervise", "ag-aaaaaaaa", "--handoff", "/x", "--handoff-timeout", "1", *tail]
    host = [
        "supervise",
        "ag-aaaaaaaa",
        "--host",
        "--handoff",
        "/x",
        "--handoff-timeout",
        "1",
        *tail,
    ]
    detached = ["supervise", "ag-aaaaaaaa", *tail]
    with monkeypatch.context() as scoped:
        scoped.setenv("AGENT_SPAWN_TOMBSTONE_DAYS", "not-a-number")
        for argv in (window, host):
            inv = cli.parse_invocation(argv)
            assert inv.config is not None and inv.config_file == str(wenv.config)
        with pytest.raises(cli.ConfigFailure):  # a detached supervisor still refuses
            cli.parse_invocation(detached)


def test_a_terminal_that_hung_up_never_makes_a_write_fail() -> None:
    import errno

    from agent_spawn import cli

    class Hung:
        def write(self, text: str) -> int:
            raise OSError(errno.EIO, "Input/output error")

        def flush(self) -> None:
            raise OSError(errno.EIO, "Input/output error")

        def fileno(self) -> int:
            raise ValueError("not a real descriptor")

    cli._write(Hung(), "text")  # pyright: ignore[reportPrivateUsage]


def script_two_opens(fakebin: Fakebin, **real: Any) -> None:
    """The preflight succeeds; the real open is answered as ``real`` says."""
    fakebin.script("termctl", "open", times=1)
    fakebin.script("termctl", "open", **real)


def run_teardown_of_the_parked_agent(wenv: WindowEnv) -> dict[str, Any]:
    (agent,) = wenv.agents()
    done = wenv.run("teardown", agent["agent_id"], "--json")
    assert done.returncode == 0, (done.stdout, done.stderr)
    body: dict[str, Any] = result_json(done)
    return body


@pytest.mark.parametrize(
    "real",
    [
        pytest.param(
            {"after_create": True, "exit": 12, "error": {"code": "locked", "message": "m"}},
            id="fails-after-creating-the-pane",
        ),
        pytest.param({"merge": {"id": "not-an-id"}}, id="unparseable-reply"),
    ],
)
def test_a_pane_whose_open_did_not_report_back_is_found_and_closed_by_teardown(
    wenv: WindowEnv, fakebin: Fakebin, real: dict[str, Any]
) -> None:
    script_two_opens(fakebin, **real)
    done = wenv.run(*LAUNCH, extra_env={"AGENT_SPAWN_HANDOFF_TIMEOUT": "2"})
    assert done.returncode == 12, (done.stdout, done.stderr)
    (agent,) = wenv.agents()
    assert agent["status"] == "parked" and agent["park_reason"] == "launch_failed"
    # the pane exists but the launcher never heard of it: it was found by its label and recorded
    (pane,) = wenv.panes()
    with wenv.store() as store:
        run = store.get_run(agent["agent_id"], agent["run"])
    assert run is not None
    assert (run["term_id"], run["socket_path"]) == (pane["id"], str(wenv.tmux.socket))
    # the window process never got an environment: it fails and tmux keeps the dead pane
    wenv.wait_for(lambda: all(p["state"] == "dead" for p in wenv.panes()), "dead pane")
    body = run_teardown_of_the_parked_agent(wenv)
    assert body["windows_closed"] == [pane["id"]]
    assert wenv.panes() == []


def test_a_pre_creation_failure_records_no_window(wenv: WindowEnv, fakebin: Fakebin) -> None:
    script_two_opens(fakebin, exit=12, error={"code": "tmux_missing", "message": "m"})
    assert wenv.run(*LAUNCH).returncode == 12
    (agent,) = wenv.agents()
    with wenv.store() as store:
        run = store.get_run(agent["agent_id"], agent["run"])
    assert run is not None and run["term_id"] is None and run["socket_path"] is None
    assert wenv.panes() == []


@pytest.mark.parametrize("creates_a_pane", [True, False], ids=["pane-created", "no-pane-created"])
def test_a_resume_whose_open_fails_records_only_its_own_pane(
    wenv: WindowEnv, fakebin: Fakebin, creates_a_pane: bool
) -> None:
    """An older run's dead pane carries the same agent label: recovery must not stamp it on the
    new run (the resume's pane if it created one, no pane id otherwise)."""
    agent_id = dead_pane_agent(wenv, fakebin)
    (old,) = wenv.panes()
    script_two_opens(
        fakebin, exit=12, error={"code": "locked", "message": "m"}, after_create=creates_a_pane
    )
    done = wenv.run("resume", agent_id, "--json", extra_env={"AGENT_SPAWN_HANDOFF_TIMEOUT": "2"})
    assert done.returncode == 12, (done.stdout, done.stderr)
    row = wenv.agent(agent_id)
    with wenv.store() as store:
        run = store.get_run(agent_id, row["run"])
    assert run is not None and row["run"] == 2
    created = [p["id"] for p in wenv.panes() if p["id"] != old["id"]]
    assert len(created) == (1 if creates_a_pane else 0)
    assert run["term_id"] == (created[0] if creates_a_pane else None)


def test_recovery_without_a_before_list_records_only_the_socket_when_panes_are_ambiguous(
    wenv: WindowEnv, fakebin: Fakebin
) -> None:
    agent_id = dead_pane_agent(wenv, fakebin)
    script_two_opens(fakebin, exit=12, error={"code": "locked", "message": "m"}, after_create=True)
    fakebin.script("termctl", "list", times=1, exit=12, error={"code": "x", "message": "m"})
    done = wenv.run("resume", agent_id, "--json", extra_env={"AGENT_SPAWN_HANDOFF_TIMEOUT": "2"})
    assert done.returncode == 12, (done.stdout, done.stderr)
    assert len(wenv.panes()) == 2  # the older dead pane and the new one: which is which is unknown
    with wenv.store() as store:
        run = store.get_run(agent_id, 2)
    assert run is not None
    assert (run["term_id"], run["socket_path"]) == (None, str(wenv.tmux.socket))


def test_a_launcher_killed_while_termctl_opens_leaves_a_pane_teardown_closes(
    wenv: WindowEnv, fakebin: Fakebin, proc_tracker: ProcTracker, tmp_path: Path
) -> None:
    gate = tmp_path / "open-gate"
    script_two_opens(fakebin, after_create=True, gate=str(gate))  # the pane exists, then hangs
    launcher = proc_tracker.popen(
        wenv.argv(*LAUNCH), env=wenv.env({"AGENT_SPAWN_HANDOFF_TIMEOUT": "2"}),
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )  # fmt: skip
    try:
        fakebin.wait_for_gate(gate)
        launcher.send_signal(signal.SIGTERM)
        out, _err = launcher.communicate(timeout=60)
    finally:
        gate.touch()  # let the hung termctl go
    assert launcher.returncode == 143 and json.loads(out)["error"]["details"] == {
        "cause": "interrupted"
    }
    (agent,) = wenv.agents()
    assert agent["status"] == "parked" and agent["park_reason"] == "killed"
    (pane,) = wenv.panes()
    with wenv.store() as store:
        run = store.get_run(agent["agent_id"], agent["run"])
    assert run is not None and run["term_id"] == pane["id"]
    wenv.wait_for(lambda: all(p["state"] == "dead" for p in wenv.panes()), "dead pane")
    assert run_teardown_of_the_parked_agent(wenv)["windows_closed"] == [pane["id"]]


def test_a_signal_after_the_handoff_is_forwarded_even_before_the_listener_closes(
    wenv: WindowEnv, fakebin: Fakebin, proc_tracker: ProcTracker, tmp_path: Path
) -> None:
    """SIGTERM after the handoff completed (the launcher's own close of the listener) reaches the
    supervisor, which now owns the run: the launcher is not interrupted and parks nothing."""
    script_hung_exec(fakebin)
    script_live_run(fakebin)
    wenv.capture_supervisor(
        tmp_path / "capture",
        launcher_prologue=(
            "import signal, time\n"
            "from agent_spawn import handoff\n"
            "def hung():  # the fake exec logs hang_ready once its signal handlers are installed\n"
            "    deadline = time.monotonic() + 60\n"
            "    while time.monotonic() < deadline:\n"
            "        try:\n"
            "            with open(os.environ['FAKEBIN_LOG']) as log:\n"
            "                if any('\"hang_ready\"' in line for line in log):\n"
            "                    return\n"
            "        except OSError:\n"
            "            pass\n"
            "        time.sleep(0.01)\n"
            "    raise RuntimeError('the fake exec never reached its hang')\n"
            "_close, calls = handoff.HandoffListener.close, []\n"
            "def close(self):\n"
            "    calls.append(1)\n"
            "    if len(calls) == 2:  # serve() closed it once: this is the launcher's own close\n"
            "        hung()  # a SIGTERM that beats exec's handlers kills it with no outcome\n"
            "        os.kill(os.getpid(), signal.SIGTERM)\n"
            "    _close(self)\n"
            "handoff.HandoffListener.close = close\n"
        ),
    )
    launcher = proc_tracker.popen(
        wenv.argv(*LAUNCH, "--wait"), env=wenv.env(), stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )  # fmt: skip
    out, err = launcher.communicate(timeout=120)
    (row,) = wenv.agents()
    track_agent(wenv, proc_tracker, row["agent_id"])
    # not the launcher's own interrupt (exit 143, "interrupted"): the supervisor was signaled
    assert launcher.returncode != 143 and "interrupted" not in out.decode(), (out, err)
    supervisor = wenv.agent(row["agent_id"])["supervisor"]
    assert supervisor is not None  # still recorded: the launcher did not clear it
    wenv.wait_for(lambda: wenv.agent(row["agent_id"])["status"] == "parked", "the agent to park")
    assert wenv.agent(row["agent_id"])["park_reason"] == "killed"
    wenv.wait_for(lambda: not procs.is_alive(supervisor), "the supervisor to exit")
    assert not [e for e in log_events(wenv) if e["event"] == "interrupted"]


def test_an_interrupt_during_the_handoff_parks_killed_and_clears_the_supervisor(
    wenv: WindowEnv, fakebin: Fakebin, proc_tracker: ProcTracker
) -> None:
    other = int(wenv.tmux.run("display-message", "-p", "#{pane_pid}").stdout.strip())
    script_two_opens(fakebin, merge={"pid": other})  # the handoff never completes
    launcher = proc_tracker.popen(
        wenv.argv(*LAUNCH), env=wenv.env({"AGENT_SPAWN_HANDOFF_TIMEOUT": "60"}),
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )  # fmt: skip
    wenv.wait_for(
        lambda: any(e["event"] == "handoff_rejected" for e in log_events(wenv)),
        "the handoff to be under way",
        timeout=60,
    )
    launcher.send_signal(signal.SIGTERM)
    out, _err = launcher.communicate(timeout=60)
    assert launcher.returncode == 143
    assert json.loads(out)["error"]["details"] == {"cause": "interrupted"}
    (agent,) = wenv.agents()
    assert agent["status"] == "parked" and agent["park_reason"] == "killed"
    assert agent["needs_attention"] is None and agent["supervisor"] is None


def test_an_undefined_term_placement_is_a_preflight_usage_error(
    wenv: WindowEnv, fakebin: Fakebin
) -> None:
    done = wenv.run(*LAUNCH, "--term-placement", "nowhere")
    error = result_json(done)["error"]
    assert done.returncode == 2
    assert (error["code"], error["tool"], error["lower_code"]) == ("usage", "termctl", "usage")
    assert wenv.agents() == [] and open_calls(fakebin, dry_run=False) == []


@pytest.mark.parametrize("fork", [False, True], ids=["resume", "fork"])
def test_a_resume_preflights_the_placement_its_open_uses(
    wenv: WindowEnv, fakebin: Fakebin, fork: bool
) -> None:
    script_done_run(fakebin)
    extra = {"FAKEBIN_PLACEMENTS": "side"}
    launched = result_json(wenv.run(*LAUNCH, "--wait", "--term-placement", "side", extra_env=extra))
    agent_id = launched["agent"]["agent_id"]
    done = wenv.run("resume", agent_id, *(["--fork"] if fork else []), "--json", "--wait",
                    extra_env=extra)  # fmt: skip
    assert done.returncode == 0, (done.stdout, done.stderr)
    dry, real = open_calls(fakebin, dry_run=True)[-1], open_calls(fakebin, dry_run=False)[-1]
    assert "--placement=side" in dry and "--placement=side" in real  # stored, not termctl's rule
    assert [a for a in dry if a.startswith("--placement")] == [
        a for a in real if a.startswith("--placement")
    ]


def test_a_label_added_while_the_window_opens_reaches_the_pane(
    wenv: WindowEnv, fakebin: Fakebin, proc_tracker: ProcTracker, tmp_path: Path
) -> None:
    script_hung_exec(fakebin)
    script_live_run(fakebin)
    gate = tmp_path / "open-gate"
    script_two_opens(fakebin, gate=str(gate))
    launcher = proc_tracker.popen(
        wenv.argv(*LAUNCH, "--label", "a=1"), env=wenv.env(), stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )  # fmt: skip
    try:
        fakebin.wait_for_gate(gate)  # termctl is opening; the labels were read before it
        (agent,) = wenv.agents()
        added = wenv.run("label", "add", agent["agent_id"], "late=1", "--json")
        assert added.returncode == 0, (added.stdout, added.stderr)
        removed = wenv.run("label", "remove", agent["agent_id"], "a=1", "--json")
        assert removed.returncode == 0, (removed.stdout, removed.stderr)
    finally:
        gate.touch()
    out, err = launcher.communicate(timeout=120)
    assert launcher.returncode == 0, (out, err)
    track_agent(wenv, proc_tracker, agent["agent_id"])
    labels = pane_labels(wenv)
    assert "late=1" in labels and "a=1" not in labels
    assert f"agent={agent['agent_id']}" in labels


def test_a_partial_batch_keeps_only_the_failed_pair_queued(
    wenv: WindowEnv, fakebin: Fakebin, proc_tracker: ProcTracker
) -> None:
    script_hung_exec(fakebin)
    script_live_run(fakebin)
    labels = [arg for i in range(29) for arg in ("--label", f"k{i:02d}=v")]
    agent = result_json(wenv.run(*LAUNCH, *labels))["agent"]
    track_agent(wenv, proc_tracker, agent["agent_id"])
    agent_id = agent["agent_id"]
    # `a=1` becomes the 30th label (pushed, and scripted to fail); `extra=1` the 31st (never)
    fakebin.script(
        "termctl", "label add", times=1, exit=12, error={"code": "locked", "message": "m"}
    )
    done = wenv.run("label", "add", agent_id, "a=1", "extra=1", "--json")
    body = result_json(done)
    assert done.returncode == 0, (done.stdout, done.stderr)
    assert any("beyond the first 30" in w for w in body["warnings"])
    assert any("queued for reconcile" in w for w in body["warnings"])
    with wenv.store() as store:
        assert [(str(p), op) for p, op in store.mirror_pending(agent_id)] == [("a=1", "add")]
    (call,) = label_calls(fakebin, "add")
    assert call[3:4] == ["a=1"] and "extra=1" not in call


def test_a_termctl_that_cannot_be_found_does_not_complete_the_window_step(
    wenv: WindowEnv, fakebin: Fakebin
) -> None:
    from agent_spawn.lifecycle import lease_key

    agent_id = dead_pane_agent(wenv, fakebin)
    with wenv.store() as store:
        with store.lease(lease_key(agent_id)) as lease:
            store.update_agent(lease, agent_id, {"termctl_path": "/nonexistent/termctl"})
    fakebin.script("claude-session", "forget", times=1, exit=20,
                   error={"code": "refused", "message": "live", "reason": "live"})  # fmt: skip
    first = wenv.run("teardown", agent_id, "--json", extra_env={"PATH": "/usr/bin:/bin"})
    body = result_json(first)
    assert first.returncode == 20, (first.stdout, first.stderr)
    assert wenv.agent(agent_id)["status"] == "tearing_down"
    assert wenv.agent(agent_id)["teardown_step"] != "windows_done"
    assert wenv.panes()[0]["state"] == "dead"
    assert body["error"]["code"] == "refused"
    second = wenv.run("teardown", agent_id, "--json")  # termctl is found now: the pane is closed
    assert second.returncode == 0, (second.stdout, second.stderr)
    assert len(result_json(second)["windows_closed"]) == 1
    assert wenv.panes() == []


def test_a_failed_list_does_not_complete_the_window_step_and_a_retry_closes_the_pane(
    wenv: WindowEnv, fakebin: Fakebin
) -> None:
    agent_id = dead_pane_agent(wenv, fakebin)
    # both servers of the step fail (the recorded socket and the default one, which is the same
    # private server here): the step is not done and nothing is closed
    fakebin.script("termctl", "list", times=2, exit=12,
                   error={"code": "tmux_missing", "message": "x"})  # fmt: skip
    fakebin.script("claude-session", "forget", times=1, exit=20,
                   error={"code": "refused", "message": "live", "reason": "live"})  # fmt: skip
    first = wenv.run("teardown", agent_id, "--json")
    assert first.returncode == 20, (first.stdout, first.stderr)
    assert wenv.agent(agent_id)["teardown_step"] != "windows_done"
    second = wenv.run("teardown", agent_id, "--json")
    assert second.returncode == 0, (second.stdout, second.stderr)
    assert len(result_json(second)["windows_closed"]) == 1
    assert wenv.panes() == []
