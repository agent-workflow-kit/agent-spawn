"""Golden and behavior tests of ``launch`` (headless, no worktree) against the fake binaries."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

import pytest

from agent_spawn import procs
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

LAUNCH = ("launch", "--mode", "headless", "--skill", "do-it", "--json")


@pytest.fixture
def env(fakebin: Fakebin, isolation_env: Isolation) -> Env:
    return Env(fakebin, isolation_env)


def test_launch_headless_done(env: Env) -> None:
    done = env.run("launch", "--mode", "headless", "--skill", "do-it", "--json", "--wait")
    body = result_json(done)
    assert done.returncode == 0, (done.stdout, done.stderr)
    schema_registry.validate(body, "agent-spawn/cli/launch.json")
    assert body["agent"]["status"] == "parked"
    assert body["outcome"] == "done"


def test_supervisor_exits_cleanly_and_logs_no_error(env: Env) -> None:
    body = result_json(env.run(*LAUNCH, "--wait"))
    agent_id = body["agent"]["agent_id"]
    env.wait_for(lambda: not procs.is_alive(env.agent(agent_id)["supervisor"]), "supervisor exit")
    log = (env.state_dir / "supervise" / f"{agent_id}.log").read_text()
    assert "error" not in log.lower() and "Traceback" not in log


def test_start_argv(env: Env, fakebin: Fakebin, tmp_path: Path) -> None:
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("do the thing")
    done = env.run(
        "launch", "--mode", "headless", "--skill", "do-it", "--prompt-file", str(prompt),
        "--label", "ticket=T-1", "--json", "--wait",
    )  # fmt: skip
    body = result_json(done)
    agent_id = body["agent"]["agent_id"]
    argv = fakebin.invocations("claude-session")[1].argv  # [0] is the classification dry run
    assert argv[0] == "start"
    assert "--mode=headless" in argv
    assert f"--label=agent={agent_id}" in argv
    assert "--label=ticket=T-1" in argv
    copies = [a for a in argv if a.startswith("--prompt-file=")]
    assert copies and str(prompt) not in " ".join(argv)
    assert copies[0].startswith(f"--prompt-file={env.state_dir}/prompts/{agent_id}/")
    assert any(a.startswith("--hook=post-state-change=") for a in argv)
    assert any(a.startswith("--outcome-file=" + str(env.state_dir)) for a in argv)


def started(env: Env, agent_id: str) -> bool:
    return env.agent(agent_id)["status"] == "live"


def launch_hung(
    env: Env, fakebin: Fakebin, proc_tracker: ProcTracker, *extra: str
) -> dict[str, Any]:
    """Launch (without --wait) a run whose `exec` hangs; return the result object."""
    script_live_run(fakebin)
    script_hung_exec(fakebin)
    done = env.run(*LAUNCH, *extra)
    body = result_json(done)
    assert done.returncode == 0, (done.stdout, done.stderr)
    track_agent(env, proc_tracker, body["agent"]["agent_id"])
    return body


def test_started_without_wait_leaves_agent_running(
    env: Env, fakebin: Fakebin, proc_tracker: ProcTracker
) -> None:
    body = launch_hung(env, fakebin, proc_tracker)
    schema_registry.validate(body, "agent-spawn/cli/launch.json")
    assert body["run"] is None and body["phase"] is None and body["timed_out"] is False
    assert body["agent"]["status"] == "live"
    row = env.agent(body["agent"]["agent_id"])
    assert procs.is_alive(row["supervisor"])
    # The same process that the launcher recorded is the supervisor (the grandchild exec'd).
    pid = row["supervisor"].pid
    cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
    assert cmdline[2:4] == [b"supervise", body["agent"]["agent_id"].encode()]  # python, script
    assert procs.read_identity(pid) == row["supervisor"]


def test_detached_supervisor_holds_no_inherited_descriptor(
    env: Env, fakebin: Fakebin, proc_tracker: ProcTracker, tmp_path: Path
) -> None:
    # The launcher holds a marker descriptor (and a pipe for its output) that the detached
    # supervisor must not inherit: the check fails if the grandchild keeps any of them.
    script_live_run(fakebin)
    script_hung_exec(fakebin)
    marker = tmp_path / "marker-held-by-the-launcher"
    marker.write_text("x")
    marker_fd = os.open(marker, os.O_RDONLY)
    try:
        launcher = proc_tracker.popen(
            env.argv(*LAUNCH),
            env=env.env(),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            pass_fds=(marker_fd,),
        )
        out, _err = launcher.communicate(timeout=60)  # returns only if no pipe end is held
    finally:
        os.close(marker_fd)
    body = json.loads(out)
    agent_id = body["agent"]["agent_id"]
    track_agent(env, proc_tracker, agent_id)
    pid = env.agent(agent_id)["supervisor"].pid
    targets = {fd: os.readlink(f"/proc/{pid}/fd/{fd}") for fd in os.listdir(f"/proc/{pid}/fd")}
    assert targets["0"] == os.devnull
    log = str(env.state_dir / "supervise" / f"{agent_id}.log")
    assert targets["1"] == targets["2"] == log
    assert str(marker) not in targets.values()
    database = str(env.state_dir / "state.db")
    own = [t for fd, t in targets.items() if int(fd) > 2 and t.startswith(database)]
    assert len(own) <= 3, own  # the database and its -wal and -shm, opened by the supervisor
    for fd, target in targets.items():
        assert not target.startswith(("pipe:", "socket:")), (fd, target)
        if int(fd) > 2:  # its own state database, or interpreter internals
            allowed = (database, "/sys/", "/dev/urandom", "anon_inode:")
            assert target.startswith(allowed), (fd, target)


def test_wait_timeout_leaves_agent_running(
    env: Env, fakebin: Fakebin, proc_tracker: ProcTracker
) -> None:
    script_live_run(fakebin)
    script_hung_exec(fakebin)
    done = env.run(*LAUNCH, "--wait", "--wait-timeout", "1s")
    body = result_json(done)
    assert done.returncode == 22
    schema_registry.validate(body, "agent-spawn/cli/launch.json")
    assert body["timed_out"] is True and body["run"] is None
    agent_id = body["agent"]["agent_id"]
    track_agent(env, proc_tracker, agent_id)
    assert started(env, agent_id)
    assert procs.is_alive(env.agent(agent_id)["supervisor"])


def test_sigkilled_waiter_leaves_agent_running(
    env: Env, fakebin: Fakebin, proc_tracker: ProcTracker
) -> None:
    script_live_run(fakebin)
    script_hung_exec(fakebin)
    waiter = proc_tracker.popen(
        env.argv(*LAUNCH, "--wait"),
        env=env.env(),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    env.wait_for(lambda: any(a["status"] == "live" for a in _agents(env)), "a live agent")
    agent_id = env.agents()[0]["agent_id"]
    track_agent(env, proc_tracker, agent_id)
    waiter.kill()
    waiter.wait(timeout=10)
    time.sleep(0.5)
    assert started(env, agent_id)
    assert procs.is_alive(env.agent(agent_id)["supervisor"])


def _agents(env: Env) -> list[dict[str, Any]]:
    try:
        return env.agents()
    except Exception:  # noqa: BLE001 - the database may not exist yet
        return []


# The fake `exec` has no child of its own here; the chain down to a real child of `exec` is
# test_supervise_chain.py::test_sigterm_chain_reaches_exec_and_its_child_even_when_exec_is_stopped.
def test_sigterm_chain_launcher_supervisor_exec(
    env: Env, fakebin: Fakebin, proc_tracker: ProcTracker
) -> None:
    script_live_run(fakebin)
    script_hung_exec(fakebin)
    waiter = proc_tracker.popen(
        env.argv(*LAUNCH, "--wait"),
        env=env.env(),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    env.wait_for(lambda: any(a["status"] == "live" for a in _agents(env)), "a live agent")
    agent_id = env.agents()[0]["agent_id"]
    track_agent(env, proc_tracker, agent_id)
    waiter.send_signal(signal.SIGTERM)  # the exact Popen child
    out, _err = waiter.communicate(timeout=30)
    body = json.loads(out)
    assert waiter.returncode == 15  # the run's outcome code: killed
    assert body["outcome"] == "killed" and body["end_reason"] == "forwarded_term"
    row = env.agent(agent_id)
    assert row["status"] == "parked" and row["park_reason"] == "killed"
    assert row["needs_attention"] is None
    execs = [i for i in fakebin.invocations("claude-session") if i.argv[:1] == ["exec"]]
    assert len(execs) == 1 and signal.SIGTERM in execs[0].signals


def test_wait_waiter_sighup_exits_130_and_leaves_agent(
    env: Env, fakebin: Fakebin, proc_tracker: ProcTracker
) -> None:
    script_live_run(fakebin)
    script_hung_exec(fakebin)
    waiter = proc_tracker.popen(
        env.argv(*LAUNCH, "--wait"),
        env=env.env(),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    env.wait_for(lambda: any(a["status"] == "live" for a in _agents(env)), "a live agent")
    agent_id = env.agents()[0]["agent_id"]
    track_agent(env, proc_tracker, agent_id)
    waiter.send_signal(signal.SIGHUP)
    out, _err = waiter.communicate(timeout=30)
    assert waiter.returncode == 130 and out == b""
    assert started(env, agent_id)


# --- failures of `start` ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("exit_code", "code", "park_reason", "attention", "escalated_by"),
    [
        (21, "ambiguous", "launch_failed", "launch_failed", None),
        (10, "not_found", "launch_failed", "launch_failed", None),
        (12, "launch_failed", "launch_failed", "launch_failed", None),
        (20, "refused", "launch_failed", None, None),
        (11, "vetoed", "vetoed", None, None),
        (13, "escalated", "escalated", "escalated", "claude-session"),
    ],
)
def test_failed_start_parks(
    env: Env,
    fakebin: Fakebin,
    exit_code: int,
    code: str,
    park_reason: str,
    attention: str | None,
    escalated_by: str | None,
) -> None:
    error: dict[str, Any] = {"code": code, "message": "scripted"}
    if code == "refused":
        error["reason"] = "live"
    fakebin.script("claude-session", "start", argv_contains=["--dry-run"])  # classification passes
    fakebin.script("claude-session", "start", error=error, exit=exit_code)
    done = env.run(*LAUNCH)
    body = result_json(done)
    assert done.returncode == exit_code, (done.stdout, done.stderr)
    schema_registry.validate(body, "agent-spawn/cli/error.json")
    assert body["error"]["tool"] == "claude-session"
    (row,) = env.agents()
    assert body["error"]["agent_id"] == row["agent_id"]
    assert row["status"] == "parked"
    assert row["park_reason"] == park_reason
    assert row["needs_attention"] == attention
    assert row["escalated_by"] == escalated_by
    assert not fakebin.invocations("claude-session")[-1].argv[:1] == ["exec"]


def non_dry(fakebin: Fakebin, command: str) -> list[list[str]]:
    return [a for a in fakebin.argvs("claude-session") if a[0] == command and "--dry-run" not in a]


def test_launcher_interrupted_before_supervisor_parks_killed(
    env: Env, fakebin: Fakebin, proc_tracker: ProcTracker
) -> None:
    fakebin.script("claude-session", "start", argv_contains=["--dry-run"])
    fakebin.script("claude-session", "start", sleep=30)
    launcher = proc_tracker.popen(
        env.argv(*LAUNCH),
        env=env.env(),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    env.wait_for(lambda: bool(non_dry(fakebin, "start")), "the real start")
    launcher.send_signal(signal.SIGTERM)
    out, _err = launcher.communicate(timeout=30)
    body = json.loads(out)
    assert launcher.returncode == 143
    schema_registry.validate(body, "agent-spawn/cli/error.json")
    error = body["error"]
    assert error["code"] == "launch_failed" and error["details"] == {"cause": "interrupted"}
    (row,) = env.agents()
    assert error["agent_id"] == row["agent_id"]
    assert row["status"] == "parked" and row["park_reason"] == "killed"
    assert row["needs_attention"] is None
    assert not non_dry(fakebin, "exec")


def test_launcher_sigint_before_supervisor_exits_130(
    env: Env, fakebin: Fakebin, proc_tracker: ProcTracker
) -> None:
    fakebin.script("claude-session", "start", argv_contains=["--dry-run"])
    fakebin.script("claude-session", "start", sleep=30)
    launcher = proc_tracker.popen(
        env.argv(*LAUNCH),
        env=env.env(),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    env.wait_for(lambda: bool(non_dry(fakebin, "start")), "the real start")
    launcher.send_signal(signal.SIGINT)
    out, _err = launcher.communicate(timeout=30)
    assert launcher.returncode == 130
    assert json.loads(out)["error"]["details"] == {"cause": "interrupted"}


def test_interrupt_before_the_record_exists_creates_nothing(
    env: Env, fakebin: Fakebin, proc_tracker: ProcTracker
) -> None:
    fakebin.script("claude-session", "start", sleep=30)  # the classification dry run hangs
    launcher = proc_tracker.popen(
        env.argv(*LAUNCH),
        env=env.env(),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    env.wait_for(lambda: bool(fakebin.invocations("claude-session")), "the dry run")
    launcher.send_signal(signal.SIGTERM)
    out, _err = launcher.communicate(timeout=30)
    assert launcher.returncode == 143
    assert json.loads(out)["error"]["agent_id"] is None
    assert env.agents() == []
    assert not list((env.state_dir / "prompts").glob("*"))


# --- pre-launch hooks, dry run, usage -----------------------------------------------------


def test_pre_launch_hook_vetoes(env: Env, fakebin: Fakebin) -> None:
    done = env.run(*LAUNCH, "--hook", "pre-launch=echo nope >&2; exit 1")
    body = result_json(done)
    assert done.returncode == 11
    schema_registry.validate(body, "agent-spawn/cli/error.json")
    assert body["error"]["hook"] == "pre-launch" and "nope" in body["error"]["hook_stderr"]
    assert env.agents() == []
    assert not non_dry(fakebin, "start")
    assert not list((env.state_dir / "prompts").glob("*"))


def test_pre_launch_hook_escalates_and_is_overridden(env: Env, fakebin: Fakebin) -> None:
    done = env.run(*LAUNCH, "--hook", "pre-launch=exit 13")
    assert done.returncode == 13 and env.agents() == []
    done = env.run(*LAUNCH, "--hook", "pre-launch=exit 13", "--override-escalation",
                   "--actor", "human:tester", "--wait")  # fmt: skip
    assert done.returncode == 0, (done.stdout, done.stderr)
    audit = (env.state_dir / "audit.jsonl").read_text()
    assert "launch_escalation_override" in audit


def test_pre_launch_payload_matches_schema(env: Env, tmp_path: Path) -> None:
    sink = tmp_path / "payload.json"
    done = env.run(*LAUNCH, "--hook", f"pre-launch=cat > {sink}", "--label", "a=b", "--wait")
    assert done.returncode == 0, (done.stdout, done.stderr)
    payload = json.loads(sink.read_text())
    schema_registry.validate(payload, "agent-spawn/hooks/pre-launch.json")
    assert payload["labels"] == ["a=b"] and payload["object_id"] is None
    assert payload["mode"] == "headless" and payload["settings"]["skill"] == "do-it"


def test_dry_run_prints_plan_and_creates_nothing(env: Env, fakebin: Fakebin) -> None:
    done = env.run(*LAUNCH, "--dry-run", "--label", "x=y")
    body = result_json(done)
    assert done.returncode == 0, (done.stdout, done.stderr)
    schema_registry.validate(body, "agent-spawn/cli/launch-dry-run.json")
    assert body["would_refuse"] is None and body["resume"] is False
    assert body["session_start_argv"][1] == "start"
    assert env.agents() == [] and not list((env.state_dir / "prompts").glob("*"))
    assert not non_dry(fakebin, "start")


def test_dry_run_third_party_outside_session_dir_is_refused(env: Env, fakebin: Fakebin) -> None:
    done = env.run(*LAUNCH, "--dry-run", "--reads-third-party", "--cwd", "/tmp")
    body = result_json(done)
    assert done.returncode == 20
    schema_registry.validate(body, "agent-spawn/cli/launch-dry-run.json")
    assert body["would_refuse"] == "third_party_cwd"


def test_third_party_run_outside_session_dir_exits_20_before_allocation(
    env: Env, fakebin: Fakebin
) -> None:
    done = env.run(*LAUNCH, "--reads-third-party", "--cwd", "/tmp")
    body = result_json(done)
    assert done.returncode == 20
    assert body["error"]["reason"] == "third_party_cwd" and body["error"]["tool"] == "agent-spawn"
    assert env.agents() == [] and not non_dry(fakebin, "start")


@pytest.mark.parametrize(
    "argv",
    [
        ("launch", "--mode", "headless"),  # neither a skill nor a prompt
        ("launch", "--mode", "headless", "--skill", "s", "--branch", "b"),  # needs --worktree new
        ("launch", "--mode", "headless", "--skill", "s", "--worktree", "relative/path"),
        ("launch", "--mode", "headless", "--skill", "s", "--worktree", "new"),  # needs --repo
        ("launch", "--mode", "headless", "--skill", "s", "--hook", "post-park=true"),
        ("launch", "--mode", "headless", "--skill", "s", "--max-turns", "many"),
        ("launch", "--mode", "headless", "--skill", "s", "--prompt-file", "/nonexistent/p"),
        (
            "launch",
            "--mode",
            "interactive",
            "--skill",
            "s",
            "--placement",
            "here",
            "--wait-timeout",
            "5s",
        ),
    ],
)
def test_usage_errors_exit_2_and_create_nothing(
    env: Env, fakebin: Fakebin, argv: tuple[str, ...]
) -> None:
    done = env.run(*argv, "--json")
    assert done.returncode == 2, (done.stdout, done.stderr)
    assert result_json(done)["error"]["code"] == "usage"
    assert env.agents() == [] and not non_dry(fakebin, "start")


def test_placement_is_ignored_with_a_warning_for_headless(env: Env) -> None:
    done = env.run(*LAUNCH, "--placement", "window", "--wait")
    body = result_json(done)
    assert done.returncode == 0
    assert any("placement" in w for w in body["warnings"])


def test_missing_claude_session_exits_12(env: Env) -> None:
    done = env.run(*LAUNCH, extra_env={"PATH": "/usr/bin:/bin"})
    body = result_json(done)
    assert done.returncode == 12 and body["error"]["code"] == "tool_missing"
    assert body["error"]["tool"] == "claude-session"
    assert env.agents() == []


def test_job_label_from_the_environment(env: Env, fakebin: Fakebin) -> None:
    done = env.run(*LAUNCH, "--wait", extra_env={"JOBSCTL_JOB_ID": "job-7"})
    assert "job=job-7" in result_json(done)["agent"]["labels"]
    done = env.run(
        *LAUNCH, "--wait", extra_env={"JOBSCTL_JOB_ID": "job-8", "CSESSION_ID": "cs-aaaaaaaa"}
    )
    assert "job=job-8" not in result_json(done)["agent"]["labels"]  # inside a session
    done = env.run(*LAUNCH, "--wait", extra_env={"JOBSCTL_JOB_ID": "bad value!"})
    body = result_json(done)
    assert not [label for label in body["agent"]["labels"] if label.startswith("job=")]
    assert any("JOBSCTL_JOB_ID" in w for w in body["warnings"])


def test_prompt_copies_are_private_and_removed_once_a_session_exists(
    env: Env, fakebin: Fakebin, tmp_path: Path
) -> None:
    secret = tmp_path / "in.txt"
    secret.write_text("input")
    done = env.run(
        *LAUNCH, "--prompt", "hello", "--input-file", str(secret), "--wait",
    )  # fmt: skip
    body = result_json(done)
    start = non_dry(fakebin, "start")[0]
    assert str(secret) not in " ".join(start)
    assert not (env.state_dir / "prompts" / body["agent"]["agent_id"]).exists()
    assert "hello" not in " ".join(" ".join(a) for a in fakebin.argvs("claude-session"))


def test_result_file_gets_the_json(env: Env, tmp_path: Path) -> None:
    target = tmp_path / "result.json"
    done = env.run(*LAUNCH, "--wait", "--result-file", str(target))
    assert json.loads(target.read_text()) == result_json(done)
    assert oct(target.stat().st_mode & 0o777) == "0o600"


def test_hooks_are_delivered_through_the_outbox(env: Env, fakebin: Fakebin, tmp_path: Path) -> None:
    config = "hooks:\n"
    for name in ("post-launch", "post-park", "post-state-change"):
        sink = f"{tmp_path}/{name}.jsonl"
        config += f'  {name}:\n    - cmd: "cat >> {sink}; echo >> {sink}"\n'
    env = Env(fakebin, env.iso, config)
    done = env.run(*LAUNCH, "--wait")
    assert done.returncode == 0, (done.stdout, done.stderr)
    for name in ("post-launch", "post-park", "post-state-change"):
        sink = tmp_path / f"{name}.jsonl"
        env.wait_for(sink.exists, f"the {name} hook")
        payload = json.loads(sink.read_text().splitlines()[0])
        schema_registry.validate(payload, f"agent-spawn/hooks/{name}.json")
        assert payload["agent"]["agent_id"] == payload["object_id"]
    park = json.loads((tmp_path / "post-park.jsonl").read_text().splitlines()[0])
    assert park["reason"] == "done" and park["outcome"]["outcome"] == "done"
    launched = json.loads((tmp_path / "post-launch.jsonl").read_text().splitlines()[0])
    assert launched["run"] == 1 and launched["resumed"] is False


def test_failed_launch_delivers_its_park_events(env: Env, fakebin: Fakebin, tmp_path: Path) -> None:
    sink = tmp_path / "park.json"
    env = Env(fakebin, env.iso, f'hooks:\n  post-park:\n    - cmd: "cat > {sink}"\n')
    fakebin.script("claude-session", "start", argv_contains=["--dry-run"])
    fakebin.script("claude-session", "start", error={"code": "not_found", "message": "x"}, exit=10)
    done = env.run(*LAUNCH)
    assert done.returncode == 10
    payload = json.loads(sink.read_text())
    schema_registry.validate(payload, "agent-spawn/hooks/post-park.json")
    assert payload["reason"] == "launch_failed" and payload["outcome"] is None
