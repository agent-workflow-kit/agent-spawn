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
    env: Env, fakebin: Fakebin, proc_tracker: ProcTracker
) -> None:
    body = launch_hung(env, fakebin, proc_tracker)  # subprocess.run returned: no pipe was held
    pid = env.agent(body["agent"]["agent_id"])["supervisor"].pid
    targets = {fd: os.readlink(f"/proc/{pid}/fd/{fd}") for fd in os.listdir(f"/proc/{pid}/fd")}
    assert targets["0"] == os.devnull
    assert (
        targets["1"]
        == targets["2"]
        == str(env.state_dir / "supervise" / f"{body['agent']['agent_id']}.log")
    )
    for fd, target in targets.items():
        assert not target.startswith(("pipe:", "socket:")), (fd, target)
        if int(fd) > 2:  # its own state database, or interpreter internals
            allowed = (str(env.state_dir / "state.db"), "/sys/", "/dev/urandom", "anon_inode:")
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
