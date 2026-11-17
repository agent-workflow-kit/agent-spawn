"""Lifecycle entries race as compare-and-sets (interface.md, Lifecycle transitions): of two
concurrent entries on a parked agent exactly one wins; the loser exits 20 without any lower-tool
call or hook of its own.

Deterministic: both commands start on a barrier (real processes), the winner is held at its first
claude-session call by a scripted gate, so the loser has finished (or not) before the winner
moves on, and the log of claude-session calls shows who called what.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from agent_spawn import lifecycle
from agent_spawn.store import Store
from tests.conftest import Isolation
from tests.fakebin.harness import Fakebin
from tests.launch_support import Env, result_json, track_agent
from tests.procguard import ProcTracker

AGENT = "ag-abcdefgh"
SESSION = "cs-abcdefgh"

BARRIER = """
import os, sys, time
barrier = sys.argv[1]
while not os.path.exists(barrier):
    time.sleep(0.002)
from agent_spawn.cli import main
sys.exit(main(sys.argv[2:]))
"""


@pytest.fixture
def env(fakebin: Fakebin, isolation_env: Isolation) -> Env:
    fakebin.seed_session(SESSION)
    return Env(fakebin, isolation_env)


def seed_parked(env: Env) -> None:
    with Store.open(env.state_dir) as store:
        store.insert_agent(
            AGENT,
            {
                "name": "seeded",
                "mode": "headless",
                "status": "parked",
                "park_reason": "done",
                "launch_kind": "implement",
                "csession_id": SESSION,
                "run": 1,
                "claude_session_path": str(env.fakebin.bin_dir / "claude-session"),
            },
        )
        with store.lease(lifecycle.lease_key(AGENT), wait=5.0) as lease:
            store.put_run(lease, AGENT, 1, {"run_kind": "implement"})


def race(
    env: Env,
    proc_tracker: ProcTracker,
    tmp_path: Path,
    commands: list[list[str]],
) -> tuple[subprocess.Popen[str], list[subprocess.Popen[str]], Path]:
    """Start the commands together; hold whichever calls claude-session first at a gate. Returns
    the loser (finished while the winner is held), the winner(s) and the gate."""
    gate = tmp_path / "gate"
    # The first mutating call is held: `start` (a resume's classification) or `forget` (a
    # teardown's last step). The read-only `status` of the live checks is not a mutation.
    env.fakebin.script("claude-session", "start", times=1, gate=str(gate))
    env.fakebin.script("claude-session", "forget", times=1, gate=str(gate))
    script = tmp_path / "barrier.py"
    script.write_text(BARRIER)
    barrier = tmp_path / "go"
    started: list[subprocess.Popen[str]] = [
        proc_tracker.popen(  # type: ignore[misc]
            [sys.executable, str(script), str(barrier), *command],
            env=env.env(),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for command in commands
    ]
    barrier.write_text("go")
    env.fakebin.wait_for_gate(gate, timeout=60)
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        finished = [p for p in started if p.poll() is not None]
        if finished:
            break
        time.sleep(0.02)
    assert len(finished) == 1, "exactly one command ends while the winner is held"
    loser = finished[0]
    return loser, [p for p in started if p is not loser], gate


def finish(proc: subprocess.Popen[str]) -> tuple[int, Any, str]:
    out, err = proc.communicate(timeout=120)
    return proc.returncode, out, err


def claude_session_calls(fakebin: Fakebin) -> list[list[str]]:
    return fakebin.argvs("claude-session")


@pytest.mark.parametrize(
    ("first", "second", "reasons"),
    [
        (["resume", AGENT, "--json"], ["resume", AGENT, "--json"], {"live"}),
        (["teardown", AGENT, "--json"], ["teardown", AGENT, "--json"], {"tearing_down"}),
        (["resume", AGENT, "--json"], ["teardown", AGENT, "--json"], {"live", "tearing_down"}),
    ],
    ids=["two-resumes", "two-teardowns", "resume-versus-teardown"],
)
def test_exactly_one_entry_wins_and_the_loser_calls_nothing(
    env: Env,
    fakebin: Fakebin,
    proc_tracker: ProcTracker,
    tmp_path: Path,
    first: list[str],
    second: list[str],
    reasons: set[str],
) -> None:
    seed_parked(env)
    loser, winners, gate = race(env, proc_tracker, tmp_path, [first, second])
    code, out, err = finish(loser)
    assert code == 20, (out, err)
    error = result_json(subprocess.CompletedProcess([], code, out, err))["error"]
    assert error["code"] == "refused" and error["reason"] in reasons
    # the only mutating claude-session call so far is the winner's, held at the gate: the
    # loser made none (it may have read `status`, which changes nothing)
    mutating = [c for c in claude_session_calls(fakebin) if c[0] != "status"]
    assert len(mutating) == 1, mutating
    assert [e for e in loser_hooks(env)] == []
    gate.write_text("open")
    (winner,) = winners
    code, out, err = finish(winner)
    assert code == 0, (out, err)
    track_agent(env, proc_tracker, AGENT)
    assert env.agent(AGENT)["status"] in ("live", "parked", "torn_down")


def loser_hooks(env: Env) -> list[str]:
    path = env.state_dir / "log.jsonl"
    if not path.exists():
        return []
    return [line for line in path.read_text().splitlines() if "hook_delivered" in line]


def test_stop_composes_with_a_resume_that_holds_the_entry(
    env: Env, fakebin: Fakebin, proc_tracker: ProcTracker, tmp_path: Path
) -> None:
    seed_parked(env)
    gate = tmp_path / "gate"
    fakebin.script("claude-session", "start", times=1, gate=str(gate))
    resumer = proc_tracker.popen(
        env.argv("resume", AGENT, "--json"),
        env=env.env(),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    fakebin.wait_for_gate(gate, timeout=60)
    assert env.agent(AGENT)["status"] == "launching"
    done = env.run("stop", AGENT, "--dry-run", "--json")
    assert done.returncode == 0, done.stderr
    body = result_json(done)
    assert body["action"] == "signal_launcher" and body["dry_run"] is True
    assert env.agent(AGENT)["status"] == "launching"  # a dry run signals nothing
    gate.write_text("open")
    code, out, err = finish(resumer)  # type: ignore[arg-type]
    assert code == 0, (out, err)
    track_agent(env, proc_tracker, AGENT)
