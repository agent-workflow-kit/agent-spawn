"""``--override-escalation`` routing (#32; interface.md, Escalation overrides): the flag reaches
exactly one tool, asserted by the fakes' argv, and is refused without a target."""

from __future__ import annotations

from typing import Any

import pytest

from agent_spawn.lifecycle import lease_key
from tests.conftest import Isolation
from tests.fakebin.harness import Fakebin
from tests.launch_support import Env, result_json
from tests.test_resume import launched, starts
from tests.test_worktrees_launch import REPO, wtctl_calls

FLAG = "--override-escalation"
HUMAN = ("--actor", "human:tester")


@pytest.fixture
def env(fakebin: Fakebin, isolation_env: Isolation) -> Env:
    return Env(fakebin, isolation_env)


def carrying_flag(fakebin: Fakebin) -> list[tuple[str, list[str]]]:
    return [
        (tool, argv)
        for tool in ("claude-session", "wtctl", "termctl", "tmux")
        for argv in fakebin.argvs(tool)
        if FLAG in argv
    ]


def set_row(env: Env, agent_id: str, **fields: Any) -> None:
    with env.store() as store, store.lease(lease_key(agent_id)) as lease:
        store.update_agent(lease, agent_id, fields)


def test_launch_override_overrides_the_pre_launch_hook_only(env: Env, fakebin: Fakebin) -> None:
    done = env.run(
        "launch", "--mode", "headless", "--skill", "do-it", "--json", "--wait",
        "--repo", REPO, "--worktree", "new", "--branch", "feat/x", "--base", "main",
        "--hook", "pre-launch=exit 13", FLAG, *HUMAN,
    )  # fmt: skip
    assert done.returncode == 0, (done.stdout, done.stderr)
    assert wtctl_calls(fakebin, "create")  # the worktree step ran
    assert carrying_flag(fakebin) == []


def test_launch_override_does_not_override_a_veto(env: Env, fakebin: Fakebin) -> None:
    done = env.run(
        "launch", "--mode", "headless", "--skill", "do-it", "--json",
        "--hook", "pre-launch=exit 1", FLAG, *HUMAN,
    )  # fmt: skip
    assert done.returncode == 11 and env.agents() == []


def test_claude_session_target_only_when_its_last_run_ended_escalated(
    env: Env, fakebin: Fakebin
) -> None:
    agent_id = launched(env)
    set_row(env, agent_id, escalated_by="claude-session", park_reason="escalated",
            last_outcome={"outcome": "done", "phase": "session"})  # fmt: skip
    before = len(starts(fakebin))
    done = env.run("resume", agent_id, FLAG, *HUMAN, "--json")
    assert done.returncode == 2, (done.stdout, done.stderr)
    assert len(starts(fakebin)) == before and carrying_flag(fakebin) == []
    assert env.agent(agent_id)["status"] == "parked"


def test_a_termctl_escalation_without_a_window_has_no_target(env: Env, fakebin: Fakebin) -> None:
    agent_id = launched(env)
    set_row(env, agent_id, escalated_by="termctl", park_reason="escalated")
    done = env.run("resume", agent_id, "--mode", "headless", FLAG, *HUMAN, "--json")
    assert done.returncode == 2 and carrying_flag(fakebin) == []


def test_a_headless_fork_has_no_target_for_the_override(env: Env, fakebin: Fakebin) -> None:
    agent_id = launched(env)
    done = env.run("resume", agent_id, "--fork", "--mode", "headless", FLAG, *HUMAN, "--json")
    assert done.returncode == 2, (done.stdout, done.stderr)
    assert carrying_flag(fakebin) == [] and len(env.agents()) == 1
    assert result_json(done)["error"]["code"] == "usage"


def test_a_here_fork_has_no_target_for_the_override(env: Env, fakebin: Fakebin) -> None:
    agent_id = launched(env)
    done = env.run("resume", agent_id, "--fork", "--placement", "here", FLAG, *HUMAN, "--json")
    assert done.returncode == 2, (done.stdout, done.stderr)
    assert carrying_flag(fakebin) == [] and len(env.agents()) == 1
    assert result_json(done)["error"]["code"] == "usage"
