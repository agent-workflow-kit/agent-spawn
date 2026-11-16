"""The cwd rule and the throwaway check on ``resume`` and ``resume --fork`` (interface.md,
Third-party runs): the pre-resume row stays unchanged, and a run in a throwaway holds nothing."""

from __future__ import annotations

import json

import pytest

from tests.conftest import Isolation
from tests.fakebin.harness import Fakebin
from tests.launch_support import Env, result_json
from tests.test_resume import launched, starts
from tests.test_worktrees_launch import fake_throwaway, wtctl_calls


@pytest.fixture
def env(fakebin: Fakebin, isolation_env: Isolation) -> Env:
    return Env(fakebin, isolation_env)


def third_party_dry_run(fakebin: Fakebin, cwd: str) -> None:
    fakebin.script(
        "claude-session",
        "start",
        argv_contains=["--dry-run"],
        merge={"profile": {"reads_third_party": True}, "cwd": cwd},
    )


def test_a_trusted_plain_resume_in_a_throwaway_is_refused_and_the_row_is_unchanged(
    env: Env, fakebin: Fakebin
) -> None:
    throwaway = fake_throwaway(fakebin)
    agent_id = launched(env, "--cwd", throwaway["path"], "--reads-third-party")
    before = env.agent(agent_id)
    done = env.run("resume", agent_id, "--json", "--wait")  # the plan says trusted
    assert done.returncode == 20, (done.stdout, done.stderr)
    error = json.loads(done.stdout)["error"]
    assert (error["reason"], error["tool"]) == ("third_party_run", "agent-spawn")
    assert (env.agent(agent_id)["status"], env.agent(agent_id)["updated"]) == (
        before["status"],
        before["updated"],
    )
    assert len(starts(fakebin)) == 1


def test_a_trusted_fork_in_a_throwaway_is_refused_and_leaves_no_record(
    env: Env, fakebin: Fakebin
) -> None:
    throwaway = fake_throwaway(fakebin)
    agent_id = launched(env, "--cwd", throwaway["path"], "--reads-third-party")
    done = env.run("resume", agent_id, "--fork", "--json", "--wait")
    assert done.returncode == 20, (done.stdout, done.stderr)
    assert json.loads(done.stdout)["error"]["reason"] == "third_party_run"
    assert [a["agent_id"] for a in env.agents()] == [agent_id]
    assert len(starts(fakebin)) == 1


def test_a_third_party_plain_resume_and_fork_at_a_throwaway_root_hold_nothing(
    env: Env, fakebin: Fakebin
) -> None:
    throwaway = fake_throwaway(fakebin)
    agent_id = launched(env, "--cwd", throwaway["path"], "--reads-third-party")
    third_party_dry_run(fakebin, throwaway["path"])
    plain = env.run("resume", agent_id, "--json", "--wait")
    assert plain.returncode == 0, (plain.stdout, plain.stderr)
    fork = env.run("resume", agent_id, "--fork", "--json", "--wait")
    assert fork.returncode == 0, (fork.stdout, fork.stderr)
    assert result_json(fork)["agent"]["holds"] == []
    assert wtctl_calls(fakebin, "adopt") == []
    assert len(starts(fakebin)) == 3


def test_a_resume_with_a_failing_records_lookup_changes_nothing(env: Env, fakebin: Fakebin) -> None:
    agent_id = launched(env)
    before = env.agent(agent_id)
    fakebin.script("wtctl", "list", error={"code": "internal", "message": "x"}, exit=1)
    done = env.run("resume", agent_id, "--json", "--wait")
    assert done.returncode == 1, (done.stdout, done.stderr)
    assert json.loads(done.stdout)["error"]["tool"] == "wtctl"
    assert (env.agent(agent_id)["status"], env.agent(agent_id)["updated"]) == (
        before["status"],
        before["updated"],
    )
    assert len(starts(fakebin)) == 1
