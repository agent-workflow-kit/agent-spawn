"""``show``: ``third_party_run`` and ``repo_config_changed`` from wtctl (found by the Stage 2 flow
scenarios: both were left as seams, null with a warning, after their owning issues landed)."""

from __future__ import annotations

import json
from typing import Any

import pytest

from tests import schema_registry
from tests.conftest import Isolation
from tests.fakebin.harness import Fakebin
from tests.inspect_support import A1, seed_agent
from tests.launch_support import Env
from tests.test_worktrees_launch import fake_throwaway
from tests.test_worktrees_resume import launched_in_worktree

SHOW = "agent-spawn/cli/show.json"


@pytest.fixture
def env(fakebin: Fakebin, isolation_env: Isolation) -> Env:
    return Env(fakebin, isolation_env)


def show(env: Env, agent_id: str) -> Any:
    done = env.run("show", agent_id, "--json")
    assert done.returncode == 0, (done.stdout, done.stderr)
    body = json.loads(done.stdout)
    schema_registry.validate(body, SHOW)
    return body


def test_third_party_run_is_true_inside_a_throwaway(env: Env, fakebin: Fakebin) -> None:
    throwaway = fake_throwaway(fakebin)
    seed_agent(env, A1, cwd=throwaway["path"] + "/pkg")
    body = show(env, A1)
    assert body["third_party_run"] is True and body["warnings"] == []


def test_third_party_run_is_false_outside_every_throwaway(env: Env, fakebin: Fakebin) -> None:
    fake_throwaway(fakebin)
    seed_agent(env, A1, cwd="/work/elsewhere")
    body = show(env, A1)
    assert body["third_party_run"] is False and body["warnings"] == []


def test_third_party_run_is_null_with_a_warning_when_the_lookup_fails(
    env: Env, fakebin: Fakebin
) -> None:
    fakebin.script("wtctl", "list", error={"code": "busy", "message": "x"}, exit=12)
    seed_agent(env, A1, cwd="/work/x")
    body = show(env, A1)
    assert body["third_party_run"] is None
    assert any("throwaway" in w for w in body["warnings"])


def test_repo_config_changed_is_null_without_a_worktree(env: Env) -> None:
    seed_agent(env, A1, cwd="/work/x")
    assert show(env, A1)["repo_config_changed"] is None


def test_repo_config_changed_reports_an_unchanged_baseline(env: Env, fakebin: Fakebin) -> None:
    agent_id = launched_in_worktree(env)
    body = show(env, agent_id)
    assert body["repo_config_changed"] == {"changed": False, "changed_classes": [], "items": []}
    assert body["third_party_run"] is False and body["warnings"] == []
    (check,) = [a for a in fakebin.argvs("wtctl") if a[:2] == ["baseline", "check"]]
    assert check[2] == env.agent(agent_id)["wt_record_id"]


def test_repo_config_changed_reports_a_changed_baseline(env: Env, fakebin: Fakebin) -> None:
    agent_id = launched_in_worktree(env)
    item = {"class": "git", "path": ".git/hooks/pre-commit"}
    fakebin.script(
        "wtctl",
        "baseline check",
        exit=20,
        error={
            "code": "refused",
            "reason": "repo_config_changed",
            "message": "changed",
            "items": [item],
            "details": {"digest": "d" * 64, "recorded_digest": "c" * 64,
                        "changed_classes": ["git"]},
        },
    )  # fmt: skip
    body = show(env, agent_id)
    assert body["repo_config_changed"] == {
        "changed": True,
        "changed_classes": ["git"],
        "items": [item],
    }


def test_repo_config_changed_is_null_with_a_warning_when_the_check_cannot_run(
    env: Env, fakebin: Fakebin
) -> None:
    agent_id = launched_in_worktree(env)
    fakebin.script("wtctl", "baseline check", exit=12, error={"code": "busy", "message": "x"})
    body = show(env, agent_id)
    assert body["repo_config_changed"] is None
    assert any("baseline" in w for w in body["warnings"])
