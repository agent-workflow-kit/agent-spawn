"""``show``: ``third_party_run`` and ``repo_config_changed`` from wtctl (found by the Stage 2 flow
scenarios: both were left as seams, null with a warning, after their owning issues landed)."""

from __future__ import annotations

import json
import os
import time
from types import SimpleNamespace
from typing import Any

import pytest

from agent_spawn import gateway as gw
from agent_spawn import showing
from agent_spawn.worktrees import WtctlBackend
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


# --- hardening: what a lower tool may print is untrusted ----------------------------------------

RECORD = "wt-aaaaaaaa"


def changed_check(fakebin: Fakebin, error: dict[str, Any]) -> None:
    fakebin.script(
        "wtctl",
        "baseline check",
        exit=20,
        error={"code": "refused", "reason": "repo_config_changed", "message": "changed", **error},
    )


def test_an_anchors_only_change_omits_the_items_and_the_classes(env: Env, fakebin: Fakebin) -> None:
    agent_id = launched_in_worktree(env)
    changed_check(fakebin, {"details": {"digest": "d" * 64, "recorded_digest": "c" * 64}})
    body = show(env, agent_id)
    assert body["repo_config_changed"] == {"changed": True, "changed_classes": [], "items": []}
    assert body["warnings"] == []


def test_a_digest_only_change_omits_the_details_and_the_items(env: Env, fakebin: Fakebin) -> None:
    agent_id = launched_in_worktree(env)
    changed_check(fakebin, {})
    body = show(env, agent_id)
    assert body["repo_config_changed"] == {"changed": True, "changed_classes": [], "items": []}


@pytest.mark.parametrize(
    "error",
    [
        {"items": "x"},
        {"items": [1]},
        {"details": {"changed_classes": "git"}},
        {"details": {"changed_classes": ["other"]}},
        {"details": {"changed_classes": ["git", "git"]}},
        {"details": "x"},
    ],
)
def test_a_present_but_malformed_change_is_null_with_a_warning(
    env: Env, fakebin: Fakebin, error: dict[str, Any]
) -> None:
    agent_id = launched_in_worktree(env)
    changed_check(fakebin, error)
    body = show(env, agent_id)
    assert body["repo_config_changed"] is None
    assert any("baseline" in w for w in body["warnings"])


def test_an_unchanged_baseline_with_a_non_object_body_is_null_with_a_warning(
    env: Env, fakebin: Fakebin
) -> None:
    agent_id = launched_in_worktree(env)
    fakebin.script("wtctl", "baseline check", exit=0, stdout="[]")
    body = show(env, agent_id)
    assert body["repo_config_changed"] is None
    assert any("baseline" in w for w in body["warnings"])


def test_a_deeply_nested_reply_is_malformed_not_a_crash(env: Env, fakebin: Fakebin) -> None:
    agent_id = launched_in_worktree(env)
    deep = "[" * 100_000
    fakebin.script("wtctl", "baseline check", exit=20, stdout=deep)
    fakebin.script("wtctl", "list", exit=0, stdout=deep)
    body = show(env, agent_id)
    assert body["repo_config_changed"] is None and body["third_party_run"] is None
    assert len(body["warnings"]) == 2


def test_non_finite_numbers_in_a_reply_are_malformed_not_a_crash(
    env: Env, fakebin: Fakebin
) -> None:
    agent_id = launched_in_worktree(env)
    fakebin.script(
        "wtctl",
        "baseline check",
        exit=20,
        stdout='{"error": {"reason": "repo_config_changed", "items": [{"n": NaN}]}}',
    )
    fakebin.script("wtctl", "list", exit=0, stdout='{"records": [{"n": Infinity}]}')
    body = show(env, agent_id)
    assert body["repo_config_changed"] is None and body["third_party_run"] is None
    assert len(body["warnings"]) == 2


def test_show_bounds_each_wtctl_read_with_its_own_timeout(
    env: Env, fakebin: Fakebin, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent_id = launched_in_worktree(env)
    row = env.agent(agent_id)
    fakebin.script("wtctl", "list", sleep=5)
    fakebin.script("wtctl", "baseline check", sleep=5)
    real = gw.get_gateway()
    fake_env = fakebin.env()

    class FakeEnvGateway:  # runs the fakes without changing this process's environment
        def run(self, argv: Any, **kwargs: Any) -> Any:
            return real.run(argv, **{**kwargs, "env": fake_env})

    monkeypatch.setattr(
        showing,
        "WtctlBackend",
        lambda argv: WtctlBackend(argv, gateway=FakeEnvGateway()),  # type: ignore[arg-type]
    )
    monkeypatch.setenv("PATH", fake_env["PATH"])
    monkeypatch.setattr(showing, "SHOW_TIMEOUT", 0.3)
    inv: Any = SimpleNamespace(config=SimpleNamespace(wtctl_cmd=["wtctl"]))
    warnings: list[str] = []
    started = time.monotonic()
    assert showing.third_party_run(inv, row, warnings) is None
    assert showing.repo_config_changed(inv, row, warnings) is None
    assert time.monotonic() - started < 4
    assert len(warnings) == 2
    assert "not looked up" in warnings[0] and "not checked" in warnings[1]


# --- wtctl resolution -----------------------------------------------------------------------------


def test_without_wtctl_the_cwd_is_no_third_party_run_and_the_baseline_warns(
    fakebin: Fakebin, isolation_env: Isolation
) -> None:
    env = Env(fakebin, isolation_env)
    agent_id = launched_in_worktree(env)
    env.config.parent.mkdir(parents=True, exist_ok=True)
    env.config.write_text("wtctl_cmd: null\n")
    body = show(env, agent_id)
    assert body["third_party_run"] is False and body["repo_config_changed"] is None
    assert body["warnings"] == ["the repo config baseline was not checked (no wtctl)"]


def test_a_record_agent_without_wtctl_warns_but_one_without_a_record_does_not(
    fakebin: Fakebin, isolation_env: Isolation
) -> None:
    env = Env(fakebin, isolation_env, "wtctl_cmd: null\n")
    seed_agent(env, A1, cwd="/work/x")
    body = show(env, A1)
    assert body["repo_config_changed"] is None and body["warnings"] == []


def test_the_stored_wtctl_path_is_the_fallback_when_the_configured_command_is_missing(
    fakebin: Fakebin, isolation_env: Isolation
) -> None:
    env = Env(fakebin, isolation_env)
    agent_id = launched_in_worktree(env)
    stored = env.agent(agent_id)["wtctl_path"]
    assert os.path.isabs(stored)
    env.config.parent.mkdir(parents=True, exist_ok=True)
    env.config.write_text("wtctl_cmd: [/nonexistent/wtctl]\n")
    body = show(env, agent_id)
    assert body["third_party_run"] is False
    assert body["repo_config_changed"] == {"changed": False, "changed_classes": [], "items": []}
    assert body["warnings"] == []


@pytest.mark.parametrize("stored", ["   ", "", "'unbalanced"])
def test_an_unusable_stored_wtctl_path_is_tool_missing_with_warnings(
    fakebin: Fakebin, isolation_env: Isolation, stored: str
) -> None:
    env = Env(fakebin, isolation_env, "wtctl_cmd: [/nonexistent/wtctl]\n")
    seed_agent(env, A1, cwd="/work/x", wt_record_id=RECORD, wtctl_path=stored)
    body = show(env, A1)
    assert body["third_party_run"] is None and body["repo_config_changed"] is None
    assert len(body["warnings"]) == 2 and all("tool_missing" in w for w in body["warnings"])
