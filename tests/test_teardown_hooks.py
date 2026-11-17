"""The ``pre-teardown`` hook (#32; interface.md, Teardown order step 3 and Escalation overrides):
veto 11, escalation 13 and the override, the payload, the return to ``parked`` unchanged, the
retry rule, the live re-check after a slow hook, and the supervisor's ephemeral teardown."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tests import schema_registry
from tests.conftest import Isolation
from tests.fakebin.harness import Fakebin
from tests.launch_support import Env, result_json
from tests.test_teardown import dead, events, forget_calls, seed, td
from tests.test_teardown_worktrees import audit_records
from tests.test_worktrees_launch import wtctl_calls
from tests.test_worktrees_resume import launched_in_worktree

HUMAN = ("--actor", "human:tester")
VETO = "pre-teardown=echo nope >&2; exit 1"
ESCALATE = "pre-teardown=echo careful >&2; exit 13"


@pytest.fixture
def env(fakebin: Fakebin, isolation_env: Isolation) -> Env:
    return Env(fakebin, isolation_env)


def unchanged(env: Env, agent_id: str, before: dict[str, Any], files: bool = True) -> None:
    """The agent is ``parked`` exactly as before the attempt (but for the bookkeeping fields)."""
    after = env.agent(agent_id)
    for key in ("updated",):
        after.pop(key, None)
        before = {k: v for k, v in before.items() if k != key}
    assert after["status"] == "parked"
    assert after["teardown_step"] is None and after["teardown_proc"] is None
    assert after == before
    if files:
        assert (env.state_dir / "prompts" / agent_id).exists()
        assert (env.state_dir / "outcomes" / agent_id).exists()


# --- veto and escalation ------------------------------------------------------------------------


def test_a_veto_exits_11_and_leaves_the_agent_parked_unchanged(env: Env, fakebin: Fakebin) -> None:
    agent_id = seed(env, fakebin)
    before = env.agent(agent_id)
    done = td(env, agent_id, "--hook", VETO)
    body = result_json(done)
    assert done.returncode == 11, (done.stdout, done.stderr)
    schema_registry.validate(body, "agent-spawn/cli/error.json")
    assert body["error"]["code"] == "vetoed" and body["error"]["hook"] == "pre-teardown"
    assert "nope" in body["error"]["hook_stderr"] and body["error"]["agent_id"] == agent_id
    unchanged(env, agent_id, before)
    assert env.agent(agent_id)["hooks"] == before["hooks"]  # this call's --hook goes with it
    assert forget_calls(fakebin) == []


def test_an_escalation_exits_13_and_leaves_the_agent_parked_unchanged(
    env: Env, fakebin: Fakebin
) -> None:
    agent_id = seed(env, fakebin)
    before = env.agent(agent_id)
    done = td(env, agent_id, "--hook", ESCALATE)
    body = result_json(done)
    assert done.returncode == 13, (done.stdout, done.stderr)
    schema_registry.validate(body, "agent-spawn/cli/error.json")
    assert body["error"]["code"] == "escalated" and "careful" in body["error"]["hook_stderr"]
    unchanged(env, agent_id, before)
    assert forget_calls(fakebin) == []


def test_a_hook_timeout_vetoes(fakebin: Fakebin, isolation_env: Isolation) -> None:
    config = 'hooks:\n  pre-teardown:\n    - cmd: "sleep 30"\n      timeout: 1\n'
    cfg_env = Env(fakebin, isolation_env, config)
    agent_id = seed(cfg_env, fakebin)
    done = td(cfg_env, agent_id)
    assert done.returncode == 11, (done.stdout, done.stderr)
    assert cfg_env.agent(agent_id)["status"] == "parked"
    assert forget_calls(fakebin) == []


def test_a_config_defined_hook_vetoes(fakebin: Fakebin, isolation_env: Isolation) -> None:
    cfg_env = Env(fakebin, isolation_env, 'hooks:\n  pre-teardown:\n    - cmd: "exit 3"\n')
    agent_id = seed(cfg_env, fakebin)
    before = cfg_env.agent(agent_id)
    done = td(cfg_env, agent_id)
    assert done.returncode == 11
    unchanged(cfg_env, agent_id, before)


def test_a_veto_with_a_worktree_makes_no_wtctl_removal_call(env: Env, fakebin: Fakebin) -> None:
    agent_id = launched_in_worktree(env)
    before = env.agent(agent_id)
    mark = len(fakebin.argvs("wtctl"))
    done = td(env, agent_id, "--hook", VETO)
    assert done.returncode == 11
    assert fakebin.argvs("wtctl")[mark:] == []  # not even a `show`
    unchanged(env, agent_id, before, files=False)
    assert forget_calls(fakebin) == [] and wtctl_calls(fakebin, "remove") == []
    assert audit_records(env) == []


def test_a_passing_hook_lets_the_teardown_finish(env: Env, fakebin: Fakebin) -> None:
    agent_id = seed(env, fakebin)
    done = td(env, agent_id, "--hook", "pre-teardown=true")
    assert done.returncode == 0, (done.stdout, done.stderr)
    assert env.agent(agent_id)["status"] == "torn_down"


def test_a_later_veto_overrides_an_earlier_escalation(
    fakebin: Fakebin, isolation_env: Isolation
) -> None:
    config = 'hooks:\n  pre-teardown:\n    - cmd: "exit 13"\n    - cmd: "exit 1"\n'
    cfg_env = Env(fakebin, isolation_env, config)
    agent_id = seed(cfg_env, fakebin)
    done = td(cfg_env, agent_id, *HUMAN, "--override-escalation")
    assert done.returncode == 11
    assert cfg_env.agent(agent_id)["status"] == "parked"


def test_decision_ids_a_hook_prints_are_logged(env: Env, fakebin: Fakebin) -> None:
    agent_id = seed(env, fakebin)
    cmd = """pre-teardown=echo '{"decision_id": "d-9"}'; exit 1"""
    assert td(env, agent_id, "--hook", cmd).returncode == 11
    log = (env.state_dir / "log.jsonl").read_text()
    assert "hook_veto" in log


# --- the override --------------------------------------------------------------------------------


def test_the_override_lets_an_escalation_through(env: Env, fakebin: Fakebin) -> None:
    agent_id = seed(env, fakebin)
    done = td(env, agent_id, "--hook", ESCALATE, *HUMAN, "--override-escalation")
    assert done.returncode == 0, (done.stdout, done.stderr)
    assert env.agent(agent_id)["status"] == "torn_down"
    assert [a[1] for a in forget_calls(fakebin)]


def test_the_override_does_not_override_a_veto(env: Env, fakebin: Fakebin) -> None:
    agent_id = seed(env, fakebin)
    done = td(env, agent_id, "--hook", VETO, *HUMAN, "--override-escalation")
    assert done.returncode == 11
    assert env.agent(agent_id)["status"] == "parked"


def test_the_override_flag_needs_a_human_actor(env: Env, fakebin: Fakebin) -> None:
    agent_id = seed(env, fakebin)
    ran = env.fakebin.bin_dir / "ran"
    done = td(env, agent_id, "--hook", f"pre-teardown=touch {ran}", "--override-escalation")
    assert done.returncode == 2
    done = td(env, agent_id, "--actor", "auto:bot", "--override-escalation")
    assert done.returncode == 2
    assert not ran.exists() and env.agent(agent_id)["status"] == "parked"


def test_the_override_reaches_the_hook_and_wtctl_remove_and_nothing_else(
    env: Env, fakebin: Fakebin, tmp_path: Path
) -> None:
    agent_id = launched_in_worktree(env)
    sink = tmp_path / "payload.json"
    done = td(env, agent_id, "--hook", f"pre-teardown=cat > {sink}; exit 13",
              *HUMAN, "--override-escalation")  # fmt: skip
    assert done.returncode == 0, (done.stdout, done.stderr)
    payload = json.loads(sink.read_text())
    assert payload["override_escalation"] is True
    (remove,) = wtctl_calls(fakebin, "remove")
    assert "--override-escalation" in remove
    others = [a for a in fakebin.argvs("wtctl") if a[0] != "remove"]
    assert all("--override-escalation" not in a for a in others)
    for argv in fakebin.argvs("claude-session") + fakebin.argvs("termctl"):
        assert "--override-escalation" not in argv
    # one audit record per wtctl remove call that carried the flag
    actions = sorted(a["action"] for a in audit_records(env))
    assert actions == ["teardown", "teardown_escalation_override"]


# --- the payload ---------------------------------------------------------------------------------


def test_the_payload_matches_its_schema_and_travels_on_stdin(
    env: Env, fakebin: Fakebin, tmp_path: Path
) -> None:
    agent_id = seed(env, fakebin)
    sink = tmp_path / "payload.json"
    envdump = tmp_path / "env"
    cmd = f"pre-teardown=cat > {sink}; env > {envdump}"
    done = td(env, agent_id, "--hook", cmd, "--merged-head", "b" * 40, "--decision-id", "d-1")
    assert done.returncode == 0, (done.stdout, done.stderr)
    payload = json.loads(sink.read_text())
    schema_registry.validate(payload, "agent-spawn/hooks/pre-teardown.json")
    assert payload["object_id"] == agent_id and payload["agent"]["agent_id"] == agent_id
    assert payload["merged_head"] == "b" * 40 and payload["force"] is False
    assert payload["override_escalation"] is False
    assert payload["decision_ids"] == ["d-1"] and payload["actor"].startswith("auto:")
    assert payload["status"] == "tearing_down"
    variables = dict(line.split("=", 1) for line in envdump.read_text().splitlines() if "=" in line)
    assert variables["AGENT_SPAWN_HOOK"] == "pre-teardown"
    assert variables["AGENT_SPAWN_ID"] == agent_id


def test_a_selector_that_does_not_match_skips_the_entry(
    fakebin: Fakebin, isolation_env: Isolation
) -> None:
    config = 'hooks:\n  pre-teardown:\n    - cmd: "exit 1"\n      labels: ["only=this"]\n'
    cfg_env = Env(fakebin, isolation_env, config)
    agent_id = seed(cfg_env, fakebin)
    assert td(cfg_env, agent_id).returncode == 0


# --- retries ---


def test_a_retry_that_never_completed_the_hooks_runs_them(env: Env, fakebin: Fakebin) -> None:
    agent_id = seed(
        env, fakebin, status="tearing_down", teardown_step="started", teardown_proc=dead()
    )
    done = td(env, agent_id, "--hook", VETO)
    assert done.returncode == 11
    row = env.agent(agent_id)
    assert row["status"] == "parked" and row["teardown_step"] is None


def test_a_retry_past_the_hooks_step_skips_them_and_logs_it(
    env: Env, fakebin: Fakebin, tmp_path: Path
) -> None:
    agent_id = seed(
        env, fakebin, status="tearing_down", teardown_step="hooks_done", teardown_proc=dead()
    )
    ran = tmp_path / "ran"
    done = td(env, agent_id, "--hook", f"pre-teardown=touch {ran}; exit 1")
    assert done.returncode == 0, (done.stdout, done.stderr)
    assert not ran.exists()
    assert '"teardown_retry"' in (env.state_dir / "log.jsonl").read_text()


def test_a_retry_without_hook_flags_runs_the_stored_hook(env: Env, fakebin: Fakebin) -> None:
    agent_id = seed(
        env,
        fakebin,
        status="tearing_down",
        teardown_step="started",
        teardown_proc=dead(),
        hooks=[{"hook": "pre-teardown", "cmd": "exit 1"}],
    )
    assert td(env, agent_id).returncode == 11


# --- liveness after a slow hook ---


def test_a_run_that_went_live_during_the_hook_returns_the_agent_to_parked(
    env: Env, fakebin: Fakebin, tmp_path: Path
) -> None:
    from tests.launch_support import script_live_run

    agent_id = seed(env, fakebin)
    before = env.agent(agent_id)
    # The first `status` (step 2's live check) answers like the fake's default, every later one
    # (step 3's re-check, after the hook) reports a live run.
    fakebin.script("claude-session", "status", times=1)
    script_live_run(fakebin)
    done = td(env, agent_id, "--hook", "pre-teardown=true")
    body = result_json(done)
    assert done.returncode == 20, (done.stdout, done.stderr)
    assert body["error"]["reason"] == "live"
    unchanged(env, agent_id, before)
    assert forget_calls(fakebin) == []


# --- events ---


def test_a_veto_queues_state_changes_but_no_post_teardown(env: Env, fakebin: Fakebin) -> None:
    agent_id = seed(env, fakebin)
    assert td(env, agent_id, "--hook", VETO).returncode == 11
    names = [n for n, _ in events(env, agent_id)]
    assert "post-teardown" not in names
    states = [p["agent"]["status"] for n, p in events(env, agent_id) if n == "post-state-change"]
    assert states == ["tearing_down", "parked"]
