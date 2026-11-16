"""Resume step 5 (worktree of an agent without a session), the holds of a resume and a fork,
and the restore rules, against the fake binaries."""

from __future__ import annotations

import json
import os
import signal
import subprocess
from pathlib import Path
from typing import Any

import pytest

from tests import schema_registry
from tests.conftest import Isolation
from tests.fakebin.harness import Fakebin
from tests.launch_support import Env, result_json
from tests.procguard import ProcTracker

REPO = "example.org/team/project"
LAUNCH = ("launch", "--mode", "headless", "--skill", "do-it", "--json", "--wait")
BUSY = {"error": {"code": "busy", "message": "later"}, "exit": 12}


@pytest.fixture
def env(fakebin: Fakebin, isolation_env: Isolation) -> Env:
    return Env(fakebin, isolation_env)


def wtctl_calls(fakebin: Fakebin, command: str | None = None) -> list[list[str]]:
    return [c for c in fakebin.argvs("wtctl") if command is None or c[0] == command]


def starts(fakebin: Fakebin) -> list[list[str]]:
    return [a for a in fakebin.argvs("claude-session") if a[0] == "start" and "--dry-run" not in a]


def records_path(fakebin: Fakebin) -> Path:
    return Path(str(fakebin.log) + ".records.json")


def records(fakebin: Fakebin) -> dict[str, Any]:
    path = records_path(fakebin)
    return json.loads(path.read_text()) if path.exists() else {}  # type: ignore[no-any-return]


def set_state(fakebin: Fakebin, record_id: str, state: str) -> None:
    data = records(fakebin)
    data[record_id]["state"] = state
    if state in ("removed", "forgotten", "gone"):
        data[record_id]["removed_at"] = "2026-01-01T00:00:00Z"
    records_path(fakebin).write_text(json.dumps(data))


def only_agent(env: Env) -> dict[str, Any]:
    agents = env.agents()
    assert len(agents) == 1, agents
    return agents[0]


def failed_launch(env: Env, fakebin: Fakebin, *extra: str, **respond: Any) -> str:
    """A launch whose `wtctl create` fails (busy by default): a parked agent without a session."""
    fakebin.script("wtctl", "create", times=1, **(respond or BUSY))
    done = env.run(*LAUNCH, "--repo", REPO, "--worktree", "new", "--prompt", "PROMPT-TEXT", *extra)
    assert done.returncode != 0 and done.returncode != 22, (done.stdout, done.stderr)
    return str(only_agent(env)["agent_id"])


def crashed_create(fakebin: Fakebin, agent_id: str) -> dict[str, Any]:
    """The record a `create` left behind when the launcher died before storing it."""
    done = fakebin.run("wtctl", "create", "--repo", REPO, "--holder", agent_id,
                       "--name", agent_id, "--json")  # fmt: skip
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)["record"]  # type: ignore[no-any-return]


def test_busy_at_launch_then_resume_creates_and_starts_the_stored_prompt(
    env: Env, fakebin: Fakebin
) -> None:
    agent_id = failed_launch(env, fakebin)
    row = env.agent(agent_id)
    assert (row["status"], row["park_reason"], row["needs_attention"]) == (
        "parked", "launch_failed", None)  # fmt: skip
    done = env.run("resume", agent_id, "--json", "--wait")
    assert done.returncode == 0, (done.stdout, done.stderr)
    body = result_json(done)
    schema_registry.validate(body, "agent-spawn/cli/resume.json")
    creates = wtctl_calls(fakebin, "create")
    assert len(creates) == 2 and creates[0][:4] == creates[1][:4]  # the same deterministic name
    assert f"--name={agent_id}" in creates[1] and f"--holder={agent_id}" in creates[1]
    record = next(iter(records(fakebin).values()))
    assert body["agent"]["worktree"] == {
        "record_id": record["id"],
        "path": record["path"],
        "owned": True,
    }
    argv = starts(fakebin)[-1]
    assert f"--cwd={os.path.realpath(record['path'])}" in argv and f"--repo={REPO}" in argv
    prompt = next(a for a in argv if a.startswith("--prompt-file="))
    assert "PROMPT-TEXT" not in " ".join(argv) and prompt  # the stored copy travels in a file
    after = env.agent(agent_id)
    assert after["csession_id"] is not None and after["needs_attention"] is None
    assert (
        after["cwd"] == os.path.realpath(record["path"]) and after["wt_record_id"] == record["id"]
    )


def test_resume_finds_the_record_a_crashed_launch_left_by_name(env: Env, fakebin: Fakebin) -> None:
    agent_id = failed_launch(env, fakebin)
    left = crashed_create(fakebin, agent_id)
    before = len(wtctl_calls(fakebin, "create"))
    done = env.run("resume", agent_id, "--json", "--wait")
    assert done.returncode == 0, (done.stdout, done.stderr)
    assert len(wtctl_calls(fakebin, "create")) == before  # no second worktree
    assert len(records(fakebin)) == 1
    assert wtctl_calls(fakebin, "list")[0][:2] == ["list", f"--repo={REPO}"] or any(
        c[:2] == ["list", f"--repo={REPO}"] for c in wtctl_calls(fakebin, "list")
    )
    row = env.agent(agent_id)
    assert row["wt_record_id"] == left["id"] and row["wt_owned"] is True
    with env.store() as store:
        assert [h["record_id"] for h in store.get_holds(agent_id)] == [left["id"]]


def test_a_record_of_the_same_name_held_by_someone_else_is_not_ours(
    env: Env, fakebin: Fakebin
) -> None:
    agent_id = failed_launch(env, fakebin)
    other = fakebin.run("wtctl", "create", "--repo", REPO, "--holder", "ag-zzzzzzzz",
                        "--name", agent_id, "--json")  # fmt: skip
    assert other.returncode == 0
    done = env.run("resume", agent_id, "--json", "--wait")
    # not ours: agent-spawn tries to create, and wtctl refuses the taken name (path_exists)
    assert done.returncode == 20, (done.stdout, done.stderr)
    assert len(starts(fakebin)) == 0
    assert env.agent(agent_id)["wt_record_id"] is None


def test_a_creating_record_parks_with_the_reconcile_hint(env: Env, fakebin: Fakebin) -> None:
    agent_id = failed_launch(env, fakebin)
    left = crashed_create(fakebin, agent_id)
    set_state(fakebin, left["id"], "creating")
    before = len(wtctl_calls(fakebin, "create"))
    done = env.run("resume", agent_id, "--json")
    assert done.returncode == 12, (done.stdout, done.stderr)
    error = json.loads(done.stdout)["error"]
    assert error["code"] == "launch_failed" and "wtctl reconcile" in error["details"]["hint"]
    assert error["agent_id"] == agent_id
    row = env.agent(agent_id)
    assert (row["status"], row["park_reason"], row["needs_attention"]) == (
        "parked", "launch_failed", "launch_failed")  # fmt: skip
    assert row["pre_resume"] is None  # parked, never restored
    assert len(wtctl_calls(fakebin, "create")) == before and starts(fakebin) == []


@pytest.mark.parametrize("state", ["unlinked", "missing", "clone_missing", "partial"])
def test_another_live_state_parks_worktree_state(env: Env, fakebin: Fakebin, state: str) -> None:
    agent_id = failed_launch(env, fakebin)
    left = crashed_create(fakebin, agent_id)
    set_state(fakebin, left["id"], state)
    done = env.run("resume", agent_id, "--json")
    assert done.returncode == 12, (done.stdout, done.stderr)
    error = json.loads(done.stdout)["error"]
    assert error["code"] == "failed" and error["tool"] == "agent-spawn"
    assert error["details"]["cause"] == "worktree_state" and error["details"]["retryable"] is False
    assert left["id"] in error["details"]["hint"]
    schema_registry.validate(json.loads(done.stdout), "agent-spawn/cli/error.json")
    row = env.agent(agent_id)
    assert (row["status"], row["park_reason"], row["needs_attention"]) == (
        "parked", "launch_failed", "launch_failed")  # fmt: skip


def test_a_stored_id_in_a_live_state_is_judged_by_show(env: Env, fakebin: Fakebin) -> None:
    agent_id = failed_launch(env, fakebin, bootstrap_failed="install")  # type: ignore[arg-type]
    row = env.agent(agent_id)
    assert row["wt_record_id"] is not None
    set_state(fakebin, row["wt_record_id"], "unlinked")
    done = env.run("resume", agent_id, "--json")
    assert done.returncode == 12
    assert json.loads(done.stdout)["error"]["details"]["cause"] == "worktree_state"
    assert wtctl_calls(fakebin, "show")  # by id, not by the name lookup
    assert not [c for c in wtctl_calls(fakebin, "list") if c[1:2] == [f"--repo={REPO}"]]


@pytest.mark.parametrize("state", ["removed", "forgotten", "gone"])
def test_a_tombstone_is_worktree_gone_and_the_row_is_restored(
    env: Env, fakebin: Fakebin, state: str
) -> None:
    fakebin.script("wtctl", "create", bootstrap_failed="install")
    done = env.run(*LAUNCH, "--repo", REPO, "--worktree", "new")
    assert done.returncode == 12
    agent_id = str(only_agent(env)["agent_id"])
    before = env.agent(agent_id)
    set_state(fakebin, before["wt_record_id"], state)
    done = env.run("resume", agent_id, "--json")
    assert done.returncode == 12, (done.stdout, done.stderr)
    assert json.loads(done.stdout)["error"]["code"] == "worktree_gone"
    after = env.agent(agent_id)
    for field in ("status", "park_reason", "needs_attention", "updated", "last_outcome"):
        assert after[field] == before[field], field
    assert after["launcher"] is None and after["pre_resume"] is None
    assert starts(fakebin) == []


def test_a_record_wtctl_does_not_know_is_worktree_gone(env: Env, fakebin: Fakebin) -> None:
    agent_id = failed_launch(env, fakebin, bootstrap_failed="x")  # type: ignore[arg-type]
    data = records(fakebin)
    records_path(fakebin).write_text(json.dumps({}))
    assert data
    done = env.run("resume", agent_id, "--json")
    assert done.returncode == 12
    assert json.loads(done.stdout)["error"]["code"] == "worktree_gone"


def test_bootstrap_failed_at_launch_continues_at_step_5_after_a_human_fixed_it(
    env: Env, fakebin: Fakebin
) -> None:
    fakebin.script("wtctl", "create", bootstrap_failed="install")
    done = env.run(*LAUNCH, "--repo", REPO, "--worktree", "new")
    assert done.returncode == 12
    agent_id = str(only_agent(env)["agent_id"])
    creates = len(wtctl_calls(fakebin, "create"))
    done = env.run("resume", agent_id, "--json", "--wait")
    assert done.returncode == 0, (done.stdout, done.stderr)
    assert len(wtctl_calls(fakebin, "create")) == creates  # the record is kept: no new create
    assert len(records(fakebin)) == 1
    assert env.agent(agent_id)["csession_id"] is not None


# --- failures of step 5's create --------------------------------------------------------------


def test_a_busy_wtctl_at_resume_restores_the_row_and_is_retryable(
    env: Env, fakebin: Fakebin
) -> None:
    agent_id = failed_launch(env, fakebin)
    before = env.agent(agent_id)
    fakebin.script("wtctl", "create", times=1, **BUSY)
    done = env.run("resume", agent_id, "--json")
    assert done.returncode == 12, (done.stdout, done.stderr)
    assert json.loads(done.stdout)["error"]["details"]["retryable"] is True
    after = env.agent(agent_id)
    for field in ("status", "park_reason", "needs_attention", "updated"):
        assert after[field] == before[field], field
    assert after["launcher"] is None and after["pre_resume"] is None


def test_an_escalated_agent_that_hits_a_busy_wtctl_keeps_its_attention(
    env: Env, fakebin: Fakebin
) -> None:
    agent_id = failed_launch(env, fakebin, error={"code": "escalated", "message": "ask"}, exit=13)
    before = env.agent(agent_id)
    assert (before["needs_attention"], before["escalated_by"]) == ("escalated", "wtctl")
    fakebin.script("wtctl", "create", times=1, **BUSY)
    done = env.run("resume", agent_id, "--json")
    assert done.returncode == 12, (done.stdout, done.stderr)
    after = env.agent(agent_id)
    assert (after["status"], after["park_reason"], after["needs_attention"]) == (
        "parked", "escalated", "escalated")  # fmt: skip
    assert after["escalated_by"] == "wtctl" and after["updated"] == before["updated"]
    assert after["launcher"] is None and after["pre_resume"] is None


@pytest.mark.parametrize(
    ("respond", "exit_code", "park", "attention", "by"),
    [
        ({"error": {"code": "vetoed", "message": "no"}, "exit": 11}, 11, "vetoed", None, None),
        (
            {"error": {"code": "escalated", "message": "ask"}, "exit": 13},
            13,
            "escalated",
            "escalated",
            "wtctl",
        ),
    ],
)
def test_a_veto_or_escalation_of_create_at_resume_parks(
    env: Env,
    fakebin: Fakebin,
    respond: dict[str, Any],
    exit_code: int,
    park: str,
    attention: str | None,
    by: str | None,
) -> None:
    agent_id = failed_launch(env, fakebin)
    fakebin.script("wtctl", "create", times=1, **respond)
    done = env.run("resume", agent_id, "--json")
    assert done.returncode == exit_code, (done.stdout, done.stderr)
    row = env.agent(agent_id)
    assert (row["status"], row["park_reason"], row["needs_attention"], row["escalated_by"]) == (
        "parked", park, attention, by)  # fmt: skip
    assert row["pre_resume"] is None


def test_other_create_failures_at_resume_restore_the_row(env: Env, fakebin: Fakebin) -> None:
    agent_id = failed_launch(env, fakebin)
    before = env.agent(agent_id)
    fakebin.script("wtctl", "create", times=1,
                   error={"code": "fetch_failed", "message": "x"}, exit=12)  # fmt: skip
    done = env.run("resume", agent_id, "--json")
    assert done.returncode == 12
    after = env.agent(agent_id)
    assert after["updated"] == before["updated"] and after["status"] == "parked"
    assert after["needs_attention"] == before["needs_attention"]


def test_bootstrap_failed_at_resume_stores_the_record_and_restores_the_row(
    env: Env, fakebin: Fakebin
) -> None:
    agent_id = failed_launch(env, fakebin)
    fakebin.script("wtctl", "create", times=1, bootstrap_failed="install")
    done = env.run("resume", agent_id, "--json")
    assert done.returncode == 12
    row = env.agent(agent_id)
    record = next(iter(records(fakebin).values()))
    assert row["wt_record_id"] == record["id"] and row["status"] == "parked"
    assert row["needs_attention"] is None and row["pre_resume"] is None  # restored, not replaced


def test_override_escalation_goes_to_create_only_and_is_audited(env: Env, fakebin: Fakebin) -> None:
    agent_id = failed_launch(env, fakebin, error={"code": "escalated", "message": "ask"}, exit=13)
    done = env.run("resume", agent_id, "--override-escalation", "--actor", "human:me", "--json",
                   "--wait")  # fmt: skip
    assert done.returncode == 0, (done.stdout, done.stderr)
    create = wtctl_calls(fakebin, "create")[-1]
    assert "--override-escalation" in create and "--actor=human:me" in create
    assert "--override-escalation" not in starts(fakebin)[-1]
    audit = [json.loads(x) for x in (env.state_dir / "audit.jsonl").read_text().splitlines()]
    assert [r["action"] for r in audit] == ["resume_escalation_override"]
    assert audit[0]["target"] == agent_id and audit[0]["result"] == "ok"


def test_override_escalation_needs_a_human_and_a_wtctl_escalation(
    env: Env, fakebin: Fakebin
) -> None:
    agent_id = failed_launch(env, fakebin)  # busy: escalated_by is null
    done = env.run("resume", agent_id, "--override-escalation", "--actor", "human:me", "--json")
    assert done.returncode == 2
    escalated = failed_launch_again(env, fakebin)
    refused = env.run("resume", escalated, "--override-escalation", "--actor", "auto:bot", "--json")
    assert refused.returncode == 2
    assert not [c for c in wtctl_calls(fakebin, "create") if "--override-escalation" in c]


def failed_launch_again(env: Env, fakebin: Fakebin) -> str:
    fakebin.script("wtctl", "create", times=1, error={"code": "escalated", "message": "a"}, exit=13)
    done = env.run(*LAUNCH, "--repo", REPO, "--worktree", "new")
    assert done.returncode == 13
    return str(max(env.agents(), key=lambda a: a["created"])["agent_id"])


# --- order and classification -------------------------------------------------------------------


def test_a_third_party_run_with_a_stored_worktree_is_refused_before_any_wtctl_call(
    env: Env, fakebin: Fakebin
) -> None:
    agent_id = failed_launch(env, fakebin)
    calls = len(fakebin.argvs("wtctl"))
    before = env.agent(agent_id)
    done = env.run("resume", agent_id, "--reads-third-party", "--json")
    assert done.returncode == 20, (done.stdout, done.stderr)
    assert json.loads(done.stdout)["error"]["reason"] == "third_party_cwd"
    assert len(fakebin.argvs("wtctl")) == calls
    after = env.agent(agent_id)
    assert after["updated"] == before["updated"] and after["launcher"] is None


def test_a_dry_run_of_a_sessionless_worktree_agent_touches_no_record(
    env: Env, fakebin: Fakebin
) -> None:
    agent_id = failed_launch(env, fakebin)
    calls = len(fakebin.argvs("wtctl"))
    done = env.run("resume", agent_id, "--dry-run", "--json")
    assert done.returncode == 0, (done.stdout, done.stderr)
    schema_registry.validate(result_json(done), "agent-spawn/cli/launch-dry-run.json")
    assert len(fakebin.argvs("wtctl")) == calls and records(fakebin) == {}
    third = env.run("resume", agent_id, "--dry-run", "--reads-third-party", "--json")
    assert third.returncode == 20 and result_json(third)["would_refuse"] == "third_party_cwd"


def test_wtctl_null_after_a_worktree_launch_exits_2_and_restores(
    env: Env, fakebin: Fakebin, isolation_env: Isolation
) -> None:
    agent_id = failed_launch(env, fakebin)
    before = env.agent(agent_id)
    env.config.parent.mkdir(parents=True, exist_ok=True)
    env.config.write_text("wtctl_cmd: null\n")
    done = env.run("resume", agent_id, "--json")
    assert done.returncode == 2, (done.stdout, done.stderr)
    assert env.agent(agent_id)["updated"] == before["updated"]


def test_the_stored_wtctl_path_is_the_fallback_when_config_cannot_resolve_it(
    env: Env, fakebin: Fakebin, tmp_path: Path
) -> None:
    agent_id = failed_launch(env, fakebin)
    env.config.parent.mkdir(parents=True, exist_ok=True)
    env.config.write_text(f"wtctl_cmd: ['{tmp_path / 'nowhere' / 'wtctl'}']\n")
    done = env.run("resume", agent_id, "--json", "--wait")
    assert done.returncode == 0, (done.stdout, done.stderr)


def test_wtctl_missing_everywhere_exits_12_and_restores(
    env: Env, fakebin: Fakebin, tmp_path: Path
) -> None:
    agent_id = failed_launch(env, fakebin)
    with env.store() as store:
        from agent_spawn.lifecycle import lease_key

        with store.lease(lease_key(agent_id)) as lease:
            store.update_agent(lease, agent_id, {"wtctl_path": str(tmp_path / "gone")})
    env.config.parent.mkdir(parents=True, exist_ok=True)
    env.config.write_text(f"wtctl_cmd: ['{tmp_path / 'nowhere' / 'wtctl'}']\n")
    done = env.run("resume", agent_id, "--json")
    assert done.returncode == 12
    assert json.loads(done.stdout)["error"]["code"] == "tool_missing"
    assert env.agent(agent_id)["launcher"] is None


# --- adopted worktrees ----------------------------------------------------------------------------


def test_a_failed_adopt_launch_is_retried_by_resume(
    env: Env, fakebin: Fakebin, tmp_path: Path
) -> None:
    root = tmp_path / "tree"
    (root / "pkg").mkdir(parents=True)
    fakebin.script("wtctl", "adopt", times=1, **BUSY)
    done = env.run(*LAUNCH, "--worktree", str(root), "--cwd", "pkg", "--prompt", "P")
    assert done.returncode == 12
    agent_id = str(only_agent(env)["agent_id"])
    done = env.run("resume", agent_id, "--json", "--wait")
    assert done.returncode == 0, (done.stdout, done.stderr)
    assert len(wtctl_calls(fakebin, "adopt")) == 2
    assert f"--cwd={root.resolve() / 'pkg'}" in starts(fakebin)[-1]
    row = env.agent(agent_id)
    assert row["wt_owned"] is False and row["wt_record_id"] is not None
    again = env.run("resume", agent_id, "--json", "--wait")
    assert again.returncode == 0
    assert len(wtctl_calls(fakebin, "adopt")) == 2  # a session exists now: no step 5


# --- holds of a resume and a fork ---------------------------------------------------------------


def launched_in_worktree(env: Env) -> str:
    done = env.run(*LAUNCH, "--repo", REPO, "--worktree", "new")
    assert done.returncode == 0, (done.stdout, done.stderr)
    return str(result_json(done)["agent"]["agent_id"])


def test_a_plain_resume_of_a_worktree_agent_takes_no_new_hold(env: Env, fakebin: Fakebin) -> None:
    agent_id = launched_in_worktree(env)
    done = env.run("resume", agent_id, "--json", "--wait")
    assert done.returncode == 0, (done.stdout, done.stderr)
    assert wtctl_calls(fakebin, "adopt") == []
    assert len(result_json(done)["agent"]["holds"]) == 1


def test_a_released_hold_is_taken_again_by_the_next_run(env: Env, fakebin: Fakebin) -> None:
    agent_id = launched_in_worktree(env)
    record_id = env.agent(agent_id)["wt_record_id"]
    # What teardown does to a hold it released (and wtctl forgot): the next run re-adopts it.
    fakebin.run("wtctl", "forget", record_id, "--holder", agent_id, "--json")
    with env.store() as store:
        from agent_spawn.lifecycle import lease_key

        with store.lease(lease_key(agent_id)) as lease:
            store.release_hold(lease, agent_id, record_id)
        assert store.get_holds(agent_id) == []
    done = env.run("resume", agent_id, "--json", "--wait")
    assert done.returncode == 0, (done.stdout, done.stderr)
    assert len(wtctl_calls(fakebin, "adopt")) == 1
    holds = result_json(done)["agent"]["holds"]
    assert [h["record_id"] for h in holds] == [record_id]
    assert agent_id in records(fakebin)[record_id]["holders"]


def test_a_fork_in_place_holds_the_record_under_its_own_id(env: Env, fakebin: Fakebin) -> None:
    source = launched_in_worktree(env)
    done = env.run("resume", source, "--fork", "--json", "--wait")
    assert done.returncode == 0, (done.stdout, done.stderr)
    fork = result_json(done)["agent"]["agent_id"]
    assert fork != source
    adopt = wtctl_calls(fakebin, "adopt")
    assert len(adopt) == 1 and f"--holder={fork}" in adopt[0]
    record_id = env.agent(source)["wt_record_id"]
    assert sorted(records(fakebin)[record_id]["holders"]) == sorted([source, fork])
    assert [h["record_id"] for h in result_json(done)["agent"]["holds"]] == [record_id]
    assert result_json(done)["agent"]["worktree"]["owned"] is False


def test_a_fork_refused_before_its_session_releases_its_holds(env: Env, fakebin: Fakebin) -> None:
    source = launched_in_worktree(env)
    fakebin.script("claude-session", "start", argv_contains=["--dry-run"])
    fakebin.script("claude-session", "start", exit=20,
                   error={"code": "refused", "message": "no", "reason": "live"})  # fmt: skip
    done = env.run("resume", source, "--fork", "--json")
    assert done.returncode == 20, (done.stdout, done.stderr)
    assert [a["agent_id"] for a in env.agents()] == [source]  # no fork remains
    record_id = env.agent(source)["wt_record_id"]
    assert records(fakebin)[record_id]["holders"] == [source]  # the fork's hold was released
    forgets = wtctl_calls(fakebin, "forget")
    assert len(forgets) == 1 and forgets[0][1] == record_id
    assert forgets[0][2].startswith("--holder=ag-") and forgets[0][2] != f"--holder={source}"


def test_a_failed_hold_of_a_resume_restores_the_row(
    env: Env, fakebin: Fakebin, tmp_path: Path
) -> None:
    agent_id = launched_in_worktree(env)
    before = env.agent(agent_id)
    record = records(fakebin)[before["wt_record_id"]]
    fakebin.run("wtctl", "forget", record["id"], "--holder", agent_id, "--json")
    with env.store() as store:
        from agent_spawn.lifecycle import lease_key

        with store.lease(lease_key(agent_id)) as lease:
            store.release_hold(lease, agent_id, record["id"])
    fakebin.script("wtctl", "adopt", times=1, **BUSY)
    done = env.run("resume", agent_id, "--json")
    assert done.returncode == 12, (done.stdout, done.stderr)
    assert json.loads(done.stdout)["error"]["details"]["retryable"] is True
    after = env.agent(agent_id)
    assert after["status"] == "parked" and after["launcher"] is None and after["pre_resume"] is None
    assert after["run"] == before["run"]


def test_a_hold_wtctl_has_but_no_local_row_is_adopted_again_and_stored(
    env: Env, fakebin: Fakebin, tmp_path: Path
) -> None:
    done = env.run(*LAUNCH, "--cwd", str(tmp_path))
    agent_id = result_json(done)["agent"]["agent_id"]
    # A crash after `adopt` and before the hold was stored: wtctl lists the agent as a holder.
    adopted = fakebin.run("wtctl", "adopt", str(tmp_path), "--holder", agent_id, "--json")
    record_id = json.loads(adopted.stdout)["record"]["id"]
    with env.store() as store:
        assert store.get_holds(agent_id) == []
    again = env.run("resume", agent_id, "--json", "--wait")
    assert again.returncode == 0, (again.stdout, again.stderr)
    assert [h["record_id"] for h in result_json(again)["agent"]["holds"]] == [record_id]
    # `adopt` is idempotent: the hold is re-asserted rather than trusted
    assert len([c for c in wtctl_calls(fakebin, "adopt") if f"--holder={agent_id}" in c]) == 1


def test_holders_emptied_externally_are_adopted_again_on_resume(env: Env, fakebin: Fakebin) -> None:
    agent_id = launched_in_worktree(env)
    record_id = env.agent(agent_id)["wt_record_id"]
    fakebin.run("wtctl", "forget", record_id, "--holder", agent_id, "--json")
    assert agent_id not in records(fakebin)[record_id]["holders"]
    with env.store() as store:
        assert [h["record_id"] for h in store.get_holds(agent_id)] == [record_id]  # row remains
    done = env.run("resume", agent_id, "--json", "--wait")
    assert done.returncode == 0, (done.stdout, done.stderr)
    assert len(wtctl_calls(fakebin, "adopt")) == 1
    assert agent_id in records(fakebin)[record_id]["holders"]
    assert [h["record_id"] for h in result_json(done)["agent"]["holds"]] == [record_id]


def test_a_busy_forget_keeps_the_failed_fork_parked_with_its_holds(
    env: Env, fakebin: Fakebin
) -> None:
    source = launched_in_worktree(env)
    fakebin.script("claude-session", "start", argv_contains=["--dry-run"])
    fakebin.script("claude-session", "start", exit=20,
                   error={"code": "refused", "message": "no", "reason": "live"})  # fmt: skip
    fakebin.script("wtctl", "forget", times=1, **BUSY)
    done = env.run("resume", source, "--fork", "--json")
    assert done.returncode == 12, (done.stdout, done.stderr)
    error = json.loads(done.stdout)["error"]
    assert error["tool"] == "wtctl" and error["details"]["retryable"] is True
    (fork,) = [a for a in env.agents() if a["agent_id"] != source]
    assert error["agent_id"] == fork["agent_id"]
    assert (fork["status"], fork["park_reason"]) == ("parked", "launch_failed")
    with env.store() as store:
        assert [h["record_id"] for h in store.get_holds(fork["agent_id"])] == [
            env.agent(source)["wt_record_id"]
        ]
    record = records(fakebin)[env.agent(source)["wt_record_id"]]
    assert fork["agent_id"] in record["holders"]  # still held: the row is the handle
    # The failure that aborted the fork is not lost behind the release failure.
    original = error["details"]["original_failure"]
    assert (original["code"], original["reason"]) == ("refused", "live")
    assert original["message"] == "claude-session refused the request: live"


def test_adopt_retry_at_resume_passes_the_repo_given_at_launch_not_the_derived_one(
    env: Env, fakebin: Fakebin, tmp_path: Path
) -> None:
    root = tmp_path / "tree"
    root.mkdir()
    derived = "example.org/derived/repo"
    fakebin.script("claude-session", "start", times=1, argv_contains=["--dry-run"],
                   merge={"repo": derived})  # fmt: skip
    fakebin.script("wtctl", "adopt", times=1, **BUSY)
    done = env.run(*LAUNCH, "--worktree", str(root), "--prompt", "P")
    assert done.returncode == 12
    agent_id = str(only_agent(env)["agent_id"])
    assert env.agent(agent_id)["repo"] == derived  # the row has a repo the user never gave
    first = wtctl_calls(fakebin, "adopt")[0]
    assert not [a for a in first if a.startswith("--repo")]
    done = env.run("resume", agent_id, "--json", "--wait")
    assert done.returncode == 0, (done.stdout, done.stderr)
    retry = wtctl_calls(fakebin, "adopt")[1]
    assert not [a for a in retry if a.startswith("--repo")]


def test_the_literal_repo_given_with_a_worktree_path_is_passed_again_at_resume(
    env: Env, fakebin: Fakebin, tmp_path: Path
) -> None:
    root = tmp_path / "tree"
    root.mkdir()
    fakebin.script("wtctl", "adopt", times=1, **BUSY)
    done = env.run(*LAUNCH, "--worktree", str(root), "--repo", REPO, "--prompt", "P")
    assert done.returncode == 12
    agent_id = str(only_agent(env)["agent_id"])
    env.run("resume", agent_id, "--json", "--wait")
    assert f"--repo={REPO}" in wtctl_calls(fakebin, "adopt")[1]


def test_a_record_found_by_name_takes_wtctls_owned_flag(env: Env, fakebin: Fakebin) -> None:
    agent_id = failed_launch(env, fakebin)
    path = Path(str(fakebin.log)).parent / "adopted-tree"
    path.mkdir()
    done = fakebin.run("wtctl", "adopt", str(path), "--repo", REPO, "--name", agent_id,
                       "--holder", agent_id, "--json")  # fmt: skip
    assert done.returncode == 0, done.stderr
    left = json.loads(done.stdout)["record"]
    assert left["owned"] is False
    done = env.run("resume", agent_id, "--json", "--wait")
    assert done.returncode == 0, (done.stdout, done.stderr)
    row = env.agent(agent_id)
    assert row["wt_record_id"] == left["id"] and row["wt_owned"] is False


def test_a_plan_cwd_equal_to_the_session_work_dir_needs_no_records_lookup(
    fakebin: Fakebin, isolation_env: Isolation, tmp_path: Path
) -> None:
    first = Env(fakebin, isolation_env)
    done = first.run(*LAUNCH, "--cwd", "session-dir", "--reads-third-party")
    assert done.returncode == 0, (done.stdout, done.stderr)
    agent_id = str(result_json(done)["agent"]["agent_id"])
    session = first.agent(agent_id)["csession_id"]
    status = fakebin.run("claude-session", "status", session, "--json", env=first.env())
    work = os.path.realpath(os.path.join(json.loads(status.stdout)["csession_dir"], "work"))
    broken = Env(fakebin, isolation_env, f"wtctl_cmd: ['{tmp_path / 'no' / 'wtctl'}']\n")
    fakebin.script("claude-session", "start", times=1, argv_contains=["--dry-run"],
                   merge={"profile": {"reads_third_party": True}, "taints": True,
                          "cwd": work})  # fmt: skip
    done = broken.run("resume", agent_id, "--allow-tainted", "--actor", "human:me", "--json",
                      "--wait")  # fmt: skip
    assert done.returncode == 0, (done.stdout, done.stderr)
    assert wtctl_calls(fakebin) == []


def test_a_failed_override_create_is_audited_once_with_the_real_result(
    env: Env, fakebin: Fakebin
) -> None:
    agent_id = failed_launch(env, fakebin, error={"code": "escalated", "message": "ask"}, exit=13)
    fakebin.script("wtctl", "create", times=1, **BUSY)
    done = env.run("resume", agent_id, "--override-escalation", "--actor", "human:me", "--json")
    assert done.returncode == 12, (done.stdout, done.stderr)
    audit = [json.loads(x) for x in (env.state_dir / "audit.jsonl").read_text().splitlines()]
    assert [(r["action"], r["result"]) for r in audit] == [("resume_escalation_override", "error")]


def test_a_refused_override_create_is_audited_as_refused(env: Env, fakebin: Fakebin) -> None:
    agent_id = failed_launch(env, fakebin, error={"code": "escalated", "message": "ask"}, exit=13)
    fakebin.script("wtctl", "create", times=1,
                   error={"code": "refused", "message": "x", "reason": "path_exists"},
                   exit=20)  # fmt: skip
    done = env.run("resume", agent_id, "--override-escalation", "--actor", "human:me", "--json")
    assert done.returncode == 20, (done.stdout, done.stderr)
    audit = [json.loads(x) for x in (env.state_dir / "audit.jsonl").read_text().splitlines()]
    assert [(r["action"], r["result"]) for r in audit] == [
        ("resume_escalation_override", "refused")
    ]


def test_a_sessionless_agent_with_a_stored_worktree_path_adopts_and_shows_nothing(
    env: Env, fakebin: Fakebin, tmp_path: Path
) -> None:
    root = tmp_path / "tree"
    root.mkdir()
    fakebin.script("wtctl", "adopt", times=1, **BUSY)
    done = env.run(*LAUNCH, "--worktree", str(root), "--prompt", "P")
    assert done.returncode == 12
    agent_id = str(only_agent(env)["agent_id"])
    assert env.agent(agent_id)["wt_record_id"] is None  # nothing stored yet: no show, no id
    done = env.run("resume", agent_id, "--json", "--wait")
    assert done.returncode == 0, (done.stdout, done.stderr)
    assert wtctl_calls(fakebin, "show") == []
    assert len(wtctl_calls(fakebin, "adopt")) == 2


def test_a_stored_record_of_a_sessionless_agent_is_shown_and_adopt_is_skipped(
    env: Env, fakebin: Fakebin, tmp_path: Path
) -> None:
    root = tmp_path / "tree"
    root.mkdir()
    # the adopt succeeded and the launch then failed before a session: the record id is stored
    fakebin.script("claude-session", "start", argv_contains=["--dry-run"])
    fakebin.script("claude-session", "start", exit=12, times=1,
                   error={"code": "launch_failed", "message": "x"})  # fmt: skip
    done = env.run(*LAUNCH, "--worktree", str(root), "--prompt", "P")
    assert done.returncode != 0
    agent_id = str(only_agent(env)["agent_id"])
    row = env.agent(agent_id)
    assert row["wt_record_id"] is not None and row["csession_id"] is None
    adopts = len(wtctl_calls(fakebin, "adopt"))
    done = env.run("resume", agent_id, "--json", "--wait")
    assert done.returncode == 0, (done.stdout, done.stderr)
    assert [c[1] for c in wtctl_calls(fakebin, "show")] == [row["wt_record_id"]]
    assert len(wtctl_calls(fakebin, "adopt")) == adopts  # known record: adopt is skipped


def test_reacquiring_a_hold_replaces_the_id_of_a_re_adopted_record(
    env: Env, fakebin: Fakebin
) -> None:
    from agent_spawn.lifecycle import lease_key

    agent_id = launched_in_worktree(env)
    old = env.agent(agent_id)["wt_record_id"]
    path = env.agent(agent_id)["wt_path"]
    with env.store() as store:
        with store.lease(lease_key(agent_id)) as lease:
            store.release_hold(lease, agent_id, old)
            store.reacquire_hold(lease, agent_id, "wt-newrecrd", path, owned=False)
        (hold,) = store.get_holds(agent_id)
        assert (
            hold["record_id"] == "wt-newrecrd" and hold["owned"] is True
        )  # the row keeps its place
        assert store.get_agent(agent_id)["wt_record_id"] == "wt-newrecrd"  # type: ignore[index]
        with store.lease(lease_key(agent_id)) as lease:
            store.reacquire_hold(lease, agent_id, "wt-otherrec", "/elsewhere", owned=False)
        assert [h["record_id"] for h in store.get_holds(agent_id)] == ["wt-newrecrd", "wt-otherrec"]


def test_a_signal_during_step_5_restores_the_pre_resume_row(
    env: Env, fakebin: Fakebin, proc_tracker: ProcTracker
) -> None:
    agent_id = failed_launch(env, fakebin)
    before = env.agent(agent_id)
    fakebin.script("wtctl", "create", hang=True, on_signal={"SIGTERM": {"exit": 143}})
    resumer = proc_tracker.popen(
        env.argv("resume", agent_id, "--json", "--wait"),
        env=env.env(),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    env.wait_for(lambda: len(wtctl_calls(fakebin, "create")) == 2, "the in-flight create")
    create = [i for i in fakebin.invocations("wtctl") if i.argv[0] == "create"][-1]
    fakebin.wait_for_hang(create.pid)
    resumer.send_signal(signal.SIGTERM)
    resumer.communicate(timeout=30)
    assert resumer.returncode == 143
    after = env.agent(agent_id)
    for field in ("status", "park_reason", "needs_attention", "updated"):
        assert after[field] == before[field], field
    assert after["launcher"] is None and after["pre_resume"] is None
    assert starts(fakebin) == [] and after["wt_record_id"] is None


def test_a_fork_hold_wtctl_has_but_no_row_stores_is_found_by_the_cwd_lookup(
    fakebin: Fakebin, tmp_path: Path
) -> None:
    from agent_spawn import worktrees as wtlib
    from agent_spawn.resume import Resume
    from tests.session_backend_support import FakebinGateway

    tree = tmp_path / "tree"
    tree.mkdir()
    fork = "ag-fffffff2"
    done = fakebin.run("wtctl", "adopt", str(tree), "--holder", fork, "--json")
    record_id = json.loads(done.stdout)["record"]["id"]

    class NoRows:
        def get_holds(self, _agent_id: str) -> list[dict[str, Any]]:
            return []  # the crash came between `adopt` and the stored row

    resume = Resume.__new__(Resume)
    resume.store = NoRows()  # type: ignore[assignment]
    resume.holds_started = True
    resume.resume_cwd = str(tree)
    resume.stored_wtctl = None
    resume.log = lambda *_a, **_k: None  # type: ignore[method-assign]
    resume.wtctl_backend = lambda _stored=None: wtlib.WtctlBackend(  # type: ignore[method-assign]
        [str(fakebin.bin_dir / "wtctl")], gateway=FakebinGateway(fakebin)
    )
    assert resume.release_fork_holds(fork) is None
    assert fork not in records(fakebin)[record_id]["holders"]
