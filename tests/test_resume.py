"""``resume`` and ``resume --fork`` (headless, no worktree) against the fake binaries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tests import schema_registry
from tests.conftest import Isolation
from tests.fakebin.harness import Fakebin
from tests.launch_support import Env, result_json

LAUNCH = ("launch", "--mode", "headless", "--skill", "do-it", "--json", "--wait")


@pytest.fixture
def env(fakebin: Fakebin, isolation_env: Isolation) -> Env:
    return Env(fakebin, isolation_env)


def launched(env: Env, *extra: str) -> str:
    done = env.run(*LAUNCH, *extra)
    assert done.returncode == 0, (done.stdout, done.stderr)
    return str(result_json(done)["agent"]["agent_id"])


def starts(fakebin: Fakebin) -> list[list[str]]:
    return [a for a in fakebin.argvs("claude-session") if a[0] == "start" and "--dry-run" not in a]


def pass_dry_run(fakebin: Fakebin) -> None:
    fakebin.script("claude-session", "start", argv_contains=["--dry-run"])


def events(env: Env, agent_id: str) -> list[dict[str, Any]]:
    with env.store() as store:
        return store.pending_events(agent_id)


def test_plain_resume_runs_a_second_run(env: Env, fakebin: Fakebin) -> None:
    agent_id = launched(env)
    done = env.run("resume", agent_id, "--json", "--wait")
    body = result_json(done)
    assert done.returncode == 0, (done.stdout, done.stderr)
    schema_registry.validate(body, "agent-spawn/cli/resume.json")
    assert body["agent"]["agent_id"] == agent_id
    assert body["agent"]["status"] == "parked"
    argv = starts(fakebin)[-1]
    assert argv[1].startswith("--resume=cs-")
    assert "--mode=headless" in argv
    assert not any(a.startswith("--kind") for a in argv)
    assert f"--label=agent={agent_id}" in argv
    assert env.agent(agent_id)["run"] == 2
    assert env.agent(agent_id)["pre_resume"] is None


def test_resume_by_select(env: Env) -> None:
    agent_id = launched(env, "--label", "ticket=T-1")
    done = env.run("resume", "--select", "ticket=T-1", "--json", "--wait")
    assert done.returncode == 0, (done.stdout, done.stderr)
    assert result_json(done)["agent"]["agent_id"] == agent_id
    missing = env.run("resume", "--select", "ticket=nope", "--json")
    assert missing.returncode == 10


def test_resume_labels_are_stored_and_passed(env: Env, fakebin: Fakebin) -> None:
    agent_id = launched(env, "--label", "a=1")
    env.run("resume", agent_id, "--label", "b=2", "--label", "a=1", "--json", "--wait")
    argv = starts(fakebin)[-1]
    assert "--label=b=2" in argv and "--label=a=1" not in argv
    with env.store() as store:
        assert store.get_labels(agent_id).texts() == ["a=1", "b=2"]


def test_resume_refuses_hook_pre_launch_and_cwd_without_fork(env: Env) -> None:
    agent_id = launched(env)
    assert env.run("resume", agent_id, "--hook", "pre-launch=true", "--json").returncode == 2
    assert env.run("resume", agent_id, "--cwd", "/tmp", "--json").returncode == 2
    assert (
        env.run("resume", agent_id, "--hook", "post-launch=true", "--json", "--wait").returncode
        == 0
    )


def test_resume_of_an_unknown_or_torn_down_agent_exits_10(env: Env) -> None:
    assert env.run("resume", "ag-aaaaaaaa", "--json").returncode == 10
    agent_id = launched(env)
    env.run("resume", agent_id, "--json", "--wait")
    with env.store() as store:
        from agent_spawn.lifecycle import lease_key

        with store.lease(lease_key(agent_id)) as lease:
            store.enter_teardown(lease, agent_id, teardown_proc=store.holder_identity())
            store.tombstone(lease, agent_id)
    done = env.run("resume", agent_id, "--json")
    assert done.returncode == 10
    assert json.loads(done.stdout)["error"]["details"]["status"] == "torn_down"


def test_fork_creates_a_new_agent(env: Env, fakebin: Fakebin) -> None:
    source = launched(env)
    before = env.agent(source)
    done = env.run("resume", source, "--fork", "--cwd", "session-dir", "--json", "--wait",
                   "--label", "x=1")  # fmt: skip
    body = result_json(done)
    assert done.returncode == 0, (done.stdout, done.stderr)
    schema_registry.validate(body, "agent-spawn/cli/resume.json")
    fork = body["agent"]["agent_id"]
    assert fork != source and body["agent"]["forked_from"] == source
    argv = starts(fakebin)[-1]
    assert "--fork" in argv and f"--label=agent={fork}" in argv and "--label=x=1" in argv
    assert not any(a.startswith("--label=agent=" + source) for a in argv)
    after = env.agent(source)
    assert after["run"] == before["run"] and after["status"] == "parked"
    assert env.agent(fork)["csession_id"] != before["csession_id"]
    assert env.agent(fork)["ephemeral"] is False


def test_fork_without_a_session_exits_12(env: Env, fakebin: Fakebin) -> None:
    pass_dry_run(fakebin)
    fakebin.script(
        "claude-session", "start", exit=20, error={"code": "refused", "message": "x", "reason": "x"}
    )
    done = env.run(*LAUNCH)
    assert done.returncode == 20
    agent_id = env.agents()[0]["agent_id"]
    assert env.agent(agent_id)["csession_id"] is None
    fork = env.run("resume", agent_id, "--fork", "--json")
    assert fork.returncode == 12
    assert json.loads(fork.stdout)["error"]["code"] == "no_run"
    assert len(env.agents()) == 1


def test_refused_start_restores_the_pre_resume_row(env: Env, fakebin: Fakebin) -> None:
    agent_id = launched(env)
    before = env.agent(agent_id)
    pass_dry_run(fakebin)
    fakebin.script("claude-session", "start", exit=20,
                   error={"code": "refused", "message": "no", "reason": "live"})  # fmt: skip
    done = env.run("resume", agent_id, "--json", "--name", "renamed")
    assert done.returncode == 20, (done.stdout, done.stderr)
    after = env.agent(agent_id)
    for key in ("status", "name", "park_reason", "needs_attention", "last_outcome", "updated",
                "run"):  # fmt: skip
        assert after[key] == before[key], key
    assert after["launcher"] is None and after["pre_resume"] is None
    assert after["expect_run_after"] is None
    assert events(env, agent_id) == [] or all(
        e["name"] != "post-park" for e in events(env, agent_id)
    )


def test_would_refuse_restores_and_exits_20(env: Env, fakebin: Fakebin) -> None:
    agent_id = launched(env)
    before = env.agent(agent_id)
    fakebin.script("claude-session", "start", argv_contains=["--dry-run"],
                   exit=20, merge={"would_refuse": "tainted"})  # fmt: skip
    done = env.run("resume", agent_id, "--json")
    assert done.returncode == 20
    assert env.agent(agent_id)["status"] == "parked"
    assert env.agent(agent_id)["updated"] == before["updated"]
    assert len(starts(fakebin)) == 1  # only the launch's start


def test_session_missing_parks_with_attention(env: Env, fakebin: Fakebin) -> None:
    agent_id = launched(env)
    fakebin.script("claude-session", "start", argv_contains=["--dry-run"],
                   exit=10, error={"code": "not_found", "message": "gone"})  # fmt: skip
    done = env.run("resume", agent_id, "--json")
    assert done.returncode == 10
    assert json.loads(done.stdout)["error"]["details"]["status"] == "session_missing"
    row = env.agent(agent_id)
    assert (row["status"], row["park_reason"], row["needs_attention"]) == (
        "parked", "session_missing", "session_missing")  # fmt: skip


def test_start_exit_12_allocated_run_parks_launch_failed(env: Env, fakebin: Fakebin) -> None:
    agent_id = launched(env)
    session = env.agent(agent_id)["csession_id"]
    pass_dry_run(fakebin)
    fakebin.script("claude-session", "start", exit=12,
                   error={"code": "launch_failed", "message": "x", "csession_id": session,
                          "run": 2, "end_reason": "launch_failed"})  # fmt: skip
    done = env.run("resume", agent_id, "--json")
    assert done.returncode == 12
    row = env.agent(agent_id)
    assert (row["status"], row["park_reason"], row["needs_attention"]) == (
        "parked", "launch_failed", "launch_failed")  # fmt: skip
    assert row["run"] == 2


def test_failed_fork_leaves_no_record_and_keeps_the_id_reserved(env: Env, fakebin: Fakebin) -> None:
    source = launched(env)
    pass_dry_run(fakebin)
    fakebin.script("claude-session", "start", exit=20,
                   error={"code": "refused", "message": "no", "reason": "live"})  # fmt: skip
    done = env.run("resume", source, "--fork", "--cwd", "session-dir", "--json")
    assert done.returncode == 20
    assert json.loads(done.stdout)["error"].get("agent_id") is None
    assert [a["agent_id"] for a in env.agents()] == [source]
    with env.store() as store:
        n = store._conn.execute("SELECT count(*) FROM allocated_ids").fetchone()[0]
    assert n == 2
    assert not (env.state_dir / "prompts").exists() or not list(
        (env.state_dir / "prompts").iterdir()
    )


def test_allow_tainted_is_human_only_and_audited(env: Env, fakebin: Fakebin) -> None:
    agent_id = launched(env)
    refused = env.run("resume", agent_id, "--allow-tainted", "--json", "--actor", "auto:bot")
    assert refused.returncode == 2
    assert len(starts(fakebin)) == 1
    done = env.run("resume", agent_id, "--allow-tainted", "--json", "--wait",
                   "--actor", "human:me")  # fmt: skip
    assert done.returncode == 0, (done.stdout, done.stderr)
    argv = starts(fakebin)[-1]
    assert "--allow-tainted" in argv
    audit = [json.loads(x) for x in (env.state_dir / "audit.jsonl").read_text().splitlines()]
    assert [r["action"] for r in audit] == ["resume_tainted"]
    assert audit[0]["target"] == agent_id and audit[0]["actor"] == "human:me"


def test_override_escalation_without_target_exits_2(env: Env) -> None:
    agent_id = launched(env)
    done = env.run("resume", agent_id, "--override-escalation", "--json", "--actor", "human:me")
    assert done.returncode == 2
    assert env.agent(agent_id)["status"] == "parked"


def test_prompt_travels_in_a_file_and_is_removed(
    env: Env, fakebin: Fakebin, tmp_path: Path
) -> None:
    agent_id = launched(env)
    prompt = tmp_path / "answer.txt"
    prompt.write_text("SECRET-ANSWER; rm -rf /")
    done = env.run("resume", agent_id, "--prompt-file", str(prompt), "--json", "--wait")
    assert done.returncode == 0, (done.stdout, done.stderr)
    argv = starts(fakebin)[-1]
    copy = [a for a in argv if a.startswith("--prompt-file=")]
    assert copy and str(prompt) not in " ".join(argv) and "SECRET" not in " ".join(argv)
    assert not (env.state_dir / "prompts" / agent_id).exists()


def test_dry_run_changes_nothing(env: Env, fakebin: Fakebin) -> None:
    agent_id = launched(env)
    before = env.agent(agent_id)
    done = env.run("resume", agent_id, "--dry-run", "--json")
    body = result_json(done)
    assert done.returncode == 0, (done.stdout, done.stderr)
    schema_registry.validate(body, "agent-spawn/cli/launch-dry-run.json")
    assert body["resume"] is True and body["would_refuse"] is None
    assert env.agent(agent_id) == before
    assert len(starts(fakebin)) == 1
