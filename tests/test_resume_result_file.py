"""The result-file rule on ``resume`` when the classification agrees (issue #19).

Launch cannot know a new session's directory before ``start``, so it checks the path only when
the classes disagree. ``resume`` knows the directory up front (``claude-session status``), so a
third-party-reading resume checks the path in every case.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_spawn.lifecycle import lease_key
from tests import schema_registry
from tests.conftest import Isolation
from tests.fakebin.harness import Fakebin
from tests.launch_support import Env, result_json

LAUNCH = ("launch", "--mode", "headless", "--skill", "do-it", "--json", "--wait")


@pytest.fixture
def env(fakebin: Fakebin, isolation_env: Isolation) -> Env:
    return Env(fakebin, isolation_env)


def third_party_agent(env: Env, fakebin: Fakebin, csd: Path) -> str:
    """A finished third-party-reading agent in its session directory."""
    # dry runs plan the session directory (a third-party resume's default cwd)
    fakebin.script(
        "claude-session", "start", argv_contains=["--dry-run"], merge={"cwd": "session-dir"}
    )
    fakebin.script("claude-session", "start", merge={"csession_dir": str(csd)})
    fakebin.script("claude-session", "status", merge={"csession_dir": str(csd)})
    done = env.run(*LAUNCH, "--reads-third-party", "--cwd", "session-dir")
    assert done.returncode == 0, (done.stdout, done.stderr)
    return str(result_json(done)["agent"]["agent_id"])


def test_resume_of_a_third_party_run_refuses_a_result_file_in_its_session_dir(
    env: Env, fakebin: Fakebin, tmp_path: Path
) -> None:
    csd = tmp_path / "csd"
    csd.mkdir()
    agent_id = third_party_agent(env, fakebin, csd)
    target = csd / "r.json"
    done = env.run(
        "resume", agent_id, "--reads-third-party", "--json", "--wait", "--result-file", str(target)
    )
    assert done.returncode == 2, (done.stdout, done.stderr)
    schema_registry.validate(result_json(done), "agent-spawn/cli/error.json")
    assert not target.exists()  # not even the exit-2 envelope is written there
    assert env.agent(agent_id)["status"] == "parked" and env.agent(agent_id)["run"] == 1
    starts = [
        a for a in fakebin.argvs("claude-session") if a[0] == "start" and "--dry-run" not in a
    ]
    assert len(starts) == 1  # the refusal came before the second start


def test_resume_of_a_third_party_run_accepts_a_result_file_elsewhere(
    env: Env, fakebin: Fakebin, tmp_path: Path
) -> None:
    csd = tmp_path / "csd"
    csd.mkdir()
    agent_id = third_party_agent(env, fakebin, csd)
    target = tmp_path / "out" / "r.json"
    target.parent.mkdir()
    done = env.run(
        "resume", agent_id, "--reads-third-party", "--json", "--wait", "--result-file", str(target)
    )
    assert done.returncode == 0, (done.stdout, done.stderr)
    assert target.read_text().strip() == done.stdout.strip()


def test_resume_of_a_trusted_run_may_write_into_its_session_dir(
    env: Env, fakebin: Fakebin, tmp_path: Path
) -> None:
    csd = tmp_path / "csd"
    csd.mkdir()
    fakebin.script("claude-session", "status", merge={"csession_dir": str(csd)})
    agent_id = str(result_json(env.run(*LAUNCH))["agent"]["agent_id"])
    target = csd / "r.json"
    done = env.run("resume", agent_id, "--json", "--wait", "--result-file", str(target))
    assert done.returncode == 0, (done.stdout, done.stderr)
    assert target.exists()


def test_resume_dry_run_refuses_a_result_file_in_the_session_dir(
    env: Env, fakebin: Fakebin, tmp_path: Path
) -> None:
    csd = tmp_path / "csd"
    csd.mkdir()
    agent_id = third_party_agent(env, fakebin, csd)
    target = csd / "r.json"
    done = env.run(
        "resume",
        agent_id,
        "--reads-third-party",
        "--dry-run",
        "--json",
        "--result-file",
        str(target),
    )
    assert done.returncode == 2, (done.stdout, done.stderr)
    assert not target.exists()


def test_resume_dry_run_accepts_a_result_file_elsewhere(
    env: Env, fakebin: Fakebin, tmp_path: Path
) -> None:
    csd = tmp_path / "csd"
    csd.mkdir()
    agent_id = third_party_agent(env, fakebin, csd)
    target = tmp_path / "out" / "r.json"
    target.parent.mkdir()
    done = env.run(
        "resume",
        agent_id,
        "--reads-third-party",
        "--dry-run",
        "--json",
        "--result-file",
        str(target),
    )
    assert done.returncode == 0, (done.stdout, done.stderr)


def test_the_cwd_rule_comes_before_the_session_dir_result_file_check(
    env: Env, fakebin: Fakebin, tmp_path: Path
) -> None:
    csd = tmp_path / "csd"
    csd.mkdir()
    agent_id = third_party_agent(env, fakebin, csd)
    with env.store() as store:
        with store.lease(lease_key(agent_id)) as lease:
            store.update_agent(lease, agent_id, {"cwd": "session-dir"}, touch=False)
    fakebin.clear_script("claude-session", "start")
    fakebin.script(
        "claude-session",
        "start",
        argv_contains=["--dry-run"],
        merge={"profile": {"reads_third_party": True}, "taints": True, "cwd": "/srv/elsewhere"},
    )
    done = env.run("resume", agent_id, "--json", "--wait", "--result-file", str(csd / "r.json"))
    assert done.returncode == 20, (done.stdout, done.stderr)
    assert result_json(done)["error"]["reason"] == "third_party_cwd"


def test_a_relative_session_dir_from_status_fails_the_resume_check_closed(
    env: Env, fakebin: Fakebin, tmp_path: Path
) -> None:
    csd = tmp_path / "csd"
    csd.mkdir()
    agent_id = third_party_agent(env, fakebin, csd)
    fakebin.clear_script("claude-session", "status")
    fakebin.script("claude-session", "status", merge={"csession_dir": "csd"})
    target = tmp_path / "out" / "r.json"
    target.parent.mkdir()
    done = env.run(
        "resume", agent_id, "--reads-third-party", "--json", "--wait", "--result-file", str(target)
    )
    assert done.returncode != 0, (done.stdout, done.stderr)
    assert not target.exists()


def test_a_fork_refuses_a_result_file_in_the_source_sessions_dir(
    env: Env, fakebin: Fakebin, tmp_path: Path
) -> None:
    csd = tmp_path / "csd"
    csd.mkdir()
    agent_id = third_party_agent(env, fakebin, csd)
    target = csd / "r.json"
    done = env.run(
        "resume",
        agent_id,
        "--fork",
        "--cwd",
        "session-dir",
        "--reads-third-party",
        "--json",
        "--wait",
        "--result-file",
        str(target),
    )
    assert done.returncode == 2, (done.stdout, done.stderr)
    assert not target.exists()
