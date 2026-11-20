"""A plain resume of a session no run ever recorded a cwd for (#119; interface.md, Resume order,
step 4 Cwd): claude-session resolves such a session's cwd as a new session would (the caller's cwd)
and checks its git directory against the stored one (``git_dir_changed``), so agent-spawn passes
the agent's recorded cwd with ``--cwd``. A session whose run recorded a cwd is unchanged."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from tests.conftest import Isolation
from tests.fakebin.harness import Fakebin
from tests.launch_support import result_json, script_done_run
from tests.private_tmux import PrivateTmux
from tests.test_windows import LAUNCH, fail_real_open
from tests.window_support import WindowEnv

HUMAN = ("--actor", "human:tester")


@pytest.fixture
def wenv(fakebin: Fakebin, isolation_env: Isolation, private_tmux: PrivateTmux) -> WindowEnv:
    private_tmux.start()
    return WindowEnv(fakebin, isolation_env, private_tmux)


def make_repo(root: Path, name: str) -> Path:
    path = root / name
    (path / ".git").mkdir(parents=True)
    return path


def resume_starts(fakebin: Fakebin) -> list[list[str]]:
    return [
        a
        for a in fakebin.argvs("claude-session")
        if a[0] == "start" and "--dry-run" not in a and any(x.startswith("--resume") for x in a)
    ]


def escalated_agent(wenv: WindowEnv, fakebin: Fakebin, repo: Path, other: Path) -> str:
    script_done_run(fakebin)
    fail_real_open(fakebin, 13, "escalated")
    done = wenv.run(*LAUNCH, "--cwd", str(repo), cwd=other)
    assert done.returncode == 13, (done.stdout, done.stderr)
    (agent,) = wenv.agents()
    assert agent["escalated_by"] == "termctl"
    return str(agent["agent_id"])


def test_an_escalation_override_from_another_cwd_runs_in_the_agents_cwd(
    wenv: WindowEnv, fakebin: Fakebin, tmp_path: Path
) -> None:
    repo, other = make_repo(tmp_path, "repo"), make_repo(tmp_path, "other")
    agent_id = escalated_agent(wenv, fakebin, repo, other)
    done = wenv.run(
        "resume", agent_id, "--override-escalation", *HUMAN, "--json", "--wait", cwd=other
    )
    assert done.returncode == 0, (done.stdout, done.stderr)
    assert f"--cwd={repo.resolve()}" in resume_starts(fakebin)[-1]


def test_a_resume_after_a_failed_open_runs_in_the_agents_cwd(
    wenv: WindowEnv, fakebin: Fakebin, tmp_path: Path
) -> None:
    repo, other = make_repo(tmp_path, "repo"), make_repo(tmp_path, "other")
    script_done_run(fakebin)
    fail_real_open(fakebin, 12)
    assert wenv.run(*LAUNCH, "--cwd", str(repo), cwd=other).returncode == 12
    (agent,) = wenv.agents()
    done = wenv.run("resume", agent["agent_id"], "--json", "--wait", cwd=other)
    assert done.returncode == 0, (done.stdout, done.stderr)
    assert f"--cwd={repo.resolve()}" in resume_starts(fakebin)[-1]


def test_a_resume_of_a_session_with_a_recorded_cwd_passes_no_cwd(
    wenv: WindowEnv, fakebin: Fakebin, tmp_path: Path
) -> None:
    repo, other = make_repo(tmp_path, "repo"), make_repo(tmp_path, "other")
    script_done_run(fakebin)
    body = result_json(wenv.run(*LAUNCH, "--wait", "--cwd", str(repo), cwd=other))
    agent_id = body["agent"]["agent_id"]
    done = wenv.run("resume", agent_id, "--json", "--wait", cwd=other)
    assert done.returncode == 0, (done.stdout, done.stderr)
    assert not any(a.startswith("--cwd") for a in resume_starts(fakebin)[-1])


def test_a_missing_recorded_cwd_is_a_launch_failure_naming_it(
    wenv: WindowEnv, fakebin: Fakebin, tmp_path: Path
) -> None:
    repo, other = make_repo(tmp_path, "repo"), make_repo(tmp_path, "other")
    agent_id = escalated_agent(wenv, fakebin, repo, other)
    (repo / ".git").rmdir()
    repo.rmdir()
    before = wenv.agent(agent_id)
    done = wenv.run(
        "resume", agent_id, "--override-escalation", *HUMAN, "--json", "--wait", cwd=other
    )
    error = json.loads(done.stdout)["error"]
    assert done.returncode == 12 and error["code"] == "launch_failed", done.stdout
    assert error["details"]["cause"] == "cwd_missing"
    assert str(repo) in error["message"]
    assert resume_starts(fakebin) == []
    after = wenv.agent(agent_id)
    assert (after["status"], after["park_reason"]) == (before["status"], before["park_reason"])


def test_a_recorded_cwd_that_became_a_symlink_is_refused(
    wenv: WindowEnv, fakebin: Fakebin, tmp_path: Path
) -> None:
    repo, other = make_repo(tmp_path, "repo"), make_repo(tmp_path, "other")
    agent_id = escalated_agent(wenv, fakebin, repo, other)
    (repo / ".git").rmdir()
    repo.rmdir()
    os.symlink(other, repo)
    done = wenv.run(
        "resume", agent_id, "--override-escalation", *HUMAN, "--json", "--wait", cwd=other
    )
    error = json.loads(done.stdout)["error"]
    assert done.returncode == 12 and error["details"]["cause"] == "cwd_changed", done.stdout
    assert resume_starts(fakebin) == []
