"""A plain resume of a session no run ever recorded a cwd for (#119; interface.md, Resume order,
step 4 Cwd): claude-session resolves such a session's cwd as a new session would (the caller's cwd)
and checks its git directory against the stored one (``git_dir_changed``), so agent-spawn passes
the agent's recorded cwd with ``--cwd``. A session whose run recorded a cwd is unchanged."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import pytest

from agent_spawn.lifecycle import lease_key
from tests.conftest import Isolation
from tests.fakebin.harness import Fakebin
from tests.launch_support import Env, result_json, script_done_run
from tests.private_tmux import PrivateTmux
from tests.test_resume_hardening import LAUNCH as PLAIN_LAUNCH
from tests.test_resume_hardening import refuse_start
from tests.test_windows import LAUNCH, fail_real_open
from tests.test_worktrees_launch import REPO, fake_throwaway
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


def test_a_sessionless_relaunch_with_a_missing_cwd_is_refused_not_run_in_the_callers(
    fakebin: Fakebin, isolation_env: Isolation, tmp_path: Path
) -> None:
    env = Env(fakebin, isolation_env)
    repo, other = make_repo(tmp_path, "repo"), make_repo(tmp_path, "other")
    refuse_start(fakebin)
    first = env.run(*PLAIN_LAUNCH, "--cwd", str(repo), cwd=other)
    assert first.returncode == 20, (first.stdout, first.stderr)
    (agent,) = env.agents()
    assert agent["csession_id"] is None
    (repo / ".git").rmdir()
    repo.rmdir()
    refuse_start(fakebin)
    done = env.run("resume", agent["agent_id"], "--json", "--wait", cwd=other)
    starts = [
        a for a in fakebin.argvs("claude-session") if a[0] == "start" and "--dry-run" not in a
    ]
    error = json.loads(done.stdout)["error"]
    assert done.returncode == 12 and error["details"]["cause"] == "cwd_missing", done.stdout
    assert len(starts) == 1  # only the refused first launch reached claude-session


# --- round 1 of the review of #119 -----------------------------------------------------------

OVERRIDE = ("--override-escalation", *HUMAN, "--json", "--wait")


def set_cwd(env: Env, agent_id: str, cwd: str) -> None:
    with env.store() as store:
        with store.lease(lease_key(agent_id)) as lease:
            store.update_agent(lease, agent_id, {"cwd": cwd}, touch=False)


def real_starts(fakebin: Fakebin) -> list[list[str]]:
    return [a for a in fakebin.argvs("claude-session") if a[0] == "start" and "--dry-run" not in a]


def error_of(done: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    return dict(json.loads(done.stdout)["error"])


def sessions_path(fakebin: Fakebin) -> Path:
    return Path(str(fakebin.log) + ".sessions.json")


def worktree_resume_agent(env: Env, fakebin: Fakebin, root: Path) -> str:
    """A sessionless agent stored with `--worktree <root> --cwd pkg` (its adopt was busy)."""
    fakebin.script("wtctl", "adopt", times=1, error={"code": "busy", "message": "later"}, exit=12)
    done = env.run("launch", "--mode", "headless", "--skill", "do-it", "--json", "--wait",
                   "--worktree", str(root), "--cwd", "pkg", "--prompt", "P")  # fmt: skip
    assert done.returncode == 12, (done.stdout, done.stderr)
    (agent,) = env.agents()
    assert agent["csession_id"] is None
    return str(agent["agent_id"])


def test_a_sessionless_worktree_relaunch_with_a_cwd_missing_in_the_worktree_is_refused(
    fakebin: Fakebin, isolation_env: Isolation, tmp_path: Path
) -> None:
    env = Env(fakebin, isolation_env)
    root = tmp_path / "tree"
    (root / "pkg").mkdir(parents=True)
    agent_id = worktree_resume_agent(env, fakebin, root)
    (root / "pkg").rmdir()
    adopts = len([a for a in fakebin.argvs("wtctl") if a[0] == "adopt"])
    done = env.run("resume", agent_id, "--json", "--wait", cwd=tmp_path)
    error = error_of(done)
    assert done.returncode == 12 and error["code"] == "launch_failed", done.stdout
    assert error["details"]["cause"] == "cwd_missing" and error["agent_id"] == agent_id
    assert str(root / "pkg") in error["message"]
    assert real_starts(fakebin) == []
    # the dry run kept its fallback (it went on to step 5: the worktree was adopted again)
    assert len([a for a in fakebin.argvs("wtctl") if a[0] == "adopt"]) == adopts + 1


def test_a_sessionless_worktree_relaunch_with_the_cwd_present_runs_in_it(
    fakebin: Fakebin, isolation_env: Isolation, tmp_path: Path
) -> None:
    env = Env(fakebin, isolation_env)
    root = tmp_path / "tree"
    (root / "pkg").mkdir(parents=True)
    agent_id = worktree_resume_agent(env, fakebin, root)
    done = env.run("resume", agent_id, "--json", "--wait", cwd=tmp_path)
    assert done.returncode == 0, (done.stdout, done.stderr)
    assert f"--cwd={root.resolve() / 'pkg'}" in real_starts(fakebin)[-1]


def test_a_sessionless_new_worktree_relaunch_resolves_the_cwd_at_step_5(
    fakebin: Fakebin, isolation_env: Isolation, tmp_path: Path
) -> None:
    env = Env(fakebin, isolation_env)
    fakebin.script("wtctl", "create", times=1, error={"code": "busy", "message": "later"}, exit=12)
    done = env.run("launch", "--mode", "headless", "--skill", "do-it", "--json", "--wait",
                   "--repo", REPO, "--worktree", "new", "--worktree-name", "pkg-tree",
                   "--cwd", "pkg", "--prompt", "P")  # fmt: skip
    assert done.returncode == 12, (done.stdout, done.stderr)
    (agent,) = env.agents()
    root = Path(fake_throwaway(fakebin)["path"]).parent.parent
    tree_dir = root / "worktrees" / REPO.split("/", 1)[0] / "team" / "project" / "pkg-tree"
    (tree_dir / "pkg").mkdir(parents=True)  # the worktree comes with the directory at step 5
    done = env.run("resume", agent["agent_id"], "--json", "--wait", cwd=tmp_path)
    assert done.returncode == 0, (done.stdout, done.stderr)
    assert f"--cwd={tree_dir.resolve() / 'pkg'}" in real_starts(fakebin)[-1]


def test_a_sessionless_new_worktree_relaunch_with_the_cwd_absent_is_refused_at_step_5(
    fakebin: Fakebin, isolation_env: Isolation, tmp_path: Path
) -> None:
    env = Env(fakebin, isolation_env)
    fakebin.script("wtctl", "create", times=1, error={"code": "busy", "message": "later"}, exit=12)
    done = env.run("launch", "--mode", "headless", "--skill", "do-it", "--json", "--wait",
                   "--repo", REPO, "--worktree", "new", "--worktree-name", "pkg-tree",
                   "--cwd", "pkg", "--prompt", "P")  # fmt: skip
    assert done.returncode == 12, (done.stdout, done.stderr)
    (agent,) = env.agents()
    done = env.run("resume", agent["agent_id"], "--json", "--wait", cwd=tmp_path)
    assert done.returncode == 12 and error_of(done)["details"]["cause"] == "cwd_missing"
    assert real_starts(fakebin) == []


def test_an_empty_recorded_cwd_is_refused_not_replaced_by_the_callers(
    wenv: WindowEnv, fakebin: Fakebin, tmp_path: Path
) -> None:
    repo, other = make_repo(tmp_path, "repo"), make_repo(tmp_path, "other")
    agent_id = escalated_agent(wenv, fakebin, repo, other)
    set_cwd(wenv, agent_id, "")
    done = wenv.run("resume", agent_id, *OVERRIDE, cwd=other)
    assert done.returncode == 12, (done.stdout, done.stderr)
    assert error_of(done)["details"]["cause"] == "cwd_changed"
    assert resume_starts(fakebin) == []


def test_a_relative_recorded_cwd_is_refused(
    wenv: WindowEnv, fakebin: Fakebin, tmp_path: Path
) -> None:
    repo, other = make_repo(tmp_path, "repo"), make_repo(tmp_path, "other")
    agent_id = escalated_agent(wenv, fakebin, repo, other)
    set_cwd(wenv, agent_id, "repo")
    done = wenv.run("resume", agent_id, *OVERRIDE, cwd=tmp_path)
    error = error_of(done)
    assert done.returncode == 12 and error["details"]["cause"] == "cwd_changed", done.stdout
    assert "not an absolute path" in error["message"]
    assert resume_starts(fakebin) == []


def test_a_recorded_cwd_with_a_nul_byte_is_refused_with_its_own_message(
    wenv: WindowEnv, fakebin: Fakebin, tmp_path: Path
) -> None:
    repo, other = make_repo(tmp_path, "repo"), make_repo(tmp_path, "other")
    agent_id = escalated_agent(wenv, fakebin, repo, other)
    set_cwd(wenv, agent_id, f"{repo}\0x")
    done = wenv.run("resume", agent_id, *OVERRIDE, cwd=other)
    error = error_of(done)
    assert done.returncode == 12 and error["details"]["cause"] == "cwd_changed", done.stdout
    assert "NUL byte" in error["message"] and "absolute" not in error["message"]
    assert resume_starts(fakebin) == []


def test_a_recorded_cwd_that_is_a_regular_file_is_missing(
    wenv: WindowEnv, fakebin: Fakebin, tmp_path: Path
) -> None:
    repo, other = make_repo(tmp_path, "repo"), make_repo(tmp_path, "other")
    agent_id = escalated_agent(wenv, fakebin, repo, other)
    file = tmp_path / "a-file"
    file.write_text("x")
    set_cwd(wenv, agent_id, str(file))
    done = wenv.run("resume", agent_id, *OVERRIDE, cwd=other)
    assert done.returncode == 12 and error_of(done)["details"]["cause"] == "cwd_missing"
    assert resume_starts(fakebin) == []


@pytest.mark.skipif(os.geteuid() == 0, reason="root can search any directory")
def test_a_recorded_cwd_behind_an_unsearchable_parent_is_changed_not_missing(
    wenv: WindowEnv, fakebin: Fakebin, tmp_path: Path
) -> None:
    repo, other = make_repo(tmp_path, "repo"), make_repo(tmp_path, "other")
    agent_id = escalated_agent(wenv, fakebin, repo, other)
    parent = tmp_path / "locked"
    inner = parent / "inner"
    inner.mkdir(parents=True)
    set_cwd(wenv, agent_id, str(inner))
    parent.chmod(0)
    try:
        done = wenv.run("resume", agent_id, *OVERRIDE, cwd=other)
    finally:
        parent.chmod(0o700)
    error = error_of(done)
    assert done.returncode == 12 and error["details"]["cause"] == "cwd_changed", done.stdout
    assert "cannot be accessed" in error["message"]
    assert resume_starts(fakebin) == []


def test_a_recorded_cwd_with_a_trailing_slash_is_passed_normalised(
    wenv: WindowEnv, fakebin: Fakebin, tmp_path: Path
) -> None:
    repo, other = make_repo(tmp_path, "repo"), make_repo(tmp_path, "other")
    agent_id = escalated_agent(wenv, fakebin, repo, other)
    set_cwd(wenv, agent_id, f"{repo.resolve()}/")
    done = wenv.run("resume", agent_id, *OVERRIDE, cwd=other)
    assert done.returncode == 0, (done.stdout, done.stderr)
    assert f"--cwd={repo.resolve()}" in resume_starts(fakebin)[-1]


def test_a_dry_run_of_an_agent_with_a_missing_cwd_names_the_agent(
    wenv: WindowEnv, fakebin: Fakebin, tmp_path: Path
) -> None:
    repo, other = make_repo(tmp_path, "repo"), make_repo(tmp_path, "other")
    agent_id = escalated_agent(wenv, fakebin, repo, other)
    (repo / ".git").rmdir()
    repo.rmdir()
    done = wenv.run("resume", agent_id, "--dry-run", "--json", cwd=other)
    error = error_of(done)
    assert done.returncode == 12 and error["details"]["cause"] == "cwd_missing", done.stdout
    assert error["agent_id"] == agent_id


def test_the_hint_for_a_missing_cwd_of_a_sessionless_agent_does_not_offer_a_fork(
    fakebin: Fakebin, isolation_env: Isolation, tmp_path: Path
) -> None:
    env = Env(fakebin, isolation_env)
    repo, other = make_repo(tmp_path, "repo"), make_repo(tmp_path, "other")
    refuse_start(fakebin)
    assert env.run(*PLAIN_LAUNCH, "--cwd", str(repo), cwd=other).returncode == 20
    (agent,) = env.agents()
    (repo / ".git").rmdir()
    repo.rmdir()
    done = env.run("resume", agent["agent_id"], "--json", "--wait", cwd=other)
    hint = error_of(done)["details"]["hint"]
    assert "fork" not in hint and "--cwd" in hint


def test_the_hint_for_a_missing_cwd_of_a_session_offers_a_fork(
    wenv: WindowEnv, fakebin: Fakebin, tmp_path: Path
) -> None:
    repo, other = make_repo(tmp_path, "repo"), make_repo(tmp_path, "other")
    agent_id = escalated_agent(wenv, fakebin, repo, other)
    (repo / ".git").rmdir()
    repo.rmdir()
    done = wenv.run("resume", agent_id, *OVERRIDE, cwd=other)
    assert "fork" in error_of(done)["details"]["hint"]


def failing_status(fakebin: Fakebin, exit_code: int, code: str) -> None:
    fakebin.script("claude-session", "status", replace=True, exit=exit_code,
                   error={"code": code, "message": "unreadable"})  # fmt: skip


@pytest.mark.parametrize(("exit_code", "code"), [(10, "not_found"), (1, "failed")])
def test_an_unreadable_status_is_not_an_unrecorded_cwd(
    wenv: WindowEnv, fakebin: Fakebin, tmp_path: Path, exit_code: int, code: str
) -> None:
    repo, other = make_repo(tmp_path, "repo"), make_repo(tmp_path, "other")
    agent_id = escalated_agent(wenv, fakebin, repo, other)
    failing_status(fakebin, exit_code, code)
    wenv.run("resume", agent_id, *OVERRIDE, cwd=other)
    attempted = resume_starts(fakebin)
    assert attempted, "a start of the resume was expected"
    assert not any(a.startswith("--cwd") for argv in attempted for a in argv)


def test_a_session_with_many_unrecorded_runs_passes_no_cwd_and_bounds_the_status_calls(
    wenv: WindowEnv, fakebin: Fakebin, tmp_path: Path
) -> None:
    repo, other = make_repo(tmp_path, "repo"), make_repo(tmp_path, "other")
    agent_id = escalated_agent(wenv, fakebin, repo, other)
    path = sessions_path(fakebin)
    data = json.loads(path.read_text())
    for entry in data.values():
        entry["run"] = 60  # sixty runs, none of which recorded a cwd
    path.write_text(json.dumps(data))
    wenv.run("resume", agent_id, *OVERRIDE, cwd=other)  # claude-session then refuses: no --cwd
    (attempt,) = resume_starts(fakebin)
    assert not any(a.startswith("--cwd") for a in attempt)
    per_run = [
        a
        for a in fakebin.argvs("claude-session")
        if a[0] == "status" and any(x.startswith("--run") for x in a)
    ]
    assert 0 < len(per_run) <= 50


def test_an_unrecorded_session_dir_agent_passes_session_dir(
    wenv: WindowEnv, fakebin: Fakebin, tmp_path: Path
) -> None:
    repo, other = make_repo(tmp_path, "repo"), make_repo(tmp_path, "other")
    agent_id = escalated_agent(wenv, fakebin, repo, other)
    set_cwd(wenv, agent_id, "session-dir")
    wenv.run("resume", agent_id, *OVERRIDE, cwd=other)
    assert "--cwd=session-dir" in resume_starts(fakebin)[-1]
