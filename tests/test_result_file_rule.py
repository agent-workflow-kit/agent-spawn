"""The third-party path rule of ``--result-file`` (interface.md, Result file): a third-party-reading
run may not have its result file inside its cwd, the throwaway whose root is that cwd, or its
session directory (``<csession_dir>``, ``outbox/`` included)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.conftest import Isolation
from tests.fakebin.harness import Fakebin
from tests.launch_support import Env
from tests.test_resume import launched
from tests.test_worktrees_launch import LAUNCH, fake_throwaway, real_starts


@pytest.fixture
def env(fakebin: Fakebin, isolation_env: Isolation) -> Env:
    return Env(fakebin, isolation_env)


def third_party_dry_run(fakebin: Fakebin, **plan: object) -> None:
    fakebin.script(
        "claude-session",
        "start",
        argv_contains=["--dry-run"],
        merge={"profile": {"reads_third_party": True}, "taints": True, **plan},
    )


def session_dir(fakebin: Fakebin, csession_id: str) -> str:
    done = fakebin.run("claude-session", "status", csession_id, "--json")
    return json.loads(done.stdout)["csession_dir"]  # type: ignore[no-any-return]


def test_a_launch_at_a_throwaway_root_refuses_a_result_file_inside_it(
    env: Env, fakebin: Fakebin
) -> None:
    throwaway = fake_throwaway(fakebin)
    inside = Path(throwaway["path"]) / "out" / "r.json"
    done = env.run(
        *LAUNCH, "--cwd", throwaway["path"], "--reads-third-party", "--result-file", str(inside)
    )
    assert done.returncode == 2, (done.stdout, done.stderr)
    assert env.agents() == [] and real_starts(fakebin) == []
    assert not inside.exists()  # a refused path is never written, not even the envelope


def test_a_launch_at_a_throwaway_root_accepts_a_result_file_outside_it(
    env: Env, fakebin: Fakebin, tmp_path: Path
) -> None:
    throwaway = fake_throwaway(fakebin)
    out = tmp_path / "r.json"
    done = env.run(
        *LAUNCH, "--cwd", throwaway["path"], "--reads-third-party", "--result-file", str(out)
    )
    assert done.returncode == 0, (done.stdout, done.stderr)
    assert json.loads(out.read_text())["agent"]["status"] == "parked"


def test_a_symlink_into_the_throwaway_does_not_hide_a_result_file_inside_it(
    env: Env, fakebin: Fakebin, tmp_path: Path
) -> None:
    throwaway = fake_throwaway(fakebin)
    link = tmp_path / "link"
    link.symlink_to(throwaway["path"])
    done = env.run(
        *LAUNCH,
        "--cwd",
        throwaway["path"],
        "--reads-third-party",
        "--result-file",
        str(link / "r.json"),
    )
    assert done.returncode == 2, (done.stdout, done.stderr)
    assert not (Path(throwaway["path"]) / "r.json").exists()


def test_a_resume_at_a_throwaway_root_refuses_a_result_file_inside_it(
    env: Env, fakebin: Fakebin
) -> None:
    throwaway = fake_throwaway(fakebin)
    agent_id = launched(env, "--cwd", throwaway["path"], "--reads-third-party")
    before = env.agent(agent_id)
    third_party_dry_run(fakebin)
    target = Path(throwaway["path"]) / "r.json"
    done = env.run("resume", agent_id, "--json", "--wait", "--result-file", str(target))
    assert done.returncode == 2, (done.stdout, done.stderr)
    assert not target.exists()
    after = env.agent(agent_id)
    assert (after["status"], after["updated"]) == ("parked", before["updated"])
    assert len(real_starts(fakebin)) == 1


@pytest.mark.parametrize("where", ["", "outbox"])
def test_a_resume_refuses_a_result_file_in_the_session_dir(
    env: Env, fakebin: Fakebin, where: str
) -> None:
    agent_id = launched(env, "--cwd", "session-dir", "--reads-third-party")
    before = env.agent(agent_id)
    directory = Path(session_dir(fakebin, before["csession_id"])) / where
    third_party_dry_run(
        fakebin, cwd=str(Path(session_dir(fakebin, before["csession_id"])) / "work")
    )
    done = env.run(
        "resume", agent_id, "--json", "--wait", "--result-file", str(directory / "r.json")
    )
    assert done.returncode == 2, (done.stdout, done.stderr)
    assert not (directory / "r.json").exists()
    assert env.agent(agent_id)["updated"] == before["updated"]
    assert len(real_starts(fakebin)) == 1


def test_a_resume_in_the_session_dir_accepts_a_result_file_elsewhere(
    env: Env, fakebin: Fakebin, tmp_path: Path
) -> None:
    agent_id = launched(env, "--cwd", "session-dir", "--reads-third-party")
    work = Path(session_dir(fakebin, env.agent(agent_id)["csession_id"])) / "work"
    third_party_dry_run(fakebin, cwd=str(work))
    out = tmp_path / "r.json"
    done = env.run("resume", agent_id, "--json", "--wait", "--result-file", str(out))
    assert done.returncode == 0, (done.stdout, done.stderr)
    assert out.exists()


def test_a_trusted_resume_may_write_its_result_file_anywhere(env: Env, fakebin: Fakebin) -> None:
    agent_id = launched(env)
    csession = env.agent(agent_id)["csession_id"]
    out = Path(session_dir(fakebin, csession)) / "r.json"
    done = env.run("resume", agent_id, "--json", "--wait", "--result-file", str(out))
    assert done.returncode == 0, (done.stdout, done.stderr)


def test_a_dry_run_applies_the_result_file_rule_too(env: Env, fakebin: Fakebin) -> None:
    throwaway = fake_throwaway(fakebin)
    inside = Path(throwaway["path"]) / "r.json"
    done = env.run(
        *LAUNCH,
        "--cwd",
        throwaway["path"],
        "--reads-third-party",
        "--dry-run",
        "--result-file",
        str(inside),
    )
    assert done.returncode == 2, (done.stdout, done.stderr)
    assert env.agents() == []
    assert not inside.exists()


def test_the_equals_form_of_a_refused_result_file_is_dropped_too(
    env: Env, fakebin: Fakebin
) -> None:
    throwaway = fake_throwaway(fakebin)
    inside = Path(throwaway["path"]) / "eq.json"
    done = env.run(
        *LAUNCH, "--cwd", throwaway["path"], "--reads-third-party", f"--result-file={inside}"
    )
    assert done.returncode == 2, (done.stdout, done.stderr)
    assert not inside.exists()


def test_a_resume_dry_run_drops_a_refused_result_file(env: Env, fakebin: Fakebin) -> None:
    throwaway = fake_throwaway(fakebin)
    agent_id = launched(env, "--cwd", throwaway["path"], "--reads-third-party")
    third_party_dry_run(fakebin)
    inside = Path(throwaway["path"]) / "dry.json"
    done = env.run("resume", agent_id, "--dry-run", "--json", "--result-file", str(inside))
    assert done.returncode == 2, (done.stdout, done.stderr)
    assert not inside.exists()


def test_an_ordinary_usage_error_still_writes_the_exit_2_envelope(env: Env, tmp_path: Path) -> None:
    out = tmp_path / "usage.json"
    done = env.run(
        *LAUNCH, "--cwd", "/nonexistent", "--worktree", "bogus", "--result-file", str(out)
    )
    assert done.returncode == 2, (done.stdout, done.stderr)
    assert json.loads(out.read_text())["error"]["code"] == "usage"


def test_a_resume_dry_run_refuses_a_result_file_in_the_session_dir(
    env: Env, fakebin: Fakebin
) -> None:
    agent_id = launched(env, "--cwd", "session-dir", "--reads-third-party")
    directory = Path(session_dir(fakebin, env.agent(agent_id)["csession_id"]))
    third_party_dry_run(fakebin, cwd=str(directory / "work"))
    done = env.run(
        "resume", agent_id, "--dry-run", "--json", "--result-file", str(directory / "r.json")
    )
    assert done.returncode == 2, (done.stdout, done.stderr)
    assert not (directory / "r.json").exists()


@pytest.mark.parametrize("where", ["", "outbox"])
def test_a_fork_refuses_a_result_file_in_the_sources_session_dir(
    env: Env, fakebin: Fakebin, where: str
) -> None:
    agent_id = launched(env, "--cwd", "session-dir", "--reads-third-party")
    directory = Path(session_dir(fakebin, env.agent(agent_id)["csession_id"]))
    third_party_dry_run(fakebin, cwd=str(directory / "work"))
    target = directory / where / "fork.json"
    done = env.run("resume", agent_id, "--fork", "--json", "--wait", "--result-file", str(target))
    assert done.returncode == 2, (done.stdout, done.stderr)
    assert not target.exists()
    assert [a["agent_id"] for a in env.agents()] == [agent_id]
    assert len(real_starts(fakebin)) == 1
