"""Failure, signal and security paths of ``resume`` (restore of the pre-resume row, forks that
fail, stored prompt copies, the third-party rule, races)."""

from __future__ import annotations

import json
import signal
import subprocess
from pathlib import Path

import pytest

from tests.conftest import Isolation
from tests.fakebin.harness import Fakebin
from tests.launch_support import (
    Env,
    result_json,
    script_hung_exec,
    script_live_run,
    track_agent,
)
from tests.procguard import ProcTracker
from tests.test_resume import LAUNCH, launched, pass_dry_run, starts


@pytest.fixture
def env(fakebin: Fakebin, isolation_env: Isolation) -> Env:
    return Env(fakebin, isolation_env)


def popen(env: Env, tracker: ProcTracker, *args: str) -> subprocess.Popen[bytes]:
    return tracker.popen(
        env.argv(*args),
        env=env.env(),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def real_starts(fakebin: Fakebin) -> list[list[str]]:
    return starts(fakebin)


def refuse_start(fakebin: Fakebin) -> None:
    pass_dry_run(fakebin)
    fakebin.script(
        "claude-session",
        "start",
        times=1,
        exit=20,
        error={"code": "refused", "message": "no", "reason": "live"},
    )


def test_interrupt_during_start_restores_a_plain_resume(
    env: Env, fakebin: Fakebin, proc_tracker: ProcTracker
) -> None:
    agent_id = launched(env)
    before = env.agent(agent_id)
    pass_dry_run(fakebin)
    fakebin.script("claude-session", "start", sleep=30)
    child = popen(env, proc_tracker, "resume", agent_id, "--json")
    env.wait_for(lambda: len(starts(fakebin)) == 2, "the resume's start")
    child.send_signal(signal.SIGTERM)
    child.communicate(timeout=30)
    assert child.returncode == 143
    after = env.agent(agent_id)
    assert (after["status"], after["updated"], after["run"]) == (
        "parked",
        before["updated"],
        before["run"],
    )
    assert after["launcher"] is None and after["pre_resume"] is None


def test_interrupt_during_start_deletes_a_fork(
    env: Env, fakebin: Fakebin, proc_tracker: ProcTracker
) -> None:
    source = launched(env)
    pass_dry_run(fakebin)
    fakebin.script("claude-session", "start", sleep=30)
    fakebin.script("claude-session", "list", stdout={"sessions": []})
    child = popen(env, proc_tracker, "resume", source, "--fork", "--cwd", "session-dir", "--json")
    env.wait_for(lambda: len(starts(fakebin)) == 2, "the fork's start")
    child.send_signal(signal.SIGINT)
    out, _ = child.communicate(timeout=30)
    assert child.returncode == 130
    assert [a["agent_id"] for a in env.agents()] == [source]
    assert json.loads(out)["error"]["agent_id"] is None


def test_sessionless_resume_reuses_and_then_drops_the_stored_copies(
    env: Env, fakebin: Fakebin
) -> None:
    refuse_start(fakebin)
    assert env.run(*LAUNCH, "--prompt", "first words").returncode == 20
    (row,) = env.agents()
    agent_id = row["agent_id"]
    assert row["csession_id"] is None
    stored = env.state_dir / "prompts" / agent_id / "prompt"
    assert stored.read_text() == "first words"
    # A failed resume keeps them.
    refuse_start(fakebin)
    failed = env.run("resume", agent_id, "--json")
    assert failed.returncode == 20
    assert stored.read_text() == "first words"
    assert env.agent(agent_id)["status"] == "parked"
    # A resume with a new prompt replaces the stored copy even when it fails.
    new = env.state_dir.parent / "new-prompt.txt"
    new.write_text("second words")
    refuse_start(fakebin)
    assert env.run("resume", agent_id, "--prompt-file", str(new), "--json").returncode == 20
    assert stored.read_text() == "second words"
    assert [p.name for p in stored.parent.iterdir()] == ["prompt"]
    # A good resume passes the stored copy (not the caller's path) and removes it.
    done = env.run("resume", agent_id, "--json", "--wait")
    assert done.returncode == 0, (done.stdout, done.stderr)
    argv = starts(fakebin)[-1]
    assert not any(a.startswith("--resume") for a in argv)  # a launch re-run
    assert "--skill=do-it" in argv and "--mode=headless" in argv
    assert any(a == f"--prompt-file={stored}" for a in argv)
    assert not (env.state_dir / "prompts" / agent_id).exists()
    assert env.agent(agent_id)["csession_id"] is not None


def test_sessionless_resume_with_a_missing_stored_copy_fails_closed(
    env: Env, fakebin: Fakebin
) -> None:
    refuse_start(fakebin)
    assert env.run(*LAUNCH, "--prompt", "words").returncode == 20
    (row,) = env.agents()
    (env.state_dir / "prompts" / row["agent_id"] / "prompt").unlink()
    before = env.agent(row["agent_id"])
    done = env.run("resume", row["agent_id"], "--json")
    assert done.returncode == 12
    assert env.agent(row["agent_id"])["updated"] == before["updated"]
    assert len(starts(fakebin)) == 1


def test_third_party_run_outside_its_session_dir_is_refused_and_restored(
    env: Env, fakebin: Fakebin, tmp_path: Path
) -> None:
    agent_id = launched(env)
    before = env.agent(agent_id)
    fakebin.script(
        "claude-session",
        "start",
        argv_contains=["--dry-run"],
        merge={"profile": {"reads_third_party": True}, "taints": True},
    )
    done = env.run("resume", agent_id, "--reads-third-party", "--json")
    assert done.returncode == 20, (done.stdout, done.stderr)
    assert json.loads(done.stdout)["error"]["reason"] == "third_party_cwd"
    after = env.agent(agent_id)
    assert (after["status"], after["updated"]) == ("parked", before["updated"])
    assert len(starts(fakebin)) == 1


def test_an_unusable_classification_starts_nothing_and_restores(env: Env, fakebin: Fakebin) -> None:
    agent_id = launched(env)
    fakebin.script("claude-session", "start", argv_contains=["--dry-run"], stdout={"dry_run": True})
    done = env.run("resume", agent_id, "--json")
    assert done.returncode != 0
    assert env.agent(agent_id)["status"] == "parked"
    assert len(starts(fakebin)) == 1


def test_a_second_resume_of_a_live_agent_exits_20(
    env: Env, fakebin: Fakebin, proc_tracker: ProcTracker
) -> None:
    agent_id = launched(env)
    script_live_run(fakebin)
    script_hung_exec(fakebin)
    first = env.run("resume", agent_id, "--json")
    assert first.returncode == 0, (first.stdout, first.stderr)
    assert result_json(first)["agent"]["status"] == "live"
    track_agent(env, proc_tracker, agent_id)
    second = env.run("resume", agent_id, "--json")
    assert second.returncode == 20
    assert json.loads(second.stdout)["error"]["reason"] == "live"
    assert env.agent(agent_id)["status"] == "live"
    # A fork of a live source is allowed.
    fork = env.run("resume", agent_id, "--fork", "--cwd", "session-dir", "--json", "--dry-run")
    assert fork.returncode == 0, (fork.stdout, fork.stderr)


def test_post_launch_marks_resume_and_fork(env: Env, tmp_path: Path) -> None:
    agent_id = launched(env, "--hook", f"post-launch=cat > {tmp_path}/launch.json")
    first = json.loads((tmp_path / "launch.json").read_text())
    assert (first["resumed"], first["fork"]) == (False, False)
    hook = f"post-launch=cat > {tmp_path}/resume.json"
    env.run("resume", agent_id, "--hook", hook, "--json", "--wait")
    again = json.loads((tmp_path / "resume.json").read_text())
    assert (again["resumed"], again["fork"]) == (True, False)
    hook = f"post-launch=cat > {tmp_path}/fork.json"
    env.run(
        "resume", agent_id, "--fork", "--cwd", "session-dir", "--hook", hook, "--json", "--wait"
    )
    forked = json.loads((tmp_path / "fork.json").read_text())
    assert (forked["resumed"], forked["fork"]) == (True, True)
    assert forked["agent"]["agent_id"] != agent_id


def test_job_label_comes_from_the_environment_but_not_for_a_fork(
    env: Env, fakebin: Fakebin
) -> None:
    agent_id = launched(env)
    env.run("resume", agent_id, "--json", "--wait", extra_env={"JOBSCTL_JOB_ID": "j-7"})
    assert "--label=job=j-7" in starts(fakebin)[-1]
    with env.store() as store:
        assert "job=j-7" in store.get_labels(agent_id).texts()
    env.run(
        "resume", agent_id, "--fork", "--cwd", "session-dir", "--json", "--wait",
        extra_env={"JOBSCTL_JOB_ID": "j-8"},
    )  # fmt: skip
    assert not any("job=" in a for a in starts(fakebin)[-1])


def test_a_failed_fork_deletes_its_prompt_copies_but_a_failed_resume_keeps_the_new_ones(
    env: Env, fakebin: Fakebin, tmp_path: Path
) -> None:
    source = launched(env)
    prompt = tmp_path / "p.txt"
    prompt.write_text("words")
    refuse_start(fakebin)
    assert env.run("resume", source, "--fork", "--cwd", "session-dir", "--prompt-file",
                   str(prompt), "--json").returncode == 20  # fmt: skip
    prompts = env.state_dir / "prompts"
    assert not [p for p in prompts.iterdir()] if prompts.exists() else True
    refuse_start(fakebin)
    assert env.run("resume", source, "--prompt-file", str(prompt), "--json").returncode == 20
    copies = list((prompts / source).glob("resume-*/prompt"))
    assert len(copies) == 1 and copies[0].read_text() == "words"


def test_override_escalation_reaches_claude_session_only_for_its_escalation(
    env: Env, fakebin: Fakebin
) -> None:
    agent_id = launched(env)
    with env.store() as store:
        from agent_spawn.lifecycle import lease_key

        with store.lease(lease_key(agent_id)) as lease:
            store.update_agent(
                lease,
                agent_id,
                {
                    "escalated_by": "claude-session",
                    "park_reason": "escalated",
                    "needs_attention": "escalated",
                    "last_outcome": {"outcome": "escalated", "phase": "launch"},
                },
            )
    done = env.run("resume", agent_id, "--override-escalation", "--actor", "human:me", "--json",
                   "--wait")  # fmt: skip
    assert done.returncode == 0, (done.stdout, done.stderr)
    assert "--override-escalation" in starts(fakebin)[-1]
    audit = [
        json.loads(x)["action"] for x in (env.state_dir / "audit.jsonl").read_text().splitlines()
    ]
    assert audit == ["resume_escalation_override"]


def test_a_live_agent_makes_no_copies_and_changes_nothing(
    env: Env, fakebin: Fakebin, proc_tracker: ProcTracker, tmp_path: Path
) -> None:
    agent_id = launched(env)
    script_live_run(fakebin)
    script_hung_exec(fakebin)
    assert env.run("resume", agent_id, "--json").returncode == 0
    track_agent(env, proc_tracker, agent_id)
    prompt = tmp_path / "p.txt"
    prompt.write_text("words")
    done = env.run("resume", agent_id, "--prompt-file", str(prompt), "--json")
    assert done.returncode == 20
    assert (
        not list((env.state_dir / "prompts").glob("*/resume-*"))
        if (env.state_dir / "prompts").exists()
        else True
    )
