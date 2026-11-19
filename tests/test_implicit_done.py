"""``--implicit-done`` / ``--no-implicit-done`` pass through ``launch`` and ``resume`` to
claude-session's ``start`` (claude-session 0.43.0 or newer); absent means nothing is passed."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests import schema_registry
from tests.conftest import Isolation
from tests.fakebin.harness import Fakebin
from tests.launch_support import Env, result_json

LAUNCH = ("launch", "--mode", "headless", "--skill", "do-it", "--json", "--wait")
FLAGS = ("--implicit-done", "--no-implicit-done")


@pytest.fixture
def env(fakebin: Fakebin, isolation_env: Isolation) -> Env:
    return Env(fakebin, isolation_env)


def starts(fakebin: Fakebin) -> list[list[str]]:
    """The real ``start`` calls (the classification dry run is not one)."""
    return [a for a in fakebin.argvs("claude-session") if a[0] == "start" and "--dry-run" not in a]


def implicit_args(argv: list[str]) -> list[str]:
    return [a for a in argv if a in FLAGS]


def launched(env: Env, *extra: str) -> str:
    done = env.run(*LAUNCH, *extra)
    assert done.returncode == 0, (done.stdout, done.stderr)
    return str(result_json(done)["agent"]["agent_id"])


@pytest.mark.parametrize("flag", FLAGS)
def test_launch_passes_the_flag_to_start(env: Env, fakebin: Fakebin, flag: str) -> None:
    launched(env, flag)
    (argv,) = starts(fakebin)
    assert implicit_args(argv) == [flag]


def test_launch_without_the_flag_passes_nothing(env: Env, fakebin: Fakebin) -> None:
    launched(env)
    (argv,) = starts(fakebin)
    assert implicit_args(argv) == []


def test_the_classification_dry_run_sees_the_flag_too(env: Env, fakebin: Fakebin) -> None:
    launched(env, "--no-implicit-done")
    dry = [a for a in fakebin.argvs("claude-session") if a[0] == "start" and "--dry-run" in a]
    assert dry and all(implicit_args(a) == ["--no-implicit-done"] for a in dry)


@pytest.mark.parametrize("flag", FLAGS)
def test_pre_launch_payload_carries_it_in_pass_throughs(
    env: Env, tmp_path: Path, flag: str
) -> None:
    sink = tmp_path / "payload.json"
    launched(env, flag, "--hook", f"pre-launch=cat > {sink}")
    payload = json.loads(sink.read_text())
    schema_registry.validate(payload, "agent-spawn/hooks/pre-launch.json")
    assert payload["pass_throughs"] == {"implicit_done": flag == "--implicit-done"}


def test_pre_launch_payload_omits_it_when_absent(env: Env, tmp_path: Path) -> None:
    sink = tmp_path / "payload.json"
    launched(env, "--hook", f"pre-launch=cat > {sink}")
    assert "implicit_done" not in json.loads(sink.read_text())["pass_throughs"]


def test_the_pre_launch_schema_types_the_value(env: Env, tmp_path: Path) -> None:
    sink = tmp_path / "payload.json"
    launched(env, "--no-implicit-done", "--hook", f"pre-launch=cat > {sink}")
    payload = json.loads(sink.read_text())
    payload["pass_throughs"]["implicit_done"] = "yes"
    with pytest.raises(Exception, match="boolean"):
        schema_registry.validate(payload, "agent-spawn/hooks/pre-launch.json")


@pytest.mark.parametrize("flag", FLAGS)
def test_launch_dry_run_shows_it(env: Env, fakebin: Fakebin, flag: str) -> None:
    done = env.run("launch", "--mode", "headless", "--skill", "do-it", "--json", "--dry-run", flag)
    body = result_json(done)
    assert done.returncode == 0, (done.stdout, done.stderr)
    schema_registry.validate(body, "agent-spawn/cli/launch-dry-run.json")
    assert implicit_args(body["session_start_argv"]) == [flag]
    assert not starts(fakebin)


def test_launch_dry_run_without_it_shows_nothing(env: Env) -> None:
    done = env.run("launch", "--mode", "headless", "--skill", "do-it", "--json", "--dry-run")
    assert implicit_args(result_json(done)["session_start_argv"]) == []


@pytest.mark.parametrize("flag", FLAGS)
def test_resume_forwards_the_flag_when_given(env: Env, fakebin: Fakebin, flag: str) -> None:
    agent_id = launched(env)
    done = env.run("resume", agent_id, flag, "--json", "--wait")
    assert done.returncode == 0, (done.stdout, done.stderr)
    argv = starts(fakebin)[-1]
    assert argv[1].startswith("--resume=cs-")
    assert implicit_args(argv) == [flag]


def test_resume_passes_nothing_when_not_given(env: Env, fakebin: Fakebin) -> None:
    # The launch's value is claude-session's to carry: agent-spawn never re-passes it.
    agent_id = launched(env, "--no-implicit-done")
    done = env.run("resume", agent_id, "--json", "--wait")
    assert done.returncode == 0, (done.stdout, done.stderr)
    assert implicit_args(starts(fakebin)[-1]) == []


@pytest.mark.parametrize("flag", FLAGS)
def test_resume_dry_run_shows_it(env: Env, flag: str) -> None:
    agent_id = launched(env)
    done = env.run("resume", agent_id, flag, "--dry-run", "--json")
    assert done.returncode == 0, (done.stdout, done.stderr)
    assert implicit_args(result_json(done)["session_start_argv"]) == [flag]


def test_a_fork_forwards_the_flag_when_given(env: Env, fakebin: Fakebin) -> None:
    agent_id = launched(env)
    done = env.run(
        "resume",
        agent_id,
        "--fork",
        "--cwd",
        "session-dir",
        "--no-implicit-done",
        "--json",
        "--wait",
    )
    assert done.returncode == 0, (done.stdout, done.stderr)
    assert implicit_args(starts(fakebin)[-1]) == ["--no-implicit-done"]


def test_reset_implicit_done_is_passed_on_to_claude_session(env: Env, fakebin: Fakebin) -> None:
    agent_id = launched(env, "--no-implicit-done")
    done = env.run("resume", agent_id, "--reset", "implicit-done", "--json", "--wait")
    assert done.returncode == 0, (done.stdout, done.stderr)
    assert "--reset=implicit-done" in starts(fakebin)[-1]


def test_reset_together_with_the_flag_is_claude_sessions_usage_error(env: Env) -> None:
    agent_id = launched(env)
    done = env.run(
        "resume", agent_id, "--reset", "implicit-done", "--implicit-done", "--json", "--wait"
    )
    assert done.returncode == 2, (done.stdout, done.stderr)


def test_the_flag_is_stored_for_a_sessionless_resume(env: Env, fakebin: Fakebin) -> None:
    # A launch that never reached `start` stores its pass-throughs; the re-run lays the flags
    # given now over them, so the stored override still applies (as for every pass-through).
    agent_id = launched(env, "--no-implicit-done")
    with env.store() as store:
        stored = store.get_agent(agent_id)["settings"]["pass_throughs"]
    assert stored == {"implicit_done": False}


def test_the_automatic_report_resume_passes_no_override(env: Env, fakebin: Fakebin) -> None:
    # The supervisor's report resume re-passes nothing of the launch: claude-session carries
    # --no-implicit-done to the resumed run itself (interface.md, Supervision).
    fakebin.script("claude-session", "exec", times=1, outcome="exited-unreported")
    launched(env, "--no-implicit-done")
    first, second = starts(fakebin)
    assert implicit_args(first) == ["--no-implicit-done"]
    assert second[1].startswith("--resume=cs-") and implicit_args(second) == []
