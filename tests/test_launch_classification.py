"""The post-start classification check and the supervisor's startup check, end to end."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from agent_spawn import launch
from tests import schema_registry
from tests.conftest import Isolation
from tests.fakebin.harness import Fakebin
from tests.launch_support import Env, result_json, script_hung_exec, script_live_run, track_agent
from tests.procguard import ProcTracker

LAUNCH = ("launch", "--mode", "headless", "--skill", "do-it", "--json")


@pytest.fixture
def env(fakebin: Fakebin, isolation_env: Isolation) -> Env:
    return Env(fakebin, isolation_env)


def status_says(fakebin: Fakebin, **run: Any) -> None:
    """The next `status` (the post-start check) reports these run fields."""
    fakebin.script("claude-session", "status", times=1, merge={"run": run})


def test_trusted_run_that_turns_out_third_party_outside_session_dir_is_cancelled(
    env: Env, fakebin: Fakebin
) -> None:
    status_says(fakebin, reads_third_party=True, cwd="/tmp/not-the-session-dir")
    done = env.run(*LAUNCH)
    body = result_json(done)
    assert done.returncode == 12
    schema_registry.validate(body, "agent-spawn/cli/error.json")
    error = body["error"]
    assert error["code"] == "launch_failed" and error["tool"] == "agent-spawn"
    assert error["details"] == {"cause": "classification_mismatch"}
    (row,) = env.agents()
    assert row["status"] == "parked" and row["park_reason"] == "launch_failed"
    assert row["needs_attention"] == "launch_failed"
    stops = [a for a in fakebin.argvs("claude-session") if a[0] == "stop"]
    assert len(stops) == 1  # the pending run was cancelled
    assert not [a for a in fakebin.argvs("claude-session") if a[0] == "exec"]


def test_status_failure_after_start_cancels_the_run(env: Env, fakebin: Fakebin) -> None:
    fakebin.script(
        "claude-session", "status", times=1, error={"code": "failed", "message": "x"}, exit=12
    )
    done = env.run(*LAUNCH)
    assert done.returncode == 12
    assert result_json(done)["error"]["details"] == {"cause": "classification_unknown"}
    assert [a for a in fakebin.argvs("claude-session") if a[0] == "stop"]


def test_mismatch_in_the_session_dir_continues_with_a_warning(env: Env, fakebin: Fakebin) -> None:
    fakebin.script("claude-session", "start", argv_contains=["--dry-run"])
    fakebin.script("claude-session", "start", merge={"csession_dir": "/tmp/csd-fake"})
    fakebin.script(
        "claude-session",
        "status",
        times=1,
        merge={
            "csession_dir": "/tmp/csd-fake",
            "run": {"reads_third_party": True, "cwd": "/tmp/csd-fake/work"},
        },
    )
    done = env.run(*LAUNCH, "--wait")
    assert done.returncode == 0, (done.stdout, done.stderr)
    log = [json.loads(line) for line in (env.state_dir / "log.jsonl").read_text().splitlines()]
    assert [r for r in log if r["event"] == "classification_mismatch" and r["level"] == "warn"]
    assert not [a for a in fakebin.argvs("claude-session") if a[0] == "stop"]


def test_third_party_run_in_its_session_dir_runs(env: Env, fakebin: Fakebin) -> None:
    done = env.run(*LAUNCH, "--reads-third-party", "--cwd", "session-dir", "--wait")
    body = result_json(done)
    assert done.returncode == 0, (done.stdout, done.stderr)
    assert body["agent"]["cwd"] == "session-dir"


def test_supervise_refuses_a_record_that_does_not_name_it(
    env: Env, fakebin: Fakebin, proc_tracker: ProcTracker
) -> None:
    script_live_run(fakebin)
    script_hung_exec(fakebin)
    body = result_json(env.run(*LAUNCH))
    agent_id = body["agent"]["agent_id"]
    track_agent(env, proc_tracker, agent_id)
    supervise = (
        "supervise",
        agent_id,
        "--state-dir",
        str(env.state_dir),
        "--config",
        str(env.config),
    )
    done = env.run(*supervise)  # the record names the running supervisor, not this process
    assert done.returncode == 2, (done.stdout, done.stderr)
    assert "supervisor" in done.stderr + done.stdout, (done.stdout, done.stderr)
    assert env.agent(agent_id)["status"] == "live"  # nothing changed
    unknown = env.run("supervise", "ag-aaaaaaaa", *supervise[2:])
    assert unknown.returncode == 2
    log = (env.state_dir / "log.jsonl").read_text()
    assert "supervise refused" in log


def test_supervise_exits_2_for_a_parked_agent(env: Env, fakebin: Fakebin) -> None:
    body = result_json(env.run(*LAUNCH, "--wait"))
    agent_id = body["agent"]["agent_id"]
    done = env.run(
        "supervise", agent_id, "--state-dir", str(env.state_dir), "--config", str(env.config)
    )
    assert done.returncode == 2


@pytest.mark.parametrize(
    ("path", "directory", "inside"),
    [
        ("/a/b", "/a/b", True),
        ("/a/b/c", "/a/b", True),
        ("/a/bc", "/a/b", False),
        ("/a", "/a/b", False),
        ("/x", "/", True),
    ],
)
def test_inside_is_component_wise(path: str, directory: str, inside: bool) -> None:
    assert launch._inside(path, directory) is inside  # pyright: ignore[reportPrivateUsage]


def test_result_file_inside_the_third_party_cwd_is_a_usage_error(
    env: Env, fakebin: Fakebin, tmp_path: Path
) -> None:
    work = tmp_path / "work"
    work.mkdir()
    done = env.run(
        *LAUNCH, "--reads-third-party", "--cwd", str(work), "--result-file", str(work / "r.json")
    )
    # A third-party run outside its session dir is refused first (exit 20): the cwd rule runs
    # before the result-file check and neither creates anything.
    assert done.returncode in (2, 20)
    assert env.agents() == []


def test_a_refused_result_file_is_forgotten_so_no_error_envelope_is_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from types import SimpleNamespace

    from agent_spawn import output, third_party
    from agent_spawn.output import CliError

    work = tmp_path / "work"
    work.mkdir()
    inv = SimpleNamespace(
        result_file=output.resolve_result_file(str(work / "r.json")),
        args=SimpleNamespace(cwd=str(work)),
        warnings=[],
        decision_ids=[],
        state_dir=str(tmp_path / "state"),
        config=None,
    )
    run = launch.Launch(inv, launch.SignalState())
    monkeypatch.setattr(third_party, "require_cwd_rule", lambda *_a, **_k: None)
    plan = third_party.Classification(
        reads_third_party=True,
        run_kind="implement",
        repo=None,
        cwd=str(work),
        would_refuse=None,
        would_fail=None,
    )
    with pytest.raises(CliError) as excinfo:
        run.pre_start_checks(plan)
    assert excinfo.value.exit_code == 2
    assert inv.result_file is None  # cli.main then writes nothing, not even the error


def test_a_post_start_mismatch_with_the_result_file_in_the_session_dir_writes_nothing(
    env: Env, fakebin: Fakebin, tmp_path: Path
) -> None:
    csd = tmp_path / "csd"
    csd.mkdir()
    status_says(fakebin, reads_third_party=True, cwd="/tmp/not-the-session-dir")
    fakebin.script("claude-session", "start", merge={"csession_dir": str(csd)})
    fakebin.script("claude-session", "status", merge={"csession_dir": str(csd)})
    target = csd / "r.json"
    done = env.run(*LAUNCH, "--result-file", str(target))
    assert done.returncode == 12, (done.stdout, done.stderr)
    assert result_json(done)["error"]["details"] == {"cause": "classification_mismatch"}
    assert not target.exists()  # nothing is written into the run's directory, not even the error
