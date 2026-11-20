"""Scope hosting of the detached supervisor, end to end against the fakes (issue #40).

The fake ``systemd-run`` execs its command in place, as a real scope does, so the supervisor is
a real process whose PID is the one the launcher recorded before the acknowledgement.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from agent_spawn import cli, gateway, procs, scope
from agent_spawn.launch import Launch, SignalState
from agent_spawn.output import CliError
from agent_spawn.resume import Resume
from tests.conftest import Isolation
from tests.fake_gateway import FakeGateway
from tests.fakebin.harness import FAKES, Fakebin
from tests.flow_support import Flow, schema
from tests.launch_support import agent_spawn_path

HEADLESS = ("launch", "--mode", "headless", "--skill", "do-it", "--json")
RUNTIME = {"XDG_RUNTIME_DIR": "/run/user/fake"}
PROBE_UNIT = re.compile(r"--unit=(agent-spawn-probe-ag-[a-z2-7]{8}-[0-9]+-[0-9a-f]+)")
SUPERVISE_UNIT = re.compile(r"--unit=(agent-spawn-ag-[a-z2-7]{8}-r[0-9]+)")


def make_flow(fakebin: Fakebin, iso: Isolation, tmp_path: Path, detach: str | None) -> Flow:
    flow = Flow(fakebin, iso, tmp_path)
    if detach is not None:
        config = flow.env.config
        config.write_text(config.read_text() + f"supervisor:\n  detach: {detach}\n")
    return flow


@pytest.fixture
def scope_flow(fakebin: Fakebin, isolation_env: Isolation, tmp_path: Path) -> Flow:
    return make_flow(fakebin, isolation_env, tmp_path, "scope")


def scope_calls(fakebin: Fakebin) -> list[dict[str, Any]]:
    """The logged systemd-run calls (the probe first), as {argv, pid, env}."""
    return [
        {
            "argv": c.argv,
            "pid": c.pid,
            "env": c.env,
            "unit": SUPERVISE_UNIT.search(" ".join(c.argv)),
        }
        for c in fakebin.invocations("systemd-run")
    ]


def supervisor_calls(fakebin: Fakebin) -> list[dict[str, Any]]:
    return [c for c in scope_calls(fakebin) if "supervise" in c["argv"]]


# --- the launch -------------------------------------------------------------------------------


def test_a_scope_launch_probes_then_starts_the_supervisor_in_a_scope(
    scope_flow: Flow, fakebin: Fakebin
) -> None:
    done = scope_flow.cli(
        schema("launch"),
        *HEADLESS,
        "--wait",
        extra_env={**RUNTIME, "AGENT_SPAWN_HANDOFF_TIMEOUT": "7"},
    )
    assert done.code == 0, (done.stdout, done.stderr)
    agent_id = done.body["agent"]["agent_id"]
    assert done.body["outcome"] == "done"
    probe, started = fakebin.invocations("systemd-run")
    true = probe.argv[-1]
    unit = PROBE_UNIT.fullmatch(probe.argv[4])
    assert unit is not None and f"-{agent_id}-" in unit.group(1)
    assert probe.argv == ["--user", "--scope", "--quiet", "--collect", probe.argv[4], "--", true]
    assert true.endswith("true")
    assert started.argv == [
        "--user",
        "--scope",
        "--quiet",
        "--collect",
        f"--unit=agent-spawn-{agent_id}-r1",
        f"--description=agent-spawn supervisor {agent_id} run 1",
        "--",
        agent_spawn_path(),
        "supervise",
        agent_id,
        "--state-dir",
        str(scope_flow.state_dir),
        "--config",
        str(scope_flow.env.config),
    ]
    # the supervisor got the launcher's explicit environment through the scope
    assert started.env["AGENT_SPAWN_HANDOFF_TIMEOUT"] == "7"
    assert scope_flow.settled(agent_id)
    scope_flow.validate_everything(expect_hooks=["post-launch", "post-park"])


def test_the_recorded_supervisor_is_the_process_systemd_run_became(
    scope_flow: Flow, fakebin: Fakebin
) -> None:
    done = scope_flow.cli(schema("launch"), *HEADLESS, "--wait", extra_env=RUNTIME)
    assert done.code == 0, (done.stdout, done.stderr)
    row = scope_flow.agent(done.body["agent"]["agent_id"])
    [started] = supervisor_calls(fakebin)
    # the PID systemd-run ran under is the recorded supervisor: exec keeps PID and start time
    assert row["supervisor"].pid == started["pid"]
    assert row["supervisor"].boot_id and row["supervisor"].start_time


def test_the_default_hosting_never_runs_systemd_run(
    fakebin: Fakebin, isolation_env: Isolation, tmp_path: Path
) -> None:
    flow = make_flow(fakebin, isolation_env, tmp_path, None)
    done = flow.cli(schema("launch"), *HEADLESS, "--wait", extra_env=RUNTIME)
    assert done.code == 0, (done.stdout, done.stderr)
    fakebin.assert_not_called("systemd-run")


def test_setsid_written_out_is_the_default(
    fakebin: Fakebin, isolation_env: Isolation, tmp_path: Path
) -> None:
    flow = make_flow(fakebin, isolation_env, tmp_path, "setsid")
    done = flow.cli(schema("launch"), *HEADLESS, "--wait", extra_env=RUNTIME)
    assert done.code == 0, (done.stdout, done.stderr)
    fakebin.assert_not_called("systemd-run")


def test_a_resume_starts_its_own_scope_named_after_its_run(
    scope_flow: Flow, fakebin: Fakebin
) -> None:
    first = scope_flow.cli(schema("launch"), *HEADLESS, "--wait", extra_env=RUNTIME)
    agent_id = first.body["agent"]["agent_id"]
    resumed = scope_flow.cli(
        schema("resume"), "resume", agent_id, "--wait", "--json", extra_env=RUNTIME
    )
    assert resumed.code == 0, (resumed.stdout, resumed.stderr)
    units = [c["unit"].group(1) for c in supervisor_calls(fakebin) if c["unit"] is not None]
    assert units == [f"agent-spawn-{agent_id}-r1", f"agent-spawn-{agent_id}-r2"]
    # every resume proves the scope first, too: one probe each
    probes = [c for c in scope_calls(fakebin) if "supervise" not in c["argv"]]
    assert len(probes) == 2


REPO = "example.org/team/project"
BUSY = {"error": {"code": "busy", "message": "later"}, "exit": 12}
WORKTREE = ("--repo", REPO, "--worktree", "new")


def snapshot(flow: Flow, fakebin: Fakebin) -> dict[str, Any]:
    """Everything a resume that fails its scope preflight must leave alone."""
    records = Path(str(fakebin.log) + ".records.json")
    agents = flow.env.agents()
    with flow.env.store() as store:
        holds = {a["agent_id"]: store.get_holds(a["agent_id"]) for a in agents}
    return {
        "agents": agents,
        "holds": holds,
        "worktree_records": records.read_text() if records.exists() else "",
        "wtctl": fakebin.argvs("wtctl"),
        "starts": [a for a in fakebin.argvs("claude-session") if a[0] == "start"],
        "supervisors": len(supervisor_calls(fakebin)),
    }


def failed_scope_resume(flow: Flow, agent_id: str, *flags: str) -> dict[str, Any]:
    refused = flow.cli(schema("resume"), "resume", agent_id, *flags, "--json")  # no bus
    assert refused.code == 12, (refused.stdout, refused.stderr)
    error = error_of(refused)
    assert error["code"] == "failed" and error["details"]["cause"] == "scope_unavailable"
    assert error["agent_id"] is None
    return error


def test_a_resume_whose_scope_cannot_start_changes_nothing(
    scope_flow: Flow, fakebin: Fakebin
) -> None:
    first = scope_flow.cli(schema("launch"), *HEADLESS, "--wait", extra_env=RUNTIME)
    agent_id = first.body["agent"]["agent_id"]
    before = snapshot(scope_flow, fakebin)
    failed_scope_resume(scope_flow, agent_id)
    assert snapshot(scope_flow, fakebin) == before  # the pre-resume row, holds, calls: unchanged


@pytest.mark.parametrize("flags", [(), ("--fork",)], ids=["plain", "fork"])
def test_a_resume_of_a_stored_worktree_with_a_dead_bus_touches_no_worktree_and_no_record(
    scope_flow: Flow, fakebin: Fakebin, flags: tuple[str, ...]
) -> None:
    first = scope_flow.cli(schema("launch"), *HEADLESS, *WORKTREE, "--wait", extra_env=RUNTIME)
    assert first.code == 0, (first.stdout, first.stderr)
    agent_id = first.body["agent"]["agent_id"]
    before = snapshot(scope_flow, fakebin)
    assert before["holds"][agent_id]  # the launch holds its worktree
    failed_scope_resume(scope_flow, agent_id, *flags)
    assert snapshot(scope_flow, fakebin) == before  # no fork row, no hold, no wtctl call at all


def test_a_resume_that_would_create_its_worktree_fails_the_probe_first(
    scope_flow: Flow, fakebin: Fakebin
) -> None:
    # an agent without a session whose worktree was never created (wtctl was busy)
    fakebin.script("wtctl", "create", times=1, **BUSY)
    first = scope_flow.cli(schema("launch"), *HEADLESS, *WORKTREE, extra_env=RUNTIME)
    assert first.code not in (0, 22), (first.stdout, first.stderr)
    [row] = scope_flow.env.agents()
    before = snapshot(scope_flow, fakebin)
    failed_scope_resume(scope_flow, row["agent_id"])
    after = snapshot(scope_flow, fakebin)
    assert after == before
    assert not [c for c in after["wtctl"] if c[0] == "create"][1:]  # never the resume's `create`


def test_every_probe_unit_is_unique(scope_flow: Flow, fakebin: Fakebin) -> None:
    first = scope_flow.cli(schema("launch"), *HEADLESS, "--wait", extra_env=RUNTIME)
    agent_id = first.body["agent"]["agent_id"]
    scope_flow.cli(schema("resume"), "resume", agent_id, "--wait", "--json", extra_env=RUNTIME)
    scope_flow.cli(schema("resume"), "resume", agent_id, "--wait", "--json", extra_env=RUNTIME)
    names = [
        m.group(1)
        for c in fakebin.invocations("systemd-run")
        if (m := PROBE_UNIT.search(" ".join(c.argv)))
    ]
    assert len(names) == 3 and len(set(names)) == 3


# --- failure ----------------------------------------------------------------------------------


def error_of(done: Any) -> dict[str, Any]:
    assert isinstance(done.body, dict)
    return done.body["error"]


def test_an_unreachable_user_manager_fails_the_launch_before_anything_starts(
    scope_flow: Flow, fakebin: Fakebin
) -> None:
    done = scope_flow.cli(schema("launch"), *HEADLESS)  # no XDG_RUNTIME_DIR: the bus is unreachable
    assert done.code == 12, (done.stdout, done.stderr)
    error = error_of(done)
    assert error["code"] == "failed" and error["tool"] == "agent-spawn"
    assert error["agent_id"] is None
    assert error["details"]["cause"] == "scope_unavailable"
    assert error["details"]["retryable"] is False
    assert "Failed to connect to bus" in error["message"]
    assert "XDG_RUNTIME_DIR" in error["details"]["hint"]
    assert scope_flow.env.agents() == []
    assert len(fakebin.invocations("systemd-run")) == 1  # the probe only: no supervisor
    assert not [
        a for a in fakebin.argvs("claude-session") if a[0] == "start" and "--dry-run" not in a
    ]
    assert not (scope_flow.state_dir / "supervise").exists() or not any(
        (scope_flow.state_dir / "supervise").iterdir()
    )


def test_a_scripted_probe_failure_never_falls_back_to_setsid(
    scope_flow: Flow, fakebin: Fakebin
) -> None:
    fakebin.script("systemd-run", "scope", exit=1, stderr="Failed to start: boom\n")
    done = scope_flow.cli(schema("launch"), *HEADLESS, extra_env=RUNTIME)
    assert done.code == 12
    assert error_of(done)["details"]["cause"] == "scope_unavailable"
    assert scope_flow.env.agents() == []
    assert "supervise" not in [a for call in fakebin.argvs("systemd-run") for a in call]


def test_a_missing_systemd_run_is_tool_missing(
    scope_flow: Flow, fakebin: Fakebin, tmp_path: Path
) -> None:
    bin_dir = tmp_path / "bin-without-systemd-run"
    bin_dir.mkdir()
    for name in [*FAKES, "python3"]:
        if name != "systemd-run":
            (bin_dir / name).symlink_to(fakebin.bin_dir / name)
    done = scope_flow.cli(schema("launch"), *HEADLESS, extra_env={**RUNTIME, "PATH": str(bin_dir)})
    assert done.code == 12, (done.stdout, done.stderr)
    error = error_of(done)
    assert error["code"] == "tool_missing" and error["tool"] == "agent-spawn"
    assert "systemd-run" in error["message"]
    assert scope_flow.env.agents() == []
    fakebin.assert_not_called("systemd-run")


def test_a_dry_run_checks_the_tool_but_starts_no_unit(scope_flow: Flow, fakebin: Fakebin) -> None:
    done = scope_flow.env.run(*HEADLESS, "--dry-run")  # no bus: not probed
    assert done.returncode == 0, (done.stdout, done.stderr)
    assert json.loads(done.stdout)["dry_run"] is True
    fakebin.assert_not_called("systemd-run")
    assert scope_flow.env.agents() == []


def test_a_scope_that_fails_to_start_after_the_probe_leaves_a_dead_supervisor_and_its_log(
    scope_flow: Flow, fakebin: Fakebin
) -> None:
    # the probe passes; the supervisor's own scope is refused (for example the unit name is taken)
    fakebin.script("claude-session", "status", pending_unadopted=True)  # nothing adopts the run
    fakebin.script(
        "systemd-run",
        "scope",
        argv_contains=["supervise"],
        exit=1,
        stderr="Failed to start transient scope unit: Unit x.scope already exists.\n",
    )
    done = scope_flow.cli(
        schema("launch"), *HEADLESS, "--wait", "--wait-timeout", "3s", extra_env=RUNTIME
    )
    # nothing adopted the run: the wait ends on its own timeout, with the agent still launching
    assert done.code == 22, (done.stdout, done.stderr)
    agents = scope_flow.env.agents()
    assert len(agents) == 1 and agents[0]["status"] == "launching"
    log = scope_flow.state_dir / "supervise" / f"{agents[0]['agent_id']}.log"
    assert "already exists" in log.read_text()
    assert scope_flow.settled(agents[0]["agent_id"])  # the supervisor is dead, not running
    assert json.loads(done.stdout)["timed_out"] is True


# --- the placement block raising ---------------------------------------------------------------


def test_a_value_error_while_building_the_scope_argv_parks_launch_failed_and_logs_the_cause(
    scope_flow: Flow,
    fakebin: Fakebin,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    added = scope_flow.env.env(RUNTIME)
    saved = {k: os.environ.get(k) for k in added}
    os.environ.update(added)  # not monkeypatch: the isolation guard checks before it is undone

    def refuse(*args: object) -> list[str]:
        raise ValueError("boom: the scope argv could not be built")

    monkeypatch.setattr(scope, "supervisor_argv", refuse)
    try:
        code = cli.main(["--json", *HEADLESS[:-1], "--wait"])
    finally:
        for key, old in saved.items():
            if old is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old
    body = json.loads(capsys.readouterr().out)
    assert code != 0 and body["error"]["code"] == "launch_failed", body
    assert "supervise" not in [a for call in fakebin.argvs("systemd-run") for a in call]
    [row] = scope_flow.env.agents()
    assert (row["status"], row["park_reason"]) == ("parked", "launch_failed")
    assert row["supervisor"] is None  # never a setsid fallback: nothing was started
    log = (scope_flow.state_dir / "log.jsonl").read_text()
    assert "boom: the scope argv could not be built" in log


# --- no systemd-run, no Linux, dry runs --------------------------------------------------------


def path_without_systemd_run(fakebin: Fakebin, tmp_path: Path) -> dict[str, str]:
    bin_dir = tmp_path / "bin-without-systemd-run"
    bin_dir.mkdir(exist_ok=True)
    for name in [*FAKES, "python3"]:
        if name != "systemd-run" and not (bin_dir / name).exists():
            (bin_dir / name).symlink_to(fakebin.bin_dir / name)
    return {**RUNTIME, "PATH": str(bin_dir)}


def test_a_dry_run_launch_without_systemd_run_is_tool_missing(
    scope_flow: Flow, fakebin: Fakebin, tmp_path: Path
) -> None:
    done = scope_flow.cli(
        schema("launch-dry-run"),
        *HEADLESS,
        "--dry-run",
        extra_env=path_without_systemd_run(fakebin, tmp_path),
    )
    assert done.code == 12, (done.stdout, done.stderr)
    error = error_of(done)
    assert error["code"] == "tool_missing" and error["tool"] == "agent-spawn"
    assert scope_flow.env.agents() == []


def test_a_dry_run_resume_without_systemd_run_is_tool_missing(
    scope_flow: Flow, fakebin: Fakebin, tmp_path: Path
) -> None:
    first = scope_flow.cli(schema("launch"), *HEADLESS, "--wait", extra_env=RUNTIME)
    agent_id = first.body["agent"]["agent_id"]
    before = scope_flow.agent(agent_id)
    done = scope_flow.cli(
        schema("launch-dry-run"),
        "resume",
        agent_id,
        "--dry-run",
        "--json",
        extra_env=path_without_systemd_run(fakebin, tmp_path),
    )
    assert done.code == 12, (done.stdout, done.stderr)
    assert error_of(done)["code"] == "tool_missing"
    assert scope_flow.agent(agent_id) == before


def test_a_dry_run_launch_with_scope_checks_the_tool_but_starts_no_unit(
    scope_flow: Flow, fakebin: Fakebin
) -> None:
    done = scope_flow.cli(schema("launch-dry-run"), *HEADLESS, "--dry-run")  # no bus: not probed
    assert done.code == 0, (done.stdout, done.stderr)
    assert done.body["dry_run"] is True
    fakebin.assert_not_called("systemd-run")
    assert scope_flow.env.agents() == []


def test_a_dry_run_resume_with_scope_starts_no_unit(scope_flow: Flow, fakebin: Fakebin) -> None:
    first = scope_flow.cli(schema("launch"), *HEADLESS, "--wait", extra_env=RUNTIME)
    agent_id = first.body["agent"]["agent_id"]
    calls = len(fakebin.invocations("systemd-run"))
    done = scope_flow.cli(schema("launch-dry-run"), "resume", agent_id, "--dry-run", "--json")
    assert done.code == 0, (done.stdout, done.stderr)
    assert len(fakebin.invocations("systemd-run")) == calls  # the launch's two, no more


def test_the_default_hosting_never_runs_systemd_run_for_a_resume_or_a_dry_run(
    fakebin: Fakebin, isolation_env: Isolation, tmp_path: Path
) -> None:
    flow = make_flow(fakebin, isolation_env, tmp_path, None)
    first = flow.cli(schema("launch"), *HEADLESS, "--wait", extra_env=RUNTIME)
    agent_id = first.body["agent"]["agent_id"]
    resumed = flow.cli(schema("resume"), "resume", agent_id, "--wait", "--json", extra_env=RUNTIME)
    assert resumed.code == 0, (resumed.stdout, resumed.stderr)
    dry = flow.cli(schema("launch-dry-run"), *HEADLESS, "--dry-run", extra_env=RUNTIME)
    assert dry.code == 0, (dry.stdout, dry.stderr)
    dry_resume = flow.cli(
        schema("launch-dry-run"), "resume", agent_id, "--dry-run", "--json", extra_env=RUNTIME
    )
    assert dry_resume.code == 0, (dry_resume.stdout, dry_resume.stderr)
    fakebin.assert_not_called("systemd-run")


class Args(SimpleNamespace):
    """Parsed arguments where every flag not given is unset."""

    def __getattr__(self, name: str) -> None:
        return None


@pytest.mark.parametrize("dry_run", [False, True], ids=["run", "dry-run"])
@pytest.mark.parametrize("platform", ["darwin", "freebsd14"])
def test_a_non_linux_launch_is_config_invalid_in_the_usage_checks(
    platform: str, dry_run: bool, tmp_path: Path
) -> None:
    inv = SimpleNamespace(
        args=Args(
            mode="headless",
            placement=None,
            dry_run=dry_run,
            override_escalation=False,
            worktree=None,
            cwd=None,
            skill="s",
            prompt=None,
            prompt_file=None,
            hook=None,
            max_turns=None,
            wait_timeout=None,
        ),
        actor="human:test",
        warnings=[],
        decision_ids=[],
        state_dir=str(tmp_path / "state"),
        config=SimpleNamespace(
            supervisor=SimpleNamespace(detach="scope"), wtctl_cmd=["wtctl"], hooks={}
        ),
        result_file=None,
    )
    run = Launch(inv, SignalState())
    run.gateway = FakeGateway(platform_name=platform)
    with pytest.raises(CliError) as info:
        run.usage_checks()
    assert (info.value.code, info.value.exit_code) == ("config_invalid", 2)
    assert not (tmp_path / "state").exists()


@pytest.mark.parametrize("dry_run", [False, True], ids=["run", "dry-run"])
def test_a_non_linux_resume_is_config_invalid_before_any_change(
    dry_run: bool, tmp_path: Path
) -> None:
    inv = SimpleNamespace(
        args=Args(
            mode="headless",
            placement=None,
            dry_run=dry_run,
            fork=False,
            reset=None,
            allow_tainted=False,
            wait_timeout=None,
            repo=None,
        ),
        actor="human:test",
        warnings=[],
        decision_ids=[],
        state_dir=str(tmp_path / "state"),
        config=SimpleNamespace(supervisor=SimpleNamespace(detach="scope")),
        result_file=None,
    )
    run = Resume(inv, SignalState())
    run.gateway = FakeGateway(platform_name="darwin")
    row = {"agent_id": "ag-abcdefg2", "settings": {"settings": {}, "pass_throughs": {}}}
    with pytest.raises(CliError) as info:
        run.plan_run(None, row)  # type: ignore[arg-type]
    assert (info.value.code, info.value.exit_code) == ("config_invalid", 2)


# --- the guards that keep a test off the real systemd-run ---------------------------------------


def stub_systemd_run(directory: Path) -> Path:
    stub = directory / "systemd-run"
    stub.write_text(f"#!/bin/sh\ntouch {directory}/ran\n")
    stub.chmod(0o755)
    return stub


def test_the_in_process_guard_refuses_a_real_systemd_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = stub_systemd_run(tmp_path)
    with pytest.raises(AssertionError, match="real systemd-run"):
        gateway.SystemGateway().run([str(real), "--user", "--scope", "--", "true"])
    monkeypatch.setenv("PATH", f"{tmp_path}:/usr/bin:/bin")
    with pytest.raises(AssertionError, match="real systemd-run"):
        gateway.SystemGateway().run(["systemd-run", "--user", "--scope", "--", "true"])
    assert not (tmp_path / "ran").exists()


def test_the_in_process_guard_lets_the_fake_systemd_run_answer(fakebin: Fakebin) -> None:
    done = gateway.SystemGateway().run(
        [str(fakebin.bin_dir / "systemd-run"), "--user", "--scope", "--unit=x", "--", "true"],
        env=fakebin.env(RUNTIME),
    )
    assert done.returncode == 0


def test_spawn_detached_refuses_a_real_systemd_run_argv(tmp_path: Path) -> None:
    real = stub_systemd_run(tmp_path)
    with pytest.raises(AssertionError, match="real systemd-run"):
        procs.spawn_detached([str(real), "--user", "--scope", "--", "true"])
    assert not (tmp_path / "ran").exists()
