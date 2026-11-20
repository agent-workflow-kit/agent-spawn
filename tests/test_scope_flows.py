"""Scope hosting of the detached supervisor, end to end against the fakes (issue #40).

The fake ``systemd-run`` execs its command in place, as a real scope does, so the supervisor is
a real process whose PID is the one the launcher recorded before the acknowledgement.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

from tests.conftest import Isolation
from tests.fakebin.harness import FAKES, Fakebin
from tests.flow_support import Flow, schema
from tests.launch_support import agent_spawn_path

HEADLESS = ("launch", "--mode", "headless", "--skill", "do-it", "--json")
RUNTIME = {"XDG_RUNTIME_DIR": "/run/user/fake"}
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
    assert probe.argv == [
        "--user",
        "--scope",
        "--quiet",
        "--collect",
        f"--unit=agent-spawn-probe-{agent_id}",
        "--",
        true,
    ]
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
