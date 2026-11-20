"""agent-spawn's scope-hosted detached supervisor against the real ``systemd-run`` (issue #40).

No Claude Code: ``claude`` is a shell stub, as in ``test_agent_spawn_stub.py``. The one thing this
suite touches outside its private world is the user's own systemd user manager, and only through
transient scopes: ``systemd-run --user --scope`` starts the unit ``agent-spawn-<agent_id>-r<run>``
(and the probe unit ``agent-spawn-probe-<agent_id>``), and the test stops exactly that unit by its
exact name at the end, also after a failure or a timeout. Nothing else of the manager is read or
changed.

What this verifies: the scope starts, the recorded supervisor PID is the supervisor itself (exec
in place) and is a member of the scope's cgroup, liveness sees it, ``stop`` ends the run and the
supervisor, and the unit disappears. What it cannot verify: that the supervisor survives the end
of a real login session under ``KillUserProcesses=yes`` (the host's logind keeps the default
``no``, and ending the login session would end this one); docs/live-testing.md lists it under
"Not covered". The scenario skips, with the reason, when ``systemd-run --user`` is unavailable.
"""

# ruff: noqa: E501  (assertion messages are prose: assumption, then the document line)

from __future__ import annotations

import os
import secrets
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from .billed import pid_gone
from .harness import World, check, wait_for
from .test_agent_spawn_stub import agent_of, stub_setup, wait_status

SC = pytest.mark.live_scenario

STANDARD = "/usr/local/bin:/usr/bin:/bin"


def bus_address(world: World) -> str | None:
    """The user manager's bus: ``DBUS_SESSION_BUS_ADDRESS`` of the real environment, else the
    socket in the real runtime directory. The private world never sets ``XDG_RUNTIME_DIR``, so
    the scenario hands the address over explicitly (and only to the commands that need it)."""
    given = world.environ.get("DBUS_SESSION_BUS_ADDRESS")
    if given:
        return given
    runtime = world.environ.get("XDG_RUNTIME_DIR") or f"/run/user/{os.getuid()}"
    socket = Path(runtime) / "bus"
    return f"unix:path={socket}" if socket.exists() else None


def manager(bus: str, *args: str, timeout: float = 30) -> subprocess.CompletedProcess[str]:
    """A ``systemctl --user`` or ``systemd-run --user`` call against the user manager only."""
    return subprocess.run(
        list(args),
        env={
            "PATH": STANDARD,
            "HOME": "/nonexistent",
            "LC_ALL": "C",
            "DBUS_SESSION_BUS_ADDRESS": bus,
        },
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        timeout=timeout,
        check=False,
    )


def unit_listed(bus: str, unit: str) -> bool:
    """Whether the manager still lists exactly this unit (read-only; the name has no wildcard)."""
    done = manager(bus, "systemctl", "--user", "list-units", "--all", "--no-legend", unit)
    return unit in done.stdout


def stop_unit(bus: str, unit: str) -> None:
    """Stop exactly ``unit`` (never a pattern); a unit that is already gone is fine."""
    if unit_listed(bus, unit):
        manager(bus, "systemctl", "--user", "stop", unit)


def require_scope(world: World) -> str:
    """The bus address, or a skip that says why scope hosting cannot be tried here."""
    if (
        shutil.which("systemd-run", path=STANDARD) is None
        or shutil.which("systemctl", path=STANDARD) is None
    ):
        pytest.skip("as-scope-supervisor: systemd-run or systemctl is not installed")
    bus = bus_address(world)
    if bus is None:
        pytest.skip("as-scope-supervisor: no systemd user manager bus (no XDG_RUNTIME_DIR/bus)")
    unit = f"agent-spawn-livetest-{secrets.token_hex(4)}.scope"
    try:
        probe = manager(
            bus,
            "systemd-run",
            "--user",
            "--scope",
            "--quiet",
            "--collect",
            f"--unit={unit.removesuffix('.scope')}",
            "--",
            "/bin/true",
        )
    finally:
        stop_unit(bus, unit)
    if probe.returncode != 0:
        pytest.skip(
            "as-scope-supervisor: `systemd-run --user --scope` does not work here: "
            f"{probe.stderr.strip()[:200]}"
        )
    return bus


@SC("as-scope-supervisor")
def test_a_scope_hosted_supervisor_runs_in_its_own_scope_and_stop_ends_it(world: World) -> None:
    """Issue #40 and interface.md Supervision (Scope hosting): with ``supervisor.detach: scope``
    the detached supervisor is started through ``systemd-run --user --scope`` and still is the
    recorded process: its PID (written before the exec) is the live ``supervise`` process, it is
    a member of the unit's cgroup (not of this login session's), liveness reports it alive, a
    ``stop`` ends the run and the supervisor, and the unit vanishes. Not verified here: survival
    of a real logout under ``KillUserProcesses=yes`` (docs/live-testing.md, Not covered)."""
    bus = require_scope(world)
    version = manager(bus, "systemd-run", "--version").stdout.splitlines()
    world.observe("agent-spawn.scope.systemd", version[0] if version else "?")
    wt = stub_setup(
        world,
        "while :; do sleep 0.2; done",
        agent_spawn_config="supervisor:\n  detach: scope\n",
    )
    unit = ""
    try:
        launched = world.spawn_json(
            "launch",
            "--mode",
            "headless",
            "--prompt",
            "reply with the single word ok",
            "--model",
            "haiku",
            "--worktree",
            "none",
            "--cwd",
            str(wt.repo.clone),
            "--wait",
            "--wait-timeout",
            "6s",
            ok=(22,),
            extra={"DBUS_SESSION_BUS_ADDRESS": bus},
            timeout=120,
        )
        aid = agent_of(launched)["agent_id"]
        unit = f"agent-spawn-{aid}-r1.scope"
        shown = world.spawn_json("show", aid)
        world.observe("agent-spawn.scope.status", shown["agent"]["status"])
        supervisor = shown["supervisor"]
        pid = supervisor["pid"]
        world.track(pid, tree=True)
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
        check(
            b"supervise" in cmdline
            and not any(part.endswith(b"systemd-run") for part in cmdline[:1]),
            "the recorded PID is the supervisor itself: systemd-run exec'd it in place (design: the pid reported by a scope is the command)",
            "interface.md Supervision, Scope hosting (Identity)",
            cmdline[:4],
        )
        cgroup = Path(f"/proc/{pid}/cgroup").read_text().strip()
        world.observe("agent-spawn.scope.cgroup", cgroup)
        check(
            cgroup.endswith(f"/{unit}"),
            "the supervisor is in the unit's own cgroup, not in a login session's scope",
            "interface.md Supervision, Scope hosting (Why it survives logout)",
            cgroup,
        )
        check(
            "session-" not in cgroup,
            "the supervisor is not in a session-*.scope cgroup",
            "interface.md Supervision, Scope hosting",
            cgroup,
        )
        procs_file = Path("/sys/fs/cgroup") / cgroup.split("::", 1)[1].lstrip("/") / "cgroup.procs"
        check(
            str(pid) in procs_file.read_text().split(),
            "the recorded PID is a member of the scope's cgroup.procs",
            "interface.md Supervision, Scope hosting (Identity)",
            procs_file,
        )
        active = manager(bus, "systemctl", "--user", "show", unit, "-p", "ActiveState", "--value")
        check(
            active.stdout.strip() == "active" and unit_listed(bus, unit),
            "the manager lists the unit as active",
            "interface.md Supervision, Scope hosting (Cleanup)",
            (active.stdout, active.stderr),
        )
        again = world.spawn_json("show", aid)
        check(
            again["agent"].get("supervisor_lost") is not True
            and again["agent"]["status"] in ("launching", "live")
            and again["supervisor"]["pid"] == pid,
            "liveness sees the scope-hosted supervisor as alive: same identity, no supervisor_lost",
            "interface.md Liveness",
            again["agent"],
        )
        stopped = world.spawn_json("stop", aid, "--grace", "5s", ok=(0,), timeout=120)
        world.observe(
            "agent-spawn.scope.stop", {k: stopped.get(k) for k in ("action", "already_parked")}
        )
        parked = wait_status(world, aid, "parked", 90)["agent"]
        check(
            parked["park_reason"] == "killed",
            "stop ends the run and the agent parks killed",
            "interface.md stop",
            parked["park_reason"],
        )
        wait_for(lambda: pid_gone(pid), 30, "the supervisor to exit after stop")
        wait_for(
            lambda: not unit_listed(bus, unit),
            30,
            f"the manager to drop {unit} after its last process exited",
        )
    finally:
        if unit:
            stop_unit(bus, unit)
        stuck: list[Any] = [u for u in ([unit] if unit else []) if unit_listed(bus, u)]
        check(not stuck, "no unit of this scenario is left running", "docs/live-testing.md", stuck)
