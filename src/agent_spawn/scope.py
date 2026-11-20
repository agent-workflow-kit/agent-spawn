"""Scope hosting of the detached supervisor (config ``supervisor.detach: scope``).

The detached supervisor is started as usual (double fork, ``setsid``, the PID pipe and the
acknowledgement, ``procs.spawn_detached``); only the program the grandchild execs changes, from
``agent-spawn supervise ...`` to ``systemd-run --user --scope ... -- agent-spawn supervise ...``.
``systemd-run`` in scope mode asks the user manager for a transient scope, moves itself into it
and execs the command in place, so the recorded PID and start time stay the supervisor's and the
supervisor lives in its own cgroup, outside the login session's scope that logind kills at logout
with ``KillUserProcesses=yes`` (interface.md, Supervision, Scope hosting).

Nothing here runs ``systemctl`` or stops a unit: liveness, ``stop`` and teardown signal the
recorded identity only. Every call goes through the gateway.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from collections.abc import Sequence

from . import gateway
from .output import CliError, safe_text

#: A scope unit: the agent id and the run number, so it is unique per started supervisor.
UNIT = re.compile(r"agent-spawn-ag-[a-z2-7]{8}-r[0-9]{1,9}")
#: The unit of the preflight probe.
PROBE_UNIT = re.compile(r"agent-spawn-probe-ag-[a-z2-7]{8}")
AGENT_ID = re.compile(r"ag-[a-z2-7]{8}")
PROBE_TIMEOUT = 30.0
MESSAGE_LIMIT = 500
SYSTEMD_RUN = "systemd-run"
HINT = (
    "scope hosting needs a reachable systemd user manager (XDG_RUNTIME_DIR set and "
    "user@<uid>.service running); set supervisor.detach: setsid to host without it"
)


def unit_name(agent_id: str, run: int) -> str:
    """The scope unit of ``agent_id``'s run (without the ``.scope`` systemd adds); ``ValueError``
    for anything outside the grammar, because systemd would silently escape such a name."""
    if not isinstance(run, int) or isinstance(run, bool) or run < 0:
        raise ValueError(f"invalid run number {run!r}")
    name = f"agent-spawn-{agent_id}-r{run}"
    if UNIT.fullmatch(name) is None:
        raise ValueError(f"invalid scope unit name {name!r}")
    return name


def probe_unit_name(agent_id: str) -> str:
    name = f"agent-spawn-probe-{agent_id}"
    if PROBE_UNIT.fullmatch(name) is None:
        raise ValueError(f"invalid probe unit name {name!r}")
    return name


def _flags(unit: str) -> list[str]:
    return ["--user", "--scope", "--quiet", "--collect", f"--unit={unit}"]


def supervisor_argv(systemd_run: str, agent_id: str, run: int, argv: Sequence[str]) -> list[str]:
    """``argv`` (the supervise argv) wrapped for scope hosting."""
    unit = unit_name(agent_id, run)
    if not os.path.isabs(systemd_run):
        raise ValueError("systemd-run must be an absolute path")
    description = f"--description=agent-spawn supervisor {agent_id} run {run}"
    return [systemd_run, *_flags(unit), description, "--", *argv]


def _missing() -> CliError:
    return CliError(
        "tool_missing",
        "systemd-run is not installed or not executable, but supervisor.detach is scope",
        details={"hint": "install systemd, or set supervisor.detach: setsid"},
    )


def check_host(gw: gateway.Gateway) -> None:
    """Static checks of a scope launch: Linux (exit 2 ``config_invalid``)."""
    if not gw.platform().startswith("linux"):
        raise CliError(
            "config_invalid",
            "supervisor.detach: scope needs Linux with systemd; use setsid",
        )


def resolve_systemd_run() -> str:
    """The absolute path of ``systemd-run`` on ``PATH`` (exit 12 ``tool_missing``)."""
    found = shutil.which(SYSTEMD_RUN)
    if found is None:
        raise _missing()
    path = os.path.abspath(found)
    if not os.path.isfile(path) or not os.access(path, os.X_OK):
        raise _missing()
    return path


def check_static(gw: gateway.Gateway) -> str:
    """The checks a dry run makes: the host, then the tool. Returns systemd-run's path."""
    check_host(gw)
    return resolve_systemd_run()


def _unavailable(what: str) -> CliError:
    return CliError(
        "failed",
        f"supervisor.detach is scope but a transient scope cannot be started: {what}",
        details={"cause": "scope_unavailable", "retryable": False, "hint": HINT},
    )


def _true() -> str:
    return shutil.which("true") or "/usr/bin/true"


def preflight(gw: gateway.Gateway, agent_id: str, *, env: dict[str, str] | None = None) -> str:
    """Launch step 1, scope preflight: prove a transient scope can be started before anything is
    created, by running ``true`` in a probe scope. Returns systemd-run's absolute path."""
    systemd_run = check_static(gw)
    if AGENT_ID.fullmatch(agent_id) is None:
        raise ValueError(f"invalid agent id {agent_id!r}")
    argv = [systemd_run, *_flags(probe_unit_name(agent_id)), "--", _true()]
    try:
        done = gw.run(argv, timeout=PROBE_TIMEOUT, env=dict(os.environ) if env is None else env)
    except FileNotFoundError:
        raise _missing() from None
    except subprocess.TimeoutExpired:
        raise _unavailable(f"systemd-run did not answer within {PROBE_TIMEOUT:g} s") from None
    except (OSError, subprocess.SubprocessError, gateway.OutputTooLarge) as exc:
        raise _unavailable(safe_text(str(exc))[:MESSAGE_LIMIT]) from None
    if done.returncode != 0:
        text = (done.stderr or done.stdout or "").strip()
        detail = f"systemd-run exited {done.returncode}"
        if text:
            detail += f": {safe_text(text)[:MESSAGE_LIMIT]}"
        raise _unavailable(detail)
    return systemd_run
