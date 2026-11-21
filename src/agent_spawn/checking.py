"""``validate``: an offline, read-only check of the config and the environment.

Nothing here creates, writes or locks anything, opens ``state.db`` or touches the network. Without
``--probe`` no process is started: the lower tools are only looked up. With ``--probe`` each
resolved tool is run once as ``<argv> --version`` through the gateway, with a timeout. The config
was already loaded (and refused with ``config_invalid``) by the CLI before this runs.
"""

from __future__ import annotations

import os
import re
import shutil
import stat
import subprocess
import sys
from typing import Any

from . import gateway
from .commands.base import Invocation
from .launch import resolve_tool
from .output import CliError, Outcome, PayloadKind, safe_text
from .store import default_state_dir

__all__ = ["validate_command"]

PROBE_TIMEOUT = 10.0
MINIMUM = (0, 1, 0)
IMPLICIT_DONE_MINIMUM = (0, 43, 0)
_VERSION = re.compile(r"[0-9]{1,9}\.[0-9]{1,9}\.[0-9]{1,9}")
_SHOWN = 200  # config values in messages are cut to this many characters


def _show(text: str) -> str:
    text = safe_text(text)
    return text if len(text) <= _SHOWN else text[:_SHOWN] + "..."


def _warning(code: str, message: str, key: str | None = None) -> dict[str, Any]:
    entry: dict[str, Any] = {"code": code, "message": message}
    if key is not None:
        entry["key"] = key
    return entry


def _failed(message: str) -> CliError:
    return CliError.of("failed", message)


def _usable_dir(path: str) -> bool:
    return stat.S_ISDIR(os.stat(path).st_mode) and os.access(path, os.W_OK | os.X_OK)


def check_state_dir(path: str) -> str:
    """``ok`` or ``absent`` (it would be created); ``CliError`` ``failed`` otherwise. Never
    creates anything."""
    try:
        if _usable_dir(path):
            return "ok"
        raise _failed(f"state_dir: {_show(path)} is not a directory the user can write to")
    except FileNotFoundError:
        if os.path.lexists(path):
            raise _failed(f"state_dir: {_show(path)} is a dangling symbolic link") from None
    except OSError as exc:
        raise _failed(f"state_dir: cannot examine {_show(path)}: {exc.strerror}") from None
    parent = os.path.dirname(path)
    while True:
        try:
            if _usable_dir(parent):
                return "absent"
            raise _failed(
                f"state_dir: {_show(path)} does not exist and {_show(parent)} is not a "
                "directory the user can write to"
            )
        except FileNotFoundError:
            if os.path.lexists(parent):
                raise _failed(f"state_dir: {_show(parent)} is a dangling symbolic link") from None
            parent = os.path.dirname(parent)
        except OSError as exc:
            raise _failed(f"state_dir: cannot examine {_show(parent)}: {exc.strerror}") from None


def _tuple(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in version.split("."))


def _probe(name: str, argv: list[str]) -> tuple[str | None, str | None]:
    """``(version, why not)`` of ``<argv> --version``; exactly one is None."""
    try:
        done = gateway.get_gateway().run([*argv, "--version"], timeout=PROBE_TIMEOUT, cwd="/")
    except subprocess.TimeoutExpired:
        return None, f"did not answer --version within {PROBE_TIMEOUT:g} seconds"
    except (OSError, gateway.OutputTooLarge) as exc:
        return None, f"could not run --version ({type(exc).__name__})"
    if done.returncode != 0:
        return None, f"--version exited {done.returncode}"
    lines = done.stdout.splitlines()
    parts = lines[0].split(" ") if lines else []
    if len(parts) != 2 or parts[0] != name or not _VERSION.fullmatch(parts[1]):
        return None, "--version did not print `<tool> X.Y.Z`"
    return parts[1], None


def validate_command(inv: Invocation) -> Outcome:
    config = inv.config
    assert config is not None and inv.config_file is not None
    probe = bool(inv.args.probe)
    warnings: list[dict[str, Any]] = []

    state_path = str(default_state_dir())
    state_status = check_state_dir(state_path)

    commands: list[tuple[str, str, list[str] | None]] = [
        ("claude-session", "claude_session_cmd", config.claude_session_cmd),
        ("wtctl", "wtctl_cmd", config.wtctl_cmd),
        ("termctl", "termctl_cmd", config.termctl_cmd),
    ]
    tools: dict[str, dict[str, Any]] = {}
    resolved: dict[str, list[str]] = {}
    required_missing: list[str] = []
    for name, key, argv in commands:
        entry: dict[str, Any] = {
            "status": "disabled",
            "path": None,
            "version": None,
            "minimum": ".".join(map(str, MINIMUM)),
        }
        tools[name] = entry
        if argv is None:
            continue
        try:
            resolved[name] = resolve_tool(argv, name)
        except CliError:
            if name == "termctl":
                entry["status"] = "missing"
                warnings.append(
                    _warning(
                        "termctl_missing",
                        f"{_show(argv[0])} is not installed or not executable: only "
                        "windows need it",
                        key,
                    )
                )
            else:
                required_missing.append(name)
            continue
        entry["status"] = "ok"
        entry["path"] = resolved[name][0]
    if required_missing:
        raise CliError.of(
            "tool_missing",
            f"{', '.join(required_missing)} not installed or not executable",
            tool=required_missing[0],
            details={"hint": f"install {required_missing[0]} or fix its *_cmd config key"},
        )

    if config.supervisor.detach == "scope" and shutil.which("systemd-run") is None:
        warnings.append(
            _warning(
                "systemd_run_missing",
                "supervisor.detach is scope but systemd-run is not on PATH: headless "
                "launches will fail",
                "supervisor.detach",
            )
        )

    if probe:
        for name, argv in resolved.items():
            version, why = _probe(name, argv)
            key = {"claude-session": "claude_session_cmd"}.get(name, f"{name}_cmd")
            if version is None:
                warnings.append(_warning("tool_version_unknown", f"{name}: {why}", key))
                continue
            parsed = _tuple(version)
            if parsed == (0, 0, 0):
                warnings.append(
                    _warning("tool_version_unknown", f"{name}: version 0.0.0 is unknown", key)
                )
                continue
            tools[name]["version"] = version
            if parsed < MINIMUM:
                warnings.append(
                    _warning("tool_too_old", f"{name} {version} is older than 0.1.0", key)
                )
            elif name == "claude-session" and parsed < IMPLICIT_DONE_MINIMUM:
                warnings.append(
                    _warning(
                        "implicit_done_unavailable",
                        f"claude-session {version} is older than 0.43.0: --implicit-done and "
                        "--no-implicit-done are refused",
                        key,
                    )
                )

    payload: dict[str, Any] = {
        "probe": probe,
        "config": {"path": inv.config_file, "files": list(inv.config_files)},
        "checks": {
            "state_dir": {"path": state_path, "status": state_status},
            "tools": {name: tools[name] for name in ("claude-session", "termctl", "wtctl")},
        },
        "warnings": warnings,
    }
    if not inv.quiet:
        for item in warnings:
            where = item.get("key", item["code"])
            sys.stderr.write(safe_text(f"agent-spawn: warning: {where}: {item['message']}") + "\n")
    lines = [
        f"config: {_show(inv.config_file)} ({len(inv.config_files)} file(s) read)",
        f"state dir: {_show(state_path)} ({state_status})",
    ]
    for name in ("claude-session", "termctl", "wtctl"):
        entry = tools[name]
        shown = entry["path"] or entry["status"]
        version = f" {entry['version']}" if entry["version"] else ""
        lines.append(f"{name}: {_show(str(shown))}{version}")
    lines.append("ok" if not warnings else f"ok, {len(warnings)} warning(s)")
    return Outcome(0, payload, PayloadKind.COMMAND, "\n".join(lines))
