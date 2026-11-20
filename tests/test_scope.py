"""Scope hosting of the detached supervisor: unit names, argv, preflight (unit level)."""

from __future__ import annotations

import subprocess
from collections.abc import Sequence
from pathlib import Path

import pytest

from agent_spawn import gateway, scope
from agent_spawn.output import CliError
from tests.fake_gateway import FakeGateway

AGENT = "ag-abcdefg2"
SD = "/usr/bin/systemd-run"


def done(code: int = 0, out: str = "", err: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess([], code, out, err)


def fake_with(result: object, platform: str = "linux") -> FakeGateway:
    gw = FakeGateway(platform_name=platform)

    def run(argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
        if isinstance(result, BaseException):
            raise result
        assert isinstance(result, subprocess.CompletedProcess)
        return result

    gw.run_results[SD] = run
    return gw


@pytest.fixture
def with_systemd_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    stub = tmp_path / "bin" / "systemd-run"
    stub.parent.mkdir()
    stub.write_text("#!/bin/sh\nexit 0\n")
    stub.chmod(0o755)
    monkeypatch.setenv("PATH", f"{stub.parent}:/usr/bin:/bin")
    return str(stub)


# --- unit names -------------------------------------------------------------------------------


def test_unit_name_is_derived_from_the_agent_and_the_run() -> None:
    assert scope.unit_name(AGENT, 3) == "agent-spawn-ag-abcdefg2-r3"


@pytest.mark.parametrize(
    "agent_id",
    [
        "",
        "ag-abcdefg",  # one character short
        "ag-abcdefg22",  # one character long
        "ag-ABCDEFG2",  # upper case
        "ag-abcdef18",  # 1 and 8 are not base32
        "ag-abcdefg2\n",
        "ag-abcdefg2 --unit=x",
        "ag-abcdefg2/..",
        "xx-abcdefg2",
        "ag-abcdefg2.service",
        "ag-abcdefgé",
    ],
)
def test_unit_name_refuses_an_id_outside_the_grammar(agent_id: str) -> None:
    with pytest.raises(ValueError):
        scope.unit_name(agent_id, 1)


@pytest.mark.parametrize("run", [-1, 10**9, True, "1"])
def test_unit_name_refuses_a_bad_run_number(run: object) -> None:
    with pytest.raises(ValueError):
        scope.unit_name(AGENT, run)  # type: ignore[arg-type]


def test_probe_unit_name_has_its_own_prefix_and_is_validated() -> None:
    assert scope.probe_unit_name(AGENT) == "agent-spawn-probe-ag-abcdefg2"
    with pytest.raises(ValueError):
        scope.probe_unit_name("ag-abc")


# --- argv -------------------------------------------------------------------------------------


def test_supervisor_argv_wraps_the_supervise_argv() -> None:
    argv = scope.supervisor_argv(
        SD, AGENT, 2, ["/x/agent-spawn", "supervise", AGENT, "--state-dir", "/s", "--config", "/c"]
    )
    assert argv == [
        SD,
        "--user",
        "--scope",
        "--quiet",
        "--collect",
        "--unit=agent-spawn-ag-abcdefg2-r2",
        "--description=agent-spawn supervisor ag-abcdefg2 run 2",
        "--",
        "/x/agent-spawn",
        "supervise",
        AGENT,
        "--state-dir",
        "/s",
        "--config",
        "/c",
    ]


def test_supervisor_argv_needs_an_absolute_systemd_run() -> None:
    with pytest.raises(ValueError):
        scope.supervisor_argv("systemd-run", AGENT, 1, ["/x"])


def test_supervisor_argv_keeps_a_hostile_command_after_the_separator() -> None:
    argv = scope.supervisor_argv(SD, AGENT, 1, ["/x/a b", "--unit=evil", "--scope"])
    assert argv.index("--") < argv.index("--unit=evil")
    assert argv.count("--") == 1


# --- host and tool checks ---------------------------------------------------------------------


@pytest.mark.parametrize("platform", ["darwin", "freebsd14", "win32"])
def test_a_non_linux_host_is_refused_as_config(platform: str) -> None:
    with pytest.raises(CliError) as info:
        scope.check_host(FakeGateway(platform_name=platform))
    assert info.value.code == "config_invalid" and info.value.exit_code == 2
    assert "supervisor.detach" in info.value.message


def test_a_missing_systemd_run_is_tool_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PATH", str(tmp_path))
    with pytest.raises(CliError) as info:
        scope.resolve_systemd_run()
    assert info.value.code == "tool_missing" and info.value.exit_code == 12
    assert info.value.tool == "agent-spawn"


def test_systemd_run_that_is_not_executable_is_tool_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "systemd-run").write_text("#!/bin/sh\n")
    monkeypatch.setenv("PATH", str(tmp_path))
    with pytest.raises(CliError) as info:
        scope.resolve_systemd_run()
    assert info.value.code == "tool_missing"


def test_the_static_check_runs_the_host_check_first(
    with_systemd_run: str,
) -> None:
    with pytest.raises(CliError) as info:
        scope.check_static(FakeGateway(platform_name="darwin"))
    assert info.value.code == "config_invalid"
    assert scope.check_static(FakeGateway()) == with_systemd_run


# --- the preflight probe ----------------------------------------------------------------------


def test_preflight_runs_one_probe_scope_with_the_callers_environment(with_systemd_run: str) -> None:
    gw = FakeGateway()
    gw.run_results[with_systemd_run] = lambda argv: done()
    assert scope.preflight(gw, AGENT, env={"XDG_RUNTIME_DIR": "/run/user/1"}) == with_systemd_run
    [call] = gw.run_calls
    assert call[:6] == [
        with_systemd_run,
        "--user",
        "--scope",
        "--quiet",
        "--collect",
        "--unit=agent-spawn-probe-ag-abcdefg2",
    ]
    assert call[6] == "--" and call[7].endswith("true") and len(call) == 8
    assert gw.run_envs == [{"XDG_RUNTIME_DIR": "/run/user/1"}]
    assert gw.run_timeouts == [scope.PROBE_TIMEOUT]


@pytest.mark.parametrize(
    "result",
    [
        done(1, err="Failed to connect to bus: $DBUS_SESSION_BUS_ADDRESS and $XDG_RUNTIME_DIR"),
        done(1),
        done(127, out="boom"),
        subprocess.TimeoutExpired(["systemd-run"], 30),
        PermissionError("denied"),
        gateway.OutputTooLarge("too much"),
    ],
)
def test_a_failed_probe_is_scope_unavailable_with_no_fallback(
    with_systemd_run: str, result: object
) -> None:
    gw = FakeGateway()

    def run(argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
        if isinstance(result, BaseException):
            raise result
        assert isinstance(result, subprocess.CompletedProcess)
        return result

    gw.run_results[with_systemd_run] = run
    with pytest.raises(CliError) as info:
        scope.preflight(gw, AGENT)
    error = info.value
    assert (error.code, error.exit_code, error.tool) == ("failed", 12, "agent-spawn")
    assert error.details is not None
    assert error.details["cause"] == "scope_unavailable"
    assert error.details["retryable"] is False
    assert "XDG_RUNTIME_DIR" in error.details["hint"]


def test_a_probe_message_is_shown_safely_and_truncated(with_systemd_run: str) -> None:
    gw = FakeGateway()
    gw.run_results[with_systemd_run] = lambda argv: done(1, err="bad\x1b[31m\n" + "x" * 5000)
    with pytest.raises(CliError) as info:
        scope.preflight(gw, AGENT)
    message = info.value.message
    assert "\x1b" not in message and "\n" not in message
    assert len(message) < 1000


def test_a_vanished_systemd_run_is_tool_missing(with_systemd_run: str) -> None:
    gw = fake_with(FileNotFoundError("gone"))
    gw.run_results[with_systemd_run] = gw.run_results.pop(SD)
    with pytest.raises(CliError) as info:
        scope.preflight(gw, AGENT)
    assert info.value.code == "tool_missing"


def test_preflight_refuses_a_bad_agent_id_before_running_anything(with_systemd_run: str) -> None:
    gw = FakeGateway()
    with pytest.raises(ValueError):
        scope.preflight(gw, "ag-abc; rm")
    assert gw.run_calls == []


def test_a_non_linux_preflight_runs_nothing(with_systemd_run: str) -> None:
    gw = FakeGateway(platform_name="darwin")
    with pytest.raises(CliError):
        scope.preflight(gw, AGENT)
    assert gw.run_calls == []


# --- when scope hosting applies ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("detach", "windowed", "here", "expected"),
    [
        ("scope", False, False, True),  # a detached supervisor: every headless run
        ("scope", True, False, False),  # a window hosts it
        ("scope", False, True, False),  # `--placement here` runs it in the launcher
        ("setsid", False, False, False),  # the default
    ],
)
def test_only_a_detached_supervisor_with_scope_configured_is_scope_hosted(
    detach: str, windowed: bool, here: bool, expected: bool
) -> None:
    from types import SimpleNamespace

    from agent_spawn.launch import Launch

    fake = SimpleNamespace(
        windowed=windowed,
        here=here,
        config=SimpleNamespace(supervisor=SimpleNamespace(detach=detach)),
    )
    assert Launch.scope_hosted.fget(fake) is expected  # type: ignore[attr-defined]
