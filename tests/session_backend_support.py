"""Shared helpers for the session backend tests."""

from __future__ import annotations

import subprocess
from collections.abc import Callable, Mapping, Sequence

from agent_spawn.gateway import SystemGateway
from agent_spawn.session_backend import RunSpec, SessionBackend
from tests.fakebin.harness import Fakebin

AGENT = "ag-abcdefgh"
SESSION = "cs-abcdefgh"
RELAY = "post-state-change=/opt/as/agent-spawn --state-dir /state --config /cfg/c.yaml _session-event"


class FakebinGateway(SystemGateway):
    """The real gateway, run with the fakebin environment (``PATH`` first has the fakes)."""

    def __init__(self, fakebin: Fakebin) -> None:
        super().__init__()
        self._fakebin = fakebin

    def run(
        self,
        argv: Sequence[str],
        *,
        stdin: str | None = None,
        timeout: float | None = None,
        env: Mapping[str, str] | None = None,
        cwd: str | None = None,
    ) -> subprocess.CompletedProcess[str]:
        return super().run(argv, stdin=stdin, timeout=timeout, env=self._fakebin.env(), cwd=cwd)


def fake_backend(fakebin: Fakebin, **kwargs: object) -> SessionBackend:
    return SessionBackend(
        [str(fakebin.bin_dir / "claude-session")],
        gateway=FakebinGateway(fakebin),
        **kwargs,  # type: ignore[arg-type]
    )


def spec(**changes: object) -> RunSpec:
    base = {"agent_id": AGENT, "mode": "headless", "skill": "implement-issue"}
    return RunSpec(**{**base, **changes})  # type: ignore[arg-type]


def completed(stdout: str, code: int = 0, stderr: str = "") -> Callable[..., object]:
    """A canned ``FakeGateway.run_results`` entry."""

    def run(argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(list(argv), code, stdout, stderr)

    return run
