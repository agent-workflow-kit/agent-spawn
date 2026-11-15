"""Helpers for the launch tests: run the real ``agent-spawn`` against the fake binaries."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from agent_spawn.store import Store
from tests.conftest import Isolation
from tests.fakebin.harness import Fakebin


def agent_spawn_path() -> str:
    """The console script of the interpreter running the tests (it must exec in place)."""
    path = Path(sys.executable).parent / "agent-spawn"
    assert path.exists(), f"{path} is missing: run `uv sync`"
    return str(path)


class Env:
    """One test's state dir, config and fake binaries."""

    def __init__(self, fakebin: Fakebin, iso: Isolation, config_text: str = "") -> None:
        self.fakebin = fakebin
        self.iso = iso
        self.state_dir = iso.state / "agent-spawn"
        self.config = iso.config / "agent-spawn" / "config.yaml"
        if config_text:
            self.config.parent.mkdir(parents=True, exist_ok=True)
            self.config.write_text(config_text)

    def env(self, extra: Mapping[str, str] | None = None) -> dict[str, str]:
        return self.fakebin.env({"XDG_STATE_HOME": str(self.iso.state), **(extra or {})})

    def argv(self, *args: str) -> list[str]:
        return [agent_spawn_path(), *args]

    def run(
        self, *args: str, extra_env: Mapping[str, str] | None = None, timeout: float = 60.0
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            self.argv(*args),
            capture_output=True,
            text=True,
            env=self.env(extra_env),
            stdin=subprocess.DEVNULL,
            timeout=timeout,
            check=False,
        )

    def store(self) -> Store:
        return Store.open(self.state_dir, create=False)

    def agents(self) -> list[dict[str, Any]]:
        with self.store() as store:
            return store.list_agents()

    def agent(self, agent_id: str) -> dict[str, Any]:
        with self.store() as store:
            row = store.get_agent(agent_id)
        assert row is not None
        return row

    def wait_for(
        self, predicate: Callable[[], bool], what: str, timeout: float = 20.0, step: float = 0.05
    ) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(step)
        raise AssertionError(f"timed out waiting for {what}")


def result_json(done: subprocess.CompletedProcess[str]) -> Any:
    assert done.stdout.strip(), f"no output (exit {done.returncode}): {done.stderr}"
    return json.loads(done.stdout)


def script_live_run(fakebin: Fakebin, *, times: int | None = None) -> None:
    """`claude-session status` reports the current run live in its session phase."""
    fakebin.script(
        "claude-session",
        "status",
        times=times,
        merge={
            "state": "working",
            "live": True,
            "run": {
                "outcome": None,
                "end_reason": None,
                "exit_code": None,
                "phase": "session",
                "state": "working",
                "ended_at": None,
                "report": None,
            },
        },
    )


def script_hung_exec(fakebin: Fakebin) -> None:
    """`claude-session exec` runs until it is signaled (then it ends `killed`)."""
    fakebin.script("claude-session", "exec", hang=True)


def track_agent(env: Env, proc_tracker: Any, agent_id: str) -> None:
    """Record the supervisor and `exec` of an agent so the test's cleanup kills exactly them."""
    row = env.agent(agent_id)
    for key in ("exec_proc", "supervisor"):
        if row.get(key) is not None:
            proc_tracker.record(row[key])
