"""Helpers for the window tests: the real ``agent-spawn`` against the fake binaries, with the fake
termctl starting its panes in a private, real tmux server (design.md, Live verification)."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from agent_spawn import procs
from tests.conftest import REAL_TMUX, Isolation
from tests.fakebin.harness import Fakebin
from tests.launch_support import Env
from tests.private_tmux import PrivateTmux


class WindowEnv(Env):
    """An ``Env`` whose termctl opens real tmux panes on the test's private server."""

    def __init__(
        self, fakebin: Fakebin, iso: Isolation, tmux: PrivateTmux, config_text: str = ""
    ) -> None:
        super().__init__(fakebin, iso, config_text)
        self.tmux = tmux

    def env(self, extra: Mapping[str, str] | None = None) -> dict[str, str]:
        assert REAL_TMUX is not None
        return super().env(
            {
                "FAKEBIN_TMUX": REAL_TMUX,
                "TERMCTL_SOCKET": str(self.tmux.socket),
                **(extra or {}),
            }
        )

    def panes(self) -> list[dict[str, Any]]:
        """The fake termctl's view of the panes (it syncs from tmux first)."""
        done = self.fakebin.run(
            "termctl", "list", "--socket", str(self.tmux.socket), "--json", env=self.env()
        )
        assert done.returncode == 0, done.stderr
        return json.loads(done.stdout)["panes"]

    def pane_pid(self, agent_id: str) -> int:
        """The pane process (the supervisor) of the agent's current run."""
        row = self.agent(agent_id)
        with self.store() as store:
            run = store.get_run(agent_id, row["run"])
        assert run is not None and run["pane_proc"] is not None
        return int(run["pane_proc"].pid)

    def alive(self, identity: procs.ProcessIdentity) -> bool:
        return procs.is_alive(identity)


def tmux_text(done: subprocess.CompletedProcess[str]) -> str:
    return done.stdout.strip()


def log_lines(path: Path) -> list[str]:
    return path.read_text().splitlines() if path.exists() else []
