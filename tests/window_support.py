"""Helpers for the window tests: the real ``agent-spawn`` against the fake binaries, with the fake
termctl starting its panes in a private, real tmux server (design.md, Live verification)."""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from agent_spawn import procs
from agent_spawn.store import Store
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
        self.spawn: str | None = None  # a wrapper that stands in for the agent-spawn executable

    def argv(self, *args: str) -> list[str]:
        return [self.spawn, *args] if self.spawn is not None else super().argv(*args)

    def capture_supervisor(self, directory: Path, launcher_prologue: str = "") -> Capture:
        """Run every window supervisor under a wrapper named ``agent-spawn`` (the launcher puts
        its own path in the pane's command) that is the pane process itself: it sends the
        supervisor's stdout and stderr to files and records its exit code atomically."""
        capture = Capture(directory)
        self.spawn = str(capture.write(launcher_prologue))
        return capture

    def env(self, extra: Mapping[str, str] | None = None) -> dict[str, str]:
        assert REAL_TMUX is not None
        return super().env(
            {
                "FAKEBIN_TMUX": REAL_TMUX,
                "TERMCTL_SOCKET": str(self.tmux.socket),
                **(extra or {}),
            }
        )

    def store_create(self) -> Store:
        """Create the state database (a supervisor only opens an existing one)."""
        return Store.open(self.state_dir)

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


class Capture:
    """Where the wrapper of ``WindowEnv.capture_supervisor`` leaves what the supervisor wrote."""

    def __init__(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.directory = directory
        self.out = directory / "stdout"
        self.err = directory / "stderr"
        self.rc = directory / "rc"

    def write(self, launcher_prologue: str = "") -> Path:
        """The wrapper script; ``launcher_prologue`` (Python) runs first in every process that is
        not a supervisor, to patch the launcher under test."""
        script = self.directory / "agent-spawn"
        head = (
            f"#!{sys.executable}\n"
            "import os, sys\n"
            "args = sys.argv[1:]\n"
            "supervisor = args[:1] == ['supervise']\n"
            "if supervisor:\n"
            f"    for fd, path in ((1, {str(self.out)!r}), (2, {str(self.err)!r})):\n"
            "        os.dup2(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600), fd)\n"
            "if not supervisor:\n"
        )
        tail = (
            "from agent_spawn.cli import main\n"
            "code = main(args)\n"
            "if supervisor:\n"
            f"    rc = {str(self.rc)!r} + ('.host' if '--host' in args else '')\n"
            "    with open(rc + '.tmp', 'w') as handle:\n"
            "        handle.write(str(code))\n"
            "    os.replace(rc + '.tmp', rc)\n"
            "raise SystemExit(code)\n"
        )
        script.write_text(head + textwrap.indent(launcher_prologue or "pass\n", "    ") + tail)
        script.chmod(0o755)
        return script

    def exit_code(self) -> int:
        return int(self.rc.read_text())

    def host_exit_code(self) -> int:
        """The exit code of the ``supervise --host`` process of a promotion."""
        return int(Path(str(self.rc) + ".host").read_text())


def tmux_text(done: subprocess.CompletedProcess[str]) -> str:
    return done.stdout.strip()


def log_lines(path: Path) -> list[str]:
    return path.read_text().splitlines() if path.exists() else []


def send_signal(identity: procs.ProcessIdentity, signum: int) -> None:
    """Signal exactly the recorded process (verified by pidfd, PID and start time)."""
    handle = procs.open_verified(identity)
    assert handle is not None, "the process is gone"
    with handle:
        handle.send(signum)
