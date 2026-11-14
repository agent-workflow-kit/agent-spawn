"""A private, real tmux server for process tests: ``tmux -f /dev/null -S <tmp>/sock``.

The server runs with a private ``HOME``/``XDG_*`` and no ``TMUX`` variable, never touches the
user's server or config, and is stopped by its own ``kill-server``; if it outlives that, only
the exact PID recorded at launch (verified by its start time) is signaled.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import tempfile
import time
from collections.abc import Mapping
from pathlib import Path


def process_start_time(pid: int) -> str | None:
    """Field 22 of ``/proc/<pid>/stat`` (parsed after the last ``)``), or None if unreadable."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return None
    fields = stat[stat.rindex(")") + 2 :].split()
    return fields[19] if len(fields) > 19 else None


class PrivateTmux:
    def __init__(self, tmux_bin: str, env: Mapping[str, str]) -> None:
        base = tempfile.gettempdir()
        if len(base) > 40:  # keep the socket path well under the 108-byte sun_path limit
            base = "/tmp"
        self.directory = Path(tempfile.mkdtemp(prefix="as-tmux-", dir=base))
        self.socket = self.directory / "sock"
        self.tmux_bin = tmux_bin
        self.env = {
            "HOME": env["HOME"],
            "XDG_CONFIG_HOME": env["XDG_CONFIG_HOME"],
            "XDG_STATE_HOME": env["XDG_STATE_HOME"],
            "PATH": env.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
            "TMUX_TMPDIR": str(self.directory),
            "TERM": "xterm",
            "LANG": "C.UTF-8",
        }
        self.server_pid: int | None = None
        self._server_start: str | None = None

    def argv(self, *args: str) -> list[str]:
        return [self.tmux_bin, "-f", "/dev/null", "-S", str(self.socket), *args]

    def run(self, *args: str, timeout: float = 15) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            self.argv(*args),
            capture_output=True,
            text=True,
            env=self.env,
            stdin=subprocess.DEVNULL,
            timeout=timeout,
            check=False,
        )

    def start(self, session: str = "test", command: str = "sleep 3600") -> None:
        """Start the server with one detached session, and record the server's PID."""
        done = self.run("new-session", "-d", "-s", session, "-x", "80", "-y", "24", command)
        assert done.returncode == 0, f"tmux new-session failed: {done.stderr}"
        pid = self.run("display-message", "-p", "#{pid}").stdout.strip()
        self.server_pid = int(pid)
        self._server_start = process_start_time(self.server_pid)

    def stop(self) -> None:
        """Stop the server: ``kill-server``, then SIGKILL the recorded PID only if it lingers."""
        if self.socket.exists():
            self.run("kill-server")
        pid = self.server_pid
        if pid is not None:
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline and self._alive(pid):
                time.sleep(0.05)
            if self._alive(pid) and self._start_time_matches(pid):
                os.kill(pid, signal.SIGKILL)
        shutil.rmtree(self.directory, ignore_errors=True)

    def _start_time_matches(self, pid: int) -> bool:
        return self._server_start is not None and process_start_time(pid) == self._server_start

    def _alive(self, pid: int) -> bool:
        if not self._start_time_matches(pid):
            return False  # gone, or the PID was reused by another process
        try:
            state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
        except OSError:
            return False
        return state != "Z"
