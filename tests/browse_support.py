"""Helpers for the browse tests: a scripted backend and agent objects like ``list --json``."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from datetime import UTC, datetime
from typing import Any

from agent_spawn.tui.app import BrowseApp
from agent_spawn.tui.model import CliResult

NOW = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
IDS = ["ag-aaaaaaaa", "ag-bbbbbbbb", "ag-cccccccc"]


def agent(agent_id: str, **fields: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "agent_id": agent_id,
        "status": "parked",
        "needs_attention": None,
        "kind": "work",
        "name": f"name {agent_id[3:7]}",
        "labels": [],
        "created": "2026-01-01T11:00:00.000Z",
        "updated": "2026-01-01T11:30:00.000Z",
    }
    base.update(fields)
    return base


class FakeBackend:
    """Answers ``list`` from ``agents`` and everything else from ``answers`` (by command word)."""

    def __init__(self, agents: list[dict[str, Any]], **answers: CliResult) -> None:
        self.agents = agents
        self.answers = answers
        self.calls: list[tuple[list[str], bool]] = []
        self.terminal: list[list[str]] = []
        self.closed = False
        self.list_result: CliResult | None = None

    def call(
        self, command: Sequence[str], *, acting: bool = False, timeout: float | None = None
    ) -> CliResult:
        self.calls.append((list(command), acting))
        word = command[0]
        if word == "list":
            return self.list_result or CliResult(0, {"agents": self.agents})
        if word in self.answers:
            return self.answers[word]
        if word == "show":
            return CliResult(
                0,
                {
                    "agent": next(a for a in self.agents if a["agent_id"] == command[1]),
                    "runs": [],
                    "session": None,
                    "supervisor": None,
                    "warnings": [],
                },
            )
        return CliResult(0, {"agent": None})

    def run_on_terminal(self, argv: Sequence[str]) -> int:
        self.terminal.append(list(argv))
        return 0

    def close(self) -> None:
        self.closed = True

    def commands(self) -> list[list[str]]:
        return [c for c, _ in self.calls]


async def settle(app: BrowseApp, pilot: Any) -> None:
    """Wait until every worker is done and all messages are handled (no sleeps)."""
    for _ in range(10):
        await app.workers.wait_for_complete()
        await pilot.pause()
        if not app.workers:
            break


def run_app(
    app: BrowseApp,
    script: Callable[[BrowseApp, Any], Awaitable[None]],
    size: tuple[int, int] = (140, 30),
) -> None:
    async def main() -> None:
        async with app.run_test(size=size) as pilot:
            await settle(app, pilot)
            await script(app, pilot)

    asyncio.run(main())


class PtyRun:
    """A child on a pseudo-terminal, driven with bounded waits (never sleeps for a fixed time)."""

    def __init__(self, argv: Sequence[str], env: dict[str, str]) -> None:
        import fcntl
        import os
        import pty
        import struct
        import subprocess
        import termios

        self.master, slave = pty.openpty()
        fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 30, 140, 0, 0))
        self.buffer = ""
        self._os = os
        self.proc = subprocess.Popen(
            list(argv),
            stdin=slave,
            stdout=slave,
            stderr=slave,
            env={**env, "TERM": "xterm-256color", "COLUMNS": "140", "LINES": "30"},
            start_new_session=True,
        )
        os.close(slave)

    def _pump(self, wait: float) -> bool:
        import select

        ready, _, _ = select.select([self.master], [], [], wait)
        if not ready:
            return True
        try:
            data = self._os.read(self.master, 65536)
        except OSError:
            return False
        if not data:
            return False
        self.buffer += data.decode("utf-8", errors="replace")
        return True

    def read_until(self, needle: str, timeout: float = 60.0) -> None:
        import time

        deadline = time.monotonic() + timeout
        while needle not in self.buffer:
            if time.monotonic() > deadline or not self._pump(0.1):
                raise AssertionError(f"{needle!r} never appeared; saw {self.buffer[-500:]!r}")

    def send(self, data: str) -> None:
        self._os.write(self.master, data.encode())

    def finish(self, timeout: float = 60.0) -> int:
        import time

        deadline = time.monotonic() + timeout
        try:
            while self.proc.poll() is None:
                if time.monotonic() > deadline:
                    raise AssertionError(f"still running; saw {self.buffer[-500:]!r}")
                self._pump(0.1)
            for _ in range(100):  # the output left in the pty after the child exited
                if not self._pump(0.02):
                    break
        finally:
            if self.proc.poll() is None:
                self.proc.terminate()  # the exact child this object started
                self.proc.wait(timeout=10)
            self._os.close(self.master)
        return int(self.proc.returncode)
