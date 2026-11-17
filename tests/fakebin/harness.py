"""Test-side handle on the fake binaries: scripting, running and reading the invocation log.

Use the ``fakebin`` fixture (``tests/conftest.py``). A test scripts a response, runs the code
under test with ``fakebin.env()`` (``PATH`` first has the fakes), then asserts the exact argv::

    fakebin.script("wtctl", "create", merge={"record": {"path": "/x"}})
    subprocess.run(["wtctl", "create", "--repo", "h/o/r", "--json"], env=fakebin.env())
    fakebin.assert_argv("wtctl", ["create", "--repo", "h/o/r", "--json"])
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

FAKES = ("claude-session", "termctl", "wtctl", "tmux")
FAKEBIN_DIR = Path(__file__).resolve().parent
SYSTEM_PATH = "/usr/local/bin:/usr/bin:/bin"


def read_log_records(path: Path) -> list[dict[str, Any]]:
    """The complete records of a JSON-lines log a fake may be appending to right now.

    A fake writes each record with one ``write`` and a terminating newline, but a reader that
    polls mid-write can still see the record cut short. A final line without its newline is
    that in-flight record: it is ignored (the next poll sees it whole). A malformed line that
    is newline-terminated is a real defect and still raises.
    """
    text = path.read_text()
    lines = text.split("\n")
    lines.pop()  # "" after the last newline, or the unterminated in-flight record
    return [json.loads(line) for line in lines if line]


@dataclass(frozen=True)
class Invocation:
    tool: str
    argv: list[str]
    cwd: str
    env: dict[str, str]
    pid: int
    stdin: str | None
    stdin_is_tty: bool
    exit_code: int | None
    signals: list[int] = field(default_factory=list)


def flag_values(argv: Sequence[str], flag: str) -> list[str]:
    """Every value of a repeatable `--flag value` or `--flag=value` in argv (before `--`)."""
    values: list[str] = []
    i = 0
    while i < len(argv):
        token = argv[i]
        if token == "--":
            break
        if token == flag and i + 1 < len(argv):
            values.append(argv[i + 1])
            i += 1
        elif token.startswith(flag + "="):
            values.append(token[len(flag) + 1 :])
        i += 1
    return values


class Fakebin:
    """One test's fakes: a bin directory, an invocation log and a script of canned responses."""

    def __init__(self, root: Path, home_env: Mapping[str, str]) -> None:
        self.root = root
        self.bin_dir = root / "bin"
        self.log = root / "invocations.jsonl"
        self.script_file = root / "script.json"
        self._home_env = dict(home_env)
        self._rules: list[dict[str, Any]] = []
        self.bin_dir.mkdir(parents=True)
        for name in FAKES:
            (self.bin_dir / name).symlink_to(FAKEBIN_DIR / name)
        # The stubs start with `#!/usr/bin/env python3`: put the test interpreter first, so
        # the 3.12/3.13 matrix runs the fakes under the version it is testing.
        (self.bin_dir / "python3").symlink_to(os.path.realpath(sys.executable))
        self.log.touch()
        self._write_script()

    # --- scripting ---------------------------------------------------------------------

    def script(
        self,
        tool: str,
        command: str | None = None,
        *,
        argv_contains: Sequence[str] = (),
        times: int | None = None,
        until_called: str | None = None,
        **respond: Any,
    ) -> None:
        """Script the next matching calls of ``tool``'s ``command``.

        ``until_called`` (a subcommand name) ends the rule once a call of that subcommand of
        ``tool`` is in the log, so a reply can depend on what happened rather than on a count.

        Keyword arguments are the response: ``exit`` (code), ``stdout`` (replaces the canned
        output: an object is printed as JSON), ``merge`` (deep-merged into the canned output),
        ``error`` (``{"code", "message", ...}`` printed as the error envelope), ``stderr``,
        ``sleep`` (seconds), ``hang`` (block until a signal), ``on_signal``
        (``{"SIGTERM": {"exit": 143}}`` or ``"ignore"``), ``outcome`` (claude-session: the
        outcome name of ``exec``/``resume``/``wait``/``status``), ``pending_unadopted``
        (claude-session ``status`` only: a run nothing adopted yet is reported pending, in its
        launch phase, with no outcome; unscripted, ``status`` never reports a pending run),
        ``end_reason`` (claude-session
        ``exec`` only: replaces the outcome's end reason) and ``secret``
        (claude-session ``verify-secret``). Rules match in the order added; ``times`` limits
        how many calls a rule answers.
        """
        rule: dict[str, Any] = {"tool": tool, "respond": respond}
        if command is not None:
            rule["command"] = command
        if argv_contains:
            rule["argv_contains"] = list(argv_contains)
        if times is not None:
            rule["times"] = times
        if until_called is not None:
            rule["until_called"] = until_called
        self._rules.append(rule)
        self._write_script()

    def _write_script(self) -> None:
        tmp = self.script_file.with_suffix(".tmp")
        tmp.write_text(json.dumps({"rules": self._rules}))
        os.replace(tmp, self.script_file)

    # --- running -----------------------------------------------------------------------

    def env(self, extra: Mapping[str, str] | None = None) -> dict[str, str]:
        """An environment where the fakes shadow the real tools, in the isolated home."""
        env = {
            **self._home_env,
            "PATH": f"{self.bin_dir}:{SYSTEM_PATH}",
            "FAKEBIN_LOG": str(self.log),
            "FAKEBIN_SCRIPT": str(self.script_file),
            "FAKEBIN_PARENT_PID": str(os.getpid()),  # a hung fake exits when this process is gone
        }
        env.update(extra or {})
        return env

    def run(
        self,
        tool: str,
        *args: str,
        input: str | None = None,
        cwd: Path | str | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float = 30,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [str(self.bin_dir / tool), *args],
            input=input if input is not None else None,
            stdin=None if input is not None else subprocess.DEVNULL,
            capture_output=True,
            text=True,
            cwd=cwd,
            env=self.env(env),
            timeout=timeout,
            check=False,
        )

    def seed_session(self, csession_id: str = "cs-abcdefgh", **fields: Any) -> str:
        """Make claude-session know ``csession_id`` (an unknown id exits 10, like the real one).

        By default the session is settled: its run 1 was adopted and ended ``done``. Pass
        ``adopted=False`` for a pending run, ``final=False`` for a live one, or any other field
        of the fake's session record (``labels``, ``kind``, ``run``...).
        """
        from tests.fakebin.fakelib.claude_session import new_entry

        path = Path(str(self.log) + ".sessions.json")
        data: dict[str, Any] = json.loads(path.read_text()) if path.exists() else {}
        settled = {"adopted": True, "final": True, "outcome": "done"}
        data[csession_id] = new_entry(**{**settled, **fields})
        path.write_text(json.dumps(data))
        return csession_id

    def python_versions(self) -> list[str]:
        """The Python version each logged call ran under."""
        return [r["python"] for r in self.records("invoke")]

    def records(self, event: str) -> list[dict[str, Any]]:
        """The log records of a kind other than a call (``outcome_file_failed``...)."""
        return [record for record in self._log_records() if record["event"] == event]

    # --- the log -----------------------------------------------------------------------

    def _log_records(self) -> list[dict[str, Any]]:
        return read_log_records(self.log)

    def invocations(self, tool: str | None = None) -> list[Invocation]:
        invokes: dict[str, dict[str, Any]] = {}
        exits: dict[str, dict[str, Any]] = {}
        order: list[str] = []
        for record in self._log_records():
            if record["event"] == "invoke":
                invokes[record["id"]] = record
                order.append(record["id"])
            elif record["event"] == "exit":
                exits[record["id"]] = record
        found: list[Invocation] = []
        for invoke_id in order:
            record = invokes[invoke_id]
            if tool is not None and record["tool"] != tool:
                continue
            ended = exits.get(invoke_id)
            found.append(
                Invocation(
                    tool=record["tool"],
                    argv=record["argv"],
                    cwd=record["cwd"],
                    env=record["env"],
                    pid=record["pid"],
                    stdin=ended["stdin"] if ended else None,
                    stdin_is_tty=record["stdin_is_tty"],
                    exit_code=ended["exit_code"] if ended else None,
                    signals=ended["signals"] if ended else [],
                )
            )
        return found

    def wait_for_hang(self, pid: int, timeout: float = 10) -> None:
        """Block until the fake with this PID logged ``hang_ready`` (its handlers are installed)."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.log.exists():
                for record in self._log_records():
                    if record["event"] == "hang_ready" and record["pid"] == pid:
                        return
            time.sleep(0.02)
        raise AssertionError(f"the fake (pid {pid}) never reached its hang")

    def clear_script(self, tool: str, command: str | None = None) -> None:
        """Disable the rules added so far for ``tool`` (and ``command``).

        A disabled rule keeps its place, because the fakes count answered calls by rule index.
        """
        for rule in self._rules:
            if rule["tool"] == tool and (command is None or rule.get("command") == command):
                rule["tool"] = "__cleared__"
        self._write_script()

    def wait_for_gate(self, gate: Path, timeout: float = 20) -> None:
        """Block until a call scripted with ``gate=<gate>`` is waiting at that gate."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if any(r["path"] == str(gate) for r in self.records("gate_ready")):
                return
            time.sleep(0.02)
        raise AssertionError(f"no call reached the gate {gate}")

    def argvs(self, tool: str) -> list[list[str]]:
        return [call.argv for call in self.invocations(tool)]

    def assert_argv(self, tool: str, expected: Sequence[str], *, index: int = -1) -> None:
        """Assert the exact argv (without the program name) of one call of ``tool``."""
        calls = self.invocations(tool)
        assert calls, f"no {tool} call was logged"
        actual = calls[index].argv
        assert actual == list(expected), (
            f"{tool} call {index} argv\n  got:      {actual}\n  expected: {list(expected)}"
        )

    def assert_calls(self, tool: str, expected: Sequence[Sequence[str]]) -> None:
        """Assert the exact argv of every call of ``tool``, in order."""
        actual = self.argvs(tool)
        assert actual == [list(item) for item in expected], (
            f"{tool} calls\n  got:      {actual}\n  expected: {[list(i) for i in expected]}"
        )

    def flag_values(self, tool: str, flag: str, *, index: int = -1) -> list[str]:
        """The values of a repeatable flag (``--label``, ``--decision-id``, ``--hook``...)."""
        return flag_values(self.invocations(tool)[index].argv, flag)

    def assert_not_called(self, tool: str) -> None:
        assert not self.invocations(tool), f"{tool} was called: {self.argvs(tool)}"
