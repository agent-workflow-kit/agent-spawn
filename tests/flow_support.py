"""Helpers for the end-to-end flow tests (issue #19): the real ``agent-spawn`` against the fakes.

A :class:`Flow` wraps one test's :class:`~tests.launch_support.Env` with a config whose hooks
record every payload, runs commands through :meth:`Flow.cli` (which validates each ``--json``
output against its schema) and :meth:`Flow.validate_everything` (which validates everything the
scenario left behind: hook payloads, ``log.jsonl`` and ``audit.jsonl``).
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_spawn import procs
from tests import schema_registry
from tests.conftest import Isolation
from tests.fakebin.harness import Fakebin
from tests.launch_support import Env, script_hung_exec
from tests.procguard import ProcTracker

ERROR = "agent-spawn/cli/error.json"
#: The hooks a flow records. ``pre-teardown`` is left out: teardown refuses to run it until #32.
HOOKS = (
    "pre-launch",
    "post-launch",
    "post-state-change",
    "post-escalate",
    "post-park",
    "post-teardown",
)

#: A run still in its launch phase (claude-session's hooks and validations not finished).
LIVE_LAUNCH_PHASE = {
    "phase": "launch",
    "state": "starting",
    "outcome": None,
    "end_reason": None,
    "exit_code": None,
    "report": None,
    "ended_at": None,
    "session_started_at": None,
}


def schema(name: str) -> str:
    return f"agent-spawn/cli/{name}.json"


@dataclass
class Result:
    code: int
    body: Any
    stdout: str
    stderr: str


class Flow:
    """One scenario: an isolated state dir, the fakes, recording hooks and validation."""

    def __init__(
        self, fakebin: Fakebin, iso: Isolation, tmp_path: Path, hooks: Sequence[str] = HOOKS
    ) -> None:
        self.fakebin = fakebin
        self.hooks_dir = tmp_path / "hook-payloads"
        self.hooks_dir.mkdir()
        # One file per payload (a synchronous hook; the name is unique per call).
        record = f"f=$(mktemp {self.hooks_dir}/${{AGENT_SPAWN_HOOK}}.XXXXXX) && cat > $f"
        lines = ["hooks:"]
        for name in hooks:
            lines += [f"  {name}:", f'    - cmd: "{record}"']
        self.env = Env(fakebin, iso, "\n".join(lines) + "\n")
        self.state_dir = self.env.state_dir
        self.validated = 0

    # -- running -----------------------------------------------------------------------

    def check(self, schema_id: str, stdout: str) -> Any:
        """Parse and validate one ``--json`` output (the error envelope when it is one)."""
        if not stdout.strip():
            return None
        body = json.loads(stdout)
        error = isinstance(body, dict) and "error" in body
        schema_registry.validate(body, ERROR if error else schema_id)
        self.validated += 1
        return body

    def cli(
        self, schema_id: str, *args: str, extra_env: dict[str, str] | None = None
    ) -> Result:
        done = self.env.run(*args, extra_env=extra_env)
        return Result(
            done.returncode, self.check(schema_id, done.stdout), done.stdout, done.stderr
        )

    def start(
        self, tracker: ProcTracker, *args: str, extra_env: dict[str, str] | None = None
    ) -> subprocess.Popen[str]:
        """Start a command in the background (tracked: the test's cleanup kills exactly it)."""
        return tracker.popen(
            self.env.argv(*args),
            env=self.env.env(extra_env),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

    def finish(self, schema_id: str, proc: subprocess.Popen[str], timeout: float = 60) -> Result:
        out, err = proc.communicate(timeout=timeout)
        return Result(proc.returncode, self.check(schema_id, out), out, err)

    # -- the state -----------------------------------------------------------------------

    def agent(self, agent_id: str) -> dict[str, Any]:
        return self.env.agent(agent_id)

    def settled(self, agent_id: str) -> bool:
        """Its supervisor (if any) has exited."""
        supervisor = self.agent(agent_id)["supervisor"]
        return supervisor is None or not procs.is_alive(supervisor)

    def wait_settled(self, agent_id: str) -> None:
        self.env.wait_for(lambda: self.settled(agent_id), f"the supervisor of {agent_id} to exit")

    def track(self, tracker: ProcTracker, agent_id: str) -> None:
        row = self.agent(agent_id)
        for key in ("exec_proc", "supervisor"):
            if row.get(key) is not None:
                tracker.record(row[key])

    def script_live_status(self, *, times: int | None = None, **top: Any) -> None:
        """`claude-session status` reports the current run live in its session phase (``top``
        fields, such as ``csession_dir``, are merged into the status object)."""
        live_run = {
            "outcome": None,
            "end_reason": None,
            "exit_code": None,
            "phase": "session",
            "state": "working",
            "ended_at": None,
            "report": None,
        }
        self.fakebin.script(
            "claude-session",
            "status",
            times=times,
            merge={"state": "working", "live": True, "run": live_run, **top},
        )

    def live_launch(self, tracker: ProcTracker, *args: str, **status_fields: Any) -> str:
        """Launch (no ``--wait``) a headless run that stays live: scripted ``live`` status and a
        hung ``exec``. Returns the agent id; the caller ends the run (``stop`` or a signal)."""
        self.script_live_status(**status_fields)
        script_hung_exec(self.fakebin)
        result = self.cli(
            schema("launch"), "launch", "--mode", "headless", "--skill", "do-it", "--json", *args
        )
        assert result.code == 0, (result.stdout, result.stderr)
        agent_id = str(result.body["agent"]["agent_id"])
        self.track(tracker, agent_id)
        assert self.agent(agent_id)["status"] == "live"
        return agent_id

    def end_live_run(self, agent_id: str) -> None:
        """What ``claude-session stop`` does to a live run's ``exec``: SIGTERM to the exact
        verified process; the run then ends ``killed`` and the supervisor parks the agent."""
        import signal  # noqa: PLC0415

        exec_proc = self.agent(agent_id)["exec_proc"]
        assert exec_proc is not None
        assert procs.forward_signal(exec_proc, signal.SIGTERM).sent

    def back_to_final_status(self) -> None:
        """The live script ends: `status` answers the canned final run again and `exec` runs to
        its (default) outcome instead of hanging."""
        self.fakebin.clear_script("claude-session", "status")
        self.fakebin.clear_script("claude-session", "exec")

    # -- the records -----------------------------------------------------------------------

    def hook_payloads(self) -> list[dict[str, Any]]:
        return [
            json.loads(path.read_text()) for path in sorted(self.hooks_dir.iterdir()) if path.is_file()
        ]

    def hooks_named(self, name: str, agent_id: str | None = None) -> list[dict[str, Any]]:
        return [
            p
            for p in self.hook_payloads()
            if p["hook"] == name and (agent_id is None or p["object_id"] == agent_id)
        ]

    def log(self) -> list[dict[str, Any]]:
        return self.lines("log.jsonl")

    def audit(self) -> list[dict[str, Any]]:
        return self.lines("audit.jsonl")

    def lines(self, name: str) -> list[dict[str, Any]]:
        path = self.state_dir / name
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    def validate_everything(self, *, expect_hooks: Sequence[str] = ()) -> None:
        """Every hook payload, log record and audit record the scenario wrote is schema-valid.

        ``expect_hooks`` names hooks that must have fired at least once (so the check cannot
        pass over an empty directory). Call it after the supervisors have exited.
        """
        payloads = self.hook_payloads()
        for payload in payloads:
            schema_registry.validate(payload, f"agent-spawn/hooks/{payload['hook']}.json")
        fired = {p["hook"] for p in payloads}
        missing = set(expect_hooks) - fired
        assert not missing, f"hooks that never fired: {sorted(missing)} (fired: {sorted(fired)})"
        event_ids = [p["event_id"] for p in payloads if p["hook"] != "pre-launch"]
        assert len(event_ids) == len(set(event_ids)), "an event was delivered twice"
        for record in self.log():
            schema_registry.validate(record, "agent-spawn/log.json")
        for record in self.audit():
            schema_registry.validate(record, "agent-spawn/audit.json")
