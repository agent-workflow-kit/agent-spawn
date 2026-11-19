"""End to end: a relayed event wakes the supervisor, which then sees the run started.

The supervisor polls `claude-session status` once a second; with that interval made huge, only a
relay record can make it look again, so the test cannot pass by polling alone. No clock decides
the result: the exec stand-in is held at a gate until the supervisor has been seen live.
"""

from __future__ import annotations

import json
import sqlite3
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from agent_spawn import supervise
from tests.conftest import Isolation
from tests.fakebin.harness import Fakebin
from tests.launch_support import agent_spawn_path, script_live_run
from tests.test_supervise_inprocess import Rig, SpawnGateway, rig_for  # noqa: F401


class RelayingGateway(SpawnGateway):
    """After the supervisor's first `status` poll, run the real `_session-event` once."""

    def __init__(
        self, fakebin: Fakebin, rig_ref: list[Rig], env: dict[str, str], gate: Path
    ) -> None:
        super().__init__(fakebin)
        self.fakebin = fakebin
        self.rig_ref, self.env, self.gate = rig_ref, env, gate
        self.relayed = False
        self.live_seen_while_exec_runs = False

    def sleep(self, seconds: float) -> None:
        rig = self.rig_ref[0]
        statuses = [a for a in self.fakebin.argvs("claude-session") if a[0] == "status"]
        if not self.relayed and statuses and rig.agent()["status"] == "launching":
            self.relayed = True
            body: dict[str, Any] = {
                "hook": "post-state-change",
                "object_type": "session",
                "object_id": rig.session,
                "labels": [f"agent={rig.agent_id}"],
                "event_id": 1,
                "event_ts": "2026-11-02T10:00:05.000Z",
                "run": 1,
                "phase": "session",
                "state": "working",
                "activity": "working",
                "needs_input_reason": None,
                "previous_state": "starting",
                "final": False,
                "cause": "UserPromptSubmit",
                "report": None,
                "outcome": None,
                "end_reason": None,
                "exit_code": None,
                "message": None,
                "sandbox_active": False,
            }
            done = subprocess.run(
                [
                    agent_spawn_path(),
                    "--state-dir",
                    str(rig.state),
                    "--config",
                    str(rig.state.parent / "config.yaml"),
                    "_session-event",
                ],
                input=json.dumps(body).encode(),
                capture_output=True,
                env=self.env,
                timeout=60,
                check=False,
            )
            assert done.returncode == 0, done.stderr
        elif self.relayed and rig.agent()["status"] == "live":
            self.live_seen_while_exec_runs = True  # sleep runs only while `exec` is alive
            self.gate.write_text("")  # now let `exec` finish
        super().sleep(seconds)


def test_a_relay_event_wakes_the_supervisor_which_detects_the_start(
    rig_for: Callable[..., Rig],  # noqa: F811
    fakebin: Fakebin,
    isolation_env: Isolation,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    rig = rig_for()
    monkeypatch.setattr(supervise, "STATUS_EVERY", 3600.0)
    # The first poll still sees the launch phase; every later one sees the started run.
    fakebin.script(
        "claude-session",
        "status",
        times=1,
        merge={"state": "starting", "live": True, "run": {"phase": "launch", "state": "starting"}},
    )
    script_live_run(fakebin)
    gate = tmp_path / "exec-gate"  # `exec` outlives the relay however slow it is
    fakebin.script("claude-session", "exec", gate=str(gate), outcome="done")
    gateway = RelayingGateway(fakebin, [rig], fakebin.env(), gate)
    rig.supervisor.gateway = gateway
    rig.supervisor.run()
    assert gateway.relayed and gateway.live_seen_while_exec_runs
    row = rig.agent()
    assert row["session_state"] is not None  # the relay's record was kept
    db = sqlite3.connect(f"file:{rig.state / 'state.db'}?mode=ro", uri=True)
    try:
        names = [r[0] for r in db.execute("SELECT event FROM events ORDER BY event_id")]
    finally:
        db.close()
    assert "post-launch" in names and "post-state-change" in names
