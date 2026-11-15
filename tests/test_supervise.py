"""The supervisor's signal handling, in process (the real chain is in test_launch.py)."""

from __future__ import annotations

import os
import signal
from pathlib import Path

import pytest

from agent_spawn import cli, procs
from agent_spawn.ids import new_agent_id
from agent_spawn.store import Store
from agent_spawn.supervise import Signals, Supervisor
from tests.conftest import Isolation
from tests.fakebin.harness import Fakebin
from tests.session_backend_support import FakebinGateway


def test_signal_before_exec_cancels_the_pending_run_and_parks(
    fakebin: Fakebin, isolation_env: Isolation, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # `run` gets the fakes' environment; monkeypatch restores the default gateway at teardown.
    monkeypatch.setattr("agent_spawn.gateway._default", FakebinGateway(fakebin))
    state = tmp_path / "state"
    session = fakebin.seed_session("cs-abcdefgh", adopted=False, final=False, labels=[])
    cs = str(fakebin.bin_dir / "claude-session")
    with Store.open(state) as store:
        me = store.holder_identity()
        agent_id = new_agent_id(store)
        row = store.insert_agent(
            agent_id,
            {
                "status": "launching",
                "name": "agent test",
                "mode": "headless",
                "supervisor": me,
                "csession_id": session,
                "run": 1,
                "claude_session_path": cs,
                "state_dir": str(state),
            },
        )
        with store.lease(f"agent:{agent_id}") as lease:
            store.put_run(lease, agent_id, 1, {"outcome_token": "tok", "mode": "headless"})
    assert row["status"] == "launching"
    config = str(tmp_path / "config.yaml")
    inv = cli.parse_invocation(
        ["supervise", agent_id, "--state-dir", str(state), "--config", config]
    )
    signals = Signals()
    signals.received.append(signal.SIGTERM)  # arrived before `exec` was spawned
    outcome = Supervisor(inv, signals).run()
    assert outcome.exit_code == 0
    with Store.open(state, create=False) as store:
        parked = store.get_agent(agent_id)
    assert parked is not None
    assert parked["status"] == "parked" and parked["park_reason"] == "killed"
    assert parked["needs_attention"] is None
    assert [c.argv[0] for c in fakebin.invocations("claude-session")][:1] == ["stop"]
    assert not [c for c in fakebin.invocations("claude-session") if c.argv[0] == "exec"]
    assert procs.is_alive(me) and os.getpid() == me.pid  # nothing signaled this process
