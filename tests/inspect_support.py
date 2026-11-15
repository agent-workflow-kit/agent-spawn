"""Helpers for the list, show, ack and label tests: seeded agents in a real state database."""

from __future__ import annotations

import sqlite3
import time
from collections.abc import Sequence
from typing import Any

from agent_spawn.labels import LabelSet
from agent_spawn.store import Store
from tests.launch_support import Env

A1 = "ag-aaaaaaaa"
A2 = "ag-bbbbbbbb"
A3 = "ag-cccccccc"
SESSION = "cs-abcdefgh"


def seed_agent(
    env: Env,
    agent_id: str,
    *,
    labels: Sequence[str] = (),
    created: float | None = None,
    updated: float | None = None,
    runs: Sequence[dict[str, Any]] = (),
    **fields: Any,
) -> dict[str, Any]:
    """Insert an agent (parked, headless by default) with its labels and run rows."""
    values: dict[str, Any] = {
        "status": "parked",
        "name": f"agent {agent_id[3:9]}",
        "mode": "headless",
    }
    values.update(fields)
    with Store.open(env.state_dir) as store:
        row = store.insert_agent(agent_id, values, labels=LabelSet.from_texts(labels))
        if runs:
            with store.lease(f"agent:{agent_id}") as lease:
                for number, run in enumerate(runs, start=1):
                    store.put_run(lease, agent_id, number, run)
    if created is not None or updated is not None:
        # created and updated are not writable through the store API: set them directly
        db = sqlite3.connect(env.state_dir / "state.db")
        with db:
            if created is not None:
                db.execute("UPDATE agents SET created = ? WHERE agent_id = ?", (created, agent_id))
            stamp = updated if updated is not None else created
            db.execute("UPDATE agents SET updated = ? WHERE agent_id = ?", (stamp, agent_id))
        db.close()
    return row


def now() -> float:
    return time.time()


def labels_of(env: Env, agent_id: str) -> list[str]:
    """The stored labels of an agent (the store is closed again before returning)."""
    with Store.open(env.state_dir, create=False) as store:
        return store.get_labels(agent_id).texts()
