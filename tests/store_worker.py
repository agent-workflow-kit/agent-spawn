"""A process that races on the state store (run by ``test_store_race.py``, never collected).

``store_worker.py <state dir> <mode> <go file> <release file> [<agent id>]``. It prints ``ready``,
waits for the go file, does its one operation, prints one JSON result line, then (for a winner
that must stay alive) waits for the release file.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from agent_spawn.store import (
    CasLost,
    LeaseHeld,
    OutboxEvent,
    Store,
)


def wait_for(path: Path, timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    while not path.exists():
        if time.monotonic() > deadline:
            sys.exit(3)
        time.sleep(0.005)


def say(**fields: object) -> None:
    print(json.dumps(fields), flush=True)


def loop(state_dir: str, go_file: Path, seconds: float = 3.0) -> None:
    """Open, read, write and close in a loop: every last close deletes the WAL sidecars."""
    say(event="ready")
    wait_for(go_file)
    failures: list[str] = []
    opens = 0
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        try:
            with Store.open(state_dir) as store:
                store.get_agent("ag-abcdefgh")
                store.record_relay_event(f"loop-{os.getpid()}", str(opens))
            opens += 1
        except Exception as exc:  # every failure is reported, none is retried
            failures.append(f"{type(exc).__name__}: {exc}")
    say(result="looped", opens=opens, failures=failures[:5])


def main() -> None:
    state_dir, mode, go, release = sys.argv[1:5]
    agent_id = sys.argv[5] if len(sys.argv) > 5 else "ag-aaaaaaaa"
    go_file, release_file = Path(go), Path(release)
    if mode == "open":
        say(event="ready")
        wait_for(go_file)
        with Store.open(state_dir) as store:
            say(result="opened", version=store.schema_version())
        return
    if mode == "loop":
        loop(state_dir, go_file)
        return
    with Store.open(state_dir) as store:
        say(event="ready")
        wait_for(go_file)
        if mode == "lease":
            try:
                lease = store.acquire_lease(f"agent:{agent_id}")
            except LeaseHeld:
                say(result="busy")
                return
            say(result="won")
            wait_for(release_file)
            store.release_lease(lease)
        elif mode == "cas":
            with store.lease(f"agent:{agent_id}", wait=20.0) as lease:
                try:
                    store.compare_and_set(
                        lease,
                        agent_id,
                        expect_status="parked",
                        changes={"status": "launching", "launcher": store.holder_identity()},
                        events=[OutboxEvent("status_change", {"by": os.getpid()})],
                    )
                except CasLost as lost:
                    say(result="lost", status=lost.status)
                else:
                    say(result="won")
        elif mode == "insert":
            row = store.insert_new_agent(
                {"name": f"racer {os.getpid()}", "status": "launching"},
                proposed=agent_id,
            )
            say(result="inserted", agent_id=row["agent_id"])
        elif mode == "crash":
            with store.lease(f"agent:{agent_id}") as lease:
                store._conn.execute("BEGIN IMMEDIATE")
                store._conn.execute(
                    "UPDATE agents SET status = 'live' WHERE agent_id = ?", (agent_id,)
                )
                store._add_events(agent_id, [OutboxEvent("half", {})])
                say(result="crashing")
                del lease
                os._exit(9)  # no commit, no rollback, no close
        else:
            sys.exit(4)


if __name__ == "__main__":
    main()
