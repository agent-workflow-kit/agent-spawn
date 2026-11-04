"""Real processes racing on the state store: exactly one winner, and crashes lose nothing."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from agent_spawn import procs
from agent_spawn.store import OutboxEvent, Store
from tests.procguard import ProcTracker

WORKER = str(Path(__file__).parent / "store_worker.py")
ID = "ag-abcdefgh"
RACERS = 5


class Race:
    def __init__(self, tracker: ProcTracker, state_dir: Path, tmp: Path) -> None:
        self.tracker = tracker
        self.state_dir = state_dir
        self.go = tmp / "go"
        self.release = tmp / "release"
        self.children: list[subprocess.Popen[bytes]] = []

    def start(self, mode: str, count: int, agent_id: str = ID) -> None:
        for _ in range(count):
            child = self.tracker.popen(
                [
                    sys.executable,
                    WORKER,
                    str(self.state_dir),
                    mode,
                    str(self.go),
                    str(self.release),
                    agent_id,
                ],
                stdout=subprocess.PIPE,
                stdin=subprocess.DEVNULL,
            )
            self.children.append(child)
        for child in self.children:
            assert child.stdout is not None
            line = json.loads(child.stdout.readline())
            assert line == {"event": "ready"}

    def fire(self) -> list[dict[str, object]]:
        self.go.write_text("go")
        results: list[dict[str, object]] = []
        for child in self.children:
            assert child.stdout is not None
            results.append(json.loads(child.stdout.readline()))
        return results

    def finish(self) -> None:
        self.release.write_text("release")
        for child in self.children:
            assert child.wait(timeout=30) in (0, 9, -9)


@pytest.fixture
def race(proc_tracker: ProcTracker, tmp_path: Path) -> Race:
    return Race(proc_tracker, tmp_path / "state" / "agent-spawn", tmp_path)


def seed(state_dir: Path, *, status: str = "parked") -> None:
    with Store.open(state_dir) as store:
        store.insert_agent(ID, {"name": "n", "status": status})


def test_lease_race_has_exactly_one_winner(race: Race) -> None:
    seed(race.state_dir)
    race.start("lease", RACERS)
    results = race.fire()
    outcomes = sorted(str(r["result"]) for r in results)
    assert outcomes == ["busy"] * (RACERS - 1) + ["won"]
    with Store.open(race.state_dir) as store:
        holder = store.lease_holder(f"agent:{ID}")
        assert holder is not None and procs.is_alive(holder)
    race.finish()
    with Store.open(race.state_dir) as store:
        assert store.lease_holder(f"agent:{ID}") is None


def test_a_winner_that_dies_frees_its_lease_at_once(race: Race) -> None:
    seed(race.state_dir)
    race.start("lease", 2)
    results = race.fire()
    assert sorted(str(r["result"]) for r in results) == ["busy", "won"]
    winner = race.children[[r["result"] for r in results].index("won")]
    with Store.open(race.state_dir) as store:
        holder = store.lease_holder(f"agent:{ID}")
        assert holder is not None and holder.pid == winner.pid
        winner.kill()  # the exact child this test started
        winner.wait(timeout=30)
        taken = store.acquire_lease(f"agent:{ID}")
        assert taken.took_over_from == holder
    race.finish()


def test_compare_and_set_race_has_exactly_one_winner(race: Race) -> None:
    seed(race.state_dir)
    race.start("cas", RACERS)
    results = race.fire()
    outcomes = [str(r["result"]) for r in results]
    assert outcomes.count("won") == 1
    assert outcomes.count("lost") == RACERS - 1
    assert {r.get("status") for r in results if r["result"] == "lost"} == {"launching"}
    race.finish()
    with Store.open(race.state_dir) as store:
        agent = store.get_agent(ID)
        assert agent is not None and agent["status"] == "launching"
        assert agent["launcher"] is not None
        assert len(store.pending_events(ID)) == 1  # the change and its event, once


def test_id_allocation_race_gives_every_process_its_own_id(race: Race) -> None:
    race.state_dir.mkdir(parents=True)
    with Store.open(race.state_dir):
        pass
    race.start("insert", RACERS)
    results = race.fire()
    ids = [str(r["agent_id"]) for r in results]
    assert len(set(ids)) == RACERS
    assert ID in ids  # one racer kept the proposed id, the others re-allocated
    race.finish()
    with Store.open(race.state_dir) as store:
        assert all(store.is_reserved(i) for i in ids)
        assert len(store.list_agents()) == RACERS


def test_concurrent_first_opens_create_and_migrate_once(race: Race) -> None:
    race.start("open", RACERS)
    results = race.fire()
    assert {r["version"] for r in results} == {1}
    race.finish()
    with Store.open(race.state_dir) as store:
        assert store._conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert store._conn.execute("SELECT count(*) FROM meta").fetchone()[0] == 1


def test_a_process_that_dies_mid_transaction_loses_the_whole_change(race: Race) -> None:
    seed(race.state_dir)
    race.start("crash", 1)
    assert race.fire() == [{"result": "crashing"}]
    race.children[0].wait(timeout=30)
    assert race.children[0].returncode == 9
    with Store.open(race.state_dir) as store:
        agent = store.get_agent(ID)
        assert agent is not None and agent["status"] == "parked"  # neither change nor event
        assert store.pending_events(ID) == []
        assert store._conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        # the database is not left locked, and the dead process's lease is taken over
        lease = store.acquire_lease(f"agent:{ID}")
        assert lease.took_over_from is not None
        store.update_agent(lease, ID, {"status": "live"}, events=[OutboxEvent("ok")])
        store.release_lease(lease)
