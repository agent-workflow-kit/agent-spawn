"""The state store: schema, migrations, files, leases, compare-and-set, labels, outbox, ids."""

from __future__ import annotations

import hashlib
import os
import sqlite3
import stat
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from agent_spawn import procs
from agent_spawn.ids import resolve_agent_id
from agent_spawn.labels import GrammarError, Label, LabelSet, parse_label
from agent_spawn.output import CliError
from agent_spawn.procs import ProcessIdentity
from agent_spawn.store import (
    AgentMissing,
    CasLost,
    IdConflict,
    LeaseHeld,
    LeaseLost,
    OutboxEvent,
    SchemaOlder,
    Store,
    StoreBusy,
    StoreCorrupt,
    StoreError,
    StoreMissing,
    default_state_dir,
)
from agent_spawn.store_schema import AGENT_COLUMNS, MIGRATIONS, RUN_COLUMNS, SCHEMA_VERSION
from tests.fake_gateway import BOOT, FakeGateway
from tests.procguard import ProcTracker

ID = "ag-abcdefgh"
OTHER = "ag-abcdefg2"


class ClockGateway(FakeGateway):
    """A manual clock with the real uid, for tests that open a real state dir."""

    def getuid(self) -> int:
        return os.getuid()


@pytest.fixture
def state_dir(tmp_path: Path) -> Path:
    return tmp_path / "state" / "agent-spawn"


@pytest.fixture
def store(state_dir: Path) -> Iterator[Store]:
    with Store.open(state_dir) as opened:
        yield opened


def new_agent(store: Store, agent_id: str = ID, **fields: object) -> dict[str, Any]:
    values: dict[str, object] = {"name": "agent", "status": "parked"}
    values.update(fields)
    return store.insert_agent(agent_id, values)


def me(store: Store) -> ProcessIdentity:
    return store.holder_identity()


def dead_identity(tracker: ProcTracker) -> ProcessIdentity:
    child = tracker.popen([sys.executable, "-c", "pass"])
    identity = tracker.identities[-1]
    child.wait()
    assert not procs.is_alive(identity)
    return identity


# -- schema, files, migrations ----------------------------------------------------------


def test_fresh_open_creates_private_wal_database(store: Store, state_dir: Path) -> None:
    assert stat.S_IMODE(os.stat(state_dir).st_mode) == 0o700
    assert stat.S_IMODE(os.stat(state_dir / "state.db").st_mode) == 0o600
    assert store._conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert store._conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    assert store.schema_version() == SCHEMA_VERSION


def test_wal_and_shm_files_are_private(store: Store, state_dir: Path) -> None:
    new_agent(store)
    for suffix in ("-wal", "-shm"):
        side = state_dir / f"state.db{suffix}"
        assert side.exists()
        assert stat.S_IMODE(os.stat(side).st_mode) == 0o600


def test_every_table_of_the_design_exists(store: Store) -> None:
    names = {r[0] for r in store._conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {
        "agents",
        "agent_runs",
        "agent_holds",
        "agent_labels",
        "label_mirror_pending",
        "relay_events",
        "events",
        "leases",
        "meta",
        "allocated_ids",
    } <= names


def test_column_specs_match_the_ddl(store: Store) -> None:
    for table, spec, extra in (
        ("agents", AGENT_COLUMNS, {"agent_id"}),
        ("agent_runs", RUN_COLUMNS, {"agent_id", "run"}),
    ):
        columns = {r["name"] for r in store._conn.execute(f"PRAGMA table_info({table})")}
        assert columns == set(spec) | extra


def test_reopen_is_idempotent_and_keeps_data(state_dir: Path) -> None:
    with Store.open(state_dir) as first:
        new_agent(first, name="kept")
    with Store.open(state_dir) as second:
        agent = second.get_agent(ID)
        assert agent is not None and agent["name"] == "kept"
        assert second.schema_version() == SCHEMA_VERSION


def _v2_add_column(conn: sqlite3.Connection) -> None:
    conn.execute("ALTER TABLE agents ADD COLUMN nickname TEXT")
    conn.execute("CREATE TABLE extra (k TEXT PRIMARY KEY)")


def test_additive_migration_keeps_rows_and_runs_once(state_dir: Path) -> None:
    with Store.open(state_dir) as old:
        new_agent(old, name="before")
    calls: list[int] = []

    def counted(conn: sqlite3.Connection) -> None:
        calls.append(1)
        _v2_add_column(conn)

    migrations = (*MIGRATIONS, counted)
    with Store.open(state_dir, migrations=migrations) as upgraded:
        assert upgraded.schema_version() == SCHEMA_VERSION + 1
        row = upgraded._conn.execute("SELECT name, nickname FROM agents").fetchone()
        assert (row["name"], row["nickname"]) == ("before", None)
    with Store.open(state_dir, migrations=migrations) as again:
        assert again.schema_version() == SCHEMA_VERSION + 1
    assert calls == [1]


def test_failed_migration_changes_nothing(state_dir: Path) -> None:
    with Store.open(state_dir):
        pass

    def broken(conn: sqlite3.Connection) -> None:
        conn.execute("CREATE TABLE half (k TEXT)")
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        Store.open(state_dir, migrations=(*MIGRATIONS, broken))
    with Store.open(state_dir) as after:
        assert after.schema_version() == SCHEMA_VERSION
        tables = {r[0] for r in after._conn.execute("SELECT name FROM sqlite_master")}
        assert "half" not in tables


def test_older_schema_without_migrate_is_schema_older(state_dir: Path) -> None:
    with Store.open(state_dir):
        pass
    with pytest.raises(SchemaOlder) as caught:
        Store.open(state_dir, migrate=False, migrations=(*MIGRATIONS, _v2_add_column))
    assert (caught.value.found, caught.value.expected) == (SCHEMA_VERSION, SCHEMA_VERSION + 1)


def test_newer_schema_is_refused_with_exit_12_and_writes_nothing(state_dir: Path) -> None:
    with Store.open(state_dir) as current:
        current._conn.execute(
            "UPDATE meta SET value = ? WHERE key = 'schema_version'", (str(SCHEMA_VERSION + 5),)
        )
    db = state_dir / "state.db"
    before = hashlib.sha256(db.read_bytes()).hexdigest()
    with pytest.raises(CliError) as caught:
        Store.open(state_dir)
    assert caught.value.code == "schema_newer"
    assert caught.value.exit_code == 12
    with pytest.raises(CliError):
        Store.open(state_dir, migrate=False, create=False)
    assert hashlib.sha256(db.read_bytes()).hexdigest() == before


def test_missing_database_is_not_created_when_not_asked(state_dir: Path, tmp_path: Path) -> None:
    with pytest.raises(StoreMissing):
        Store.open(state_dir, create=False)
    assert not state_dir.exists()
    state_dir.mkdir(parents=True)
    with pytest.raises(StoreMissing):
        Store.open(state_dir, create=False)
    assert not (state_dir / "state.db").exists()


def test_a_foreign_database_is_never_migrated(state_dir: Path) -> None:
    state_dir.mkdir(parents=True)
    foreign = sqlite3.connect(state_dir / "state.db")
    foreign.execute("CREATE TABLE notes (x)")
    foreign.commit()
    foreign.close()
    with pytest.raises(StoreCorrupt):
        Store.open(state_dir)


def test_a_non_database_file_is_corrupt_not_a_crash(state_dir: Path) -> None:
    state_dir.mkdir(parents=True)
    (state_dir / "state.db").write_bytes(b"this is not sqlite" * 100)
    with pytest.raises(StoreError):
        Store.open(state_dir)


def test_state_db_symlink_is_refused(state_dir: Path, tmp_path: Path) -> None:
    state_dir.mkdir(parents=True)
    target = tmp_path / "elsewhere.db"
    target.write_bytes(b"")
    (state_dir / "state.db").symlink_to(target)
    with pytest.raises(StoreError, match="symbolic link"):
        Store.open(state_dir)
    assert target.read_bytes() == b""


@pytest.mark.parametrize("suffix", ["-wal", "-shm", "-journal"])
def test_sidecar_symlink_is_refused(state_dir: Path, tmp_path: Path, suffix: str) -> None:
    with Store.open(state_dir):
        pass
    victim = tmp_path / "victim"
    victim.write_text("keep")
    (state_dir / f"state.db{suffix}").symlink_to(victim)
    with pytest.raises(StoreError, match="symbolic link"):
        Store.open(state_dir)
    assert victim.read_text() == "keep"


def test_a_symlinked_state_dir_is_followed_once(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real)
    with Store.open(link) as opened:
        new_agent(opened)
    assert (real / "state.db").exists()


def test_loose_modes_are_tightened(state_dir: Path) -> None:
    with Store.open(state_dir):
        pass
    os.chmod(state_dir, 0o755)
    os.chmod(state_dir / "state.db", 0o666)
    with Store.open(state_dir):
        pass
    assert stat.S_IMODE(os.stat(state_dir).st_mode) == 0o700
    assert stat.S_IMODE(os.stat(state_dir / "state.db").st_mode) == 0o600


def test_state_dir_is_created_private_even_with_a_loose_umask(state_dir: Path) -> None:
    old = os.umask(0)
    try:
        with Store.open(state_dir):
            pass
    finally:
        os.umask(old)
    assert stat.S_IMODE(os.stat(state_dir).st_mode) == 0o700
    assert stat.S_IMODE(os.stat(state_dir / "state.db").st_mode) == 0o600


def test_state_dir_that_is_a_file_is_an_error(tmp_path: Path) -> None:
    (tmp_path / "file").write_text("x")
    with pytest.raises(StoreError):
        Store.open(tmp_path / "file")


def test_path_with_nul_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(StoreError):
        Store.open(str(tmp_path) + "/a\0b")


def test_default_state_dir_follows_xdg_state_home() -> None:
    assert default_state_dir({"XDG_STATE_HOME": "/x/state", "HOME": "/h"}) == Path(
        "/x/state/agent-spawn"
    )
    assert default_state_dir({"HOME": "/h"}) == Path("/h/.local/state/agent-spawn")
    assert default_state_dir({"XDG_STATE_HOME": "relative", "HOME": "/h"}) == Path(
        "/h/.local/state/agent-spawn"
    )


# -- busy handling and transactions -----------------------------------------------------


def test_a_blocked_writer_gets_store_busy_after_the_timeout(state_dir: Path) -> None:
    with Store.open(state_dir) as first:
        new_agent(first)
        blocker = sqlite3.connect(state_dir / "state.db", isolation_level=None)
        blocker.execute("BEGIN IMMEDIATE")
        try:
            with Store.open(state_dir, busy_timeout=0.2) as second:
                started = time.monotonic()
                with pytest.raises(StoreBusy):
                    second.acquire_lease(f"agent:{ID}")
                assert time.monotonic() - started < 5
        finally:
            blocker.execute("ROLLBACK")
            blocker.close()
        first.acquire_lease(f"agent:{ID}")  # the lock is free again


def test_transaction_rolls_back_everything_on_error(store: Store) -> None:
    with pytest.raises(RuntimeError):
        with store.transaction():
            new_agent(store)
            raise RuntimeError("boom")
    assert store.get_agent(ID) is None
    assert not store.is_reserved(ID)


def test_nested_transaction_joins_the_outer_one(store: Store) -> None:
    with pytest.raises(RuntimeError):
        with store.transaction():
            store.record_relay_event("cs-1", "e1")
            with store.transaction():
                store.record_relay_event("cs-1", "e2")
            raise RuntimeError("boom")
    assert store.record_relay_event("cs-1", "e1")  # was rolled back, so new again


def test_text_that_cannot_be_encoded_writes_nothing(store: Store) -> None:
    with pytest.raises(StoreError):
        new_agent(store, name="lone \ud800 surrogate")
    assert store.get_agent(ID) is None
    assert not store.is_reserved(ID)


# -- agents: insert, read, hostile values ----------------------------------------------


def test_insert_and_read_back_every_kind_of_column(store: Store) -> None:
    identity = me(store)
    row = store.insert_agent(
        ID,
        {
            "name": "n",
            "status": "launching",
            "launcher": identity,
            "decision_ids": ["d1", "d2"],
            "settings": {"skill": "x", "n": 1},
            "tainted": False,
            "promote": None,
            "expect_run_after": 3,
            "ephemeral": True,
        },
    )
    assert row["launcher"] == identity
    assert row["decision_ids"] == ["d1", "d2"]
    assert row["tainted"] is False and row["promote"] is None and row["ephemeral"] is True
    assert row["expect_run_after"] == 3
    assert row["promotion_pending"] is False and row["supervisor_lost"] is False
    assert row["created"] == row["updated"]


@pytest.mark.parametrize(
    "name",
    [
        "x'); DROP TABLE agents; --",
        "Robert'); DELETE FROM leases;--",
        "‮\u0000nul and bidi",
        "emoji \U0001f600 é́",
        "a" * 100_000,
        "",
        "%_\\",
    ],
)
def test_hostile_text_is_stored_verbatim_as_a_parameter(store: Store, name: str) -> None:
    new_agent(store, name=name, cwd=name, repo=name)
    agent = store.get_agent(ID)
    assert agent is not None and agent["name"] == name and agent["cwd"] == name
    assert store._conn.execute("SELECT count(*) FROM agents").fetchone()[0] == 1


def test_unknown_or_hostile_column_names_are_rejected(store: Store) -> None:
    with pytest.raises(ValueError):
        new_agent(store, **{"name) VALUES ('x'); DROP TABLE agents; --": "x"})
    with pytest.raises(ValueError):
        store.insert_agent(ID, {"name": "n", "status": "parked", "agent_id": "ag-zzzzzzzz"})
    assert store._conn.execute("SELECT count(*) FROM agents").fetchone()[0] == 0


@pytest.mark.parametrize(
    "bad",
    ["ag-ABCDEFGH", "ag-abcdefg", "ag-abcdefghi", "ag-abcdef18", "x'; --", "", "ag-abcdefg\n"],
)
def test_malformed_ids_are_rejected(store: Store, bad: str) -> None:
    with pytest.raises(ValueError):
        new_agent(store, bad)


def test_values_of_the_wrong_python_type_are_rejected(store: Store) -> None:
    with pytest.raises(TypeError):
        new_agent(store, name=5)
    with pytest.raises(TypeError):
        new_agent(store, tainted=1)
    with pytest.raises(TypeError):
        new_agent(store, launcher={"pid": 1})
    assert store.get_agent(ID) is None


def test_a_bad_status_is_rejected_by_the_database(store: Store) -> None:
    with pytest.raises(StoreError):
        new_agent(store, status="zombie")
    assert store.get_agent(ID) is None


def test_a_record_needs_name_and_status(store: Store) -> None:
    with pytest.raises(ValueError):
        store.insert_agent(ID, {"status": "parked"})


def test_a_corrupt_stored_identity_is_store_corrupt(store: Store) -> None:
    new_agent(store)
    store._conn.execute("UPDATE agents SET supervisor = ?", ('{"pid": "x"}',))
    with pytest.raises(StoreCorrupt):
        store.get_agent(ID)


def test_list_agents_by_status(store: Store) -> None:
    new_agent(store, ID, status="parked")
    new_agent(store, OTHER, status="live")
    assert [a["agent_id"] for a in store.list_agents(statuses=["live"])] == [OTHER]
    assert [a["agent_id"] for a in store.list_agents(statuses=[])] == []
    assert len(store.list_agents()) == 2


# -- ids: AgentIndex ---------------------------------------------------------------------


def test_prefix_query_returns_live_and_tombstoned_records(store: Store) -> None:
    new_agent(store, "ag-abcdefgh")
    new_agent(store, "ag-abcdzzzz", status="torn_down")
    new_agent(store, "ag-abceaaaa")
    entries = {(e.agent_id, e.torn_down) for e in store.entries_with_prefix("ag-abcd")}
    assert entries == {("ag-abcdefgh", False), ("ag-abcdzzzz", True)}
    assert [e.agent_id for e in store.entries_with_prefix("ag-abcdefgh")] == ["ag-abcdefgh"]
    assert store.entries_with_prefix("ag-zzzz") == []


@pytest.mark.parametrize(
    "prefix", ["%", "ag-%", "ag-a_cd", "' OR 1=1 --", "\u0000", "ag-\U0010ffff"]
)
def test_hostile_prefixes_match_nothing(store: Store, prefix: str) -> None:
    new_agent(store)
    assert store.entries_with_prefix(prefix) == []


def test_prefix_query_uses_the_primary_key_index(store: Store) -> None:
    plan = store._conn.execute(
        "EXPLAIN QUERY PLAN SELECT agent_id, status FROM agents "
        "WHERE agent_id >= ? AND agent_id < ?",
        ("ag-a", "ag-b"),
    ).fetchall()
    assert any("SEARCH" in row["detail"] for row in plan)


def test_resolve_agent_id_over_the_store(store: Store) -> None:
    new_agent(store, "ag-abcdefgh")
    new_agent(store, "ag-abcdzzzz", status="torn_down")
    assert resolve_agent_id("abcde", store).agent_id == "ag-abcdefgh"
    assert resolve_agent_id("ag-abcdzzzz", store).torn_down
    with pytest.raises(CliError) as gone:
        resolve_agent_id("ag-qqqqqqqq", store)
    assert gone.value.exit_code == 10


def test_an_id_stays_reserved_after_the_record_is_deleted(store: Store) -> None:
    new_agent(store)
    with store.lease(f"agent:{ID}") as lease:
        assert store.delete_agent(lease, ID)
    assert store.get_agent(ID) is None
    assert store.is_reserved(ID)
    with pytest.raises(CliError):
        resolve_agent_id(ID, store)  # reserved but unknown
    with pytest.raises(IdConflict):
        new_agent(store)


def test_insert_conflict_does_not_touch_the_existing_record(store: Store) -> None:
    new_agent(store, name="first")
    with pytest.raises(IdConflict):
        new_agent(store, name="second")
    agent = store.get_agent(ID)
    assert agent is not None and agent["name"] == "first"


def test_insert_new_agent_reallocates_on_conflict(store: Store) -> None:
    new_agent(store, "ag-aaaaaaaa")
    row = store.insert_new_agent({"name": "x", "status": "launching"}, proposed="ag-aaaaaaaa")
    assert row["agent_id"] != "ag-aaaaaaaa"
    assert store.is_reserved(row["agent_id"])
    same = store.insert_new_agent({"name": "y", "status": "launching"}, proposed="ag-bbbbbbbb")
    assert same["agent_id"] == "ag-bbbbbbbb"


def test_reallocation_skips_reserved_ids_from_the_random_source(store: Store) -> None:
    new_agent(store, "ag-aaaaaaaa")
    # base32 of five zero bytes is "aaaaaaaa"; the next draw differs.
    draws = iter([b"\x00" * 5, b"\x00" * 5, b"\xff" * 5])
    row = store.insert_new_agent(
        {"name": "x", "status": "launching"}, random_bytes=lambda n: next(draws)
    )
    assert row["agent_id"] == "ag-77777777"


# -- leases -----------------------------------------------------------------------------


def test_lease_excludes_a_second_live_holder(store: Store, proc_tracker: ProcTracker) -> None:
    child = proc_tracker.popen([sys.executable, "-c", "import time; time.sleep(60)"])
    other = proc_tracker.identities[-1]
    first = store.acquire_lease(f"agent:{ID}", holder=other)
    with pytest.raises(LeaseHeld) as caught:
        store.acquire_lease(f"agent:{ID}")
    assert caught.value.holder == other
    assert first.holder.pid == child.pid
    assert store.lease_holder(f"agent:{ID}") == other


def test_lease_is_not_reentrant(store: Store) -> None:
    store.acquire_lease("reconcile")
    with pytest.raises(LeaseHeld):
        store.acquire_lease("reconcile")


def test_lease_wait_gives_up_with_lease_held(state_dir: Path) -> None:
    gateway = ClockGateway()
    with Store.open(state_dir, gateway=gateway, identity=ProcessIdentity(2, 7, "b")) as store:
        gateway.add(1, start=5)
        store.acquire_lease("reconcile", holder=ProcessIdentity(1, 5, BOOT))
        with pytest.raises(LeaseHeld):
            store.acquire_lease("reconcile", wait=1.0, poll=0.25)
        assert gateway.now >= 1000.0 + 1.0  # slept on the gateway clock, never wall time


def test_takeover_from_a_dead_pid(store: Store, proc_tracker: ProcTracker) -> None:
    dead = dead_identity(proc_tracker)
    store.acquire_lease(f"agent:{ID}", holder=dead)
    taken = store.acquire_lease(f"agent:{ID}")
    assert taken.took_over_from == dead
    assert store.lease_holder(f"agent:{ID}") == me(store)


def test_takeover_from_a_reused_pid_with_another_start_time(
    store: Store, proc_tracker: ProcTracker
) -> None:
    proc_tracker.popen([sys.executable, "-c", "import time; time.sleep(60)"])
    live = proc_tracker.identities[-1]
    reused = ProcessIdentity(live.pid, live.start_time + 1, live.boot_id)
    assert procs.is_alive(live) and not procs.is_alive(reused)
    store.acquire_lease("reconcile", holder=reused)
    assert store.acquire_lease("reconcile").took_over_from == reused
    # the real owner of that PID is untouched and still holds nothing
    assert procs.is_alive(live)


def test_takeover_when_the_boot_id_differs(store: Store, proc_tracker: ProcTracker) -> None:
    proc_tracker.popen([sys.executable, "-c", "import time; time.sleep(60)"])
    live = proc_tracker.identities[-1]
    other_boot = ProcessIdentity(live.pid, live.start_time, "another-boot")
    store.acquire_lease("reconcile", holder=other_boot)
    assert store.acquire_lease("reconcile").took_over_from == other_boot


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="/proc zombie state")
def test_takeover_from_a_zombie(store: Store, proc_tracker: ProcTracker) -> None:
    proc_tracker.popen([sys.executable, "-c", "pass"])  # never waited for: becomes a zombie
    zombie = proc_tracker.identities[-1]
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        info = procs.inspect(zombie.pid)
        if info is not None and info.state == "Z":
            break
        time.sleep(0.01)
    else:
        pytest.fail("child never became a zombie")
    store.acquire_lease(f"delivery:{ID}", holder=zombie)
    assert store.acquire_lease(f"delivery:{ID}").took_over_from == zombie


def test_an_unparseable_process_entry_is_never_dead(
    store: Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    holder = ProcessIdentity(os.getpid(), 12345, procs.read_boot_id())
    store.acquire_lease("reconcile", holder=holder)

    def broken(*args: object, **kwargs: object) -> None:
        raise procs.ProcessIdentityError("garbled stat")

    monkeypatch.setattr(procs, "inspect", broken)
    with pytest.raises(procs.ProcessIdentityError):
        store.acquire_lease("reconcile")
    assert store.lease_holder("reconcile") == holder  # unchanged


def test_a_corrupt_holder_row_is_never_taken_over(store: Store) -> None:
    store.acquire_lease("reconcile")
    store._conn.execute("UPDATE leases SET holder = 'garbage'")
    with pytest.raises(StoreCorrupt):
        store.acquire_lease("reconcile")


def test_release_only_deletes_the_holders_own_lease(
    store: Store, proc_tracker: ProcTracker
) -> None:
    dead = dead_identity(proc_tracker)
    stale = store.acquire_lease("reconcile", holder=dead)
    fresh = store.acquire_lease("reconcile")  # takes over
    assert not store.release_lease(stale)
    assert store.lease_holder("reconcile") == me(store)
    assert store.release_lease(fresh)
    assert store.lease_holder("reconcile") is None


@pytest.mark.parametrize(
    "key",
    ["", "agent:", "agent:ag-short", "agent:x'; --", "other", "reconcile ", "agent:AG-abcdefgh"],
)
def test_lease_keys_are_validated(store: Store, key: str) -> None:
    with pytest.raises(ValueError):
        store.acquire_lease(key)


def test_lease_context_manager_releases_on_error(store: Store) -> None:
    with pytest.raises(RuntimeError):
        with store.lease("reconcile"):
            raise RuntimeError("boom")
    assert store.lease_holder("reconcile") is None


# -- compare-and-set ---------------------------------------------------------------------


def test_cas_wins_and_writes_change_and_event_together(store: Store) -> None:
    new_agent(store)
    with store.lease(f"agent:{ID}") as lease:
        row = store.compare_and_set(
            lease,
            ID,
            expect_status="parked",
            changes={"status": "launching", "launcher": me(store), "expect_run_after": None},
            events=[OutboxEvent("status_change", {"from": "parked", "to": "launching"})],
        )
    assert row["status"] == "launching" and row["launcher"] == me(store)
    events = store.pending_events(ID)
    assert [(e["event"], e["payload"]) for e in events] == [
        ("status_change", {"from": "parked", "to": "launching"})
    ]


def test_cas_loser_changes_nothing_and_queues_no_event(store: Store) -> None:
    new_agent(store, status="live")
    with store.lease(f"agent:{ID}") as lease:
        with pytest.raises(CasLost) as caught:
            store.compare_and_set(
                lease,
                ID,
                expect_status="parked",
                changes={"status": "launching", "name": "changed"},
                events=[OutboxEvent("status_change")],
            )
    assert caught.value.status == "live"
    agent = store.get_agent(ID)
    assert agent is not None and agent["name"] == "agent" and agent["status"] == "live"
    assert store.pending_events(ID) == []


def test_cas_accepts_several_expected_statuses_and_extra_columns(store: Store) -> None:
    new_agent(store, status="launching", mode="headless")
    with store.lease(f"agent:{ID}") as lease:
        with pytest.raises(CasLost):
            store.compare_and_set(
                lease,
                ID,
                expect_status=("launching", "live"),
                expect={"mode": "interactive"},
                changes={"status": "parked"},
            )
        store.compare_and_set(
            lease,
            ID,
            expect_status=("launching", "live"),
            expect={"mode": "headless", "csession_id": None},
            changes={"status": "parked", "park_reason": "launch_failed"},
        )
    agent = store.get_agent(ID)
    assert agent is not None and agent["status"] == "parked"


def test_cas_on_a_missing_agent_is_a_loss_with_no_status(store: Store) -> None:
    with store.lease(f"agent:{ID}") as lease:
        with pytest.raises(CasLost) as caught:
            store.compare_and_set(lease, ID, expect_status="parked", changes={"status": "live"})
    assert caught.value.current is None and caught.value.status is None


def test_writes_need_the_agents_own_lease(store: Store) -> None:
    new_agent(store)
    new_agent(store, OTHER)
    with store.lease(f"agent:{OTHER}") as wrong:
        with pytest.raises(ValueError):
            store.update_agent(wrong, ID, {"name": "x"})
    with store.lease(f"delivery:{ID}") as delivery:
        with pytest.raises(ValueError):
            store.update_agent(delivery, ID, {"name": "x"})
    agent = store.get_agent(ID)
    assert agent is not None and agent["name"] == "agent"


def test_a_released_or_taken_over_lease_cannot_write(
    store: Store, proc_tracker: ProcTracker
) -> None:
    new_agent(store)
    lease = store.acquire_lease(f"agent:{ID}")
    store.release_lease(lease)
    with pytest.raises(LeaseLost):
        store.update_agent(lease, ID, {"name": "late"})
    dead = dead_identity(proc_tracker)
    stale = store.acquire_lease(f"agent:{ID}", holder=dead)
    store.acquire_lease(f"agent:{ID}")  # takes over from the dead holder
    with pytest.raises(LeaseLost):
        store.update_agent(stale, ID, {"name": "ghost"})


def test_immutable_columns_cannot_be_written(store: Store) -> None:
    new_agent(store)
    with store.lease(f"agent:{ID}") as lease:
        for column in ("agent_id", "created", "nope"):
            with pytest.raises(ValueError):
                store.update_agent(lease, ID, {column: "x"})
        with pytest.raises(ValueError):
            store.update_agent(lease, ID, {}, touch=False)


def test_update_touches_updated_unless_told_not_to(state_dir: Path) -> None:
    gateway = ClockGateway()
    with Store.open(state_dir, gateway=gateway, identity=ProcessIdentity(1, 1, "b")) as store:
        new_agent(store)
        gateway.now = 2000.0
        with store.lease(f"agent:{ID}") as lease:
            row = store.update_agent(lease, ID, {"park_reason": "done"})
            assert row["updated"] == 2000.0
            gateway.now = 3000.0
            row = store.update_agent(lease, ID, {"park_reason": "failed"}, touch=False)
            assert row["updated"] == 2000.0


def test_update_of_a_missing_agent(store: Store) -> None:
    with store.lease(f"agent:{ID}") as lease:
        with pytest.raises(AgentMissing):
            store.update_agent(lease, ID, {"name": "x"})


def test_a_failing_event_write_rolls_back_the_change(
    store: Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    new_agent(store)

    def explode(agent_id: str, events: object) -> None:
        raise sqlite3.OperationalError("disk I/O error")

    monkeypatch.setattr(store, "_add_events", explode)
    with store.lease(f"agent:{ID}") as lease:
        with pytest.raises(StoreError):
            store.compare_and_set(
                lease, ID, expect_status="parked", changes={"status": "launching"}
            )
    agent = store.get_agent(ID)
    assert agent is not None and agent["status"] == "parked"


def test_event_payloads_that_are_not_json_write_nothing(store: Store) -> None:
    new_agent(store)
    with store.lease(f"agent:{ID}") as lease:
        with pytest.raises(ValueError):
            store.compare_and_set(
                lease,
                ID,
                expect_status="parked",
                changes={"status": "launching"},
                events=[OutboxEvent("x", {"nan": float("nan")})],
            )
    agent = store.get_agent(ID)
    assert agent is not None and agent["status"] == "parked"


# -- labels ------------------------------------------------------------------------------


def lbl(*texts: str) -> list[Label]:
    return [parse_label(t) for t in texts]


def test_insert_with_labels_and_read_in_stored_order(store: Store) -> None:
    store.insert_agent(
        ID, {"name": "n", "status": "parked"}, labels=LabelSet.from_texts(["b=2", "a=1"])
    )
    assert store.get_labels(ID).texts() == ["b=2", "a=1"]


def test_edit_labels_add_remove_and_noops(store: Store) -> None:
    new_agent(store)
    with store.lease(f"agent:{ID}") as lease:
        assert store.edit_labels(lease, ID, "add", lbl("a=1", "b=2", "a=1")).changed
        assert not store.edit_labels(lease, ID, "add", lbl("a=1")).changed
        assert not store.edit_labels(lease, ID, "remove", lbl("zzz=1")).changed
        assert store.edit_labels(lease, ID, "remove", lbl("a=1")).changed
        store.edit_labels(lease, ID, "add", lbl("c=3"))
    assert store.get_labels(ID).texts() == ["b=2", "c=3"]


def test_label_cap_is_checked_against_the_stored_set(store: Store) -> None:
    new_agent(store)
    with store.lease(f"agent:{ID}") as lease:
        store.edit_labels(lease, ID, "add", lbl(*[f"k{i}=v" for i in range(60)]))
        with pytest.raises(GrammarError):
            store.edit_labels(lease, ID, "add", lbl(*[f"x{i}=v" for i in range(5)]))
        assert len(store.get_labels(ID)) == 60
        change = store.edit_labels(lease, ID, "add_run", lbl("job=j1"))
        assert change.changed
        store.edit_labels(lease, ID, "add", lbl("m1=v", "m2=v", "m3=v"))
        skipped = store.edit_labels(lease, ID, "add_run", lbl("job=j2"))
        assert skipped.skipped and len(store.get_labels(ID)) == 64


def test_mirror_queue_records_changes_and_cancels_opposites(store: Store) -> None:
    new_agent(store)
    with store.lease(f"agent:{ID}") as lease:
        store.edit_labels(lease, ID, "add", lbl("a=1", "b=2"))
        store.edit_labels(lease, ID, "remove", lbl("a=1"))  # never mirrored: cancels
        pending = store.mirror_pending(ID)
        assert pending == [(Label("b", "2"), "add")]
        store.edit_labels(lease, ID, "remove", lbl("b=2"))
        assert store.mirror_pending(ID) == []
        store.edit_labels(lease, ID, "add", lbl("c=3"))
        store.mirror_done(lease, ID, store.mirror_pending(ID))
        assert store.mirror_pending(ID) == []


def test_labels_with_hostile_values_roundtrip(store: Store) -> None:
    new_agent(store)
    with store.lease(f"agent:{ID}") as lease:
        store.edit_labels(lease, ID, "add", lbl("k=v,w:x/y#z@q+r=s"))
    assert store.get_labels(ID).texts() == ["k=v,w:x/y#z@q+r=s"]


def test_edit_labels_on_a_missing_agent(store: Store) -> None:
    with store.lease(f"agent:{ID}") as lease:
        with pytest.raises(AgentMissing):
            store.edit_labels(lease, ID, "add", lbl("a=1"))


def test_find_agents_ands_the_pairs_and_skips_tombstones(store: Store) -> None:
    for agent_id, status, texts in (
        (ID, "parked", ["a=1", "b=2"]),
        (OTHER, "parked", ["a=1"]),
        ("ag-abcdefg3", "torn_down", ["a=1", "b=2"]),
    ):
        store.insert_agent(
            agent_id, {"name": "n", "status": status}, labels=LabelSet.from_texts(texts)
        )
    assert store.find_agents(lbl("a=1", "b=2")) == [ID]
    assert store.find_agents(lbl("a=1")) == [OTHER, ID]
    assert store.find_agents(lbl("a=1", "b=2"), include_torn_down=True) == ["ag-abcdefg3", ID]
    assert store.find_agents(lbl("a=1", "b=3")) == []
    assert store.find_agents(lbl("a=1", "a=1")) == [OTHER, ID]
    assert len(store.find_agents([])) == 2


def test_deleting_an_agent_removes_its_dependents(store: Store) -> None:
    new_agent(store)
    with store.lease(f"agent:{ID}") as lease:
        store.edit_labels(lease, ID, "add", lbl("a=1"))
        store.put_run(lease, ID, 1, {"mode": "headless"})
        store.put_hold(lease, ID, "wt-1", "/p", owned=True)
        store.update_agent(lease, ID, {"status": "live"}, events=[OutboxEvent("e")])
        assert store.delete_agent(lease, ID)
        assert not store.delete_agent(lease, ID)
    for table in ("agent_labels", "agent_runs", "agent_holds", "events", "label_mirror_pending"):
        assert store._conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0] == 0


def test_foreign_keys_are_enforced(store: Store) -> None:
    with pytest.raises(sqlite3.IntegrityError):
        store._conn.execute(
            "INSERT INTO agent_labels (agent_id, pos, key, value) "
            "VALUES ('ag-aaaaaaaa', 0, 'a', 'b')"
        )


# -- runs and holds ----------------------------------------------------------------------


def test_runs_are_created_updated_and_ordered(store: Store) -> None:
    new_agent(store)
    with store.lease(f"agent:{ID}") as lease:
        store.put_run(lease, ID, 2, {"mode": "interactive", "pane_proc": me(store)})
        store.put_run(lease, ID, 1, {"mode": "headless", "reads_third_party": True})
        row = store.put_run(
            lease, ID, 1, {"outcome": {"outcome": "done"}, "exit_code": 0, "auto_resumed": True}
        )
        with pytest.raises(ValueError):
            store.put_run(lease, ID, 1, {"agent_id": "x"})
    assert row["mode"] == "headless" and row["reads_third_party"] is True
    assert row["outcome"] == {"outcome": "done"} and row["auto_resumed"] is True
    assert [r["run"] for r in store.get_runs(ID)] == [1, 2]
    assert store.get_run(ID, 2) is not None and store.get_run(ID, 9) is None


def test_holds_keep_order_and_release_and_replace(store: Store) -> None:
    new_agent(store, wt_record_id="wt-1")
    with store.lease(f"agent:{ID}") as lease:
        assert store.put_hold(lease, ID, "wt-1", "/a", owned=True)
        assert not store.put_hold(lease, ID, "wt-1", "/a", owned=True)
        assert store.put_hold(lease, ID, "wt-2", "/b", owned=False)
        store.release_hold(lease, ID, "wt-2")
        assert [h["record_id"] for h in store.get_holds(ID)] == ["wt-1"]
        assert len(store.get_holds(ID, include_released=True)) == 2
        store.replace_hold_record(lease, ID, "wt-1", "wt-9", "/c")
    holds = store.get_holds(ID)
    assert holds == [{"record_id": "wt-9", "path": "/c", "owned": True, "released": False}]
    agent = store.get_agent(ID)
    assert agent is not None and agent["wt_record_id"] == "wt-9" and agent["wt_path"] == "/c"


# -- relay dedup and outbox --------------------------------------------------------------


def test_relay_events_are_deduplicated_and_pruned_after_seven_days(state_dir: Path) -> None:
    gateway = ClockGateway()
    with Store.open(state_dir, gateway=gateway, identity=ProcessIdentity(1, 1, "b")) as store:
        assert store.record_relay_event("cs-1", "e1")
        assert not store.record_relay_event("cs-1", "e1")
        assert store.record_relay_event("cs-2", "e1")
        gateway.now += 6 * 86400
        assert store.prune_relay_events() == 0
        gateway.now += 2 * 86400
        assert store.prune_relay_events() == 2
        assert store.record_relay_event("cs-1", "e1")


def test_outbox_is_ordered_per_agent_and_delivery_needs_its_lease(store: Store) -> None:
    new_agent(store)
    new_agent(store, OTHER)
    store.add_events(ID, [OutboxEvent("one", {"n": 1}), OutboxEvent("two", {"n": 2})])
    store.add_events(OTHER, [OutboxEvent("other")])
    pending = store.pending_events(ID)
    assert [e["event"] for e in pending] == ["one", "two"]
    assert pending[0]["event_id"] < pending[1]["event_id"]
    with store.lease(f"agent:{ID}") as wrong:
        with pytest.raises(ValueError):
            store.mark_delivered(wrong, ID, pending[0]["event_id"])
    with store.lease(f"delivery:{ID}") as delivery:
        assert store.mark_delivered(delivery, ID, pending[0]["event_id"])
        assert not store.mark_delivered(delivery, ID, pending[0]["event_id"])
        assert not store.mark_delivered(delivery, ID, store.pending_events(OTHER)[0]["event_id"])
    assert [e["event"] for e in store.pending_events(ID)] == ["two"]
    assert len(store.pending_events(OTHER)) == 1


def test_events_for_an_unknown_agent_are_rejected_by_the_foreign_key(store: Store) -> None:
    with pytest.raises(StoreError):
        store.add_events(ID, [OutboxEvent("x")])
