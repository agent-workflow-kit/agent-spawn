"""Upgrades with live agents (interface.md, Upgrades and State relay; issue #35): a migration
while a supervisor is running, final bookkeeping after lost relay events, an older
``_session-event`` against another schema, and stored paths that went stale."""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from agent_spawn import chain, launch, procs, promotion
from agent_spawn.agent_lookup import backend_for
from agent_spawn.liveness import Liveness
from agent_spawn.output import CliError
from agent_spawn.session_backend import RunOutcome
from agent_spawn.stop import Stop
from agent_spawn.store import Store
from agent_spawn.store_schema import MIGRATIONS
from agent_spawn.supervise import RunCtx
from agent_spawn.teardown import Teardown
from tests.conftest import Isolation
from tests.fakebin.harness import Fakebin
from tests.test_supervise_inprocess import Rig, SpawnGateway, final_of

# --- a later, additive schema, as a future release would ship it -------------------------------


def _v2(conn: sqlite3.Connection) -> None:
    conn.execute("ALTER TABLE agents ADD COLUMN future_note TEXT")
    conn.execute("ALTER TABLE agent_runs ADD COLUMN future_flag INTEGER NOT NULL DEFAULT 0")
    conn.execute("CREATE TABLE future_things (thing_id TEXT PRIMARY KEY, body TEXT)")


MIGRATIONS_V2 = (*MIGRATIONS, _v2)


@pytest.fixture
def rig_for(
    fakebin: Fakebin, isolation_env: Isolation, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> Callable[..., Rig]:
    def make(**agent: object) -> Rig:
        return Rig(fakebin, monkeypatch, tmp_path, **agent)

    return make


def migrate_to_v2(state: Path) -> None:
    with Store.open(state, create=False, migrations=MIGRATIONS_V2) as store:
        assert store.schema_version() == 2


# --- a live agent survives a migration -----------------------------------------------------------


def test_a_live_agent_survives_a_migration_and_its_supervisor_finishes(
    rig_for: Callable[..., Rig], fakebin: Fakebin, monkeypatch: pytest.MonkeyPatch
) -> None:
    rig = rig_for()
    migrated: list[bool] = []

    class Gateway(SpawnGateway):
        def spawn(self, argv: Sequence[str], **kwargs: Any) -> subprocess.Popen[bytes]:
            # The supervisor has opened its database at schema 1 and checked its record; a newer
            # agent-spawn migrates the database before `exec` is spawned.
            if not migrated:
                migrate_to_v2(rig.state)
                migrated.append(True)
            return super().spawn(argv, **kwargs)

    rig.supervisor.gateway = Gateway(fakebin)
    outcome = rig.supervisor.run()
    assert migrated and outcome.exit_code == 0
    with Store.open(rig.state, create=False, migrations=MIGRATIONS_V2) as store:
        row = store.get_agent(rig.agent_id)
    assert row is not None
    assert row["status"] == "parked" and row["park_reason"] == "done"
    assert row["last_outcome"]["outcome"] == "done"
    db = sqlite3.connect(f"file:{rig.state / 'state.db'}?mode=ro", uri=True)
    try:
        assert db.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone() == (
            "2",
        )
        # the supervisor's writes landed in the migrated database, beside the new column
        assert db.execute(
            "SELECT outcome IS NOT NULL, future_flag FROM agent_runs WHERE agent_id = ?",
            (rig.agent_id,),
        ).fetchone() == (1, 0)
    finally:
        db.close()


Layout = dict[str, Any]


def layout_of(db: sqlite3.Connection) -> Layout:
    """What an older process depends on: each table's columns (type, NOT NULL, default) and its
    CHECK count, every index (unique or not) and every trigger."""
    master = db.execute("SELECT type, name, tbl_name, sql FROM sqlite_master").fetchall()
    tables = {r[1]: r[3] or "" for r in master if r[0] == "table"}
    unique = {
        row[1]: bool(row[2])
        for table in tables
        for row in db.execute(f"PRAGMA index_list('{table}')")
    }
    return {
        "columns": {
            table: {r[1]: (r[2], r[3], r[4]) for r in db.execute(f"PRAGMA table_info('{table}')")}
            for table in tables
        },
        "checks": {table: sql.upper().count("CHECK") for table, sql in tables.items()},
        "indexes": {r[1]: (unique.get(r[1], False), r[3], r[2]) for r in master if r[0] == "index"},
        "triggers": {r[1]: r[3] for r in master if r[0] == "trigger"},
    }


def additivity_problems(before: Layout, after: Layout) -> list[str]:
    """Why ``after`` is not an additive change of ``before`` (empty when it is)."""
    problems: list[str] = []
    for table, columns in before["columns"].items():
        if table not in after["columns"]:
            problems.append(f"{table} was dropped")
            continue
        for name, (kind, _notnull, _default) in columns.items():
            if name not in after["columns"][table]:
                problems.append(f"{table}.{name} was dropped")
            elif after["columns"][table][name][0] != kind:
                problems.append(f"{table}.{name} changed type")
        for name, (_kind, notnull, default) in after["columns"][table].items():
            if name not in columns and notnull and default is None:
                problems.append(f"{table}.{name} needs a default")
    for table, count in after["checks"].items():
        if count > before["checks"].get(table, 0):
            problems.append(f"{table} gained a CHECK constraint")
    for name, (is_unique, sql, table) in after["indexes"].items():
        if name not in before["indexes"] and is_unique and table in before["columns"]:
            # (a new table's own keys constrain nobody's older writes)
            problems.append(f"{name} is a new UNIQUE index")
        if name in before["indexes"] and before["indexes"][name] != (is_unique, sql, table):
            problems.append(f"{name} changed")
    for name in before["indexes"]:
        if name not in after["indexes"]:
            problems.append(f"index {name} was dropped")
    for name, sql in after["triggers"].items():
        if before["triggers"].get(name) != sql:
            problems.append(f"trigger {name} is new or changed")
    return problems


def test_every_migration_is_additive(tmp_path: Path) -> None:
    """A shipped schema stays readable and writable by the older agent-spawn processes that live
    agents keep running: nothing is dropped or retyped, a new column is nullable or has a
    default (an older INSERT does not name it), and no new UNIQUE index, trigger or CHECK can
    reject an older process's write."""
    state = tmp_path / "state"
    with Store.open(state, migrations=MIGRATIONS[:1]):
        pass
    before = _layout(state)
    for upto in range(2, len(MIGRATIONS_V2) + 1):
        with Store.open(state, migrations=MIGRATIONS_V2[:upto]):
            pass
        after = _layout(state)
        assert additivity_problems(before, after) == []
        before = after


BASE = "CREATE TABLE t (a INTEGER PRIMARY KEY, b TEXT)"


@pytest.mark.parametrize(
    ("change", "expected"),
    [
        ("ALTER TABLE t DROP COLUMN b", "t.b was dropped"),
        ("DROP TABLE t", "t was dropped"),
        (
            "ALTER TABLE t DROP COLUMN b; ALTER TABLE t ADD COLUMN b INTEGER",
            "t.b changed type",
        ),
        ("ALTER TABLE t ADD COLUMN c TEXT NOT NULL", "t.c needs a default"),
        ("CREATE UNIQUE INDEX t_b ON t (b)", "t_b is a new UNIQUE index"),
        (
            "CREATE TRIGGER t_ins BEFORE INSERT ON t BEGIN SELECT RAISE(ABORT, 'no'); END",
            "trigger t_ins is new or changed",
        ),
        ("ALTER TABLE t ADD COLUMN c INTEGER CHECK (c > 0)", "t gained a CHECK constraint"),
    ],
)
def test_the_additivity_check_catches_a_breaking_change(change: str, expected: str) -> None:
    db = sqlite3.connect(":memory:")
    db.execute(BASE)
    before = layout_of(db)
    db.executescript(change)
    assert expected in additivity_problems(before, layout_of(db))


@pytest.mark.parametrize(
    "change",
    [
        "ALTER TABLE t ADD COLUMN c TEXT",
        "ALTER TABLE t ADD COLUMN c INTEGER NOT NULL DEFAULT 0",
        "CREATE INDEX t_b ON t (b)",
        "CREATE TABLE u (x TEXT)",
    ],
)
def test_the_additivity_check_accepts_an_additive_change(change: str) -> None:
    db = sqlite3.connect(":memory:")
    db.execute(BASE)
    before = layout_of(db)
    db.executescript(change)
    assert additivity_problems(before, layout_of(db)) == []


def _layout(state: Path) -> Layout:
    db = sqlite3.connect(f"file:{state / 'state.db'}?mode=ro", uri=True)
    try:
        return layout_of(db)
    finally:
        db.close()


# --- the final bookkeeping after lost relay events -------------------------------------------


def seed_stale_state(rig: Rig, **agent: object) -> None:
    """The agent and its run as the relay left them: the final events never arrived."""
    with Store.open(rig.state, create=False) as store:
        with store.lease(f"agent:{rig.agent_id}") as lease:
            store.update_agent(
                lease,
                rig.agent_id,
                {
                    "session_state": "working",
                    "activity": "working",
                    "needs_input_reason": "permission",
                    **agent,
                },
            )
            store.put_run(
                lease,
                rig.agent_id,
                1,
                {"session_state": "working", "activity": "working"},
            )


def test_final_bookkeeping_records_the_final_state_after_lost_relay_events(
    rig_for: Callable[..., Rig],
) -> None:
    rig = rig_for()
    seed_stale_state(rig)
    rig.supervisor.run()
    row = rig.agent()
    assert (row["session_state"], row["activity"], row["needs_input_reason"]) == (
        "exited",
        None,
        None,
    )
    with Store.open(rig.state, create=False) as store:
        run = store.get_run(rig.agent_id, 1)
    assert run is not None
    assert (run["session_state"], run["activity"], run["needs_input_reason"]) == (
        "exited",
        None,
        None,
    )


def test_final_bookkeeping_follows_the_outcome_not_a_fixed_state(
    rig_for: Callable[..., Rig], fakebin: Fakebin
) -> None:
    rig = rig_for()
    seed_stale_state(rig)
    fakebin.script("claude-session", "exec", outcome="timeout")
    rig.supervisor.run()
    assert rig.agent()["session_state"] == "timeout"


def test_final_bookkeeping_keeps_what_the_relay_already_recorded(
    rig_for: Callable[..., Rig],
) -> None:
    """The relay delivered the final event: nothing is rewritten (the relay's own activity and
    reason stand)."""
    rig = rig_for()
    seed_stale_state(rig, session_state="exited", activity="idle", needs_input_reason=None)
    rig.put_run(1, session_state="exited", activity="idle")
    rig.supervisor.run()
    row = rig.agent()
    assert (row["session_state"], row["activity"]) == ("exited", "idle")


# --- an older _session-event against another schema --------------------------------------------


def table_counts(state: Path) -> dict[str, int]:
    db = sqlite3.connect(f"file:{state / 'state.db'}?mode=ro", uri=True)
    try:
        names = [r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type = 'table'")]
        return {n: db.execute(f'SELECT COUNT(*) FROM "{n}"').fetchone()[0] for n in names}
    finally:
        db.close()


def relay_payload() -> bytes:
    return json.dumps({"object_id": "cs-abcdefgh"}).encode()


def test_an_older_relay_drops_the_event_against_a_migrated_database(
    rig_for: Callable[..., Rig], isolation_env: Isolation
) -> None:
    rig = rig_for()
    migrate_to_v2(rig.state)
    wal = rig.state / "state.db-wal"
    assert not wal.exists() or wal.stat().st_size == 0
    before = (rig.state / "state.db").read_bytes()
    counts_before = table_counts(rig.state)
    done = subprocess.run(
        [
            sys.executable,
            "-m",
            "agent_spawn",
            "--state-dir",
            str(rig.state),
            "--config",
            str(rig.state.parent / "config.yaml"),
            "_session-event",
        ],
        input=relay_payload(),
        capture_output=True,
        env={**isolation_env.env},
        timeout=60,
        check=False,
    )
    assert done.returncode == 0 and done.stdout == b""
    log = [
        json.loads(line)
        for line in (rig.state / "log.jsonl").read_text().splitlines()
        if line.strip()
    ]
    schema = [r for r in log if r["event"] == "schema_older"]
    assert len(schema) == 1 and schema[0]["details"] == {"found": 2, "expected": 1}
    assert (rig.state / "state.db").read_bytes() == before  # not even a checkpoint
    # opening a WAL database may create the -wal and -shm files, but nothing is written to them
    assert not wal.exists() or wal.stat().st_size == 0
    assert table_counts(rig.state) == counts_before  # no relay row, no event, no run written


# --- stored paths that went stale --------------------------------------------------------------


def executable(path: Path) -> str:
    path.write_text("#!/bin/sh\nexit 0\n")
    path.chmod(0o755)
    return str(path)


def test_a_stored_self_path_is_used_while_it_runs(tmp_path: Path) -> None:
    stored = executable(tmp_path / "agent-spawn")
    assert launch.stored_self({"agent_spawn_path": stored}) == stored


def test_a_stale_stored_self_path_falls_back_to_this_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "live").mkdir()
    live = executable(tmp_path / "live" / "agent-spawn")
    monkeypatch.setattr(launch, "resolve_self", lambda: live)
    gone = str(tmp_path / "uv-cache" / "agent-spawn")  # never existed, or was cleaned up
    assert launch.stored_self({"agent_spawn_path": gone}) == live
    assert launch.stored_self({"agent_spawn_path": None}) == live
    assert launch.stored_self({}) == live


def test_a_stale_stored_self_path_with_no_replacement_is_tool_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def missing() -> str:
        raise CliError("tool_missing", "agent-spawn cannot find its own executable path")

    monkeypatch.setattr(launch, "resolve_self", missing)
    with pytest.raises(CliError) as caught:
        launch.stored_self({"agent_spawn_path": str(tmp_path / "gone")})
    assert caught.value.code == "tool_missing"


def test_a_non_executable_stored_self_path_counts_as_stale(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plain = tmp_path / "agent-spawn"
    plain.write_text("x")
    plain.chmod(0o644)
    monkeypatch.setattr(launch, "resolve_self", lambda: "/replacement")
    assert launch.stored_self({"agent_spawn_path": str(plain)}) == "/replacement"


def test_a_stored_tool_is_used_while_it_runs_else_the_configured_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stored = executable(tmp_path / "stored-tool")
    configured = executable(tmp_path / "configured-tool")
    assert launch.stored_tool(f"{stored} --flag", [configured], "claude-session") == [
        stored,
        "--flag",
    ]
    gone = str(tmp_path / "gone-tool")
    assert launch.stored_tool(f"{gone} --flag", [configured, "x"], "claude-session") == [
        configured,
        "x",
    ]
    assert launch.stored_tool(None, [configured], "claude-session") == [configured]
    with pytest.raises(CliError) as caught:
        launch.stored_tool(gone, [str(tmp_path / "also-gone")], "claude-session")
    assert caught.value.code == "tool_missing"


def test_the_supervisor_uses_the_stored_claude_session_when_it_still_runs(
    rig_for: Callable[..., Rig], fakebin: Fakebin
) -> None:
    rig = rig_for()
    row = rig.agent()
    backend = rig.supervisor.backend(row)
    assert list(backend.command)[0] == str(fakebin.bin_dir / "claude-session")


def test_the_supervisor_falls_back_to_the_configured_claude_session_when_the_stored_one_is_gone(
    rig_for: Callable[..., Rig], fakebin: Fakebin, tmp_path: Path
) -> None:
    rig = rig_for(claude_session_path=str(tmp_path / "uv-cache" / "claude-session"))
    rig.supervisor.config.claude_session_cmd = [str(fakebin.bin_dir / "claude-session")]
    outcome = rig.supervisor.run()
    assert outcome.exit_code == 0
    assert rig.agent()["status"] == "parked" and rig.agent()["park_reason"] == "done"


def test_the_internal_argvs_use_the_stored_self_path_that_still_runs(
    tmp_path: Path, rig_for: Callable[..., Rig]
) -> None:
    rig = rig_for()
    stored = executable(tmp_path / "bin-agent-spawn")
    row = {**rig.agent(), "agent_spawn_path": stored}
    argv = promotion.host_argv(rig.supervisor, row, "/run/socket")
    assert argv[0] == stored and "--state-dir" in argv and str(rig.state) in argv


def test_the_host_argv_survives_a_stale_stored_self_path(
    tmp_path: Path, rig_for: Callable[..., Rig], monkeypatch: pytest.MonkeyPatch
) -> None:
    rig = rig_for()
    live = executable(tmp_path / "live-agent-spawn")
    monkeypatch.setattr(launch, "resolve_self", lambda: live)
    row = {**rig.agent(), "agent_spawn_path": str(tmp_path / "gone")}
    assert promotion.host_argv(rig.supervisor, row, "/run/socket")[0] == live


# --- claude-session itself going stale under a running supervisor ---------------------------------


def wrapper_for(fakebin: Fakebin, path: Path) -> str:
    """A stored claude-session path that runs the fake, and can be removed like an old install."""
    path.write_text(f'#!/bin/sh\nexec {fakebin.bin_dir / "claude-session"} "$@"\n')
    path.chmod(0o755)
    return str(path)


def test_a_claude_session_removed_after_exec_started_falls_back_to_the_configured_one(
    rig_for: Callable[..., Rig], fakebin: Fakebin, tmp_path: Path
) -> None:
    stored = Path(wrapper_for(fakebin, tmp_path / "old-claude-session"))
    rig = rig_for(claude_session_path=str(stored))
    rig.supervisor.config.claude_session_cmd = [str(fakebin.bin_dir / "claude-session")]

    class Gateway(SpawnGateway):
        def spawn(self, argv: Sequence[str], **kwargs: Any) -> subprocess.Popen[bytes]:
            child = super().spawn(argv, **kwargs)
            stored.unlink()  # the upgrade removes the old path while `exec` runs
            return child

    rig.supervisor.gateway = Gateway(fakebin)
    outcome = rig.supervisor.run()
    assert outcome.exit_code == 0 and not stored.exists()
    row = rig.agent()
    assert row["status"] == "parked" and row["park_reason"] == "done"
    assert row["last_outcome"]["outcome"] == "done"


def test_neither_claude_session_running_at_start_parks_instead_of_raising(
    rig_for: Callable[..., Rig], tmp_path: Path
) -> None:
    rig = rig_for(claude_session_path=str(tmp_path / "gone" / "claude-session"))
    rig.supervisor.config.claude_session_cmd = [str(tmp_path / "also-gone")]
    outcome = rig.supervisor.run()
    assert outcome.exit_code == 0
    row = rig.agent()
    assert row["status"] == "parked" and row["park_reason"] == "launch_failed"
    assert row["needs_attention"] == "launch_failed"


def test_neither_claude_session_running_after_exec_parks_instead_of_raising(
    rig_for: Callable[..., Rig], fakebin: Fakebin, tmp_path: Path
) -> None:
    stored = Path(wrapper_for(fakebin, tmp_path / "old-claude-session"))
    rig = rig_for(claude_session_path=str(stored))
    rig.supervisor.config.claude_session_cmd = [str(tmp_path / "also-gone")]

    class Gateway(SpawnGateway):
        def spawn(self, argv: Sequence[str], **kwargs: Any) -> subprocess.Popen[bytes]:
            child = super().spawn(argv, **kwargs)
            stored.unlink()
            return child

    rig.supervisor.gateway = Gateway(fakebin)
    outcome = rig.supervisor.run()
    assert outcome.exit_code == 0
    row = rig.agent()
    assert row["status"] == "parked" and row["park_reason"] == "failed"


def test_claude_session_gone_at_the_end_of_a_chain_run_parks_as_a_chain_failure(
    rig_for: Callable[..., Rig], monkeypatch: pytest.MonkeyPatch
) -> None:
    rig = rig_for()
    sup = rig.supervisor
    ended = [final_of()]  # the first run ended exited-unreported: a report resume follows

    def end_of_run(*args: Any, **kwargs: Any) -> Any:
        if ended:
            return ended.pop()
        raise CliError("tool_missing", "claude-session is gone")  # the chain run's end

    monkeypatch.setattr(sup, "end_of_run", end_of_run)
    monkeypatch.setattr(chain, "start_report_resume", lambda *a, **k: RunCtx(2, "/x/o2.json"))
    monkeypatch.setattr(sup, "spawn_exec", lambda row, session: SimpleNamespace())
    monkeypatch.setattr(sup, "watch", lambda *a, **k: None)
    assert sup.run().exit_code == 0
    db = sqlite3.connect(f"file:{rig.state / 'state.db'}?mode=ro", uri=True)
    try:
        rows = db.execute(
            "SELECT payload FROM events WHERE agent_id = ? AND event = 'post-park'",
            (rig.agent_id,),
        ).fetchall()
    finally:
        db.close()
    assert len(rows) == 1
    failure = json.loads(rows[0][0])["chain_failure"]
    assert (failure["tool"], failure["reason"], failure["run"]) == (
        "claude-session",
        "tool_missing",
        2,
    )


def test_a_chain_run_with_no_runnable_agent_spawn_fails_before_anything_is_taken(
    rig_for: Callable[..., Rig],
    fakebin: Fakebin,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing() -> str:
        raise CliError("tool_missing", "agent-spawn cannot find its own executable path")

    monkeypatch.setattr(launch, "resolve_self", missing)
    rig = rig_for(agent_spawn_path=str(tmp_path / "gone" / "agent-spawn"))
    sup = rig.supervisor
    sup.store = Store.open(rig.state, create=False)
    try:
        backend = sup.backend(rig.agent())
        with pytest.raises(chain.ChainFailure) as caught:
            chain.start_report_resume(sup, backend, rig.agent(), rig.session, final_of())
    finally:
        sup.store.close()
    assert (caught.value.tool, caught.value.reason) == ("agent-spawn", "tool_missing")
    fakebin.assert_not_called("claude-session")  # no dry run, no hold, no run


# --- the other readers of the stored claude-session path -----------------------------------------


class StaleSession:
    """A stored claude-session path that no longer exists, and a configured one that runs."""

    def __init__(self, tmp_path: Path) -> None:
        self.gone = str(tmp_path / "uv-cache" / "claude-session")
        self.configured = executable(tmp_path / "configured-claude-session")


@pytest.fixture
def stale(tmp_path: Path) -> StaleSession:
    return StaleSession(tmp_path)


def test_liveness_falls_back_when_the_stored_claude_session_is_gone(
    tmp_path: Path, stale: StaleSession
) -> None:
    with Store.open(tmp_path / "state") as store:
        liveness = Liveness(
            store, state_dir=str(tmp_path / "state"), fallback_command=[stale.configured]
        )
        backend = liveness.backend({"claude_session_path": stale.gone})
        assert backend is not None and backend.command == [stale.configured]
        kept = liveness.backend({"claude_session_path": f"{stale.configured} --x"})
        assert kept is not None and kept.command == [stale.configured, "--x"]


def test_the_lookup_backend_falls_back_when_the_stored_claude_session_is_gone(
    stale: StaleSession,
) -> None:
    inv = SimpleNamespace(
        config=SimpleNamespace(claude_session_cmd=[stale.configured]), actor=None, decision_ids=[]
    )
    backend = backend_for(inv, {"claude_session_path": stale.gone})  # type: ignore[arg-type]
    assert backend.command == [stale.configured]


def test_stop_falls_back_when_the_stored_claude_session_is_gone(
    tmp_path: Path, stale: StaleSession
) -> None:
    inv = SimpleNamespace(
        config=SimpleNamespace(claude_session_cmd=[stale.configured]),
        args=SimpleNamespace(dry_run=False),
        state_dir=str(tmp_path),
        warnings=[],
        actor=None,
        decision_ids=[],
    )
    backend = Stop(inv).make_backend({"claude_session_path": stale.gone})
    assert backend.command == [stale.configured]


def test_teardown_falls_back_when_the_stored_claude_session_is_gone(
    tmp_path: Path, stale: StaleSession
) -> None:
    config = SimpleNamespace(claude_session_cmd=[stale.configured])
    with Store.open(tmp_path / "state") as store:
        teardown = Teardown(store, config, state_dir=str(tmp_path / "state"), actor="x")  # type: ignore[arg-type]
        backend = teardown.session_backend({"claude_session_path": stale.gone})
    assert backend.command == [stale.configured]


def test_the_final_state_of_an_earlier_run_does_not_overwrite_the_current_runs_agent_row(
    rig_for: Callable[..., Rig],
) -> None:
    rig = rig_for()
    seed_stale_state(rig, run=2, session_state="working", activity="working")
    rig.put_run(2, session_state="working", activity="working")
    sup = rig.supervisor
    sup.store = Store.open(rig.state, create=False)
    try:
        result = RunOutcome("cs-abcdefgh", 1, "session", "done", "normal", 0, None, None, "file")
        sup.record_run(
            RunCtx(1, "/x/o.json"), result, {"state": "exited", "first_working_at": None}
        )
    finally:
        sup.store.close()
    row = rig.agent()
    assert (row["run"], row["session_state"], row["activity"]) == (2, "working", "working")
    with Store.open(rig.state, create=False) as store:
        run = store.get_run(rig.agent_id, 1)
    assert run is not None and run["session_state"] == "exited"  # the run row is unconditional


# --- a real supervisor process across a migration -------------------------------------------------

GATE = """#!/bin/sh
# gate.sh <file> <command...>: wait (bounded) for <file> to exist, then become <command>.
gate="$1"; shift
i=0
while [ ! -e "$gate" ] && [ $i -lt 1200 ]; do sleep 0.05; i=$((i+1)); done
exec "$@"
"""


def wait_for_marker(path: Path, proc: subprocess.Popen[bytes], timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return
        assert proc.poll() is None, f"the supervisor exited early ({proc.returncode})"
        time.sleep(0.02)
    raise AssertionError(f"{path.name} did not appear in {timeout}s")


def test_a_real_supervisor_process_finishes_after_the_database_is_migrated_under_it(
    rig_for: Callable[..., Rig],
    fakebin: Fakebin,
    isolation_env: Isolation,
    tmp_path: Path,
) -> None:
    """The supervisor runs `exec` (blocked on a file barrier), a newer agent-spawn migrates the
    database from another process, `exec` is released, and the supervisor's final bookkeeping
    lands in the migrated database and the process exits 0."""
    work = tmp_path / "gates"
    work.mkdir()
    started, release, go = work / "exec-started", work / "release", work / "go"
    gate = work / "gate.sh"
    gate.write_text(GATE)
    gate.chmod(0o755)
    # claude-session as stored with the agent: `exec` announces itself and waits for the barrier
    stored = work / "claude-session"
    stored.write_text(
        "#!/bin/sh\n"
        f'if [ "$1" = exec ]; then : > {started}; {gate} {release} true; fi\n'
        f'exec {fakebin.bin_dir / "claude-session"} "$@"\n'
    )
    stored.chmod(0o755)
    rig = rig_for(claude_session_path=str(stored))
    log = (work / "supervisor.log").open("wb")
    # The supervisor waits for `go` as a gate shell, which then becomes `agent-spawn supervise`:
    # same PID and start time, so the identity recorded below is the supervisor's own.
    proc = subprocess.Popen(
        [
            str(gate),
            str(go),
            sys.executable,
            "-m",
            "agent_spawn",
            "supervise",
            rig.agent_id,
            "--state-dir",
            str(rig.state),
            "--config",
            str(tmp_path / "config.yaml"),
        ],
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=log,
        env=fakebin.env(),
    )
    try:
        identity = procs.read_identity(proc.pid)
        assert identity is not None
        with Store.open(rig.state, create=False) as store:
            with store.lease(f"agent:{rig.agent_id}") as lease:
                store.update_agent(lease, rig.agent_id, {"supervisor": identity})
        go.touch()
        wait_for_marker(started, proc)
        migrate_to_v2(rig.state)  # the supervisor is running with its schema-1 connection
        release.touch()
        assert proc.wait(timeout=60) == 0, (work / "supervisor.log").read_text()
    finally:
        if proc.poll() is None:
            proc.kill()  # exactly the process this test started
            proc.wait(timeout=10)
        log.close()
    with Store.open(rig.state, create=False, migrations=MIGRATIONS_V2) as store:
        row = store.get_agent(rig.agent_id)
        run = store.get_run(rig.agent_id, 1)
    assert row is not None and run is not None
    assert row["status"] == "parked" and row["park_reason"] == "done"
    assert row["last_outcome"]["outcome"] == "done" and row["exec_proc"] is None
    assert run["outcome"]["outcome"] == "done" and run["session_state"] == "exited"
