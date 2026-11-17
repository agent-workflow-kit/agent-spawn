"""Upgrades with live agents (interface.md, Upgrades and State relay; issue #35): a migration
while a supervisor is running, final bookkeeping after lost relay events, an older
``_session-event`` against another schema, and stored paths that went stale."""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pytest

from agent_spawn import launch, promotion
from agent_spawn.output import CliError
from agent_spawn.store import Store
from agent_spawn.store_schema import MIGRATIONS
from tests.conftest import Isolation
from tests.fakebin.harness import Fakebin
from tests.test_supervise_inprocess import Rig, SpawnGateway

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


def test_every_migration_is_additive(tmp_path: Path) -> None:
    """A shipped schema stays readable and writable by the older agent-spawn processes that live
    agents keep running: nothing is dropped or retyped, and a new column is nullable or has a
    default (an older INSERT does not name it)."""
    state = tmp_path / "state"
    with Store.open(state, migrations=MIGRATIONS[:1]):
        pass
    before = _layout(state)
    for upto in range(2, len(MIGRATIONS_V2) + 1):
        with Store.open(state, migrations=MIGRATIONS_V2[:upto]):
            pass
        after = _layout(state)
        for table, columns in before.items():
            assert table in after, f"{table} was dropped"
            for name, (kind, _notnull, _default) in columns.items():
                assert name in after[table], f"{table}.{name} was dropped"
                assert after[table][name][0] == kind, f"{table}.{name} changed type"
            for name, (_kind, notnull, default) in after[table].items():
                if name not in columns:
                    assert not notnull or default is not None, f"{table}.{name} needs a default"
        before = after


def _layout(state: Path) -> dict[str, dict[str, tuple[str, int, Any]]]:
    db = sqlite3.connect(f"file:{state / 'state.db'}?mode=ro", uri=True)
    try:
        tables = [
            r[0]
            for r in db.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
        ]
        return {
            table: {r[1]: (r[2], r[3], r[4]) for r in db.execute(f"PRAGMA table_info('{table}')")}
            for table in tables
        }
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


def relay_payload() -> bytes:
    return json.dumps({"object_id": "cs-abcdefgh"}).encode()


def test_an_older_relay_drops_the_event_against_a_migrated_database(
    rig_for: Callable[..., Rig], isolation_env: Isolation
) -> None:
    rig = rig_for()
    migrate_to_v2(rig.state)
    before = (rig.state / "state.db").read_bytes()
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
    assert os.path.isdir(tmp_path)


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
