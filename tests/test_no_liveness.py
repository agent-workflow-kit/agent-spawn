"""``list --no-liveness`` and ``show --no-liveness``: the stored state, read with no side effect.

Nothing is probed (no process entry, no pidfd, no lower tool) and nothing is written (state
database, log, audit records, hook outbox); the output says so with ``liveness_checked: false``.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import stat
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from agent_spawn import cli, procs
from agent_spawn import gateway as gw
from agent_spawn.procs import ProcessIdentity
from agent_spawn.store import Store
from agent_spawn.store_schema import SCHEMA_VERSION
from tests import schema_registry
from tests.conftest import Isolation
from tests.fakebin.harness import Fakebin
from tests.inspect_support import A1, A2, SESSION, run_inprocess, seed_agent
from tests.launch_support import Env

LIST = "agent-spawn/cli/list.json"
SHOW = "agent-spawn/cli/show.json"
ERROR = "agent-spawn/cli/error.json"

# Every gateway call that touches a process, a pid, a descriptor or a lower tool.
PROBES = {
    "run", "spawn", "run_on_terminal", "exec_replace", "read_file", "pidfd_open",
    "pidfd_send_signal", "kill", "getpgid", "killpg", "wait_pid_exit", "waitpid", "fork",
    "poll_readable",
}  # fmt: skip


@pytest.fixture
def env(fakebin: Fakebin, isolation_env: Isolation) -> Env:
    return Env(fakebin, isolation_env)


SIDECARS = ("-wal", "-shm", "-journal")
Snap = dict[str, tuple[str, int] | None]


def snapshot(env: Env) -> Snap:
    """sha256 and mtime_ns of every file under the state dir (database, sidecars, log, audit),
    ``None`` for a missing database sidecar, and the directory mode under ``"<dir mode>"``."""
    found: Snap = {}
    for root, _dirs, files in os.walk(env.state_dir):
        for name in files:
            path = Path(root, name)
            found[str(path.relative_to(env.state_dir))] = (
                hashlib.sha256(path.read_bytes()).hexdigest(),
                path.stat().st_mtime_ns,
            )
    for suffix in SIDECARS:
        found.setdefault(f"state.db{suffix}", None)
    found["<dir mode>"] = (oct(stat.S_IMODE(env.state_dir.stat().st_mode)), 0)
    return found


def assert_unchanged(env: Env, before: Snap) -> None:
    """Nothing was written. SQLite may create the empty ``-wal`` and the ``-shm`` of a WAL
    database for a reader (they hold no state); an existing ``-wal`` is byte- and
    mtime-identical, an existing ``-shm`` still exists (a reader updates its lock words)."""
    after = snapshot(env)
    for key, was in before.items():
        if key == "state.db-shm":
            assert (after[key] is None) == (was is None) or was is None, key
        elif key == "state.db-wal" and was is None:
            made = after[key]
            assert made is None or made[0] == hashlib.sha256(b"").hexdigest(), key
        else:
            assert after.get(key) == was, key
    assert set(after) - set(before) == set()


def dead_identity(env: Env) -> ProcessIdentity:
    with Store.open(env.state_dir) as store:
        me = store.holder_identity()
    return ProcessIdentity(pid=2**22 - 3, start_time=1, boot_id=me.boot_id)


def seed_dead(env: Env, fakebin: Fakebin, agent_id: str = A1, **fields: Any) -> None:
    dead = dead_identity(env)
    seed_agent(
        env, agent_id, status="live", csession_id=SESSION, session_state="working",
        supervisor=dead, exec_proc=dead, cwd="/work/x", wt_path="/work/x",
        wt_record_id="wt-aaaaaaaa",
        claude_session_path=str(fakebin.bin_dir / "claude-session"), **fields,
    )  # fmt: skip


def body_of(done: Any, schema: str) -> Any:
    assert done.returncode == 0, (done.stdout, done.stderr)
    body = json.loads(done.stdout)
    schema_registry.validate(body, schema)
    return body


def no_tool_called(fakebin: Fakebin) -> None:
    for tool in ("claude-session", "wtctl", "termctl", "tmux"):
        assert fakebin.argvs(tool) == [], tool


# --- the guarantee: no side effect on a dead supervisor ----------------------------------------


def test_list_no_liveness_leaves_every_file_byte_identical(env: Env, fakebin: Fakebin) -> None:
    seed_dead(env, fakebin)
    before = snapshot(env)
    body = body_of(env.run("list", "--no-liveness", "--json"), LIST)
    assert_unchanged(env, before)
    assert body["liveness_checked"] is False
    agent = body["agents"][0]
    assert agent["status"] == "live" and agent["needs_attention"] is None  # as stored
    no_tool_called(fakebin)


def test_list_without_the_flag_still_checks_and_parks(env: Env, fakebin: Fakebin) -> None:
    fakebin.seed_session(SESSION, adopted=True, final=True, outcome="done")
    seed_dead(env, fakebin)
    body = body_of(env.run("list", "--json"), LIST)
    assert "liveness_checked" not in body  # without the flag nothing changes
    assert body["agents"][0]["status"] == "parked"


def test_show_no_liveness_leaves_every_file_byte_identical(env: Env, fakebin: Fakebin) -> None:
    seed_dead(env, fakebin, runs=[{"mode": "headless"}], run=1)
    before = snapshot(env)
    body = body_of(env.run("show", A1, "--no-liveness", "--json"), SHOW)
    assert_unchanged(env, before)
    assert body["liveness_checked"] is False
    assert body["agent"]["status"] == "live"
    assert body["agent"]["park_reason"] is None
    assert [r["run"] for r in body["runs"]] == [1]
    no_tool_called(fakebin)


def test_show_without_the_flag_has_no_member_and_still_checks(env: Env, fakebin: Fakebin) -> None:
    fakebin.seed_session(SESSION, adopted=True, final=True, outcome="done")
    seed_dead(env, fakebin)
    body = body_of(env.run("show", A1, "--json"), SHOW)
    assert "liveness_checked" not in body
    assert body["supervisor"]["alive"] is False
    assert body["agent"]["status"] == "parked"


def test_show_no_liveness_does_not_park_a_missing_session(env: Env, fakebin: Fakebin) -> None:
    seed_agent(
        env, A1, status="live", csession_id="cs-zzzzzzzz",
        claude_session_path=str(fakebin.bin_dir / "claude-session"),
    )  # fmt: skip
    before = snapshot(env)
    body = body_of(env.run("show", A1, "--no-liveness", "--json"), SHOW)
    assert body["agent"]["status"] == "live"
    assert body["session"] is None
    assert_unchanged(env, before)
    no_tool_called(fakebin)


def test_show_no_liveness_names_the_stored_supervisor_without_probing(
    env: Env, fakebin: Fakebin
) -> None:
    dead = dead_identity(env)
    seed_dead(env, fakebin)
    body = body_of(env.run("show", A1, "--no-liveness", "--json"), SHOW)
    # `alive` is not evaluated: null, never a guess
    assert body["supervisor"] == {"pid": dead.pid, "alive": None, "exec_pid": dead.pid}
    assert body["session"] is None
    assert body["third_party_run"] is None
    assert body["repo_config_changed"] is None


def test_show_no_liveness_of_a_tombstone_and_a_supervisorless_agent(
    env: Env, fakebin: Fakebin
) -> None:
    seed_agent(env, A1, status="torn_down", csession_id=SESSION, torn_down_at=10.0)
    seed_agent(env, A2, status="parked")
    body = body_of(env.run("show", A1, "--no-liveness", "--json"), SHOW)
    assert body["supervisor"] is None and body["liveness_checked"] is False
    assert body_of(env.run("show", A2, "--no-liveness", "--json"), SHOW)["supervisor"] is None
    no_tool_called(fakebin)


def test_no_liveness_without_a_database_creates_nothing(env: Env) -> None:
    assert body_of(env.run("list", "--no-liveness", "--json"), LIST) == {
        "agents": [],
        "liveness_checked": False,
    }
    done = env.run("show", A1, "--no-liveness", "--json")
    assert done.returncode == 10
    assert not env.state_dir.exists()


def test_human_output_says_the_check_did_not_run(env: Env, fakebin: Fakebin) -> None:
    seed_dead(env, fakebin)
    assert "liveness not checked" in env.run("show", A1, "--no-liveness").stdout
    assert "liveness not checked" in env.run("list", "--no-liveness").stdout
    assert "liveness not checked" not in env.run("show", A1).stdout


def test_list_no_liveness_keeps_filters(env: Env, fakebin: Fakebin) -> None:
    seed_dead(env, fakebin, A1)
    seed_agent(env, A2, needs_attention="blocked")
    body = body_of(env.run("list", "--no-liveness", "--needs-attention", "--json"), LIST)
    assert [a["agent_id"] for a in body["agents"]] == [A2]


# --- no gateway probe -------------------------------------------------------------------------


class SpyGateway:
    """The real gateway, recording which method was called."""

    def __init__(self) -> None:
        self.real = gw.SystemGateway()
        self.calls: list[str] = []

    def __getattr__(self, name: str) -> Callable[..., Any]:
        target = getattr(self.real, name)
        if not callable(target):
            return target  # type: ignore[no-any-return]

        def call(*args: Any, **kwargs: Any) -> Any:
            self.calls.append(name)
            return target(*args, **kwargs)

        return call


@pytest.mark.parametrize("command", [("list",), ("show", A1)])
def test_no_gateway_probe_runs_with_the_flag_and_does_with_the_other(
    env: Env,
    fakebin: Fakebin,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    command: tuple[str, ...],
) -> None:
    fakebin.seed_session(SESSION, adopted=True, final=True, outcome="done")
    seed_dead(env, fakebin)
    for name, value in env.env().items():
        monkeypatch.setenv(name, value)
    try:
        spy = SpyGateway()
        previous = gw.set_gateway(spy)  # type: ignore[arg-type]
        try:
            assert cli.main([*command, "--no-liveness", "--json"]) == 0
        finally:
            gw.set_gateway(previous)
        assert [c for c in spy.calls if c in PROBES] == []
        assert fakebin.argvs("claude-session") == []

        # control: the same call without the flag does probe (the spy can see probes)
        spy = SpyGateway()
        previous = gw.set_gateway(spy)  # type: ignore[arg-type]
        try:
            assert cli.main([*command, "--json"]) == 0
        finally:
            gw.set_gateway(previous)
        assert [c for c in spy.calls if c in PROBES] != []
    finally:
        for name in [n for n in os.environ if n.startswith(("FAKEBIN_", "CSESSION_"))]:
            monkeypatch.delenv(name)
    capsys.readouterr()


# --- the open is read-only (#107 round 1: R1-R7, R10) ------------------------------------------

DB = "state.db"


def raw(env: Env, sql: str, *args: Any) -> None:
    db = sqlite3.connect(env.state_dir / DB)
    with db:
        db.execute(sql, args)
    db.close()


def both(env: Env, *extra: str) -> list[Any]:
    """The listing and the detail of A1 with the flag; each must exit 0 and validate."""
    return [
        body_of(env.run("list", "--no-liveness", *extra, "--json"), LIST),
        body_of(env.run("show", A1, "--no-liveness", *extra, "--json"), SHOW),
    ]


def test_a_delete_journal_database_stays_delete_mode_and_unchanged(env: Env) -> None:
    seed_agent(env, A1)
    raw(env, "PRAGMA journal_mode = DELETE")  # closes in delete mode: no sidecars left
    db = sqlite3.connect(env.state_dir / DB)
    assert db.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
    db.close()
    before = snapshot(env)
    listed, shown = both(env)
    assert listed["agents"][0]["agent_id"] == A1 and shown["agent"]["agent_id"] == A1
    assert_unchanged(env, before)
    assert not (env.state_dir / f"{DB}-wal").exists()
    db = sqlite3.connect(f"file:{env.state_dir / DB}?mode=ro", uri=True)
    assert db.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
    db.close()


WRITER = """
import os, sqlite3, sys
db = sqlite3.connect(sys.argv[1], isolation_level=None)
db.execute("PRAGMA wal_autocheckpoint = 0")
db.execute("PRAGMA journal_mode = WAL")
db.execute("UPDATE agents SET name = 'from the wal' WHERE agent_id = ?", (sys.argv[2],))
os._exit(0)
"""


def test_a_leftover_wal_is_read_and_neither_checkpointed_nor_deleted(env: Env) -> None:
    seed_agent(env, A1)
    subprocess.run(
        [sys.executable, "-c", WRITER, str(env.state_dir / DB), A1], check=True, timeout=60
    )
    wal = env.state_dir / f"{DB}-wal"
    assert wal.stat().st_size > 0  # committed frames the database file does not have
    before = snapshot(env)
    listed, shown = both(env)
    assert listed["agents"][0]["name"] == "from the wal"
    assert shown["agent"]["name"] == "from the wal"
    assert before[f"{DB}-wal"] is not None
    assert_unchanged(env, before)
    assert wal.stat().st_size > 0


def test_a_read_only_state_directory_keeps_its_mode_and_works(env: Env) -> None:
    seed_agent(env, A1)
    env.state_dir.chmod(0o555)
    try:
        before = snapshot(env)
        both(env)
        assert_unchanged(env, before)
        assert stat.S_IMODE(env.state_dir.stat().st_mode) == 0o555
    finally:
        env.state_dir.chmod(0o700)


def test_a_read_only_database_file_works_and_keeps_its_mode(env: Env) -> None:
    seed_agent(env, A1)
    db = env.state_dir / DB
    db.chmod(0o400)
    try:
        before = snapshot(env)
        both(env)
        assert_unchanged(env, before)
        assert stat.S_IMODE(db.stat().st_mode) == 0o400
    finally:
        db.chmod(0o600)


def test_a_loose_database_mode_is_not_tightened_by_a_read(env: Env) -> None:
    seed_agent(env, A1)
    (env.state_dir / DB).chmod(0o644)
    env.state_dir.chmod(0o755)
    try:
        both(env)
        assert stat.S_IMODE((env.state_dir / DB).stat().st_mode) == 0o644
        assert stat.S_IMODE(env.state_dir.stat().st_mode) == 0o755
    finally:
        env.state_dir.chmod(0o700)


def older_database(env: Env) -> None:
    """A real pre-migration database: a WAL database with nothing the latest migration added
    (the only migration so far makes every table), so schema version SCHEMA_VERSION - 1."""
    assert SCHEMA_VERSION == 1, "rebuild this fixture for the new migration"
    env.state_dir.mkdir(mode=0o700, parents=True)
    db = sqlite3.connect(env.state_dir / DB, isolation_level=None)
    db.execute("PRAGMA journal_mode = WAL")
    db.close()
    (env.state_dir / DB).chmod(0o600)


def test_no_liveness_never_migrates_an_older_database(env: Env) -> None:
    older_database(env)
    before = snapshot(env)
    for args in (("list",), ("show", A1)):
        done = env.run(*args, "--no-liveness", "--json")
        assert done.returncode == 12, (done.stdout, done.stderr)
        body = json.loads(done.stdout)
        schema_registry.validate(body, ERROR)
        assert "older schema" in body["error"]["message"]
    assert_unchanged(env, before)


def test_the_same_older_database_is_migrated_without_the_flag(env: Env) -> None:
    older_database(env)
    assert env.run("list", "--json").returncode == 0
    with Store.open(env.state_dir, create=False, migrate=False) as store:
        assert store.schema_version() == SCHEMA_VERSION


def test_a_newer_database_is_exit_12_and_unchanged(env: Env) -> None:
    seed_agent(env, A1)
    raw(env, "UPDATE meta SET value = ? WHERE key = 'schema_version'", str(SCHEMA_VERSION + 1))
    before = snapshot(env)
    for args in (("list",), ("show", A1)):
        assert env.run(*args, "--no-liveness", "--json").returncode == 12
    assert_unchanged(env, before)


def test_a_corrupt_stored_column_is_exit_12_in_both_modes_and_leaves_the_database(
    env: Env,
) -> None:
    seed_agent(env, A1)
    raw(env, "UPDATE agents SET supervisor = 'not a process' WHERE agent_id = ?", A1)
    before = snapshot(env)
    for args in (("list",), ("show", A1)):
        assert env.run(*args, "--no-liveness", "--json").returncode == 12, args
    assert_unchanged(env, before)
    assert env.run("list", "--json").returncode == 12  # same without the flag


UNREADABLE_PID = 999998


@pytest.fixture
def unreadable_entry(monkeypatch: pytest.MonkeyPatch) -> ProcessIdentity:
    real = procs.is_alive

    def is_alive(identity: ProcessIdentity, *args: Any, **kwargs: Any) -> bool:
        if identity.pid == UNREADABLE_PID:
            raise procs.ProcessIdentityError("unreadable")
        return real(identity, *args, **kwargs)

    monkeypatch.setattr(procs, "is_alive", is_alive)
    return ProcessIdentity(UNREADABLE_PID, 1, "boot")


def test_an_unreadable_process_entry_is_not_logged_with_the_flag_but_is_without(
    env: Env,
    unreadable_entry: ProcessIdentity,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    seed_agent(env, A1, status="live", supervisor=unreadable_entry, exec_proc=unreadable_entry)
    log = env.state_dir / "log.jsonl"
    before = snapshot(env)
    code, out, _err = run_inprocess(env, monkeypatch, capsys, "list", "--no-liveness", "--json")
    assert code == 0 and json.loads(out)["agents"][0]["status"] == "live"
    assert not log.exists() or "liveness_unreadable" not in log.read_text()
    assert_unchanged(env, before)
    # control: the same row, the same call without the flag, is logged
    code, out, _err = run_inprocess(env, monkeypatch, capsys, "list", "--json")
    assert code == 0
    assert "liveness_unreadable" in log.read_text()


def test_show_without_the_flag_calls_claude_session_and_wtctl_the_flag_calls_neither(
    env: Env, fakebin: Fakebin
) -> None:
    fakebin.seed_session(SESSION, adopted=True, final=True, outcome="done")
    seed_dead(env, fakebin)
    body_of(env.run("show", A1, "--no-liveness", "--json"), SHOW)
    no_tool_called(fakebin)
    body_of(env.run("show", A1, "--json"), SHOW)  # control: the calls are visible here
    assert fakebin.argvs("claude-session") != []
    assert fakebin.argvs("wtctl") != []
