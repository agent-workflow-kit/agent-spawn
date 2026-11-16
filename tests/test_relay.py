"""The state relay (`_session-event`): recording, dedup, ignoring, hostile input, exit 0 always."""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import time
from pathlib import Path
from typing import Any

import pytest

from agent_spawn import cli, relay
from agent_spawn.store import RELAY_EVENT_DAYS, Store
from tests import schema_registry
from tests.conftest import Isolation
from tests.launch_support import agent_spawn_path

SESSION = "cs-abcdefgh"
AGENT = "ag-abcdefgh"
EVENT_TS = "2026-11-02T10:00:05.000Z"


class Rig:
    def __init__(self, tmp_path: Path, iso: Isolation) -> None:
        self.state = tmp_path / "state"
        self.config = tmp_path / "config.yaml"
        self.iso = iso
        with Store.open(self.state) as store:
            store.insert_agent(
                AGENT,
                {
                    "status": "live",
                    "name": "agent test",
                    "mode": "headless",
                    "csession_id": SESSION,
                    "run": 1,
                    "launch_kind": "work",
                },
            )

    def argv(self) -> list[str]:
        return [
            agent_spawn_path(),
            "--state-dir",
            str(self.state),
            "--config",
            str(self.config),
            "_session-event",
        ]

    def send(
        self, payload: Any = None, *, raw: bytes | None = None, argv: list[str] | None = None
    ) -> subprocess.CompletedProcess[bytes]:
        data = raw if raw is not None else json.dumps(payload).encode()
        return subprocess.run(
            argv or self.argv(),
            input=data,
            capture_output=True,
            env={**self.iso.env},
            timeout=60,
            check=False,
        )

    def row(self) -> dict[str, Any]:
        with Store.open(self.state, create=False) as store:
            row = store.get_agent(AGENT)
        assert row is not None
        return row

    def run_row(self, run: int) -> dict[str, Any] | None:
        with Store.open(self.state, create=False) as store:
            return store.get_run(AGENT, run)

    def outbox(self) -> list[dict[str, Any]]:
        db = sqlite3.connect(f"file:{self.state / 'state.db'}?mode=ro", uri=True)
        try:
            rows = db.execute("SELECT payload FROM events ORDER BY event_id").fetchall()
            return [json.loads(r[0]) for r in rows]
        finally:
            db.close()

    def relay_count(self) -> int:
        db = sqlite3.connect(f"file:{self.state / 'state.db'}?mode=ro", uri=True)
        try:
            return int(db.execute("SELECT COUNT(*) FROM relay_events").fetchone()[0])
        finally:
            db.close()

    def log(self) -> list[dict[str, Any]]:
        path = self.state / "log.jsonl"
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text().splitlines()]


@pytest.fixture
def rig(tmp_path: Path, isolation_env: Isolation) -> Rig:
    return Rig(tmp_path, isolation_env)


def payload(**changes: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "tool": "claude-session",
        "hook": "post-state-change",
        "object_type": "session",
        "object_id": SESSION,
        "labels": [f"agent={AGENT}", "owner=me"],
        "event_id": 1,
        "event_ts": EVENT_TS,
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
    body.update(changes)
    return body


def validate_outbox(body: dict[str, Any]) -> None:
    full = {
        **body,
        "tool": "agent-spawn",
        "hook": "post-state-change",
        "object_type": "agent",
        "object_id": AGENT,
        "event_id": 1,
        "ts": "2026-11-14T00:00:00.000Z",
    }
    schema_registry.validate(full, "agent-spawn/hooks/post-state-change.json")


# --- recording -------------------------------------------------------------------------------


def test_an_event_updates_the_agent_the_run_and_queues_the_outbox_event(rig: Rig) -> None:
    done = rig.send(payload())
    assert done.returncode == 0 and done.stdout == b"" and done.stderr == b""
    row = rig.row()
    assert (row["run"], row["session_state"], row["activity"]) == (1, "working", "working")
    run = rig.run_row(1)
    assert run is not None
    assert (run["session_state"], run["activity"]) == ("working", "working")
    assert run["first_working_at"] is not None
    (event,) = rig.outbox()
    validate_outbox(event)
    assert event["change"] == "session" and event["state"] == "working"
    assert event["session_event_id"] == 1 and event["agent"]["agent_id"] == AGENT
    assert event["previous_state"] == "starting" and event["cause"] == "UserPromptSubmit"


def test_the_same_event_twice_is_recorded_once(rig: Rig) -> None:
    assert rig.send(payload()).returncode == 0
    assert rig.send(payload()).returncode == 0
    assert len(rig.outbox()) == 1 and rig.relay_count() == 1
    assert rig.send(payload(event_id=2, state="idle", activity="idle")).returncode == 0
    assert len(rig.outbox()) == 2 and rig.row()["session_state"] == "idle"


def test_the_same_event_id_of_another_session_is_a_different_event(rig: Rig) -> None:
    assert rig.send(payload()).returncode == 0
    assert rig.relay_count() == 1
    other = payload(object_id="cs-bbbbbbbb")
    assert rig.send(other).returncode == 0  # ignored (not the agent's session), not recorded
    assert rig.relay_count() == 1


def test_a_late_event_of_an_earlier_run_does_not_roll_the_agent_back(rig: Rig) -> None:
    with Store.open(rig.state, create=False) as store:
        with store.lease(f"agent:{AGENT}") as lease:
            store.update_agent(lease, AGENT, {"run": 2})
    assert rig.send(payload(run=1, event_id=5, state="idle", activity="idle")).returncode == 0
    row = rig.row()
    assert row["run"] == 2 and row["session_state"] is None
    run = rig.run_row(1)
    assert run is not None and run["session_state"] == "idle"
    (event,) = rig.outbox()
    assert event["run"] == 1 and event["agent"]["run"] == 2


def test_an_event_of_a_newer_run_advances_the_agent(rig: Rig) -> None:
    assert rig.send(payload(run=2, event_id=9)).returncode == 0
    assert rig.row()["run"] == 2


def test_first_working_at_is_the_first_working_event_and_is_kept(rig: Rig) -> None:
    assert rig.send(payload(event_id=1, state="starting", activity=None)).returncode == 0
    run = rig.run_row(1)
    assert run is not None and run["first_working_at"] is None
    assert rig.send(payload(event_id=2)).returncode == 0
    first = (rig.run_row(1) or {})["first_working_at"]
    assert first is not None
    assert rig.send(payload(event_id=3, event_ts="2026-11-02T11:00:00.000Z")).returncode == 0
    assert (rig.run_row(1) or {})["first_working_at"] == first


def test_needs_input_reason_and_a_final_report_are_relayed(rig: Rig) -> None:
    report = {
        "status": "done",
        "implicit": False,
        "summary": "text <script>",
        "reason": None,
        "artifact": None,
        "reported_at": EVENT_TS,
    }
    body = payload(
        event_id=3,
        state="exited",
        activity=None,
        final=True,
        cause="exit",
        report=report,
        outcome="done",
        end_reason="normal",
        exit_code=0,
    )
    assert rig.send(body).returncode == 0
    (event,) = rig.outbox()
    validate_outbox(event)
    assert event["final"] is True and event["report"]["summary"] == "text <script>"
    assert (
        rig.send(payload(event_id=4, state="needs-input", needs_input_reason="question")).returncode
        == 0
    )
    assert rig.row()["needs_input_reason"] == "question"


def test_a_malformed_report_is_replaced_by_null_in_the_outbox_event(rig: Rig) -> None:
    assert rig.send(payload(report={"status": "bogus"})).returncode == 0
    (event,) = rig.outbox()
    validate_outbox(event)
    assert event["report"] is None


def test_a_payload_missing_a_required_field_records_nothing(rig: Rig) -> None:
    body = payload()
    del body["cause"]
    assert rig.send(body).returncode == 0
    assert rig.outbox() == [] and rig.relay_count() == 0


def test_a_final_flag_inconsistent_with_the_outcome_records_nothing(rig: Rig) -> None:
    assert rig.send(payload(final=True)).returncode == 0
    assert rig.outbox() == []


def test_relay_events_older_than_seven_days_are_pruned(rig: Rig) -> None:
    db = sqlite3.connect(rig.state / "state.db")
    old = time.time() - (RELAY_EVENT_DAYS * 86400 + 60)
    db.execute("INSERT INTO relay_events VALUES (?, ?, ?)", (SESSION, "99", old))
    db.execute("INSERT INTO relay_events VALUES (?, ?, ?)", (SESSION, "98", time.time()))
    db.commit()
    db.close()
    assert rig.send(payload()).returncode == 0
    assert rig.relay_count() == 2  # the fresh pre-seeded key and the new one


# --- ignored events --------------------------------------------------------------------------


def ignored_reasons(rig: Rig) -> list[str]:
    return [r["details"]["reason"] for r in rig.log() if r["event"] == "relay_event_ignored"]


def test_an_event_whose_object_id_differs_is_logged_and_ignored(rig: Rig) -> None:
    assert rig.send(payload(object_id="cs-bbbbbbbb")).returncode == 0
    assert rig.outbox() == [] and rig.row()["session_state"] is None
    assert ignored_reasons(rig) == ["session_mismatch"]
    assert rig.relay_count() == 0


def test_an_event_for_an_agent_without_a_session_is_ignored(rig: Rig) -> None:
    with Store.open(rig.state, create=False) as store:
        with store.lease(f"agent:{AGENT}") as lease:
            store.update_agent(lease, AGENT, {"csession_id": None})
    assert rig.send(payload(state="starting", activity=None)).returncode == 0
    assert rig.outbox() == [] and ignored_reasons(rig) == ["agent_without_session"]


def test_an_event_for_an_unknown_agent_is_ignored(rig: Rig) -> None:
    assert rig.send(payload(labels=["agent=ag-zzzzzzzz"])).returncode == 0
    assert ignored_reasons(rig) == ["unknown_agent"]


def test_an_event_for_a_torn_down_agent_is_ignored(rig: Rig) -> None:
    db = sqlite3.connect(rig.state / "state.db")
    db.execute("UPDATE agents SET status = 'torn_down' WHERE agent_id = ?", (AGENT,))
    db.commit()
    db.close()
    assert rig.send(payload()).returncode == 0
    assert rig.outbox() == [] and ignored_reasons(rig) == ["torn_down"]


@pytest.mark.parametrize(
    "labels",
    [
        [],
        ["owner=me"],
        [f"agent={AGENT}", "agent=ag-zzzzzzzz"],
        ["agent=not-an-id"],
        "agent=ag-abcdefgh",
        [1, None],
    ],
)
def test_an_event_without_exactly_one_valid_agent_label_records_nothing(
    rig: Rig, labels: Any
) -> None:
    assert rig.send(payload(labels=labels)).returncode == 0
    assert rig.outbox() == [] and rig.relay_count() == 0


# --- hostile input ---------------------------------------------------------------------------

HOSTILE = {
    "empty": b"",
    "whitespace": b"  \n",
    "not json": b"{not json",
    "array": b"[]",
    "null": b"null",
    "scalar": b"7",
    "binary": b"\xff\xfe\x00\x01",
    "deep array": b"[" * 200_000,
    "deep object": b'{"a":' * 100_000,
    "huge": b'{"x":"' + b"a" * (relay.MAX_PAYLOAD_BYTES + 10) + b'"}',
    "huge int": b'{"event_id":' + b"9" * 6000 + b"}",
}


@pytest.mark.parametrize("raw", HOSTILE.values(), ids=HOSTILE.keys())
def test_hostile_stdin_exits_0_and_records_nothing(rig: Rig, raw: bytes) -> None:
    done = rig.send(raw=raw)
    assert done.returncode == 0 and done.stdout == b""
    assert rig.outbox() == [] and rig.relay_count() == 0
    assert rig.row()["session_state"] is None


BAD_FIELDS = {
    "hook": {"hook": "pre-launch"},
    "object_type": {"object_type": "agent"},
    "object_id": {"object_id": "../etc"},
    "event_id zero": {"event_id": 0},
    "event_id bool": {"event_id": True},
    "event_id str": {"event_id": "1"},
    "event_id huge": {"event_id": 2**80},
    "run zero": {"run": 0},
    "run huge": {"run": 10**9},
    "state": {"state": "dancing"},
    "activity": {"activity": "sleeping"},
    "reason": {"needs_input_reason": "bored"},
    "state type": {"state": ["working"]},
}


@pytest.mark.parametrize("changes", BAD_FIELDS.values(), ids=BAD_FIELDS.keys())
def test_invalid_fields_record_nothing(rig: Rig, changes: dict[str, Any]) -> None:
    assert rig.send(payload(**changes)).returncode == 0
    assert rig.outbox() == [] and rig.relay_count() == 0


def test_untrusted_text_never_reaches_the_log_or_stderr(rig: Rig) -> None:
    secret = "SECRET-\x1b[31m-text"
    done = rig.send(payload(object_id="cs-bbbbbbbb", message=secret, labels=[f"agent={AGENT}"]))
    assert done.returncode == 0
    assert b"SECRET" not in done.stderr
    assert "SECRET" not in (rig.state / "log.jsonl").read_text()


def test_unusual_extra_fields_and_unicode_are_harmless(rig: Rig) -> None:
    body = payload(message="é‮" * 5000, extra={"a": [1] * 1000}, summary="x")
    assert rig.send(body).returncode == 0
    (event,) = rig.outbox()
    validate_outbox(event)
    assert len(event["message"]) <= relay.MAX_MESSAGE


# --- argv, missing and foreign databases ------------------------------------------------------


def test_malformed_argv_exits_0_and_records_nothing(rig: Rig) -> None:
    bad = [
        [agent_spawn_path(), "_session-event"],
        [agent_spawn_path(), "--state-dir", "rel", "--config", "/c", "_session-event"],
        [agent_spawn_path(), "--state-dir", str(rig.state), "--config", "c", "_session-event"],
        [agent_spawn_path(), "--state-dir", str(rig.state), "_session-event"],
        [*rig.argv(), "--bogus", "extra"],
        [agent_spawn_path(), "--state-dir", "--config", "_session-event"],
    ]
    for argv in bad:
        done = rig.send(payload(), argv=argv)
        assert done.returncode == 0, argv
        assert done.stdout == b""
    # the last one carries valid flags plus junk after the command: still exit 0 (recorded or not)
    assert len(rig.outbox()) <= 1


def test_an_invalid_config_file_does_not_matter(rig: Rig) -> None:
    rig.config.write_text("tombstone_days: [unterminated\n")
    done = rig.send(payload())
    assert done.returncode == 0 and len(rig.outbox()) == 1


def test_a_missing_state_db_is_logged_to_stderr_and_creates_nothing(
    tmp_path: Path, isolation_env: Isolation
) -> None:
    missing = tmp_path / "nowhere"
    argv = [agent_spawn_path(), "--state-dir", str(missing), "--config", "/c", "_session-event"]
    done = subprocess.run(
        argv,
        input=json.dumps(payload()).encode(),
        capture_output=True,
        env={**isolation_env.env},
        timeout=60,
        check=False,
    )
    assert done.returncode == 0 and b"no state database" in done.stderr
    assert not missing.exists()
    empty = tmp_path / "emptydir"
    empty.mkdir()
    argv[2] = str(empty)
    done = subprocess.run(
        argv,
        input=b"{}",
        capture_output=True,
        env={**isolation_env.env},
        timeout=60,
        check=False,
    )
    assert done.returncode == 0 and list(empty.iterdir()) == []


def test_an_older_schema_logs_schema_older_and_records_nothing(rig: Rig) -> None:
    db = sqlite3.connect(rig.state / "state.db")
    db.execute("UPDATE meta SET value = '0' WHERE key = 'schema_version'")
    db.commit()
    db.close()
    done = rig.send(payload())
    assert done.returncode == 0
    assert [r["event"] for r in rig.log()] == ["schema_older"]
    assert rig.log()[0]["details"] == {"found": 0, "expected": 1}
    db = sqlite3.connect(rig.state / "state.db")
    assert db.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 0
    assert db.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()[0] == "0"
    db.close()


def test_a_newer_schema_is_not_written_to(rig: Rig) -> None:
    db = sqlite3.connect(rig.state / "state.db")
    db.execute("UPDATE meta SET value = '99' WHERE key = 'schema_version'")
    db.commit()
    db.close()
    done = rig.send(payload())
    assert done.returncode == 0 and b"schema_newer" in done.stderr
    # the spec's `schema_older` log event covers any schema other than ours
    assert [r["event"] for r in rig.log()] == ["schema_older"]
    assert rig.log()[0]["details"] == {"found": 99, "expected": 1}
    db = sqlite3.connect(rig.state / "state.db")
    assert db.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 0
    assert db.execute("SELECT COUNT(*) FROM relay_events").fetchone()[0] == 0
    db.close()


def test_a_foreign_database_exits_0(rig: Rig) -> None:
    (rig.state / "state.db").write_bytes(b"this is not sqlite" * 100)
    done = rig.send(payload())
    assert done.returncode == 0


def test_a_locked_database_exits_0_after_the_busy_timeout(rig: Rig) -> None:
    holder = sqlite3.connect(rig.state / "state.db", isolation_level=None)
    holder.execute("BEGIN IMMEDIATE")
    try:
        done = rig.send(payload())
    finally:
        holder.execute("ROLLBACK")
        holder.close()
    assert done.returncode == 0 and done.stdout == b""
    assert rig.outbox() == []


class _Stream:
    def __init__(self, data: bytes) -> None:
        self.data = data

    def read(self, n: int) -> bytes:
        return self.data[:n]


def test_the_relay_never_raises_in_process(
    rig: Rig, monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    def boom(*_a: object, **_k: object) -> None:
        raise RuntimeError("boom secret")

    monkeypatch.setattr("agent_spawn.relay.record", boom)
    assert relay.run(rig.argv()[1:], stdin=_Stream(json.dumps(payload()).encode())) == 0
    err = capfd.readouterr().err
    assert "RuntimeError" in err and "secret" not in err


def test_concurrent_relays_record_each_event_once(rig: Rig) -> None:
    procs = [
        subprocess.Popen(
            rig.argv(),
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env={**rig.iso.env},
        )
        for _ in range(6)
    ]
    data = json.dumps(payload()).encode()
    for proc in procs:
        assert proc.stdin is not None
        proc.stdin.write(data)
        proc.stdin.close()
    for proc in procs:
        assert proc.wait(timeout=60) == 0
    assert len(rig.outbox()) == 1


def test_main_routes_the_relay_without_parsing_or_config(
    rig: Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    from agent_spawn import cli

    monkeypatch.setattr("sys.stdin", type("S", (), {"buffer": _Stream(b"{}")})())
    assert cli.main(["--json", "--state-dir", "relative", "_session-event", "--nonsense"]) == 0


# --- review round 1 ----------------------------------------------------------------------------


def _set_third_party(rig: Rig, run: int, value: bool) -> None:
    with Store.open(rig.state, create=False) as store:
        with store.lease(f"agent:{AGENT}") as lease:
            store.put_run(lease, AGENT, run, {"reads_third_party": value})


@pytest.mark.parametrize(
    ("old_run", "current_run", "expected"),
    [(True, False, True), (False, True, True), (False, False, False)],
)
def test_a_late_event_of_an_earlier_run_reports_either_runs_third_party_flag(
    rig: Rig, old_run: bool, current_run: bool, expected: bool
) -> None:
    with Store.open(rig.state, create=False) as store:
        with store.lease(f"agent:{AGENT}") as lease:
            store.update_agent(lease, AGENT, {"run": 2})
    _set_third_party(rig, 1, old_run)
    _set_third_party(rig, 2, current_run)
    assert rig.send(payload(run=1, event_id=5)).returncode == 0
    (event,) = rig.outbox()
    assert event["reads_third_party"] is expected


@pytest.mark.parametrize(
    "argv",
    [
        ["--state-dir", "/a", "--config", "/b", "_session-event"],
        ["--state-dir", "/a", "--config", "/b", "--", "_session-event"],
        ["--", "_session-event"],
        ["--bogus", "x", "--state-dir", "/a", "--config", "/b", "_session-event"],
        ["--conf", "/b", "_session-event"],
        ["--json", "--bogus=1", "_session-event"],
        ["--prompt", "_session-event", "launch"],  # accepted trade-off, see cli._is_relay
    ],
)
def test_the_relay_is_recognized_in_every_argv_shape(argv: list[str]) -> None:
    assert cli._is_relay(argv)  # pyright: ignore[reportPrivateUsage]


@pytest.mark.parametrize(
    "argv",
    [
        [],
        ["launch", "_session-event"],
        ["--state-dir", "_session-event", "list"],  # its value, not the command
        ["--config", "_session-event"],
        ["launch", "--", "_session-event"],
        ["--", "launch", "_session-event"],
        ["list"],
    ],
)
def test_other_argv_is_not_the_relay(argv: list[str]) -> None:
    assert not cli._is_relay(argv)  # pyright: ignore[reportPrivateUsage]


def test_the_relay_after_a_double_dash_exits_0_with_an_invalid_config(rig: Rig) -> None:
    rig.config.write_text("{{{ not yaml")
    argv = [agent_spawn_path(), "--state-dir", str(rig.state), "--config", str(rig.config)]
    done = rig.send(payload(), argv=[*argv, "--", "_session-event"])
    assert done.returncode == 0 and len(rig.outbox()) == 1


def test_a_closed_stderr_pipe_does_not_change_the_exit_code(rig: Rig) -> None:
    read_end, write_end = os.pipe()
    os.close(read_end)
    try:
        done = subprocess.run(
            rig.argv(),
            input=b"{not json",
            stdout=subprocess.DEVNULL,
            stderr=write_end,
            env={**rig.iso.env},
            timeout=60,
            check=False,
        )
    finally:
        os.close(write_end)
    assert done.returncode == 0


@pytest.mark.parametrize("version", ["0", "99"])
def test_a_foreign_schema_without_sidecars_is_left_byte_for_byte(rig: Rig, version: str) -> None:
    db = sqlite3.connect(rig.state / "state.db")
    db.execute("UPDATE meta SET value = ? WHERE key = 'schema_version'", (version,))
    db.commit()
    db.close()
    before = (rig.state / "state.db").read_bytes()
    sidecars = [rig.state / "state.db-wal", rig.state / "state.db-shm"]
    assert not any(p.exists() for p in sidecars)
    assert rig.send(payload()).returncode == 0
    assert not any(p.exists() for p in sidecars)
    assert (rig.state / "state.db").read_bytes() == before
