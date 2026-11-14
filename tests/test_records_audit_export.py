"""log.jsonl / audit.jsonl writers and readers (issue #7) and the ``export`` command."""

from __future__ import annotations

import json
import multiprocessing
import os
import stat
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from agent_spawn import cli, records
from agent_spawn.commands import maintenance
from tests import schema_registry

NOW = datetime(2026, 11, 3, 10, 0, 0, tzinfo=UTC)
AGENT = "ag-abcdefgh"
INPUTS: dict[str, Any] = {
    "labels": ["pr=1"],
    "repo": None,
    "worktree": None,
    "csession_id": None,
    "merged_head": None,
    "force": False,
}


def audit(state: Path, action: str = "stop", at: datetime = NOW, **kw: Any) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "actor": "human:me",
        "decision_ids": [],
        "target": AGENT,
        "action": action,
        "inputs": INPUTS,
        "result": "ok",
        "now": at,
    }
    fields.update(kw)
    return records.append_audit(state, **fields)


def lines(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text().splitlines()]


# --- log.jsonl -----------------------------------------------------------------------------------


def test_log_record_is_schema_valid_and_file_is_private(tmp_path: Path) -> None:
    state = tmp_path / "state"
    records.append_log(state, "launch", agent_id=AGENT, now=NOW)
    path = state / "log.jsonl"
    (record,) = lines(path)
    schema_registry.validate(record, "agent-spawn/log.json")
    assert record["ts"] == "2026-11-03T10:00:00.000Z"
    assert record["level"] == "info" and record["agent_id"] == AGENT
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(state.stat().st_mode) == 0o700


def test_every_log_event_with_every_field_validates(tmp_path: Path) -> None:
    for event in sorted(records.LOG_EVENTS):
        records.append_log(
            tmp_path,
            event,
            level="warn",
            agent_id=None,
            csession_id="cs-abcdefgh",
            run=2,
            actor="auto:policy",
            decision_ids=["d1"],
            hook="pre-launch",
            pid=123,
            signal="SIGTERM",
            message="hello",
            details={"reason": "third_party_cwd", "path": "/x\ny"},
        )
    got = lines(tmp_path / "log.jsonl")
    assert len(got) == len(records.LOG_EVENTS)
    for record in got:
        schema_registry.validate(record, "agent-spawn/log.json")


def test_events_in_code_match_the_schema() -> None:
    log = schema_registry.schema("agent-spawn/log.json")["properties"]
    assert set(log["event"]["enum"]) == records.LOG_EVENTS
    assert set(log["level"]["enum"]) == records.LOG_LEVELS
    audit_schema = schema_registry.schema("agent-spawn/audit.json")["properties"]
    assert set(audit_schema["action"]["enum"]) == records.AUDIT_ACTIONS
    assert set(audit_schema["result"]["enum"]) == records.AUDIT_RESULTS
    assert set(audit_schema["inputs"]["required"]) == records.AUDIT_INPUT_KEYS


def _deep(depth: int = 100_000) -> Any:
    node: Any = []
    for _ in range(depth):
        node = [node]
    return node


@pytest.mark.parametrize(
    "kwargs",
    [
        {"event": "nope"},
        {"event": "launch", "level": "fatal"},
        {"event": "launch", "agent_id": "ag-BAD"},
        {"event": "launch", "csession_id": "x"},
        {"event": "launch", "run": 0},
        {"event": "launch", "run": True},
        {"event": "launch", "actor": "root"},
        {"event": "launch", "decision_ids": ["bad id"]},
        {"event": "launch", "hook": "pre-nothing"},
        {"event": "launch", "pid": "1"},
        {"event": "launch", "details": ["x"]},
        {"event": "launch", "details": {"x": float("nan")}},
        {"event": "launch", "details": {"x": object()}},
        {"event": "launch", "message": 5},
        {"event": ["x"]},
        {"event": "launch", "level": ["info"]},
        {"event": "launch", "hook": ["pre-launch"]},
        {"event": "launch", "decision_ids": "abc"},
        {"event": "launch", "details": {"a": _deep()}},
    ],
)
def test_invalid_log_records_are_refused_and_nothing_is_written(
    tmp_path: Path, kwargs: dict[str, Any]
) -> None:
    with pytest.raises(ValueError):
        records.append_log(tmp_path, **kwargs)
    assert not (tmp_path / "log.jsonl").exists() or (tmp_path / "log.jsonl").read_text() == ""


def test_log_record_is_one_line_even_with_hostile_text(tmp_path: Path) -> None:
    hostile = "a\nb\r\n \x00é\u0085\u2028"
    records.append_log(tmp_path, "warning", message=hostile, details={"stderr": hostile})
    raw = (tmp_path / "log.jsonl").read_bytes()
    assert raw.count(b"\n") == 1 and raw.endswith(b"\n")
    (record,) = lines(tmp_path / "log.jsonl")
    assert record["message"] == hostile
    schema_registry.validate(record, "agent-spawn/log.json")


def test_oversized_record_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="too large"):
        records.append_log(tmp_path, "warning", message="x" * (records.MAX_RECORD_BYTES + 1))


def test_tightens_a_loose_file_and_appends(tmp_path: Path) -> None:
    path = tmp_path / "log.jsonl"
    path.write_text("")
    path.chmod(0o666)
    records.append_log(tmp_path, "launch", agent_id=AGENT)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_refuses_a_symlinked_file(tmp_path: Path) -> None:
    target = tmp_path / "elsewhere"
    target.write_text("")
    (tmp_path / "log.jsonl").symlink_to(target)
    with pytest.raises(records.RecordsError, match="symbolic link"):
        records.append_log(tmp_path, "launch", agent_id=AGENT)
    assert target.read_text() == ""


def test_refuses_a_hard_linked_and_a_non_regular_file(tmp_path: Path) -> None:
    (tmp_path / "other").write_text("")
    os.link(tmp_path / "other", tmp_path / "log.jsonl")
    with pytest.raises(records.RecordsError):
        records.append_log(tmp_path, "launch", agent_id=AGENT)
    os.mkfifo(tmp_path / "audit.jsonl")
    with pytest.raises(records.RecordsError):
        audit(tmp_path)


def test_a_torn_last_line_does_not_swallow_the_next_record(tmp_path: Path) -> None:
    (tmp_path / "log.jsonl").write_text('{"ts": "torn')
    records.append_log(tmp_path, "launch", agent_id=AGENT)
    last = (tmp_path / "log.jsonl").read_text().splitlines()[-1]
    schema_registry.validate(json.loads(last), "agent-spawn/log.json")


def _writer(args: tuple[str, int, int]) -> None:
    state, worker, count = args
    for i in range(count):
        records.append_log(Path(state), "warning", message=f"{worker}:{i}:" + "x" * 5000)
        audit(Path(state), "stop", target=AGENT, decision_ids=[f"w{worker}"])


def test_concurrent_processes_never_interleave_records(tmp_path: Path) -> None:
    ctx = multiprocessing.get_context("fork")
    workers, count = 6, 40
    procs = [ctx.Process(target=_writer, args=((str(tmp_path), w, count),)) for w in range(workers)]
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join(120)
        assert proc.exitcode == 0
    got_log = lines(tmp_path / "log.jsonl")
    got_audit = lines(tmp_path / "audit.jsonl")
    assert len(got_log) == len(got_audit) == workers * count
    assert len({r["message"] for r in got_log}) == workers * count
    for record in got_audit:
        schema_registry.validate(record, "agent-spawn/audit.json")


# --- audit.jsonl ---------------------------------------------------------------------------------


def test_audit_record_is_schema_valid(tmp_path: Path) -> None:
    audit(
        tmp_path,
        "teardown",
        decision_ids=["a:b/c"],
        inputs={
            "labels": ["pr=1", "pr=1"],
            "repo": "example.org/o/r",
            "worktree": {"record_id": "wt-abcdefgh", "path": "/w", "owned": True},
            "csession_id": "cs-abcdefgh",
            "merged_head": "a" * 40,
            "force": True,
        },
        result="refused",
    )
    (record,) = lines(tmp_path / "audit.jsonl")
    schema_registry.validate(record, "agent-spawn/audit.json")
    assert (record["decision_id"], record["result"]) == (["a:b/c"], "refused")
    assert stat.S_IMODE((tmp_path / "audit.jsonl").stat().st_mode) == 0o600


@pytest.mark.parametrize(
    "override",
    [
        {"action": "launch"},
        {"result": "maybe"},
        {"actor": "bob"},
        {"target": "ag-x"},
        {"decision_ids": "d1"},
        {"decision_ids": ["bad id"]},
        {"inputs": {**INPUTS, "extra": 1}},
        {"inputs": {k: v for k, v in INPUTS.items() if k != "force"}},
        {"inputs": {**INPUTS, "force": "yes"}},
        {"inputs": {**INPUTS, "merged_head": "abc"}},
        {"inputs": {**INPUTS, "labels": ["BAD"]}},
        {"inputs": {**INPUTS, "csession_id": "x"}},
        {"inputs": {**INPUTS, "worktree": "wt-abcdefgh"}},
        {"inputs": {**INPUTS, "worktree": {"record_id": "wt-x", "path": "/w", "owned": True}}},
        {"inputs": {**INPUTS, "repo": ""}},
        {"inputs": {**INPUTS, "repo": "no-slash"}},
        {"inputs": {**INPUTS, "repo": "example.org/o/.."}},
        {"inputs": {**INPUTS, "repo": "Example.org/o/r"}},
        {"inputs": {**INPUTS, "repo": "example.org/o/r\n"}},
        {"action": ["stop"]},
        {"result": ["ok"]},
        {"now": datetime.min.replace(tzinfo=UTC)},
    ],
)
def test_invalid_audit_records_are_refused(tmp_path: Path, override: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        audit(tmp_path, **override)
    assert not (tmp_path / "audit.jsonl").exists()


# --- reading -------------------------------------------------------------------------------------


def test_read_audit_orders_by_time_and_skips_bad_lines(tmp_path: Path) -> None:
    audit(tmp_path, "stop", at=NOW)
    audit(tmp_path, "teardown", at=NOW - timedelta(hours=1))
    with open(tmp_path / "audit.jsonl", "a") as handle:
        handle.write("not json\n[]\n" + json.dumps({"ts": "x"}) + "\n\n")
        handle.write('{"torn": ')
    result = records.read_audit(tmp_path)
    assert [r["action"] for r in result.records] == ["teardown", "stop"]
    assert result.skipped == 4  # bad json, [], wrong shape, torn


def test_read_audit_missing_file_or_dir_is_empty(tmp_path: Path) -> None:
    assert records.read_audit(tmp_path / "nope").records == []
    assert records.read_audit(tmp_path).records == []


def test_read_audit_skips_records_that_do_not_fit_the_schema(tmp_path: Path) -> None:
    good = audit(tmp_path)
    bad = {**good, "action": "launch"}
    extra = {**good, "prompt": "secret text"}
    with open(tmp_path / "audit.jsonl", "a") as handle:
        handle.write(json.dumps(bad) + "\n" + json.dumps(extra) + "\n")
    result = records.read_audit(tmp_path)
    assert result.records == [good] and result.skipped == 2


def _write_raw(state: Path, *chunks: bytes) -> None:
    with open(state / "audit.jsonl", "ab") as handle:
        for chunk in chunks:
            handle.write(chunk)


def test_read_audit_skips_a_deeply_nested_line(tmp_path: Path) -> None:
    good = audit(tmp_path)
    _write_raw(tmp_path, b"[" * 100_000 + b"\n")
    result = records.read_audit(tmp_path)
    assert result.records == [good] and result.skipped == 1


@pytest.mark.parametrize("ts", ["0001-01-01T00:00:00+05:00", "9999-12-31T23:59:59-05:00"])
def test_read_audit_skips_a_ts_that_overflows_utc(tmp_path: Path, ts: str) -> None:
    good = audit(tmp_path)
    _write_raw(tmp_path, json.dumps({**good, "ts": ts}).encode() + b"\n")
    result = records.read_audit(tmp_path)
    assert result.records == [good] and result.skipped == 1


def test_format_ts_out_of_range_is_a_value_error() -> None:
    with pytest.raises(ValueError):
        records.format_ts(datetime.max.replace(tzinfo=timezone(-timedelta(hours=5))))


@pytest.mark.parametrize(
    "ts",
    [
        "2026-11-03",
        "2026-11-03T10:00:00",
        "2026-11-03 10:00:00Z",
        "2026-11-03t10:00:00z",
        "20261103T100000Z",
        "2026-11-03T10:00:00+0000",
        "2026-11-03T10:00:00\u0661Z",
        "2026-11-03T10:00:00Z\n",
    ],
)
def test_ts_must_be_a_full_rfc3339_date_time(tmp_path: Path, ts: str) -> None:
    good = audit(tmp_path)
    _write_raw(tmp_path, json.dumps({**good, "ts": ts}).encode() + b"\n")
    result = records.read_audit(tmp_path)
    assert result.records == [good] and result.skipped == 1
    with pytest.raises(ValueError):
        records.check_audit({**good, "ts": ts})


def test_read_audit_skips_an_over_cap_line_and_reads_on(tmp_path: Path) -> None:
    first = audit(tmp_path, at=NOW)
    _write_raw(tmp_path, b"x" * (records.MAX_RECORD_BYTES * 3 + 5) + b"\n")
    second = audit(tmp_path, "teardown", at=NOW + timedelta(seconds=1))
    result = records.read_audit(tmp_path)
    assert result.records == [first, second] and result.skipped == 1


def test_read_audit_cap_boundary_does_not_disturb_neighbours(tmp_path: Path) -> None:
    first = audit(tmp_path, at=NOW)
    _write_raw(
        tmp_path,
        b"x" * (records.MAX_RECORD_BYTES - 1) + b"\n",  # exactly the cap with its newline
        b"x" * records.MAX_RECORD_BYTES + b"\n",  # one over
    )
    second = audit(tmp_path, "teardown", at=NOW + timedelta(seconds=1))
    result = records.read_audit(tmp_path)
    assert result.records == [first, second] and result.skipped == 2


def test_read_audit_ties_keep_file_order(tmp_path: Path) -> None:
    for action in ("teardown", "stop", "resume_tainted", "stop"):
        audit(tmp_path, action, at=NOW)
    result = records.read_audit(tmp_path)
    assert [r["action"] for r in result.records] == ["teardown", "stop", "resume_tainted", "stop"]


def test_hand_written_patterns_match_the_schemas() -> None:
    defs = schema_registry.schema("agent-spawn/defs.json")["$defs"]

    def body(name: str) -> str:
        pattern = defs[name]["pattern"]
        assert pattern.startswith("^") and pattern.endswith("$")
        return pattern[1:-1]

    assert records._AGENT_ID.pattern == body("agent_id")
    assert records._CSESSION_ID.pattern == body("csession_id")
    assert records._WT_ID.pattern == body("wt_record_id")
    assert records._ACTOR.pattern == body("actor")
    assert records._DECISION_ID.pattern == body("decision_id")
    assert records._LABEL.pattern == body("label")
    assert records._SHA.pattern == body("sha")
    assert records._ABS_PATH.pattern == body("abs_path")
    assert records._REPO.pattern == body("repo_id")


# --- state directory -----------------------------------------------------------------------------


def test_append_tightens_a_loose_state_dir(tmp_path: Path) -> None:
    state = tmp_path / "state"
    state.mkdir(mode=0o777)
    state.chmod(0o777)
    audit(state)
    assert stat.S_IMODE(state.stat().st_mode) == 0o700


def test_append_refuses_a_symlinked_state_dir(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real)
    with pytest.raises(records.RecordsError):
        audit(link)
    assert not (real / "audit.jsonl").exists()


def test_append_refuses_a_state_dir_of_another_user(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(os, "geteuid", lambda: os.stat(tmp_path).st_uid + 1)
    with pytest.raises(records.RecordsError):
        audit(tmp_path)


# --- export command ------------------------------------------------------------------------------


def run_export(
    state: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], *argv: str
) -> tuple[int, str, str]:
    monkeypatch.setenv("XDG_STATE_HOME", str(state))
    code = cli.main(["export", *argv])
    out = capsys.readouterr()
    return code, out.out, out.err


def seeded(tmp_path: Path) -> Path:
    state = tmp_path / "xdg" / "agent-spawn"
    if state.exists():
        return state
    audit(state, "stop", at=NOW - timedelta(days=3))
    audit(state, "teardown", at=NOW - timedelta(hours=2), target="ag-bbbbbbbb")
    audit(state, "resume_tainted", at=NOW - timedelta(minutes=5))
    audit(state, "stop", at=NOW - timedelta(minutes=1), target="ag-cccccccc")
    return state


def export_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    *argv: str,
) -> list[dict[str, Any]]:
    code, out, _ = run_export(seeded(tmp_path).parent, monkeypatch, capsys, "--json", *argv)
    assert code == 0
    payload = json.loads(out)
    schema_registry.validate(payload, "agent-spawn/cli/export.json")
    return payload


def test_export_json_all_oldest_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    payload = export_json(tmp_path, monkeypatch, capsys)
    assert [r["action"] for r in payload] == ["stop", "teardown", "resume_tainted", "stop"]
    assert [r["ts"] for r in payload] == sorted(r["ts"] for r in payload)


def test_export_since_filters_by_instant_inclusive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    fixed = datetime.now(UTC)
    state = tmp_path / "xdg" / "agent-spawn"
    audit(state, "stop", at=fixed - timedelta(days=2))
    audit(state, "teardown", at=fixed - timedelta(minutes=10))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "xdg"))
    assert cli.main(["export", "--json", "--since", "1h"]) == 0
    got = json.loads(capsys.readouterr().out)
    schema_registry.validate(got, "agent-spawn/cli/export.json")
    assert [r["action"] for r in got] == ["teardown"]
    boundary = (fixed - timedelta(days=2)).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
    assert cli.main(["export", "--json", "--since", boundary]) == 0
    assert len(json.loads(capsys.readouterr().out)) == 2


def test_export_action_filter_is_ored_and_combines_with_since(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    only_stop = export_json(tmp_path, monkeypatch, capsys, "--action", "stop")
    assert [r["target"] for r in only_stop] == [AGENT, "ag-cccccccc"]
    two = export_json(tmp_path, monkeypatch, capsys, "--action", "stop", "--action", "teardown")
    assert len(two) == 3
    # NOW is in the past, so a --since duration drops everything.
    assert export_json(tmp_path, monkeypatch, capsys, "--since", "1s", "--action", "stop") == []
    since = export_json(tmp_path, monkeypatch, capsys, "--since", "2026-11-03T09:00:00Z")
    assert [r["action"] for r in since] == ["resume_tainted", "stop"]
    both = export_json(
        tmp_path, monkeypatch, capsys, "--since", "2026-11-03T09:00:00Z", "--action", "stop"
    )
    assert [r["target"] for r in both] == ["ag-cccccccc"]


def test_export_without_an_audit_file_prints_an_empty_array(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    code, out, _ = run_export(tmp_path, monkeypatch, capsys, "--json")
    assert code == 0 and json.loads(out) == []
    assert not (tmp_path / "agent-spawn").exists()  # reading creates nothing


def test_export_unknown_action_is_a_usage_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    code, out, _ = run_export(tmp_path, monkeypatch, capsys, "--json", "--action", "launch")
    assert code == 2
    schema_registry.validate(json.loads(out), "agent-spawn/cli/error.json")


def test_export_human_output_has_one_line_per_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    code, out, _ = run_export(seeded(tmp_path).parent, monkeypatch, capsys)
    assert code == 0
    rows = out.splitlines()
    assert len(rows) == 4
    assert "stop" in rows[0] and AGENT in rows[0] and "human:me" in rows[0]


def test_export_warns_about_skipped_lines_but_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    state = seeded(tmp_path)
    with open(state / "audit.jsonl", "a") as handle:
        handle.write("garbage\n")
    code, out, err = run_export(state.parent, monkeypatch, capsys, "--json")
    assert code == 0 and len(json.loads(out)) == 4
    assert "skipped 1" in err


def test_export_fails_cleanly_on_an_unreadable_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    state = tmp_path / "agent-spawn"
    state.mkdir()
    (state / "elsewhere").write_text("")
    (state / "audit.jsonl").symlink_to(state / "elsewhere")
    code, out, _ = run_export(tmp_path, monkeypatch, capsys, "--json")
    assert code == 12
    schema_registry.validate(json.loads(out), "agent-spawn/cli/error.json")


def test_export_handler_is_wired() -> None:
    from agent_spawn.commands.table import HANDLERS

    assert HANDLERS["export"] is maintenance.export
