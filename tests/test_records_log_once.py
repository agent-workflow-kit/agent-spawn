"""``append_log_unless_recent``: a log line that repeats is written once per window (issue #92).

The decision and the write happen under one exclusive lock, only the tail of the file is read, and
a log that cannot be used never raises anything but ``RecordsError``.
"""

from __future__ import annotations

import json
import multiprocessing
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from agent_spawn import records

NOW = datetime(2026, 11, 3, 10, 0, 0, tzinfo=UTC)
WINDOW = timedelta(hours=1)
A, B = "ag-abcdefgh", "ag-bcdefghi"


def entry(agent: str = A, reason: str = "r") -> dict[str, Any]:
    return {
        "event": "warning",
        "level": "warn",
        "agent_id": agent,
        "message": "liveness_unreadable",
        "details": {"cause": "liveness_unreadable", "reason": reason},
    }


def lines(state: Path) -> list[dict[str, Any]]:
    path = state / "log.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def once(state: Path, entries: list[dict[str, Any]], at: datetime = NOW, **kw: Any) -> int:
    return records.append_log_unless_recent(state, entries, within=WINDOW, now=at, **kw)


def test_a_repeat_inside_the_window_is_not_written(tmp_path: Path) -> None:
    assert once(tmp_path, [entry()]) == 1
    assert once(tmp_path, [entry()], NOW + timedelta(minutes=59)) == 0
    assert len(lines(tmp_path)) == 1


def test_a_repeat_after_the_window_is_written_again(tmp_path: Path) -> None:
    once(tmp_path, [entry()])
    assert once(tmp_path, [entry()], NOW + WINDOW) == 1
    assert len(lines(tmp_path)) == 2


def test_another_agent_or_another_reason_is_a_new_line(tmp_path: Path) -> None:
    once(tmp_path, [entry(A, "x")])
    assert once(tmp_path, [entry(B, "x"), entry(A, "y")], NOW + timedelta(seconds=1)) == 2


def test_duplicates_inside_one_batch_are_written_once(tmp_path: Path) -> None:
    assert once(tmp_path, [entry(), entry()]) == 1


def test_other_events_in_the_tail_do_not_count(tmp_path: Path) -> None:
    records.append_log(tmp_path, "warning", level="warn", agent_id=A, message="other", now=NOW)
    assert once(tmp_path, [entry()]) == 1


def test_a_record_dated_in_the_future_does_not_suppress(tmp_path: Path) -> None:
    once(tmp_path, [entry()], NOW + timedelta(days=30))
    assert once(tmp_path, [entry()]) == 1


def test_a_record_dated_slightly_ahead_still_suppresses(tmp_path: Path) -> None:
    once(tmp_path, [entry()], NOW + timedelta(seconds=30))
    assert once(tmp_path, [entry()]) == 0


def test_a_record_dated_exactly_at_the_slack_edge_suppresses(tmp_path: Path) -> None:
    once(tmp_path, [entry()], NOW + records._FUTURE_SLACK)
    assert once(tmp_path, [entry()]) == 0


def test_a_record_dated_one_second_past_the_slack_edge_does_not_suppress(tmp_path: Path) -> None:
    once(tmp_path, [entry()], NOW + records._FUTURE_SLACK + timedelta(seconds=1))
    assert once(tmp_path, [entry()]) == 1


def test_a_record_dated_beyond_the_slack_ahead_does_not_suppress(tmp_path: Path) -> None:
    once(tmp_path, [entry()], NOW + timedelta(minutes=2))
    assert once(tmp_path, [entry()]) == 1


def test_a_first_line_cut_by_the_tail_is_not_seen(tmp_path: Path) -> None:
    # The cut lands exactly on the ``{`` of a complete, identical record that follows a junk
    # prefix on the same line: only the skip to the next newline keeps it from being seen.
    once(tmp_path, [entry()])
    log = tmp_path / "log.jsonl"
    record_line = log.read_bytes()
    log.write_bytes(b"junk" + record_line)
    got = once(tmp_path, [entry()], NOW + timedelta(seconds=1), tail_bytes=len(record_line))
    assert got == 1
    assert log.read_bytes().count(b"\n") == 2


def test_a_first_line_the_tail_covers_whole_is_seen(tmp_path: Path) -> None:
    once(tmp_path, [entry()])
    size = (tmp_path / "log.jsonl").stat().st_size
    assert once(tmp_path, [entry()], NOW + timedelta(seconds=1), tail_bytes=size) == 0
    assert len(lines(tmp_path)) == 1


def test_a_torn_last_line_and_garbage_are_survived(tmp_path: Path) -> None:
    once(tmp_path, [entry(A)])
    with open(tmp_path / "log.jsonl", "ab") as handle:
        handle.write(b'not json liveness_unreadable\n{"event": "warning", "ts": ')
    assert once(tmp_path, [entry(A), entry(B)], NOW + timedelta(seconds=1)) == 1
    text = (tmp_path / "log.jsonl").read_text()
    assert text.count(f'"agent_id":"{B}"') == 1 and text.count(f'"agent_id":"{A}"') == 1


def test_only_the_tail_is_read_so_an_old_line_beyond_it_is_not_seen(tmp_path: Path) -> None:
    once(tmp_path, [entry()])
    with open(tmp_path / "log.jsonl", "ab") as handle:
        handle.write(b'{"filler":"' + b"x" * 5000 + b'"}\n')
    assert once(tmp_path, [entry()], NOW + timedelta(seconds=1), tail_bytes=1024) == 1


def test_a_huge_log_costs_one_bounded_read(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    once(tmp_path, [entry()])
    with open(tmp_path / "log.jsonl", "ab") as handle:
        handle.truncate(200 * 1024 * 1024)  # sparse: the head is never read
        handle.write(b"\n")
    once(tmp_path, [entry(B)], NOW + timedelta(seconds=1))
    read_sizes: list[int] = []
    real = os.pread

    def spy(fd: int, size: int, offset: int) -> bytes:
        read_sizes.append(size)
        return real(fd, size, offset)

    monkeypatch.setattr(records.os, "pread", spy)
    assert once(tmp_path, [entry(B)], NOW + timedelta(seconds=2)) == 0
    assert max(read_sizes) <= records.LOG_TAIL_BYTES


def test_thousands_of_rows_are_one_write_and_deduped(tmp_path: Path) -> None:
    ids = [f"ag-{n:08d}".translate(str.maketrans("0189", "abcd")) for n in range(3000)]
    batch = [entry(i) for i in ids]
    assert once(tmp_path, batch) == 3000
    assert once(tmp_path, batch, NOW + timedelta(seconds=1)) == 0
    assert len(lines(tmp_path)) == 3000


def test_a_log_that_cannot_be_written_raises_records_error(tmp_path: Path) -> None:
    (tmp_path / "log.jsonl").mkdir()
    with pytest.raises(records.RecordsError):
        once(tmp_path, [entry()])


def test_a_symlinked_log_is_refused(tmp_path: Path) -> None:
    (tmp_path / "elsewhere").write_text("")
    (tmp_path / "log.jsonl").symlink_to(tmp_path / "elsewhere")
    with pytest.raises(records.RecordsError):
        once(tmp_path, [entry()])


def _race(state: str, barrier: Any, results: Any) -> None:
    barrier.wait(timeout=30)
    results.put(
        records.append_log_unless_recent(
            Path(state), [entry()], within=WINDOW, now=NOW, tail_bytes=records.LOG_TAIL_BYTES
        )
    )


def test_concurrent_callers_write_the_line_exactly_once(tmp_path: Path) -> None:
    ctx = multiprocessing.get_context("fork")
    workers = 6
    barrier = ctx.Barrier(workers)
    results: Any = ctx.Queue()
    procs = [
        ctx.Process(target=_race, args=(str(tmp_path), barrier, results)) for _ in range(workers)
    ]
    for proc in procs:
        proc.start()
    counts = [results.get(timeout=60) for _ in procs]
    for proc in procs:
        proc.join(60)
        assert proc.exitcode == 0
    assert sum(counts) == 1
    assert len(lines(tmp_path)) == 1
