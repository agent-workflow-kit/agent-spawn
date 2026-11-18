"""The browse model: filters, cells, escaping, the intent file and the subprocess backend."""

from __future__ import annotations

import json
import stat
import sys
from pathlib import Path

import pytest

from agent_spawn.output import resolve_result_file
from agent_spawn.tui import model
from tests import schema_registry
from tests.browse_support import NOW, agent

SCHEMA = "agent-spawn/cli/browse-result.json"


def test_label_filter_ands_terms_case_insensitively() -> None:
    a = agent("ag-aaaaaaaa", labels=["ticket=ABC-1", "owner=Me"])
    assert model.FilterState(label="abc owner").matches(a)
    assert not model.FilterState(label="abc nope").matches(a)
    assert model.FilterState(label="  ").matches(a)


def test_status_and_attention_filters() -> None:
    a = agent("ag-aaaaaaaa", status="live", needs_attention="blocked")
    assert model.FilterState(status="live", attention_only=True).matches(a)
    assert not model.FilterState(status="parked").matches(a)
    assert not model.FilterState(attention_only=True).matches(agent("ag-bbbbbbbb"))


def test_status_cycle_returns_to_all() -> None:
    state = model.FilterState()
    seen = []
    for _ in range(len(model.STATUS_CYCLE)):
        state.status = state.next_status()
        seen.append(state.status)
    assert seen[-1] is None and "parked" in seen


@pytest.mark.parametrize(
    ("stamp", "expected"),
    [
        ("2026-01-01T11:59:30+00:00", "30s"),
        ("2026-01-01T11:00:00Z", "1h"),
        ("2025-12-30T12:00:00Z", "2d"),
        ("2026-01-01T12:05:00Z", "0s"),
        ("garbage", "-"),
        (None, "-"),
        ("2026-01-01T11:00:00", "-"),
    ],
)
def test_age_text(stamp: object, expected: str) -> None:
    assert model.age_text(stamp, NOW) == expected


HOSTILE = "[bold red]x[/] \x1b[2J\x1b]0;title\x07‮\x00 \udc80 " + "y" * 5000


def test_hostile_text_is_escaped_and_bounded_in_cells_and_detail() -> None:
    a = agent("ag-aaaaaaaa", name=HOSTILE, labels=[HOSTILE], kind=HOSTILE, cwd=HOSTILE)
    cells = model.agent_cells(a, NOW, marked=False)
    detail = model.detail_text(
        a, {"agent": a, "warnings": [HOSTILE], "runs": [{"run": 1, "mode": HOSTILE}]}
    )
    for text in [*cells, detail]:
        assert "\x1b" not in text and "\x00" not in text and "\x07" not in text
        assert "‮" not in text
    assert all(len(c) <= model.MAX_CELL for c in cells)
    assert "[bold red]x[/]" in cells[5]  # shown literally, not interpreted here
    assert max(len(line) for line in detail.splitlines()) < 2 * model.MAX_VALUE


def test_parse_agents_skips_malformed_entries() -> None:
    good = agent("ag-aaaaaaaa")
    payload = {
        "agents": [good, 5, {"agent_id": "nope", "status": "x"}, {"agent_id": "ag-bbbbbbbb"}]
    }
    assert model.parse_agents(payload) == [good]
    assert model.parse_agents([]) == []
    assert model.parse_agents({"agents": "x"}) == []


def test_intent_text_validates_against_the_schema() -> None:
    for intent, selected in [
        ("attach", ["ag-aaaaaaaa"]),
        ("resume", ["ag-aaaaaaaa", "ag-bbbbbbbb"]),
        ("back", []),
    ]:
        schema_registry.validate(json.loads(model.intent_text(intent, selected)), SCHEMA)
    for intent, selected in [
        ("attach", []),
        ("attach", ["ag-aaaaaaaa", "ag-bbbbbbbb"]),
        ("resume", []),
        ("x", []),
    ]:
        with pytest.raises(ValueError):
            model.intent_text(intent, selected)
    assert json.loads(model.intent_text("resume", ["ag-aaaaaaaa"] * 2))["selected"] == [
        "ag-aaaaaaaa"
    ]


def test_write_intent_is_private_atomic_and_refuses_a_symlink(tmp_path: Path) -> None:
    target = resolve_result_file(str(tmp_path / "r.json"))
    assert model.write_intent(target, "back", []) is None
    path = tmp_path / "r.json"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    schema_registry.validate(json.loads(path.read_text()), SCHEMA)
    other = tmp_path / "other"
    other.write_text("keep")
    link = tmp_path / "link.json"
    link.symlink_to(other)
    warning = model.write_intent(resolve_result_file(str(link)), "back", [])
    assert warning is not None and other.read_text() == "keep"


PRINT_ARGV = "import json,sys;print(json.dumps({'argv': sys.argv[1:]}));sys.exit(0)"


def test_subprocess_backend_builds_global_flags_and_parses_json() -> None:
    backend = model.SubprocessBackend(
        config_path="/c.yaml",
        actor="human:me",
        decision_ids=["d1"],
        prefix=[sys.executable, "-c", PRINT_ARGV],
    )
    plain = backend.call(["list"])
    assert plain.ok and plain.payload == {"argv": ["--json", "--config", "/c.yaml", "list"]}
    acting = backend.call(["teardown", "ag-aaaaaaaa", "--yes"], acting=True)
    assert acting.payload == {
        "argv": [
            "--json",
            "--config",
            "/c.yaml",
            "--actor",
            "human:me",
            "--decision-id",
            "d1",
            "teardown",
            "ag-aaaaaaaa",
            "--yes",
        ]
    }


def test_subprocess_backend_reports_errors_and_junk() -> None:
    code = "import sys;print('not json');sys.stderr.write('boom\\n');sys.exit(12)"
    backend = model.SubprocessBackend(
        config_path=None, actor="human:me", prefix=[sys.executable, "-c", code]
    )
    result = backend.call(["list"])
    assert result.exit_code == 12 and result.payload is None
    assert result.describe() == "exit 12: boom"
    missing = model.SubprocessBackend(config_path=None, actor="human:me", prefix=["/nonexistent/x"])
    assert missing.call(["list"]).exit_code == 1


def test_subprocess_backend_times_out_and_close_stops_children() -> None:
    backend = model.SubprocessBackend(
        config_path=None,
        actor="human:me",
        prefix=[sys.executable, "-c", "import time;time.sleep(60)"],
    )
    assert backend.call(["list"], timeout=0.5).exit_code == 124
    backend.close()
    assert backend.call(["list"]).exit_code == 130


def test_error_description_prefers_the_envelope_and_escapes() -> None:
    result = model.CliResult(10, {"error": {"code": "not_found", "message": HOSTILE}})
    text = result.describe()
    assert text.startswith("exit 10: ") and "\x1b" not in text and len(text) < 400
    assert result.error_code == "not_found"


def test_parse_agents_keeps_the_first_of_a_duplicate_id() -> None:
    first = agent("ag-aaaaaaaa", name="first")
    again = agent("ag-aaaaaaaa", name="second")
    other = agent("ag-bbbbbbbb")
    parsed = model.parse_agents({"agents": [first, again, other]})
    assert [a["name"] for a in parsed] == ["first", other["name"]]


def test_deeply_nested_json_is_junk_not_a_crash() -> None:
    body = b"[" * 100000 + b"]" * 100000
    result = model._result_of(0, body, b"")
    assert result.exit_code == 0 and result.payload is None


def test_shown_survives_values_json_cannot_dump() -> None:
    circular: dict[str, object] = {}
    circular["self"] = circular
    deep: list[object] = []
    for _ in range(100000):
        deep = [deep]
    for odd in (circular, deep, {("tuple", "key"): 1}):
        text = model.shown(odd)
        assert isinstance(text, str) and len(text) <= model.MAX_VALUE


def make_backend(code: str) -> model.SubprocessBackend:
    return model.SubprocessBackend(
        config_path=None, actor="human:me", prefix=[sys.executable, "-I", "-c", code]
    )


def test_a_grandchild_holding_the_pipes_does_not_block_the_call(tmp_path: Path) -> None:
    pidfile = tmp_path / "grandchild.pid"
    code = (
        "import subprocess,sys;"
        "g=subprocess.Popen([sys.executable,'-I','-c','import time;time.sleep(60)']);"
        f"open({str(pidfile)!r},'w').write(str(g.pid));"
        "print('{\"ok\": true}')"
    )
    backend = make_backend(code)
    try:
        result = backend.call(["list"], timeout=30)
        assert result.ok and result.payload == {"ok": True}
    finally:
        _reap(pidfile)


def _reap(pidfile: Path) -> None:
    import os
    import signal

    if pidfile.exists():
        pid = int(pidfile.read_text())  # the exact grandchild this test started
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def test_close_kills_a_child_that_ignores_sigterm() -> None:
    import threading
    import time

    code = (
        "import signal,sys,time;signal.signal(signal.SIGTERM, signal.SIG_IGN);"
        "print('up', flush=True);time.sleep(60)"
    )
    backend = make_backend(code)
    results: list[model.CliResult] = []
    thread = threading.Thread(target=lambda: results.append(backend.call(["list"])))
    thread.start()
    deadline = time.monotonic() + 30
    while not backend._children and time.monotonic() < deadline:  # noqa: SLF001
        time.sleep(0.01)
    time.sleep(0.5)  # let the child install its handler
    backend.close()
    thread.join(timeout=30)
    assert not thread.is_alive()
    assert results and not results[0].ok


def test_a_huge_output_is_cut_off_and_reported() -> None:
    code = "import sys;sys.stdout.write('x' * (64 * 1024 * 1024))"
    result = make_backend(code).call(["list"], timeout=60)
    assert result.exit_code != 0 and "too much output" in result.message


def test_a_timeout_does_not_wait_for_a_grandchild_either(tmp_path: Path) -> None:
    pidfile = tmp_path / "grandchild.pid"
    code = (
        "import subprocess,sys,time;"
        "g=subprocess.Popen([sys.executable,'-I','-c','import time;time.sleep(60)']);"
        f"open({str(pidfile)!r},'w').write(str(g.pid));time.sleep(60)"
    )
    try:
        result = make_backend(code).call(["list"], timeout=1)
        assert result.exit_code == 124
    finally:
        _reap(pidfile)
