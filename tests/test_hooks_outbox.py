"""Hook processes, selectors, payloads, pre-hooks and outbox delivery (issue #10)."""

from __future__ import annotations

import errno
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from agent_spawn import hooks, procs
from agent_spawn.config import Hooks
from agent_spawn.gateway import SystemGateway
from agent_spawn.hooks import (
    AsyncRunner,
    HookEntry,
    agent_event,
    deliver_outbox,
    entry_matches,
    invocation_entries,
    parse_invocation_hooks,
    redact_payload,
    run_hook_process,
    run_pre_hooks,
    verdict_error,
)
from agent_spawn.output import CliError
from agent_spawn.store import OutboxEvent, Store
from tests import schema_registry
from tests.fixtures import TS, sample_agent, sample_outcome
from tests.procguard import ProcTracker, descendants

ID = "ag-abcdefgh"
OTHER = "ag-abcdefg2"


# -- fixtures and helpers ---------------------------------------------------------------------


@pytest.fixture
def state_dir(tmp_path: Path) -> Path:
    return tmp_path / "state"


@pytest.fixture
def store(state_dir: Path) -> Iterator[Store]:
    with Store.open(state_dir) as opened:
        opened.insert_agent(ID, {"name": "agent", "status": "parked"})
        yield opened


def config(**entries: list[dict[str, Any]]) -> Hooks:
    return Hooks.model_validate({name.replace("_", "-"): value for name, value in entries.items()})


def recorder(path: Path) -> str:
    """A hook command appending its stdin (one JSON line) to ``path``."""
    return f"cat >> {shlex.quote(str(path))}; echo >> {shlex.quote(str(path))}"


def lines(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def log_records(state_dir: Path) -> list[dict[str, Any]]:
    return lines(state_dir / "log.jsonl")


def _log_time(record: dict[str, Any]) -> datetime:
    return datetime.fromisoformat(record["ts"])


def track(tracker: ProcTracker, *paths: Path) -> list[int]:
    """Record the PIDs a hook wrote (one per line) so teardown kills exactly those."""
    found: list[int] = []
    for path in paths:
        if not path.exists():
            continue
        for text in path.read_text().split():
            pid = int(text)
            found.append(pid)
            handle = procs.identify(pid)
            if handle is not None:
                with handle:
                    tracker.record(handle.identity)
    return found


def alive(pid: int) -> bool:
    handle = procs.identify(pid)
    if handle is None:
        return False
    with handle:
        return handle.is_alive()


def assert_dead(pids: list[int], timeout: float = 3.0) -> None:
    end = time.monotonic() + timeout
    while any(alive(p) for p in pids):
        assert time.monotonic() < end, f"still alive: {[p for p in pids if alive(p)]}"
        time.sleep(0.02)


def wait_for(path: Path, timeout: float = 10.0) -> None:
    end = time.monotonic() + timeout
    while not path.exists():
        assert time.monotonic() < end, f"{path} never appeared"
        time.sleep(0.02)


def wait_for_text(path: Path, timeout: float = 60.0) -> str:
    """Wait until ``path`` has content (a shell creates a file, then writes it); return it."""
    end = time.monotonic() + timeout
    while not (path.exists() and (text := path.read_text())):
        assert time.monotonic() < end, f"{path} never had content"
        time.sleep(0.02)
    return text


def quiesce(timeout: float = 8.0) -> None:
    """Wait until this process has no live (non-zombie) descendants."""
    end = time.monotonic() + timeout
    while any(info.state != "Z" for info in descendants(os.getpid()).values()):
        assert time.monotonic() < end, "descendants still running"
        time.sleep(0.02)


def open_fds() -> list[str]:
    return sorted(os.listdir("/proc/self/fd"))


def queue(store: Store, *events: OutboxEvent, agent_id: str = ID) -> None:
    store.add_events(agent_id, list(events))


def event(hook: str = "post-park", **agent: Any) -> OutboxEvent:
    if hook == "post-escalate":
        agent.setdefault("needs_attention", "blocked")
    return agent_event(
        hook,
        sample_agent(**agent),
        reads_third_party=False,
        extra=_extra(hook),
    )


def _extra(hook: str) -> dict[str, Any]:
    return {
        "post-launch": {
            "run": 1,
            "resumed": False,
            "fork": False,
            "promoted": False,
            "auto_resumed": False,
        },
        "post-state-change": {
            "change": "agent",
            "previous_status": "live",
            "previous_needs_attention": None,
        },
        "post-escalate": {"outcome": sample_outcome(), "chain_failure": None},
        "post-park": {
            "reason": "done",
            "outcome": sample_outcome(),
            "teardown_refused": False,
            "chain_failure": None,
        },
        "post-teardown": {"worktrees": [], "merged_head": None, "force": False},
    }[hook]


# -- payloads validate against the schemas ----------------------------------------------------


@pytest.mark.parametrize("hook", hooks.POST_HOOKS)
def test_every_delivered_payload_validates_and_carries_tainted(
    store: Store, state_dir: Path, tmp_path: Path, hook: str
) -> None:
    out = tmp_path / "out.jsonl"
    queue(store, event(hook))
    result = deliver_outbox(
        store, ID, config(**{hook: [{"cmd": recorder(out)}]}), state_dir=state_dir
    )
    assert result.delivered == 1 and result.hooks_failed == 0
    (payload,) = lines(out)
    schema_registry.validate(payload, f"agent-spawn/hooks/{hook}.json")
    assert payload["tainted"] is False
    assert payload["hook"] == hook and payload["object_id"] == ID
    assert isinstance(payload["event_id"], int)


def test_session_state_change_payload_validates(
    store: Store, state_dir: Path, tmp_path: Path
) -> None:
    out = tmp_path / "out.jsonl"
    extra = {
        "change": "session",
        "csession_id": "cs-abcdefgh",
        "run": 1,
        "state": "working",
        "previous_state": "starting",
        "activity": "working",
        "needs_input_reason": None,
        "final": False,
        "cause": "Stop",
        "report": None,
        "session_event_id": 3,
        "event_ts": TS,
        "phase": "session",
        "outcome": None,
        "end_reason": None,
        "exit_code": None,
        "message": None,
        "sandbox_active": False,
    }
    queue(
        store,
        agent_event("post-state-change", sample_agent(), reads_third_party=False, extra=extra),
    )
    deliver_outbox(
        store, ID, config(post_state_change=[{"cmd": recorder(out)}]), state_dir=state_dir
    )
    (payload,) = lines(out)
    schema_registry.validate(payload, "agent-spawn/hooks/post-state-change.json")


def test_pre_launch_payload_validates(state_dir: Path, tmp_path: Path) -> None:
    state_dir.mkdir()
    out = tmp_path / "pre.jsonl"
    verdict = run_pre_hooks(
        "pre-launch",
        _pre_launch_body(),
        [HookEntry(cmd=recorder(out) + "; true")],
        state_dir=state_dir,
    )
    assert verdict.verdict == "ok"
    (payload,) = lines(out)
    schema_registry.validate(payload, "agent-spawn/hooks/pre-launch.json")
    assert payload["object_id"] is None and payload["tainted"] is None


def _pre_launch_body(**overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "repo": None,
        "labels": ["pr=1"],
        "kind": "implement",
        "run_kind": "implement",
        "reads_third_party": False,
        "tainted": None,
        "mode": "headless",
        "status": None,
        "needs_attention": None,
        "placement": None,
        "term_placement": None,
        "worktree": "none",
        "branch": None,
        "cwd": None,
        "name": "implement",
        "skill": None,
        "has_prompt": True,
        "input_names": [],
        "plugin_dirs": [],
        "promote": None,
        "ephemeral": False,
        "wait": False,
        "settings": {
            "kind": "implement",
            "mode": "headless",
            "placement": None,
            "term_placement": None,
            "name": None,
            "repo": None,
            "worktree": "none",
            "branch": None,
            "base": None,
            "ref": None,
            "detach": False,
            "worktree_name": None,
            "no_bootstrap": False,
            "copy_env": False,
            "cwd": None,
            "labels": ["pr=1"],
            "skill": None,
            "has_prompt": True,
            "input_names": [],
            "plugin_dirs": [],
            "promote": None,
            "ephemeral": False,
            "wait": False,
            "wait_timeout": None,
        },
        "pass_throughs": {},
        "actor": "auto:test",
        "decision_ids": [],
        "override_escalation": False,
    }
    body.update(overrides)
    return body


# -- redaction --------------------------------------------------------------------------------


def _with_report(**flags: Any) -> dict[str, Any]:
    outcome = sample_outcome()
    return {
        "reads_third_party": False,
        "tainted": False,
        "report": dict(outcome["report"]),
        "outcome": outcome,
        "agent": {"last_outcome": "done"},
        **flags,
    }


@pytest.mark.parametrize(
    ("flags", "redacted"),
    [
        ({}, False),
        ({"reads_third_party": True}, True),
        ({"tainted": True}, True),
        ({"reads_third_party": None}, True),  # unknown is treated as untrusted
        ({"tainted": None}, True),
    ],
)
def test_report_text_is_redacted_unless_known_trusted(
    flags: dict[str, Any], redacted: bool
) -> None:
    body = _with_report(**flags)
    result = redact_payload(body)
    for report in (result["report"], result["outcome"]["report"]):
        assert (report["summary"] is None) is redacted
        assert report["status"] == "done"  # only the two free-text fields go
    assert body["report"]["summary"] is not None  # the input is not mutated


def test_redacted_text_never_reaches_a_hook_or_its_environment(
    store: Store, state_dir: Path, tmp_path: Path
) -> None:
    secret = "SECRET-INJECTION-$(touch PWNED)"
    outcome = sample_outcome()
    outcome["report"]["summary"] = secret
    outcome["report"]["reason"] = secret
    ev = agent_event(
        "post-park",
        sample_agent(tainted=True),
        reads_third_party=False,
        extra={**_extra("post-park"), "outcome": outcome},
    )
    queue(store, ev)
    out, env_dump = tmp_path / "out.jsonl", tmp_path / "env.txt"
    cmd = f"{recorder(out)}; env > {shlex.quote(str(env_dump))}"
    deliver_outbox(store, ID, config(post_park=[{"cmd": cmd}]), state_dir=state_dir)
    (payload,) = lines(out)
    assert payload["outcome"]["report"]["summary"] is None
    assert secret not in out.read_text() and "SECRET-INJECTION" not in env_dump.read_text()


def test_trusted_report_text_is_delivered_on_stdin_only(
    store: Store, state_dir: Path, tmp_path: Path
) -> None:
    secret = '"; touch PWNED; echo "'
    outcome = sample_outcome()
    outcome["report"]["summary"] = secret
    queue(
        store,
        agent_event(
            "post-park",
            sample_agent(),
            reads_third_party=False,
            extra={**_extra("post-park"), "outcome": outcome},
        ),
    )
    out, env_dump, argv_dump = tmp_path / "o.jsonl", tmp_path / "e.txt", tmp_path / "a.txt"
    cmd = (
        f"{recorder(out)}; env > {shlex.quote(str(env_dump))}; "
        f'echo "$0 $*" > {shlex.quote(str(argv_dump))}; cat /proc/$$/cmdline >> '
        f"{shlex.quote(str(argv_dump))}"
    )
    deliver_outbox(store, ID, config(post_park=[{"cmd": cmd}]), state_dir=state_dir)
    assert lines(out)[0]["outcome"]["report"]["summary"] == secret
    assert "touch PWNED" not in env_dump.read_text()
    assert "touch PWNED" not in argv_dump.read_text()
    assert not (state_dir / "PWNED").exists() and not Path("PWNED").exists()


# -- environment, cwd -------------------------------------------------------------------------


def test_environment_and_cwd(store: Store, state_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "env.txt"
    queue(store, event("post-launch"))
    cmd = (
        f'echo "$AGENT_SPAWN_HOOK|$AGENT_SPAWN_ID|$AGENT_SPAWN_OBJECT_TYPE|$PWD" > '
        f"{shlex.quote(str(out))}"
    )
    deliver_outbox(store, ID, config(post_launch=[{"cmd": cmd}]), state_dir=state_dir)
    hook, agent_id, kind, cwd = out.read_text().strip().split("|")
    assert (hook, agent_id, kind) == ("post-launch", ID, "agent")
    assert os.path.realpath(cwd) == os.path.realpath(state_dir)


def test_pre_hook_id_is_empty_and_inherits_environment(state_dir: Path, tmp_path: Path) -> None:
    state_dir.mkdir()
    out = tmp_path / "env.txt"
    cmd = f'echo "[$AGENT_SPAWN_ID][$KEEP_ME]" > {shlex.quote(str(out))}'
    run_pre_hooks(
        "pre-launch",
        _pre_launch_body(),
        [HookEntry(cmd=cmd)],
        state_dir=state_dir,
        environ={**os.environ, "KEEP_ME": "yes"},
    )
    assert out.read_text().strip() == "[][yes]"


def test_hostile_agent_id_never_reaches_the_environment(state_dir: Path) -> None:
    state_dir.mkdir()
    with pytest.raises(ValueError, match="invalid agent id"):
        run_hook_process(
            "true",
            b"{}",
            hook="post-park",
            agent_id="ag-x; touch PWNED",
            timeout=5,
            state_dir=state_dir,
        )


# -- selectors --------------------------------------------------------------------------------


def _payload(**fields: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "hook": "post-state-change",
        "run_kind": "implement",
        "labels": ["pr=1", "team=a"],
        "status": "live",
        "change": "session",
        "state": "working",
    }
    base.update(fields)
    return base


@pytest.mark.parametrize(
    ("entry", "payload", "expected"),
    [
        (HookEntry("x"), _payload(), True),
        (HookEntry("x", kinds=("implement",)), _payload(), True),
        (HookEntry("x", kinds=("review",)), _payload(), False),
        (HookEntry("x", kinds=("implement",)), _payload(run_kind=None), False),
        (HookEntry("x", exclude_kinds=("implement",)), _payload(), False),
        (HookEntry("x", exclude_kinds=("review",)), _payload(), True),
        (HookEntry("x", exclude_kinds=("review",)), _payload(run_kind=None), True),
        (HookEntry("x", labels=("pr=1",)), _payload(), True),
        (HookEntry("x", labels=("pr=1", "team=a")), _payload(), True),
        (HookEntry("x", labels=("pr=1", "team=b")), _payload(), False),
        (HookEntry("x", labels=("pr=1",)), _payload(labels=None), False),
        (HookEntry("x", states=("working", "idle")), _payload(), True),
        (HookEntry("x", states=("idle",)), _payload(), False),
        (HookEntry("x", states=("working",)), _payload(change="agent"), False),
        (HookEntry("x", states=("working",)), _payload(change="agent", state="working"), False),
        (HookEntry("x", statuses=("live", "parked")), _payload(), True),
        (HookEntry("x", statuses=("parked",)), _payload(), False),
        (HookEntry("x", kinds=("implement",), statuses=("parked",)), _payload(), False),
    ],
)
def test_selectors(entry: HookEntry, payload: dict[str, Any], expected: bool) -> None:
    assert entry_matches(entry, payload) is expected


def test_selectors_filter_deliveries(store: Store, state_dir: Path, tmp_path: Path) -> None:
    hit, miss = tmp_path / "hit", tmp_path / "miss"
    queue(store, event("post-park", labels=["pr=1"], run_kind="implement"))
    cfg = config(
        post_park=[
            {"cmd": recorder(hit), "kinds": ["implement"], "labels": ["pr=1"]},
            {"cmd": recorder(miss), "kinds": ["review"]},
            {"cmd": recorder(miss), "exclude_kinds": ["implement"]},
            {"cmd": recorder(miss), "labels": ["pr=2"]},
        ]
    )
    deliver_outbox(store, ID, cfg, state_dir=state_dir)
    assert len(lines(hit)) == 1 and lines(miss) == []


# -- per-invocation hooks ---------------------------------------------------------------------


def test_parse_invocation_hooks() -> None:
    assert parse_invocation_hooks(["post-launch=echo a=b"], ["pre-launch", "post-launch"]) == [
        {"hook": "post-launch", "cmd": "echo a=b"}
    ]
    for bad in ["nocmd", "post-park=x", "post-launch=", "post-launch=  ", "post-launch=a\0b"]:
        with pytest.raises(CliError) as caught:
            parse_invocation_hooks([bad], ["pre-launch", "post-launch"])
        assert caught.value.exit_code == 2


def test_invocation_entries_ignore_malformed_items() -> None:
    stored = [
        {"hook": "post-launch", "cmd": "ok"},
        {"hook": "post-park", "cmd": "other hook"},
        {"hook": "post-launch", "cmd": ""},
        {"hook": "post-launch", "cmd": "a\0b"},
        {"hook": "post-launch"},
        "junk",
        None,
    ]
    assert [e.cmd for e in invocation_entries(stored, "post-launch")] == ["ok"]
    assert invocation_entries("junk", "post-launch") == []
    assert invocation_entries(None, "post-launch") == []


def test_invocation_hooks_run_only_for_their_run(
    store: Store, state_dir: Path, tmp_path: Path
) -> None:
    mine, theirs = tmp_path / "mine", tmp_path / "theirs"
    with store.lease(f"agent:{ID}") as lease:
        store.put_run(lease, ID, 1, {"hooks": [{"hook": "post-launch", "cmd": recorder(mine)}]})
        store.put_run(lease, ID, 2, {"hooks": [{"hook": "post-launch", "cmd": recorder(theirs)}]})
        store.put_run(lease, ID, 3, {})
    body = event("post-launch")
    for run in (1, 3):
        payload = dict(body.payload) | {"run": run}
        queue(store, OutboxEvent("post-launch", payload))
    deliver_outbox(store, ID, config(), state_dir=state_dir)
    assert len(lines(mine)) == 1 and lines(theirs) == []


def test_invocation_hooks_run_in_addition_to_config_and_are_never_async(
    store: Store, state_dir: Path, tmp_path: Path
) -> None:
    cfg_out, inv_out = tmp_path / "cfg", tmp_path / "inv"
    with store.lease(f"agent:{ID}") as lease:
        store.put_run(lease, ID, 1, {"hooks": [{"hook": "post-park", "cmd": recorder(inv_out)}]})
    queue(store, event("post-park"))
    deliver_outbox(store, ID, config(post_park=[{"cmd": recorder(cfg_out)}]), state_dir=state_dir)
    assert len(lines(cfg_out)) == 1 and len(lines(inv_out)) == 1
    assert invocation_entries([{"hook": "post-park", "cmd": "x"}], "post-park")[0].async_ is False


def test_teardown_invocation_hooks_come_from_the_agent(
    store: Store, state_dir: Path, tmp_path: Path
) -> None:
    out = tmp_path / "out"
    with store.lease(f"agent:{ID}") as lease:
        store.update_agent(lease, ID, {"hooks": [{"hook": "post-teardown", "cmd": recorder(out)}]})
    queue(store, event("post-teardown"))
    deliver_outbox(store, ID, config(), state_dir=state_dir)
    assert len(lines(out)) == 1


# -- failures are logged, never raised --------------------------------------------------------


def test_failing_hook_is_logged_and_does_not_stop_later_hooks_or_events(
    store: Store, state_dir: Path, tmp_path: Path
) -> None:
    out = tmp_path / "out"
    queue(store, event("post-park"), event("post-park"))
    cfg = config(post_park=[{"cmd": "echo boom >&2; exit 3"}, {"cmd": recorder(out)}])
    result = deliver_outbox(store, ID, cfg, state_dir=state_dir)
    assert result.delivered == 2 and result.hooks_failed == 2
    assert len(lines(out)) == 2 and store.pending_events(ID) == []
    failed = [r for r in log_records(state_dir) if r["event"] == "hook_failed"]
    assert failed[0]["details"]["exit_code"] == 3 and failed[0]["details"]["stderr"] == "boom\n"
    assert "cmd" not in json.dumps(failed[0])
    for record in log_records(state_dir):
        schema_registry.validate(record, "agent-spawn/log.json")


def test_huge_stderr_and_stdout_are_capped_and_the_log_survives(
    store: Store, state_dir: Path
) -> None:
    queue(store, event("post-park"))
    cmd = (
        "head -c 20000000 /dev/zero | tr '\\0' 'x' >&2; "
        "head -c 20000000 /dev/zero | tr '\\0' 'y'; exit 2"
    )
    result = deliver_outbox(store, ID, config(post_park=[{"cmd": cmd}]), state_dir=state_dir)
    assert result.delivered == 1
    (record,) = [r for r in log_records(state_dir) if r["event"] == "hook_failed"]
    assert len(record["details"]["stderr"].encode()) <= hooks.LOG_TEXT_LIMIT


def test_capture_is_capped_in_the_result(state_dir: Path) -> None:
    state_dir.mkdir()
    res = run_hook_process(
        "head -c 5000000 /dev/zero >&2; head -c 5000000 /dev/zero",
        b"{}",
        hook="post-park",
        agent_id=ID,
        timeout=20,
        state_dir=state_dir,
    )
    assert len(res.stderr) == hooks.CAPTURE_LIMIT and res.stderr_truncated
    assert len(res.stdout) == hooks.CAPTURE_LIMIT and res.stdout_truncated


def test_hook_that_never_reads_a_huge_stdin_does_not_block(store: Store, state_dir: Path) -> None:
    ev = event("post-park")
    payload = dict(ev.payload)
    payload["agent"] = {**payload["agent"], "name": "n" * 3_000_000}
    queue(store, OutboxEvent("post-park", payload))
    start = time.monotonic()
    result = deliver_outbox(store, ID, config(post_park=[{"cmd": "true"}]), state_dir=state_dir)
    assert result.delivered == 1 and result.hooks_failed == 0
    assert time.monotonic() - start < 10


def test_hook_reads_a_huge_stdin_completely(store: Store, state_dir: Path, tmp_path: Path) -> None:
    ev = event("post-park")
    payload = dict(ev.payload)
    payload["agent"] = {**payload["agent"], "name": "é" * 500_000}
    queue(store, OutboxEvent("post-park", payload))
    out = tmp_path / "out.jsonl"
    deliver_outbox(store, ID, config(post_park=[{"cmd": recorder(out)}]), state_dir=state_dir)
    (got,) = lines(out)
    assert got["agent"]["name"] == "é" * 500_000


def test_cmd_with_nul_and_unusual_unicode(state_dir: Path) -> None:
    state_dir.mkdir()
    res = run_hook_process(
        "a\0b", b"{}", hook="post-park", agent_id=ID, timeout=5, state_dir=state_dir
    )
    assert res.error and res.exit_code is None
    res = run_hook_process(
        "printf 'ü\\377' >&2; exit 1",
        b"{}",
        hook="post-park",
        agent_id=ID,
        timeout=5,
        state_dir=state_dir,
    )
    assert res.exit_code == 1 and res.stderr == b"\xc3\xbc\xff"


def test_malformed_and_unknown_events_are_skipped_and_marked_delivered(
    store: Store, state_dir: Path, tmp_path: Path
) -> None:
    out = tmp_path / "out"
    store._conn.execute(  # noqa: SLF001 - plant rows the writers would never write
        "INSERT INTO events (agent_id, event, payload, created) "
        "VALUES (?, 'post-park', '\"x\"', 1)",
        (ID,),
    )
    store._conn.execute(
        "INSERT INTO events (agent_id, event, payload, created) VALUES (?, 'bogus', '{}', 1)",
        (ID,),
    )
    store._conn.commit()
    queue(store, event("post-park"))
    result = deliver_outbox(
        store, ID, config(post_park=[{"cmd": recorder(out)}]), state_dir=state_dir
    )
    assert result.delivered == 3 and len(lines(out)) == 1


# -- timeouts and process groups --------------------------------------------------------------


def _script(tmp_path: Path, body: str) -> str:
    path = tmp_path / "hook.sh"
    path.write_text(body)
    return f"/bin/sh {shlex.quote(str(path))}"


def test_timeout_kills_the_whole_group_including_children(
    tracker_dir: Path, tracker: ProcTracker, state_dir: Path
) -> None:
    state_dir.mkdir()
    pids = tracker_dir / "pids"
    cmd = (
        f"echo $$ >> {shlex.quote(str(pids))}; "
        f"sleep 60 & echo $! >> {shlex.quote(str(pids))}; "
        f"sleep 60 & echo $! >> {shlex.quote(str(pids))}; wait"
    )
    sentinel = tracker.popen(["sleep", "60"])  # another group of ours: must survive
    try:
        res = run_hook_process(
            cmd, b"{}", hook="post-park", agent_id=ID, timeout=0.5, state_dir=state_dir
        )
        found = track(tracker, pids)
    finally:
        pass
    assert res.timed_out and len(found) == 3
    assert_dead(found)
    assert sentinel.poll() is None
    signals = [r for r in log_records(state_dir) if r["event"] == "signal_sent"]
    assert signals and all(r["pid"] == found[0] for r in signals)
    assert signals[0]["signal"] == "SIGTERM"


def test_hook_ignoring_sigterm_is_killed(
    tracker_dir: Path, tracker: ProcTracker, state_dir: Path
) -> None:
    state_dir.mkdir()
    pids = tracker_dir / "pids"
    cmd = f"trap '' TERM; echo $$ >> {shlex.quote(str(pids))}; while :; do sleep 1; done"
    start = time.monotonic()
    res = run_hook_process(
        cmd, b"{}", hook="post-park", agent_id=ID, timeout=0.4, state_dir=state_dir
    )
    found = track(tracker, pids)
    assert res.timed_out and res.signal == signal.SIGKILL
    # Whole-test hang guard only; loose so that a loaded machine (parallel workers) cannot fail it.
    assert time.monotonic() - start < hooks.TERM_GRACE + 30
    assert_dead(found)
    sent = [r for r in log_records(state_dir) if r["event"] == "signal_sent"]
    assert [r["signal"] for r in sent] == ["SIGTERM", "SIGKILL"]
    # The grace itself: the logged SIGKILL follows the logged SIGTERM by about TERM_GRACE (the
    # timestamps have millisecond precision; the upper bound leaves room for a loaded machine).
    gap = (_log_time(sent[1]) - _log_time(sent[0])).total_seconds()
    assert hooks.TERM_GRACE - 0.1 <= gap < hooks.TERM_GRACE + 10


def test_cooperative_hook_exits_on_sigterm_without_waiting_for_the_grace(
    state_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state_dir.mkdir()
    # A long grace makes the margin independent of machine speed: waiting it out would take 60 s.
    monkeypatch.setattr(hooks, "TERM_GRACE", 60.0)
    start = time.monotonic()
    res = run_hook_process(
        "sleep 120", b"{}", hook="post-park", agent_id=ID, timeout=0.3, state_dir=state_dir
    )
    assert res.timed_out and time.monotonic() - start < 30


def test_leader_exits_but_child_holds_the_pipe(
    tracker_dir: Path, tracker: ProcTracker, state_dir: Path, tmp_path: Path
) -> None:
    state_dir.mkdir()
    pids = tracker_dir / "pids"
    exited = tmp_path / "exited"
    cmd = (
        f"sleep 60 & echo $! > {shlex.quote(str(pids))}; echo done; "
        f"date +%s.%N > {shlex.quote(str(exited))}; exit 0"
    )
    res = run_hook_process(
        cmd, b"{}", hook="post-park", agent_id=ID, timeout=50, state_dir=state_dir
    )
    returned = time.time()
    track(tracker, pids)
    assert res.ok and res.stdout == b"done\n"
    # The child's 60 s sleep (and the 50 s timeout) is not waited for: the call returns about
    # DRAIN_GRACE after the leader exited. The lower bound proves the drain happens; the upper
    # bound is a few seconds, so a drain that overstays fails while a loaded machine passes.
    drain = returned - float(exited.read_text())
    assert hooks.DRAIN_GRACE - 0.1 <= drain < hooks.DRAIN_GRACE + 3


def test_exit_status_and_signal_death(state_dir: Path) -> None:
    state_dir.mkdir()
    res = run_hook_process(
        "exit 7", b"{}", hook="post-park", agent_id=ID, timeout=5, state_dir=state_dir
    )
    assert res.exit_code == 7 and not res.ok
    res = run_hook_process(
        "kill -9 $$", b"{}", hook="post-park", agent_id=ID, timeout=5, state_dir=state_dir
    )
    assert res.exit_code is None and res.signal == signal.SIGKILL and not res.timed_out


def test_group_signal_refuses_anything_unverified(
    tracker: ProcTracker,
) -> None:
    gw = hooks.get_gateway()
    # Not a group leader (shares pytest's group): never signalled, so pytest survives.
    shared = tracker.popen(["sleep", "60"])
    group = hooks._Group(shared, gw)  # noqa: SLF001
    assert group.identity is not None
    assert group.signal(signal.SIGKILL) is False
    assert shared.poll() is None
    # A leader whose recorded identity no longer matches (a reused PID).
    leader = tracker.popen(["sleep", "60"], start_new_session=True)
    group = hooks._Group(leader, gw)  # noqa: SLF001
    assert group.identity is not None
    group.identity = procs.ProcessIdentity(
        group.identity.pid, group.identity.start_time + 1, group.identity.boot_id
    )
    assert group.signal(signal.SIGKILL) is False and leader.poll() is None
    # Never identified (it was already gone at spawn): nothing is sent.
    group.identity = None
    assert group.signal(signal.SIGKILL) is False and leader.poll() is None
    # A reaped leader's PID may be anyone's.
    done = tracker.popen(["true"], start_new_session=True)
    group = hooks._Group(done, gw)  # noqa: SLF001
    done.wait()
    assert group.signal(signal.SIGKILL) is False


@pytest.fixture
def tracker(proc_tracker: ProcTracker) -> ProcTracker:
    return proc_tracker


@pytest.fixture
def tracker_dir(tmp_path: Path) -> Path:
    return tmp_path


# -- async hooks ------------------------------------------------------------------------------


def test_async_hook_is_detached_and_counts_as_delivered(
    store: Store, state_dir: Path, tmp_path: Path, tracker: ProcTracker
) -> None:
    out = tmp_path / "out.jsonl"
    queue(store, event("post-park"))
    runner = AsyncRunner(state_dir)
    gate = tmp_path / "gate"  # the hook waits for this file, so it is still running on return
    cmd = f"while [ ! -e {shlex.quote(str(gate))} ]; do sleep 0.05; done; {recorder(out)}"
    result = deliver_outbox(
        store,
        ID,
        config(post_park=[{"cmd": cmd, "async": True}]),
        state_dir=state_dir,
        runner=runner,
    )
    try:
        assert result.delivered == 1
        assert not out.exists()  # delivery did not wait for the hook
        assert runner.poll() == 1
    finally:
        gate.touch()  # open the gate even when an assertion failed: no polling shell is leaked
    assert runner.wait(10) == 0
    assert len(lines(out)) == 1
    assert track(tracker) == []


def test_async_hook_timeout_is_enforced_while_the_starter_lives(
    tmp_path: Path, state_dir: Path, tracker: ProcTracker
) -> None:
    state_dir.mkdir()
    pids = tmp_path / "pids"
    runner = AsyncRunner(state_dir)
    error = runner.start(
        f"echo $$ > {shlex.quote(str(pids))}; sleep 60 & echo $! >> {shlex.quote(str(pids))}; wait",
        b"{}",
        hook="post-park",
        agent_id=ID,
        timeout=0.3,
    )
    assert error is None
    end = time.monotonic() + 60
    while len(pids.read_text().split() if pids.exists() else []) < 2:  # shell and its sleep
        assert time.monotonic() < end, "the hook never wrote both pids"
        time.sleep(0.02)
    found = track(tracker, pids)
    assert found and runner.wait(10) == 0
    assert_dead(found)
    assert any(r["event"] == "hook_timeout" for r in log_records(state_dir))


def test_async_timeout_is_reported_even_when_the_watchdog_fires_before_poll(
    state_dir: Path, tmp_path: Path, tracker: ProcTracker
) -> None:
    state_dir.mkdir()
    pids = tmp_path / "pids"
    runner = AsyncRunner(state_dir)
    cmd = f"echo $$ > {shlex.quote(str(pids))}; sleep 60"
    assert runner.start(cmd, b"{}", hook="post-park", agent_id=ID, timeout=0.3) is None
    wait_for_text(pids)
    found = track(tracker, pids)
    # The watchdog (1 s, rounded up) kills the group before the first poll: wait for the hook's
    # shell to be dead (nothing has polled or reaped it: it is a zombie) before the first look.
    assert_dead(found, timeout=60)
    assert runner.wait(10) == 0
    events = [r["event"] for r in log_records(state_dir)]
    assert "hook_timeout" in events and "hook_failed" not in events
    quiesce()  # the watchdog subshell lingers through its SIGKILL grace


def test_async_command_killed_by_a_signal_reports_the_signal(state_dir: Path) -> None:
    state_dir.mkdir()
    runner = AsyncRunner(state_dir)
    runner.start("kill -s KILL $$", b"{}", hook="post-park", agent_id=ID, timeout=30)
    assert runner.wait(10) == 0
    (record,) = [r for r in log_records(state_dir) if r["event"] == "hook_failed"]
    assert record["details"]["signal"] == 9


def test_async_failure_is_logged(state_dir: Path) -> None:
    state_dir.mkdir()
    runner = AsyncRunner(state_dir)
    runner.start("exit 4", b"{}", hook="post-park", agent_id=ID, timeout=5)
    runner.wait(5)
    (record,) = [r for r in log_records(state_dir) if r["event"] == "hook_failed"]
    assert record["details"]["exit_code"] == 4 and record["details"]["async"] is True


def test_async_without_a_runner_still_runs_the_hook(
    store: Store, state_dir: Path, tmp_path: Path
) -> None:
    out = tmp_path / "out.jsonl"
    queue(store, event("post-park"))
    result = deliver_outbox(
        store, ID, config(post_park=[{"cmd": recorder(out), "async": True}]), state_dir=state_dir
    )
    assert result.delivered == 1 and result.hooks_failed == 0
    quiesce()
    assert len(lines(out)) == 1


# -- pre-hooks --------------------------------------------------------------------------------


def _pre(state_dir: Path, *cmds: str, **body: Any) -> hooks.PreHookVerdict:
    state_dir.mkdir(exist_ok=True)
    return run_pre_hooks(
        "pre-launch",
        _pre_launch_body(**body),
        [HookEntry(cmd=c, timeout=5) for c in cmds],
        state_dir=state_dir,
    )


def test_pre_hook_veto_exits_11_creates_nothing_and_cuts_stderr(state_dir: Path) -> None:
    verdict = _pre(state_dir, "head -c 20000 /dev/zero | tr '\\0' 'e' >&2; exit 1")
    assert verdict.verdict == "veto" and verdict.cause == "exit"
    error = verdict_error(verdict)
    assert error is not None and error.exit_code == 11 and error.code == "vetoed"
    assert error.hook == "pre-launch"
    assert error.hook_stderr is not None and len(error.hook_stderr.encode()) == 4096
    schema_registry.validate(error.envelope(), "agent-spawn/cli/error.json")


def test_pre_hook_escalation_exits_13_after_running_the_remaining_hooks(
    state_dir: Path, tmp_path: Path
) -> None:
    out = tmp_path / "later"
    verdict = _pre(state_dir, "echo why >&2; exit 13", recorder(out))
    assert verdict.verdict == "escalate"
    assert len(lines(out)) == 1  # the remaining hook still ran
    error = verdict_error(verdict)
    assert error is not None and error.exit_code == 13 and error.code == "escalated"
    assert error.hook_stderr == "why\n"
    schema_registry.validate(error.envelope(), "agent-spawn/cli/error.json")


def test_a_later_veto_beats_an_earlier_escalation_and_stops_the_rest(
    state_dir: Path, tmp_path: Path
) -> None:
    out = tmp_path / "never"
    verdict = _pre(state_dir, "exit 13", "exit 2", recorder(out))
    assert verdict.verdict == "veto" and verdict.exit_code == 2
    assert lines(out) == []


def test_pre_hook_timeout_vetoes(state_dir: Path) -> None:
    state_dir.mkdir()
    verdict = run_pre_hooks(
        "pre-launch",
        _pre_launch_body(),
        [HookEntry(cmd="sleep 30", timeout=0.3)],
        state_dir=state_dir,
    )
    assert verdict.verdict == "veto" and verdict.cause == "timeout"
    error = verdict_error(verdict)
    assert error is not None and error.exit_code == 11
    assert any(r["event"] == "hook_timeout" for r in log_records(state_dir))


def test_pre_hook_killed_by_a_signal_vetoes(state_dir: Path) -> None:
    verdict = _pre(state_dir, "kill -9 $$")
    assert verdict.verdict == "veto" and verdict.cause == "signal"


def test_pre_hook_passes_and_collects_decision_ids(state_dir: Path) -> None:
    verdict = _pre(
        state_dir,
        """echo '{"decision_id": "d-1"}'""",
        """echo '{"decision_id": ["d-2", "bad id!", 7, "d-1"]}'""",
        "echo not json",
        """echo '{"decision_id": "d-3"}'; exit 13""",
    )
    assert verdict.verdict == "escalate" and verdict.decision_ids == ("d-1", "d-2", "d-3")
    ok = _pre(state_dir, """echo '{"decision_id": "d-9"}'""")
    assert ok.verdict == "ok" and ok.decision_ids == ("d-9",) and verdict_error(ok) is None


def test_pre_hook_selectors_apply(state_dir: Path) -> None:
    state_dir.mkdir(exist_ok=True)
    entries = [HookEntry(cmd="exit 1", kinds=("review",)), HookEntry(cmd="exit 1", labels=("x=1",))]
    verdict = run_pre_hooks("pre-launch", _pre_launch_body(), entries, state_dir=state_dir)
    assert verdict.verdict == "ok"


def test_pre_hook_names_are_checked(state_dir: Path) -> None:
    with pytest.raises(ValueError):
        run_pre_hooks("post-park", {}, [], state_dir=state_dir)
    with pytest.raises(ValueError):
        agent_event("pre-launch", sample_agent(), reads_third_party=None)
    with pytest.raises(ValueError):
        hooks.config_entries(config(), "nope")


def test_config_entries_carry_selectors_and_async() -> None:
    cfg = config(
        post_state_change=[
            {
                "cmd": "c",
                "async": True,
                "timeout": "5s",
                "states": ["idle"],
                "statuses": ["live"],
                "kinds": ["review"],
                "labels": ["a=b"],
            }
        ]
    )
    (entry,) = hooks.config_entries(cfg, "post-state-change")
    assert entry == HookEntry(
        "c",
        timeout=5.0,
        async_=True,
        kinds=("review",),
        labels=("a=b",),
        states=("idle",),
        statuses=("live",),
    )


# -- outbox: lease, order, at-least-once ------------------------------------------------------

_DELIVER = """
import sys
from agent_spawn.config import Hooks
from agent_spawn.hooks import deliver_outbox
from agent_spawn.store import Store
state_dir, agent_id, cmd = sys.argv[1:4]
hooks = Hooks.model_validate({"post-park": [{"cmd": cmd, "timeout": 60}]})
with Store.open(state_dir) as store:
    result = deliver_outbox(store, agent_id, hooks, state_dir=state_dir)
    print(result.delivered, result.lease_held_elsewhere, flush=True)
"""


def _deliverer(tracker: ProcTracker, state_dir: Path, cmd: str) -> subprocess.Popen[bytes]:
    return tracker.popen(
        [sys.executable, "-c", _DELIVER, str(state_dir), ID, cmd],
        stdout=subprocess.PIPE,
        env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
    )


def numbered_events(count: int) -> list[OutboxEvent]:
    return [event("post-park") for _ in range(count)]


def test_events_are_delivered_in_event_id_order(
    store: Store, state_dir: Path, tmp_path: Path
) -> None:
    queue(store, *numbered_events(5))
    out = tmp_path / "out.jsonl"
    deliver_outbox(store, ID, config(post_park=[{"cmd": recorder(out)}]), state_dir=state_dir)
    ids = [p["event_id"] for p in lines(out)]
    assert ids == sorted(ids) and len(set(ids)) == 5


def test_an_event_is_delivered_once_to_a_second_pass(
    store: Store, state_dir: Path, tmp_path: Path
) -> None:
    queue(store, event("post-park"))
    out = tmp_path / "out.jsonl"
    cfg = config(post_park=[{"cmd": recorder(out)}])
    deliver_outbox(store, ID, cfg, state_dir=state_dir)
    again = deliver_outbox(store, ID, cfg, state_dir=state_dir)
    assert again.delivered == 0 and len(lines(out)) == 1


def test_a_held_lease_leaves_events_to_the_holder(
    store: Store, state_dir: Path, tmp_path: Path
) -> None:
    queue(store, event("post-park"))
    out = tmp_path / "out.jsonl"
    with store.lease(f"delivery:{ID}"):
        result = deliver_outbox(
            store, ID, config(post_park=[{"cmd": recorder(out)}]), state_dir=state_dir
        )
    assert result.lease_held_elsewhere and result.delivered == 0
    assert lines(out) == [] and len(store.pending_events(ID)) == 1


def test_two_delivering_processes_keep_the_order(
    store: Store, state_dir: Path, tmp_path: Path, tracker: ProcTracker
) -> None:
    queue(store, *numbered_events(3))
    out, started = tmp_path / "out.jsonl", tmp_path / "started"
    slow = f"touch {shlex.quote(str(started))}; sleep 0.3; {recorder(out)}"
    first = _deliverer(tracker, state_dir, slow)
    wait_for(started)
    # A second process while the first holds the lease: it leaves the events to the holder.
    second = deliver_outbox(
        store, ID, config(post_park=[{"cmd": recorder(out)}]), state_dir=state_dir
    )
    assert second.lease_held_elsewhere and second.delivered == 0
    assert first.wait(timeout=30) == 0
    ids = [p["event_id"] for p in lines(out)]
    assert len(ids) == 3 and ids == sorted(ids)
    assert store.pending_events(ID) == []


def test_two_processes_started_together_deliver_each_event_once_in_order(
    store: Store, state_dir: Path, tmp_path: Path, tracker: ProcTracker
) -> None:
    queue(store, *numbered_events(4))
    out = tmp_path / "out.jsonl"
    cmd = f"sleep 0.1; {recorder(out)}"
    both = [_deliverer(tracker, state_dir, cmd) for _ in range(2)]
    for child in both:
        assert child.wait(timeout=60) == 0
    ids = [p["event_id"] for p in lines(out)]
    assert len(ids) == 4 and ids == sorted(ids)


def test_redelivery_after_the_delivering_process_was_killed(
    store: Store, state_dir: Path, tmp_path: Path, tracker: ProcTracker
) -> None:
    queue(store, *numbered_events(2))
    out, started, pids = tmp_path / "out.jsonl", tmp_path / "started", tmp_path / "pids"
    # The first hook run records the event, then hangs: the deliverer dies before marking it.
    hang = (
        f"{recorder(out)}; echo $$ >> {shlex.quote(str(pids))}; "
        f"sleep 60 & echo $! >> {shlex.quote(str(pids))}; "
        f"touch {shlex.quote(str(started))}; wait"
    )
    victim = _deliverer(tracker, state_dir, hang)
    wait_for(started)
    time.sleep(0.2)
    victim.send_signal(signal.SIGKILL)  # the exact child this test started
    victim.wait(timeout=10)
    track(tracker, pids)
    assert len(store.pending_events(ID)) == 2  # nothing was marked delivered
    # The next deliverer takes over the dead holder's lease and delivers both, again.
    result = deliver_outbox(
        store, ID, config(post_park=[{"cmd": recorder(out)}]), state_dir=state_dir
    )
    assert result.delivered == 2 and not result.lease_held_elsewhere
    seen = [p["event_id"] for p in lines(out)]
    assert seen[0] == seen[1]  # the interrupted event came a second time (at least once)
    assert len(set(seen)) == 2
    assert store.pending_events(ID) == []


def test_events_queued_while_delivering_are_picked_up_by_the_holder(
    store: Store, state_dir: Path, tmp_path: Path
) -> None:
    out = tmp_path / "out.jsonl"
    queue(store, event("post-park"))
    late = tmp_path / "late"

    # The first hook queues another event from a second connection while the lease is held.
    script = tmp_path / "add.py"
    script.write_text(
        "import sys\n"
        "from agent_spawn.store import Store, OutboxEvent\n"
        "import json\n"
        "payload = json.loads(sys.stdin.read())\n"
        "payload.pop('event_id'); payload.pop('ts')\n"
        "from agent_spawn.hooks import agent_event\n"
        "with Store.open(sys.argv[1]) as s:\n"
        f"    s.add_events('{ID}', [OutboxEvent('post-park', payload)])\n"
    )
    marker, saved = shlex.quote(str(late)), shlex.quote(str(tmp_path / "saved"))
    cmd = (
        f"cat > {saved}; if [ ! -e {marker} ]; then touch {marker}; "
        f"PYTHONPATH={shlex.quote(os.pathsep.join(sys.path))} "
        f"{shlex.quote(sys.executable)} {shlex.quote(str(script))} "
        f"{shlex.quote(str(state_dir))} < {saved}; fi; "
        f"cat {saved} >> {shlex.quote(str(out))}; echo >> {shlex.quote(str(out))}"
    )
    result = deliver_outbox(store, ID, config(post_park=[{"cmd": cmd}]), state_dir=state_dir)
    assert result.delivered == 2 and len(lines(out)) == 2


def test_store_errors_are_logged_not_raised(
    store: Store, state_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from agent_spawn.store import StoreBusy

    def boom(*_a: object, **_k: object) -> None:
        raise StoreBusy("database is locked")

    monkeypatch.setattr(store, "pending_events", boom)
    result = deliver_outbox(store, ID, config(post_park=[{"cmd": "true"}]), state_dir=state_dir)
    assert result.error and "StoreBusy" in result.error
    assert any(r["event"] == "warning" for r in log_records(state_dir))
    assert store.lease_holder(f"delivery:{ID}") is None  # the lease was released


def test_event_ids_are_unique_across_agents(store: Store, state_dir: Path, tmp_path: Path) -> None:
    store.insert_agent(OTHER, {"name": "other", "status": "parked"})
    queue(store, event("post-park"))
    queue(store, event("post-park", agent_id=OTHER), agent_id=OTHER)
    out = tmp_path / "out.jsonl"
    cfg = config(post_park=[{"cmd": recorder(out)}])
    deliver_outbox(store, ID, cfg, state_dir=state_dir)
    deliver_outbox(store, OTHER, cfg, state_dir=state_dir)
    got = lines(out)
    assert len({p["event_id"] for p in got}) == 2
    assert {p["object_id"] for p in got} == {ID, OTHER}


# todo(#11): test_read_only_commands_do_not_deliver belongs here once `list`, `show` and `wait`
# exist: run each against a store holding a pending outbox event and assert the event stays
# undelivered (the commands are stubs today, so there is nothing behavioral to run).


# -- review round 1: process handling ---------------------------------------------------------

linux_only = pytest.mark.skipif(not sys.platform.startswith("linux"), reason="needs /proc")


def _boom(*_a: object, **_k: object) -> None:
    raise OSError(errno.EMFILE, "too many open files")


def _no_identity_for_hooks(monkeypatch: pytest.MonkeyPatch) -> None:
    """``procs.identify`` finds nothing for the hook module (gateway given), the tests still can."""
    real = procs.identify
    monkeypatch.setattr(
        procs,
        "identify",
        lambda pid, **kw: None if kw.get("gateway") is not None else real(pid, **kw),
    )


def test_identity_lookup_failure_neither_leaks_nor_raises(
    state_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state_dir.mkdir()
    monkeypatch.setattr(procs, "identify", _boom)
    before = open_fds()
    res = run_hook_process(
        "exec sleep 60", b"{}", hook="post-park", agent_id=ID, timeout=0.4, state_dir=state_dir
    )
    assert res.timed_out and not res.unreaped
    assert open_fds() == before
    quiesce(3)


def test_a_failure_while_supervising_kills_the_hook_and_closes_the_pipes(
    state_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state_dir.mkdir()
    monkeypatch.setattr(hooks, "_read_capped", _boom)
    before = open_fds()
    start = time.monotonic()
    res = run_hook_process(
        "sleep 60", b"{}", hook="post-park", agent_id=ID, timeout=30, state_dir=state_dir
    )
    assert res.error is not None and "OSError" in res.error and not res.ok
    assert time.monotonic() - start < 10
    assert open_fds() == before
    quiesce(3)


def test_an_exception_from_one_hook_is_logged_and_delivery_continues(
    store: Store, state_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "out"
    real = hooks.run_hook_process
    calls: list[str] = []

    def flaky(cmd: str, *args: Any, **kwargs: Any) -> Any:
        calls.append(cmd)
        if len(calls) == 1:
            raise RuntimeError("boom")
        return real(cmd, *args, **kwargs)

    monkeypatch.setattr(hooks, "run_hook_process", flaky)
    queue(store, event("post-park"))
    result = deliver_outbox(
        store,
        ID,
        config(post_park=[{"cmd": "true"}, {"cmd": recorder(out)}]),
        state_dir=state_dir,
    )
    assert result.delivered == 1 and result.hooks_failed == 1 and len(lines(out)) == 1
    assert any(r["event"] == "hook_failed" for r in log_records(state_dir))


def test_unidentified_group_still_times_out_and_only_the_leader_is_signalled(
    tracker_dir: Path, tracker: ProcTracker, state_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state_dir.mkdir()
    pids = tracker_dir / "pids"
    _no_identity_for_hooks(monkeypatch)
    cmd = (
        f"echo $$ >> {shlex.quote(str(pids))}; sleep 60 & echo $! >> {shlex.quote(str(pids))}; wait"
    )
    start = time.monotonic()
    res = run_hook_process(
        cmd, b"{}", hook="post-park", agent_id=ID, timeout=0.5, state_dir=state_dir
    )
    found = track(tracker, pids)
    # The child sleeps 60 s: returning in under half of that shows nothing waited for it.
    assert res.timed_out and not res.unreaped and time.monotonic() - start < 30
    leader, child = found
    assert not alive(leader)
    assert alive(child)  # unverified group members are never signalled
    names = [r for r in log_records(state_dir) if r["event"] == "signal_sent"]
    assert names and all(r["details"]["group"] is False for r in names)


def test_unidentified_pre_hook_timeout_is_a_timeout_verdict(
    state_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state_dir.mkdir()
    _no_identity_for_hooks(monkeypatch)
    verdict = run_pre_hooks(
        "pre-launch",
        _pre_launch_body(),
        [HookEntry(cmd="exec sleep 60", timeout=0.4)],
        state_dir=state_dir,
    )
    assert verdict.verdict == "veto" and verdict.cause == "timeout"


def test_an_unreaped_hook_is_never_labelled_a_signal_death(
    state_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state_dir.mkdir()
    stuck = hooks.ProcessResult(unreaped=True, error="the hook could not be reaped")
    monkeypatch.setattr(hooks, "run_hook_process", lambda *_a, **_k: stuck)
    verdict = run_pre_hooks(
        "pre-launch", _pre_launch_body(), [HookEntry(cmd="x")], state_dir=state_dir
    )
    assert verdict.verdict == "veto" and verdict.cause != "signal"


class _Skewed(SystemGateway):
    """The real gateway whose clock jumps far ahead once ``skew`` is set."""

    skew = 0.0

    def monotonic(self) -> float:
        return super().monotonic() + self.skew


def test_a_hook_that_finished_by_the_deadline_is_not_timed_out(
    state_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state_dir.mkdir()
    gw = _Skewed()
    real = hooks._read_capped  # noqa: SLF001
    pid_file, gate = tmp_path / "pid", tmp_path / "gate"

    def read_then_expire(*args: Any, **kwargs: Any) -> None:
        real(*args, **kwargs)
        gate.touch()  # the hook was still running at the first look: now it may exit
        assert_dead(
            [int(wait_for_text(pid_file))], timeout=60
        )  # it exits (a zombie until reaped) ...
        gw.skew = 1000.0  # ... and the deadline passes before the next look

    monkeypatch.setattr(hooks, "_read_capped", read_then_expire)
    cmd = (
        f"echo $$ > {shlex.quote(str(pid_file))}; "
        f"while [ ! -e {shlex.quote(str(gate))} ]; do sleep 0.02; done; echo ok"
    )
    res = run_hook_process(
        cmd, b"{}", hook="post-park", agent_id=ID, timeout=5, state_dir=state_dir, gateway=gw
    )
    assert res.ok and not res.timed_out and res.stdout == b"ok\n"


@linux_only
def test_hooks_do_not_inherit_ignored_signals_or_a_blocked_mask(state_dir: Path) -> None:
    state_dir.mkdir()
    sigs = (signal.SIGINT, signal.SIGQUIT, signal.SIGHUP, signal.SIGTERM)
    old = {sig: signal.signal(sig, signal.SIG_IGN) for sig in sigs}
    signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGUSR1})
    try:
        res = run_hook_process(
            "grep -E '^Sig(Ign|Blk)' /proc/self/status",
            b"{}",
            hook="post-park",
            agent_id=ID,
            timeout=5,
            state_dir=state_dir,
        )
    finally:
        signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGUSR1})
        for sig, handler in old.items():
            signal.signal(sig, handler)
    masks = {
        line.split(":")[0]: int(line.split()[1], 16) for line in res.stdout.decode().splitlines()
    }
    for sig in sigs:
        assert not masks["SigIgn"] & (1 << (sig - 1)), sig.name
    assert masks["SigBlk"] == 0


def test_json_nesting_bomb_in_hook_output_is_ignored(state_dir: Path) -> None:
    assert hooks._decision_ids(b"[" * 60000) == []  # noqa: SLF001
    verdict = _pre(state_dir, "head -c 60000 /dev/zero | tr '\\0' '['")
    assert verdict.verdict == "ok" and verdict.decision_ids == ()


def test_invocation_hooks_follow_the_event_snapshot_not_the_live_run(
    store: Store, state_dir: Path, tmp_path: Path
) -> None:
    mine, theirs = tmp_path / "mine", tmp_path / "theirs"
    with store.lease(f"agent:{ID}") as lease:
        store.put_run(
            lease, ID, 1, {"hooks": [{"hook": "post-state-change", "cmd": recorder(mine)}]}
        )
        store.put_run(
            lease, ID, 2, {"hooks": [{"hook": "post-state-change", "cmd": recorder(theirs)}]}
        )
        store.update_agent(lease, ID, {"run": 2})  # bumped after the event was queued
    queue(store, event("post-state-change", run=1))
    deliver_outbox(store, ID, config(), state_dir=state_dir)
    assert len(lines(mine)) == 1 and lines(theirs) == []


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores directory permissions")
def test_an_unwritable_state_dir_is_an_error_not_a_silent_fallback(
    store: Store, state_dir: Path, tmp_path: Path
) -> None:
    marker = tmp_path / "ran"
    locked = tmp_path / "locked"
    locked.mkdir()
    locked.chmod(0o500)
    try:
        res = run_hook_process(
            f"touch {shlex.quote(str(marker))}",
            b"{}",
            hook="post-park",
            agent_id=ID,
            timeout=5,
            state_dir=locked,
        )
        queue(store, event("post-park"))
        result = deliver_outbox(
            store,
            ID,
            config(post_park=[{"cmd": f"touch {shlex.quote(str(marker))}"}]),
            state_dir=locked,
        )
    finally:
        locked.chmod(0o700)
    assert res.error is not None and not res.ok and not marker.exists()
    assert result.delivered == 1 and result.hooks_failed == 1


def test_a_bad_agent_id_skips_the_hook_and_delivery_continues(
    store: Store, state_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "out"
    stored = {
        "event_id": 7,
        "event": "post-park",
        "payload": event("post-park").payload,
        "created": 0.0,
    }
    monkeypatch.setattr(store, "get_agent", lambda *_a, **_k: None)
    deliverer = hooks._Deliverer(  # noqa: SLF001
        store,
        config(post_park=[{"cmd": recorder(out)}, {"cmd": recorder(out)}]),
        state_dir,
        AsyncRunner(state_dir),
        None,
        None,
    )
    result = hooks.DeliveryResult()
    deliverer.deliver("ag-x; touch PWNED", stored, result)
    assert result.hooks_run == 2 and result.hooks_failed == 2 and lines(out) == []
    assert any(r["event"] == "hook_failed" for r in log_records(state_dir))


# -- review round 1: async hooks --------------------------------------------------------------


def _group_members(pgid: int) -> list[int]:
    """Live (non-zombie) processes of process group ``pgid`` (Linux ``/proc`` scan)."""
    found: list[int] = []
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        try:
            text = Path(f"/proc/{name}/stat").read_text()
        except OSError:
            continue
        fields = text[text.rindex(")") + 2 :].split()
        if int(fields[2]) == pgid and fields[0] != "Z":
            found.append(int(name))
    return found


def _wait_group_gone(pgid: int, timeout: float) -> None:
    end = time.monotonic() + timeout
    while _group_members(pgid):
        assert time.monotonic() < end, f"group still has {_group_members(pgid)}"
        time.sleep(0.02)


@linux_only
def test_async_timeout_holds_after_the_starting_process_exited(
    tmp_path: Path, state_dir: Path, tracker: ProcTracker
) -> None:
    state_dir.mkdir()
    pids = tmp_path / "pids"
    script = (
        "import sys\n"
        "from agent_spawn.hooks import AsyncRunner\n"
        "r = AsyncRunner(sys.argv[1])\n"
        "err = r.start(sys.argv[2], b'{}', hook='post-park', agent_id=None, timeout=1)\n"
        "print(err or r.running[0].group.pid)\n"
    )
    cmd = (
        f"echo $PPID >> {shlex.quote(str(pids))}; echo $$ >> {shlex.quote(str(pids))}; "
        f"sleep 30 & echo $! >> {shlex.quote(str(pids))}; wait"
    )
    done = subprocess.run(
        [sys.executable, "-c", script, str(state_dir), cmd],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    wrapper = int(done.stdout.strip())
    end = time.monotonic() + 10
    while len(pids.read_text().split() if pids.exists() else []) < 3:
        assert time.monotonic() < end
        time.sleep(0.02)
    found = track(tracker, pids)
    assert wrapper == found[0]
    assert_dead(found, timeout=4.0)
    _wait_group_gone(wrapper, 8.0)
    quiesce()


@linux_only
def test_async_watchdog_is_removed_when_the_hook_ends(state_dir: Path) -> None:
    state_dir.mkdir()
    runner = AsyncRunner(state_dir)
    assert runner.start("true", b"{}", hook="post-park", agent_id=None, timeout=30) is None
    wrapper = runner.running[0].group.pid
    assert runner.wait(5) == 0
    _wait_group_gone(wrapper, 2.0)
    assert not any(r["event"] == "hook_failed" for r in log_records(state_dir))


@linux_only
def test_async_wrapper_never_orphans_its_watchdog_sleep() -> None:
    # The wrapper signals the watchdog subshell as soon as a quick command ends, possibly before
    # the subshell installed its trap; the default action must not leave ``sleep`` in the group.
    leaked: list[int] = []
    for _ in range(300):
        proc = subprocess.Popen(
            [hooks._SH, "-c", hooks._ASYNC_WRAPPER, "agent-spawn-hook", "true", "30", "2"],
            start_new_session=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        proc.wait()
        end = time.monotonic() + 2.0
        while _group_members(proc.pid) and time.monotonic() < end:
            time.sleep(0.02)
        left = _group_members(proc.pid)
        if left:
            leaked.extend(left)
            for pid in left:  # exact pids just read from this probe's own group
                os.kill(pid, signal.SIGKILL)
    assert not leaked, f"watchdog processes left in the group: {leaked}"


def test_async_command_is_a_positional_parameter_never_interpolated(
    state_dir: Path, tmp_path: Path
) -> None:
    state_dir.mkdir()
    marker = tmp_path / "marker"
    runner = AsyncRunner(state_dir)
    cmd = f"echo '\"; touch {marker}; \"' > /dev/null; exit 3"
    runner.start(cmd, b"{}", hook="post-park", agent_id=None, timeout=5)
    runner.wait(5)
    (record,) = [r for r in log_records(state_dir) if r["event"] == "hook_failed"]
    assert record["details"]["exit_code"] == 3 and not marker.exists()


def test_running_async_hooks_are_capped(state_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    state_dir.mkdir()
    monkeypatch.setattr(hooks, "MAX_ASYNC", 2)
    monkeypatch.setattr(hooks, "ASYNC_CAP_WAIT", 0.3)
    runner = AsyncRunner(state_dir)
    try:
        for _ in range(2):
            assert (
                runner.start("sleep 20", b"{}", hook="post-park", agent_id=None, timeout=1) is None
            )
        error = runner.start("true", b"{}", hook="post-park", agent_id=None, timeout=1)
        assert error is not None and "too many" in error
        assert len(runner.running) == 2
    finally:
        runner.wait(10)
    quiesce()


def test_a_slot_frees_when_a_capped_hook_finishes(
    state_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state_dir.mkdir()
    monkeypatch.setattr(hooks, "MAX_ASYNC", 1)
    runner = AsyncRunner(state_dir)
    assert runner.start("sleep 0.3", b"{}", hook="post-park", agent_id=None, timeout=5) is None
    assert runner.start("true", b"{}", hook="post-park", agent_id=None, timeout=5) is None
    runner.wait(5)
    quiesce()


def test_deliver_outbox_reaps_the_async_runner_after_each_event(
    store: Store, state_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = AsyncRunner(state_dir)
    polls: list[int] = []
    real = runner.poll

    def counting() -> int:
        polls.append(1)
        return real()

    monkeypatch.setattr(runner, "poll", counting)
    queue(store, event("post-park"), event("post-park"))
    deliver_outbox(
        store,
        ID,
        config(post_park=[{"cmd": "true", "async": True}]),
        state_dir=state_dir,
        runner=runner,
    )
    assert len(polls) >= 2
    runner.wait(5)
    quiesce()
