"""``reconcile`` (#33): the lease, the order of the steps, the changes it reports, the
housekeeping deletions and their confinement, ``--dry-run``, idempotence and the golden body."""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from agent_spawn import labels as labellib
from agent_spawn import lifecycle, procs
from agent_spawn.agent_view import agent_object
from agent_spawn.hooks import agent_event
from agent_spawn.lifecycle import LEASE_WAIT, lease_key
from agent_spawn.store import Store
from tests import schema_registry
from tests.conftest import Isolation
from tests.fakebin.harness import Fakebin
from tests.launch_support import Env, result_json, script_live_run
from tests.procguard import ProcTracker
from tests.test_teardown import dead as dead_identity
from tests.test_teardown import me, seed

GOLDEN = Path(__file__).parent / "golden"
DAY = 86400.0
SESSION = "cs-abcdefgh"


@pytest.fixture
def env(fakebin: Fakebin, isolation_env: Isolation) -> Env:
    return Env(fakebin, isolation_env)


def reconcile(env: Env, *flags: str) -> Any:
    done = env.run("reconcile", "--json", *flags)
    assert done.returncode == 0, (done.stdout, done.stderr)
    body = result_json(done)
    schema_registry.validate(body, "agent-spawn/cli/reconcile.json")
    return body


def kinds(body: Any) -> list[tuple[str, str]]:
    return [(c["agent_id"], c["kind"]) for c in body["changes"]]


def only(body: Any, kind: str) -> list[dict[str, Any]]:
    return [c for c in body["changes"] if c["kind"] == kind]


def calls(fakebin: Fakebin, tool: str, command: str) -> list[list[str]]:
    return [a for a in fakebin.argvs(tool) if a and a[0] == command]


def db_rows(env: Env, sql: str, *params: Any) -> list[tuple[Any, ...]]:
    db = sqlite3.connect(f"file:{env.state_dir / 'state.db'}?mode=ro", uri=True)
    try:
        return db.execute(sql, params).fetchall()
    finally:
        db.close()


def tombstone(env: Env, fakebin: Fakebin, age_days: float | None = 40.0, **fields: Any) -> str:
    """A torn-down agent, ``age_days`` old (``None``: no ``torn_down_at`` at all)."""
    agent_id = seed(
        env,
        fakebin,
        session=False,
        status="torn_down",
        park_reason=None,
        torn_down_at=None if age_days is None else time.time() - age_days * DAY,
        **fields,
    )
    return agent_id


def put_run(env: Env, agent_id: str, run: int, **fields: Any) -> None:
    with Store.open(env.state_dir) as store:
        with store.lease(lease_key(agent_id), wait=LEASE_WAIT) as lease:
            store.put_run(lease, agent_id, run, {"run_kind": "implement", **fields})


def log_events(env: Env, event: str) -> list[dict[str, Any]]:
    path = env.state_dir / "log.jsonl"
    if not path.exists():
        return []
    lines = [json.loads(x) for x in path.read_text().splitlines()]
    return [x for x in lines if x["event"] == event]


def launching_dead(env: Env, fakebin: Fakebin, proc_tracker: ProcTracker, **fields: Any) -> str:
    """A ``launching`` agent whose launcher is dead and that has no session yet."""
    child = proc_tracker.popen([sys.executable, "-c", "pass"], stdin=subprocess.DEVNULL)
    identity = procs.read_identity(child.pid)
    child.wait(timeout=30)
    assert identity is not None
    return seed(
        env,
        fakebin,
        session=False,
        status="launching",
        park_reason=None,
        launcher=identity,
        supervisor=None,
        csession_id=None,
        run=None,
        **fields,
    )


# --- the empty and the unknown ------------------------------------------------------------------


def test_without_a_state_database_nothing_happens_and_nothing_is_created(env: Env) -> None:
    body = reconcile(env)
    assert body == {"changes": [], "warnings": [], "already_running": False, "dry_run": False}
    assert not env.state_dir.exists()


def test_a_quiet_state_reports_nothing_and_leaves_no_lease_behind(
    env: Env, fakebin: Fakebin
) -> None:
    seed(env, fakebin)
    body = reconcile(env)
    assert body["changes"] == [] and body["already_running"] is False
    assert db_rows(env, "SELECT key FROM leases") == []
    assert len(calls(fakebin, "claude-session", "reconcile")) == 1  # step 1 ran


def test_a_failing_claude_session_reconcile_is_a_warning_and_the_pass_goes_on(
    env: Env, fakebin: Fakebin
) -> None:
    fakebin.script("claude-session", "reconcile", exit=12, error={"code": "failed", "message": "x"})
    old = tombstone(env, fakebin)
    body = reconcile(env)
    assert any("claude-session reconcile failed" in w for w in body["warnings"])
    assert kinds(body) == [(old, "tombstone_deleted")]


def test_reconcile_calls_claude_session_as_the_internal_actor(env: Env, fakebin: Fakebin) -> None:
    seed(env, fakebin)
    done = env.run("reconcile", "--json", "--actor", "human:alice")
    assert done.returncode == 0
    (call,) = calls(fakebin, "claude-session", "reconcile")
    assert not any("human:alice" in part for part in call)


# --- step 2: liveness ----------------------------------------------------------------------------


def test_a_launcher_that_died_parks_launch_failed_and_the_event_is_delivered(
    env: Env, fakebin: Fakebin, proc_tracker: ProcTracker, tmp_path: Path
) -> None:
    sink = tmp_path / "sink.jsonl"
    hooked = Env(
        fakebin,
        env.iso,
        f'hooks:\n  post-state-change:\n    - cmd: "cat >> {sink}; echo >> {sink}"\n',
    )
    agent_id = launching_dead(hooked, fakebin, proc_tracker)
    body = reconcile(hooked)
    assert kinds(body)[0] == (agent_id, "launch_failed")
    assert body["changes"][0]["details"]["cause"] == "launcher_died"
    assert [k for _, k in kinds(body)] == ["launch_failed", "events_delivered"]
    row = hooked.agent(agent_id)
    assert (row["status"], row["park_reason"]) == ("parked", "launch_failed")
    assert sink.read_text().count('"hook"') == 1  # delivered by reconcile, once
    assert db_rows(hooked, "SELECT count(*) FROM events WHERE delivered = 0") == [(0,)]


def test_a_live_agent_with_a_dead_supervisor_is_flagged_and_never_signaled_or_stopped(
    env: Env, fakebin: Fakebin, proc_tracker: ProcTracker
) -> None:
    survivor = proc_tracker.popen(["sleep", "300"], stdin=subprocess.DEVNULL)
    survivor_identity = procs.read_identity(survivor.pid)
    assert survivor_identity is not None
    agent_id = seed(
        env,
        fakebin,
        status="live",
        park_reason=None,
        supervisor=dead_identity(),
        exec_proc=survivor_identity,
    )
    script_live_run(fakebin)
    body = reconcile(env)
    assert (agent_id, "supervisor_lost") in kinds(body)
    row = env.agent(agent_id)
    assert row["status"] == "live" and row["supervisor_lost"] is True
    assert survivor.poll() is None  # nothing was signaled
    assert calls(fakebin, "claude-session", "stop") == []
    assert reconcile(env)["changes"] == []  # idempotent


def test_a_session_claude_session_does_not_know_is_parked_session_missing(
    env: Env, fakebin: Fakebin
) -> None:
    agent_id = seed(env, fakebin, status="live", park_reason=None, supervisor=dead_identity())
    fakebin.script("claude-session", "status", exit=10, error={"code": "not_found", "message": "x"})
    body = reconcile(env)
    assert (agent_id, "session_missing") in kinds(body)
    row = env.agent(agent_id)
    assert (row["status"], row["park_reason"]) == ("parked", "session_missing")


def test_a_pending_run_of_a_dead_supervisor_is_reported_and_left_alone(
    env: Env, fakebin: Fakebin
) -> None:
    agent_id = seed(env, fakebin, status="live", park_reason=None, supervisor=dead_identity())
    put_run(env, agent_id, 1)
    fakebin.seed_session(SESSION, adopted=False, final=False, outcome=None)
    fakebin.script("claude-session", "status", pending_unadopted=True)
    body = reconcile(env)
    assert (agent_id, "pending_run") in kinds(body)
    assert env.agent(agent_id)["status"] == "live"
    assert calls(fakebin, "claude-session", "stop") == []


def test_two_sessions_for_one_launch_are_ambiguous(
    env: Env, fakebin: Fakebin, proc_tracker: ProcTracker
) -> None:
    agent_id = launching_dead(env, fakebin, proc_tracker)
    fakebin.seed_session("cs-aaaaaaaa", labels=[f"agent={agent_id}"])
    fakebin.seed_session("cs-bbbbbbbb", labels=[f"agent={agent_id}"])
    body = reconcile(env)
    assert [k for _, k in kinds(body)][:2] == ["ambiguous_session", "launch_failed"]
    assert only(body, "launch_failed")[0]["details"]["cause"] == "ambiguous_session"
    assert env.agent(agent_id)["park_reason"] == "launch_failed"


def test_a_fork_whose_launcher_was_killed_before_start_returned_is_removed_with_its_holds(
    env: Env, fakebin: Fakebin, proc_tracker: ProcTracker
) -> None:
    source = seed(env, fakebin)
    fork = launching_dead(env, fakebin, proc_tracker, forked_from=source)
    with Store.open(env.state_dir) as store:
        with store.lease(lifecycle.lease_key(fork), wait=5) as lease:
            store.put_hold(lease, fork, "wt-aaaaaaaa", "/x/y", owned=False)
    body = reconcile(env)
    assert (fork, "fork_abandoned") in kinds(body)
    with Store.open(env.state_dir) as store:
        assert store.get_agent(fork) is None and store.is_reserved(fork)
        assert store.get_agent(source) is not None
    assert not (env.state_dir / "prompts" / fork).exists()
    forgets = calls(fakebin, "wtctl", "forget")
    assert forgets and forgets[0][1] == "wt-aaaaaaaa" and f"--holder={fork}" in forgets[0]
    assert reconcile(env)["changes"] == []


def test_a_failure_judging_one_agent_does_not_stop_the_others(
    env: Env, fakebin: Fakebin, proc_tracker: ProcTracker
) -> None:
    broken = seed(env, fakebin, status="live", park_reason=None, supervisor=dead_identity())
    fakebin.script(
        "claude-session",
        "status",
        times=1,
        exit=12,
        error={"code": "failed", "message": "x"},
    )
    other = launching_dead(env, fakebin, proc_tracker)
    body = reconcile(env)
    assert (other, "launch_failed") in kinds(body)
    assert any(w.startswith(broken) for w in body["warnings"])
    assert env.agent(broken)["status"] == "live"


# --- step 3: the outbox -------------------------------------------------------------------------


def queue_event(env: Env, agent_id: str) -> None:
    with Store.open(env.state_dir) as store:
        row = store.get_agent(agent_id)
        assert row is not None
        store.add_events(
            agent_id,
            [
                agent_event(
                    "post-state-change",
                    agent_object(store, row),
                    reads_third_party=False,
                    extra={
                        "change": "agent",
                        "previous_status": "parked",
                        "previous_needs_attention": None,
                    },
                )
            ],
        )


def hooked_env(env: Env, tmp_path: Path, hook: str = "post-state-change") -> tuple[Env, Path]:
    sink = tmp_path / "sink.jsonl"
    config = f'hooks:\n  {hook}:\n    - cmd: "cat >> {sink}; echo >> {sink}"\n'
    return Env(env.fakebin, env.iso, config), sink


def test_undelivered_events_are_delivered_once_in_order_and_reported(
    env: Env, fakebin: Fakebin, tmp_path: Path
) -> None:
    hooked, sink = hooked_env(env, tmp_path)
    first = seed(hooked, fakebin)
    second = seed(hooked, fakebin)
    for agent_id in (first, first, second):
        queue_event(hooked, agent_id)
    body = reconcile(hooked)
    assert [(c["agent_id"], c["details"]["count"]) for c in only(body, "events_delivered")] == [
        (first, 2),
        (second, 1),
    ]
    assert sink.read_text().count('"hook"') == 3
    assert reconcile(hooked)["changes"] == []  # at most once more is nothing
    assert sink.read_text().count('"hook"') == 3


def test_a_tombstones_pending_events_are_delivered_too(
    env: Env, fakebin: Fakebin, tmp_path: Path
) -> None:
    hooked, sink = hooked_env(env, tmp_path)
    agent_id = tombstone(hooked, fakebin, age_days=1.0)
    queue_event(hooked, agent_id)
    body = reconcile(hooked)
    assert kinds(body) == [(agent_id, "events_delivered")]


def test_an_agent_whose_delivery_lease_is_held_is_left_to_its_holder(
    env: Env, fakebin: Fakebin, tmp_path: Path
) -> None:
    hooked, sink = hooked_env(env, tmp_path)
    agent_id = seed(hooked, fakebin)
    queue_event(hooked, agent_id)
    with Store.open(hooked.state_dir) as store:
        lease = store.acquire_lease(f"delivery:{agent_id}")  # this (live) process holds it
        try:
            body = reconcile(hooked)
        finally:
            store.release_lease(lease)
    assert body["changes"] == [] and not sink.exists()
    assert db_rows(hooked, "SELECT count(*) FROM events WHERE delivered = 0") == [(1,)]


# --- step 4: teardown_incomplete -----------------------------------------------------------------


def tearing_down(env: Env, fakebin: Fakebin, step: str, proc: Any, **fields: Any) -> str:
    return seed(
        env,
        fakebin,
        status="tearing_down",
        park_reason=None,
        teardown_step=step,
        teardown_proc=proc,
        **fields,
    )


def test_a_dead_teardown_is_reported_and_never_continued(env: Env, fakebin: Fakebin) -> None:
    agent_id = tearing_down(env, fakebin, "hooks_done", dead_identity())
    before = env.agent(agent_id)
    body = reconcile(env)
    assert kinds(body) == [(agent_id, "teardown_incomplete")]
    assert body["changes"][0]["details"] == {"teardown_step": "hooks_done", "teardown_alive": False}
    assert env.agent(agent_id) == before
    assert calls(fakebin, "claude-session", "forget") == []
    assert (env.state_dir / "prompts" / agent_id).exists()


def test_a_running_teardown_is_reported_alive(env: Env, fakebin: Fakebin) -> None:
    agent_id = tearing_down(env, fakebin, "started", me())
    (change,) = reconcile(env)["changes"]
    assert change["details"] == {"teardown_step": "started", "teardown_alive": True}
    assert change["agent_id"] == agent_id


def test_a_dead_teardown_at_the_hook_step_says_a_hook_may_be_running(
    env: Env, fakebin: Fakebin
) -> None:
    tearing_down(env, fakebin, "started", dead_identity())
    (change,) = reconcile(env)["changes"]
    assert change["details"]["hook_may_be_running"] is True


def test_a_partial_worktree_removal_names_its_record(env: Env, fakebin: Fakebin) -> None:
    tearing_down(
        env, fakebin, "worktree_partial", dead_identity(), teardown_partial_record="wt-aaaaaaaa"
    )
    (change,) = reconcile(env)["changes"]
    assert change["details"]["partial_record"] == "wt-aaaaaaaa"
    assert calls(fakebin, "wtctl", "remove") == []


def test_a_tearing_down_agent_without_a_marker_is_not_reported(env: Env, fakebin: Fakebin) -> None:
    seed(env, fakebin, status="tearing_down", park_reason=None, teardown_proc=dead_identity())
    assert reconcile(env)["changes"] == []


# --- step 5: label mirror ------------------------------------------------------------------------


def add_labels(env: Env, agent_id: str, *texts: str) -> None:
    with Store.open(env.state_dir) as store:
        with store.lease(lease_key(agent_id), wait=LEASE_WAIT) as lease:
            store.edit_labels(lease, agent_id, "add", [labellib.parse_label(t) for t in texts])


def remove_labels(env: Env, agent_id: str, *texts: str) -> None:
    with Store.open(env.state_dir) as store:
        with store.lease(lease_key(agent_id), wait=LEASE_WAIT) as lease:
            store.edit_labels(lease, agent_id, "remove", [labellib.parse_label(t) for t in texts])


def pending(env: Env, agent_id: str) -> list[tuple[str, str]]:
    with Store.open(env.state_dir) as store:
        return [(str(pair), op) for pair, op in store.mirror_pending(agent_id)]


def test_pending_label_changes_are_pushed_to_the_session_and_dropped(
    env: Env, fakebin: Fakebin
) -> None:
    agent_id = seed(env, fakebin)
    add_labels(env, agent_id, "team=a", "owner=flow")
    remove_labels(env, agent_id, "owner=flow")
    assert pending(env, agent_id) == [("team=a", "add"), ("owner=flow", "remove")]
    body = reconcile(env)
    (change,) = only(body, "labels_resynced")
    assert change["details"] == {"added": ["team=a"], "removed": ["owner=flow"]}
    assert pending(env, agent_id) == []
    pushed = [a for a in fakebin.argvs("claude-session") if a[:1] == ["label"]]
    assert any("team=a" in a for a in pushed) and any("owner=flow" in a for a in pushed)
    assert reconcile(env)["changes"] == []


def test_a_failed_push_stays_queued_with_a_warning_and_no_change(
    env: Env, fakebin: Fakebin
) -> None:
    agent_id = seed(env, fakebin)
    add_labels(env, agent_id, "team=a")
    fakebin.script("claude-session", "label add", exit=12, error={"code": "failed", "message": "x"})
    body = reconcile(env)
    assert only(body, "labels_resynced") == []
    assert any(w.startswith(agent_id) and "queued for reconcile" in w for w in body["warnings"])
    assert pending(env, agent_id) == [("team=a", "add")]


def test_labels_of_an_agent_without_a_session_stay_queued(env: Env, fakebin: Fakebin) -> None:
    agent_id = seed(env, fakebin, session=False)
    add_labels(env, agent_id, "team=a")
    assert reconcile(env)["changes"] == []
    assert pending(env, agent_id) == [("team=a", "add")]


def test_a_live_agents_pane_gets_the_labels_too(env: Env, fakebin: Fakebin) -> None:
    agent_id = seed(env, fakebin, status="live", park_reason=None)
    put_run(env, agent_id, 1, term_id="tc-aaaaaaaa", socket_path="/sock/x")
    script_live_run(fakebin)
    fakebin.script("termctl", "label add", stdout={"id": "tc-aaaaaaaa"})  # the pane took it
    add_labels(env, agent_id, "team=a")
    body = reconcile(env)
    assert only(body, "labels_resynced")[0]["details"]["added"] == ["team=a"]
    pane = [a for a in fakebin.argvs("termctl") if a[:2] == ["label", "add"]]
    assert pane and "tc-aaaaaaaa" in pane[0] and "team=a" in pane[0]
    assert pending(env, agent_id) == []


# --- step 6: tombstones ---------------------------------------------------------------------------


def test_an_old_tombstone_is_deleted_with_everything_it_owns_and_its_id_stays_reserved(
    env: Env, fakebin: Fakebin
) -> None:
    agent_id = tombstone(env, fakebin, 40.0)
    put_run(env, agent_id, 1)
    add_labels(env, agent_id, "team=a")
    queue_event(env, agent_id)
    with Store.open(env.state_dir) as store:
        with store.lease(lease_key(agent_id), wait=LEASE_WAIT) as lease:
            store.put_hold(lease, agent_id, "wt-aaaaaaaa", "/x/y", owned=False)
    body = reconcile(env)
    # the event is delivered (step 3) before the tombstone goes (step 6)
    assert kinds(body) == [(agent_id, "events_delivered"), (agent_id, "tombstone_deleted")]
    with Store.open(env.state_dir) as store:
        assert store.get_agent(agent_id) is None and store.is_reserved(agent_id)
    for table in ("agent_runs", "agent_holds", "agent_labels", "label_mirror_pending", "events"):
        assert db_rows(env, f"SELECT count(*) FROM {table} WHERE agent_id = ?", agent_id) == [(0,)]
    assert not (env.state_dir / "prompts" / agent_id).exists()
    assert not (env.state_dir / "outcomes" / agent_id).exists()
    assert [x["details"]["kind"] for x in log_events(env, "reconcile_change")] == [
        "events_delivered",
        "files_removed",
        "files_removed",
        "tombstone_deleted",
    ]
    assert reconcile(env)["changes"] == []


def test_a_young_tombstone_and_one_without_a_time_stay(env: Env, fakebin: Fakebin) -> None:
    young = tombstone(env, fakebin, 1.0)
    untimed = tombstone(env, fakebin, None)
    assert reconcile(env)["changes"] == []
    with Store.open(env.state_dir) as store:
        assert store.get_agent(young) is not None and store.get_agent(untimed) is not None


def test_tombstone_days_comes_from_the_config(fakebin: Fakebin, isolation_env: Isolation) -> None:
    cfg = Env(fakebin, isolation_env, "tombstone_days: 100\n")
    agent_id = tombstone(cfg, fakebin, 40.0)
    assert reconcile(cfg)["changes"] == []
    short = Env(fakebin, isolation_env, "tombstone_days: 10\n")
    assert kinds(reconcile(short)) == [(agent_id, "tombstone_deleted")]


def test_a_tombstone_being_delivered_is_kept_for_the_next_pass(env: Env, fakebin: Fakebin) -> None:
    agent_id = tombstone(env, fakebin, 40.0)
    with Store.open(env.state_dir) as store:
        lease = store.acquire_lease(f"delivery:{agent_id}")
        try:
            assert reconcile(env)["changes"] == []
            assert store.get_agent(agent_id) is not None
        finally:
            store.release_lease(lease)
    assert kinds(reconcile(env)) == [(agent_id, "tombstone_deleted")]


def test_a_row_that_is_not_a_tombstone_is_never_deleted(env: Env, fakebin: Fakebin) -> None:
    parked = seed(env, fakebin)
    db = sqlite3.connect(env.state_dir / "state.db")
    try:
        with db:
            db.execute("UPDATE agents SET created = ?", (time.time() - 400 * DAY,))
    finally:
        db.close()
    assert reconcile(env)["changes"] == []
    assert env.agent(parked)["status"] == "parked"
    assert (env.state_dir / "prompts" / parked).exists()


# --- step 6: prompt copies and outcome files -----------------------------------------------------


def backdate(path: Path, age: float) -> None:
    moment = time.time() - age
    os.utime(path, (moment, moment))


def make_dir(env: Env, sub: str, agent_id: str, age: float = 0.0) -> Path:
    directory = env.state_dir / sub / agent_id
    directory.mkdir(parents=True, mode=0o700)
    (directory / "file").write_text("x")
    backdate(directory, age)
    return directory


def init_state(env: Env, fakebin: Fakebin) -> None:
    """A state database to reconcile (the command does nothing without one)."""
    seed(env, fakebin)


def test_prompt_copies_without_a_record_go_only_after_an_hour(env: Env, fakebin: Fakebin) -> None:
    init_state(env, fakebin)
    old = make_dir(env, "prompts", "ag-aaaaaaaa", age=2 * 3600)
    young = make_dir(env, "prompts", "ag-bbbbbbbb", age=600)
    assert reconcile(env)["changes"] == []  # files are not agent changes
    assert not old.exists() and young.exists()
    logged = [
        x for x in log_events(env, "reconcile_change") if x["details"]["kind"] == "files_removed"
    ]
    assert [(x["agent_id"], x["details"]["reason"]) for x in logged] == [(None, "no_record")]


@pytest.mark.parametrize("status", ["parked", "launching", "live", "tearing_down"])
def test_an_agent_with_a_record_keeps_its_old_prompt_copies(
    env: Env, fakebin: Fakebin, status: str
) -> None:
    agent_id = seed(env, fakebin, status=status, park_reason=None, teardown_step=None)
    backdate(env.state_dir / "prompts" / agent_id, 10 * DAY)
    backdate(env.state_dir / "outcomes" / agent_id, 10 * DAY)
    script_live_run(fakebin)
    reconcile(env)
    assert (env.state_dir / "prompts" / agent_id / "prompt").exists()
    assert (env.state_dir / "outcomes" / agent_id / "tok.json").exists()


def test_a_tombstones_leftover_files_go_whatever_their_age(env: Env, fakebin: Fakebin) -> None:
    agent_id = tombstone(env, fakebin, 1.0)
    reconcile(env)
    assert not (env.state_dir / "prompts" / agent_id).exists()
    assert not (env.state_dir / "outcomes" / agent_id).exists()
    with Store.open(env.state_dir) as store:
        assert store.get_agent(agent_id) is not None  # the row itself is young


def test_outcome_files_without_a_record_are_not_touched(env: Env, fakebin: Fakebin) -> None:
    init_state(env, fakebin)
    stray = make_dir(env, "outcomes", "ag-aaaaaaaa", age=10 * DAY)
    reconcile(env)
    assert stray.exists()


def test_deletion_never_follows_a_symlink(env: Env, fakebin: Fakebin, tmp_path: Path) -> None:
    init_state(env, fakebin)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "precious").write_text("keep")
    # a symlink named like an orphan directory, one inside an orphan, one for a tombstone
    (env.state_dir / "prompts").mkdir(exist_ok=True)
    os.symlink(outside, env.state_dir / "prompts" / "ag-aaaaaaaa")
    orphan = make_dir(env, "prompts", "ag-bbbbbbbb", age=2 * 3600)
    os.symlink(outside, orphan / "link")
    backdate(orphan, 2 * 3600)  # adding the link touched it
    gone = tombstone(env, fakebin, 1.0)
    os.symlink(outside, env.state_dir / "outcomes" / gone / "link")
    reconcile(env)
    assert (outside / "precious").read_text() == "keep"
    assert os.path.islink(env.state_dir / "prompts" / "ag-aaaaaaaa")  # not ours to delete
    assert not orphan.exists() and not (env.state_dir / "outcomes" / gone).exists()


def test_a_prompts_directory_that_is_a_symlink_is_not_swept(
    env: Env, fakebin: Fakebin, tmp_path: Path
) -> None:
    init_state(env, fakebin)
    real = tmp_path / "elsewhere"
    victim = real / "ag-aaaaaaaa"
    victim.mkdir(parents=True)
    (victim / "f").write_text("x")
    backdate(victim, 10 * DAY)
    os.rename(env.state_dir / "prompts", tmp_path / "moved") if (
        env.state_dir / "prompts"
    ).exists() else None
    os.symlink(real, env.state_dir / "prompts")
    body = reconcile(env)
    assert (victim / "f").exists()
    assert any("prompts/" in w for w in body["warnings"])


def test_names_that_are_not_agent_ids_are_left_alone(env: Env, fakebin: Fakebin) -> None:
    init_state(env, fakebin)
    odd = make_dir(env, "prompts", "not-an-id", age=10 * DAY)
    reconcile(env)
    assert odd.exists()


# --- step 6: handoff directories -----------------------------------------------------------------


def handoff_dir(env: Env, name: str, owner: Any, extra: tuple[str, ...] = ()) -> Path:
    directory = env.state_dir / "handoff" / name
    directory.mkdir(parents=True, mode=0o700)
    (directory / "s").write_text("")
    if owner is not None:
        (directory / "owner").write_text(json.dumps(owner.to_dict()))
    for leaf in extra:
        (directory / leaf).write_text("x")
    return directory


def test_a_handoff_directory_of_a_dead_owner_is_removed_and_a_live_ones_stays(
    env: Env, fakebin: Fakebin
) -> None:
    init_state(env, fakebin)
    gone = handoff_dir(env, "aaaaaaaaaaaa", dead_identity())
    live = handoff_dir(env, "bbbbbbbbbbbb", me())
    other_boot = procs.ProcessIdentity(me().pid, me().start_time, "another-boot")
    rebooted = handoff_dir(env, "cccccccccccc", other_boot)
    reconcile(env)
    assert not gone.exists() and not rebooted.exists()
    assert live.exists() and (live / "s").exists() and (live / "owner").exists()
    removed = [
        x for x in log_events(env, "reconcile_change") if x["details"]["kind"] == "handoff_removed"
    ]
    assert len(removed) == 2


def test_a_zombie_owner_counts_as_dead(
    env: Env, fakebin: Fakebin, proc_tracker: ProcTracker
) -> None:
    init_state(env, fakebin)
    child = proc_tracker.popen([sys.executable, "-c", "pass"], stdin=subprocess.DEVNULL)
    env.wait_for(
        lambda: (info := procs.inspect(child.pid)) is not None and not info.alive,
        "the child to exit (it is not reaped yet)",
    )
    identity = procs.read_identity(child.pid)
    assert identity is not None  # a zombie: still in the process table
    directory = handoff_dir(env, "dddddddddddd", identity)
    reconcile(env)
    assert not directory.exists()
    child.wait(timeout=30)


def test_a_handoff_directory_with_other_files_or_odd_shape_stays(
    env: Env, fakebin: Fakebin, tmp_path: Path
) -> None:
    init_state(env, fakebin)
    extra = handoff_dir(env, "aaaaaaaaaaaa", dead_identity(), extra=("surprise",))
    oddly_named = handoff_dir(env, "not-a-handoff-name", dead_identity())
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "owner").write_text(json.dumps(dead_identity().to_dict()))
    os.symlink(outside, env.state_dir / "handoff" / "bbbbbbbbbbbb")
    body = reconcile(env)
    assert extra.exists() and oddly_named.exists() and (outside / "owner").exists()
    assert any("holds other files" in w for w in body["warnings"])


def test_a_handoff_directory_without_a_readable_owner_goes_only_when_old(
    env: Env, fakebin: Fakebin
) -> None:
    init_state(env, fakebin)
    fresh = handoff_dir(env, "aaaaaaaaaaaa", None)  # a listener between mkdir and the file
    stale = handoff_dir(env, "bbbbbbbbbbbb", None)
    backdate(stale, 2 * 3600)
    garbled = handoff_dir(env, "cccccccccccc", None)
    (garbled / "owner").write_text("{not json")
    backdate(garbled, 2 * 3600)
    reconcile(env)
    assert fresh.exists() and not stale.exists() and not garbled.exists()


def test_a_handoff_root_that_is_a_symlink_is_not_swept(
    env: Env, fakebin: Fakebin, tmp_path: Path
) -> None:
    init_state(env, fakebin)
    real = tmp_path / "real"
    victim = real / "aaaaaaaaaaaa"
    victim.mkdir(parents=True)
    (victim / "owner").write_text(json.dumps(dead_identity().to_dict()))
    os.symlink(real, env.state_dir / "handoff")
    body = reconcile(env)
    assert victim.exists()
    assert any("handoff directory" in w for w in body["warnings"])


# --- the lease ------------------------------------------------------------------------------------


def hold_gate_env(env: Env, tmp_path: Path) -> tuple[Env, Path, Path, Path]:
    """An environment whose ``post-state-change`` hook blocks until a gate file exists, so a
    reconcile that delivers an event holds its lease for as long as the test wants."""
    started, gate, sink = tmp_path / "started", tmp_path / "gate", tmp_path / "sink"
    cmd = (
        f"touch {started}; while [ ! -e {gate} ]; do sleep 0.05; done; "
        f"cat >> {sink}; echo >> {sink}"
    )
    config = f'hooks:\n  post-state-change:\n    - cmd: "{cmd}"\n      timeout: 120\n'
    return Env(env.fakebin, env.iso, config), started, gate, sink


def test_a_second_caller_exits_0_already_running_and_does_nothing(
    env: Env, fakebin: Fakebin, tmp_path: Path
) -> None:
    hooked, started, gate, sink = hold_gate_env(env, tmp_path)
    busy = seed(hooked, fakebin)
    queue_event(hooked, busy)
    other = tombstone(hooked, fakebin, 40.0)  # the first pass would delete this at step 6
    first = subprocess.Popen(
        hooked.argv("reconcile", "--json"),
        env=hooked.env(),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        hooked.wait_for(started.exists, "the first reconcile to deliver (holding its lease)")
        for _ in range(2):
            body = reconcile(hooked)
            assert body == {
                "changes": [],
                "warnings": [],
                "already_running": True,
                "dry_run": False,
            }
        with Store.open(hooked.state_dir) as store:
            assert store.get_agent(other) is not None  # the second callers deleted nothing
            assert len(store.pending_events(busy)) == 1  # ... and delivered nothing
    finally:
        gate.write_text("")
        out, err = first.communicate(timeout=120)
    assert first.returncode == 0, err
    done = json.loads(out)
    schema_registry.validate(done, "agent-spawn/cli/reconcile.json")
    assert done["already_running"] is False
    assert (busy, "events_delivered") in kinds(done) and (other, "tombstone_deleted") in kinds(done)
    assert sink.read_text().count('"hook"') == 1
    assert db_rows(hooked, "SELECT key FROM leases") == []


def test_concurrent_reconciles_released_together_deliver_every_event_once(
    env: Env, fakebin: Fakebin, tmp_path: Path
) -> None:
    sink = tmp_path / "sink"
    config = f'hooks:\n  post-state-change:\n    - cmd: "cat >> {sink}; echo >> {sink}"\n'
    racing = Env(fakebin, env.iso, config)
    agents = [seed(racing, fakebin) for _ in range(3)]
    for agent_id in agents:
        for _ in range(3):
            queue_event(racing, agent_id)
    barrier = tmp_path / "barrier"
    script = 'while [ ! -e "$0" ]; do :; done; exec "$@"'
    procs_ = [
        subprocess.Popen(
            ["/bin/sh", "-c", script, str(barrier), *racing.argv("reconcile", "--json")],
            env=racing.env(),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for _ in range(4)
    ]
    barrier.write_text("")  # all four go at once
    bodies: list[Any] = []
    for proc in procs_:
        out, err = proc.communicate(timeout=120)
        assert proc.returncode == 0, err
        body = json.loads(out)
        schema_registry.validate(body, "agent-spawn/cli/reconcile.json")
        bodies.append(body)
    delivered = sum(c["details"]["count"] for b in bodies for c in only(b, "events_delivered"))
    assert delivered == 9 and sink.read_text().count('"hook"') == 9
    assert db_rows(racing, "SELECT count(*) FROM events WHERE delivered = 0") == [(0,)]
    assert all(b["changes"] == [] for b in bodies if b["already_running"])


def test_a_lease_of_a_dead_holder_is_taken_over(env: Env, fakebin: Fakebin) -> None:
    seed(env, fakebin)
    with Store.open(env.state_dir) as store:
        store.acquire_lease("reconcile", holder=dead_identity())  # never released
    assert reconcile(env)["already_running"] is False
    assert db_rows(env, "SELECT key FROM leases") == []


def test_a_lease_of_a_live_holder_is_not_taken(env: Env, fakebin: Fakebin) -> None:
    old = tombstone(env, fakebin, 40.0)
    with Store.open(env.state_dir) as store:
        lease = store.acquire_lease("reconcile")  # this process: alive
        try:
            body = reconcile(env)
        finally:
            store.release_lease(lease)
    assert body["already_running"] is True and body["changes"] == []
    with Store.open(env.state_dir) as store:
        assert store.get_agent(old) is not None


def test_a_lease_row_that_cannot_be_parsed_is_an_error_and_is_left_alone(
    env: Env, fakebin: Fakebin
) -> None:
    seed(env, fakebin)
    db = sqlite3.connect(env.state_dir / "state.db")
    try:
        with db:
            db.execute(
                "INSERT INTO leases (key, holder, acquired) VALUES ('reconcile', 'garbage', 0)"
            )
    finally:
        db.close()
    done = env.run("reconcile", "--json")
    assert done.returncode == 12, (done.stdout, done.stderr)
    schema_registry.validate(result_json(done), "agent-spawn/cli/error.json")
    assert db_rows(env, "SELECT holder FROM leases WHERE key = 'reconcile'") == [("garbage",)]


# --- dry run, isolation of steps, idempotence ----------------------------------------------------


def snapshot(env: Env) -> tuple[list[str], list[tuple[str, int, float]], str]:
    db = sqlite3.connect(f"file:{env.state_dir / 'state.db'}?mode=ro", uri=True)
    try:
        dump = list(db.iterdump())
    finally:
        db.close()
    files: list[tuple[str, int, float]] = []
    for root, dirs, names in os.walk(env.state_dir):
        for name in sorted(dirs + names):
            full = Path(root) / name
            if name.startswith("state.db"):
                continue
            info = os.lstat(full)
            files.append((str(full), info.st_size, info.st_mtime))
    log = env.state_dir / "log.jsonl"
    return dump, sorted(files), log.read_text() if log.exists() else ""


def rich_state(env: Env, fakebin: Fakebin, proc_tracker: ProcTracker) -> dict[str, str]:
    ids: dict[str, str] = {}
    ids["launching"] = launching_dead(env, fakebin, proc_tracker)
    ids["old"] = tombstone(env, fakebin, 40.0)
    ids["labelled"] = seed(env, fakebin)
    add_labels(env, ids["labelled"], "team=a")
    queue_event(env, ids["labelled"])
    ids["tearing"] = tearing_down(env, fakebin, "hooks_done", dead_identity())
    make_dir(env, "prompts", "ag-aaaaaaaa", age=2 * 3600)
    handoff_dir(env, "aaaaaaaaaaaa", dead_identity())
    return ids


def test_a_dry_run_reports_and_changes_nothing_and_takes_no_lease(
    env: Env, fakebin: Fakebin, proc_tracker: ProcTracker, tmp_path: Path
) -> None:
    hooked, sink = hooked_env(env, tmp_path)
    ids = rich_state(hooked, fakebin, proc_tracker)
    before = snapshot(hooked)
    body = reconcile(hooked, "--dry-run")
    assert body["dry_run"] is True and body["already_running"] is False
    assert sorted(kinds(body)) == sorted(
        [
            (ids["launching"], "launch_failed"),
            (ids["labelled"], "events_delivered"),
            (ids["tearing"], "teardown_incomplete"),
            (ids["labelled"], "labels_resynced"),
            (ids["old"], "tombstone_deleted"),
        ]
    )
    assert snapshot(hooked) == before
    assert not sink.exists()  # no hook ran
    assert calls(fakebin, "claude-session", "reconcile")[0][-2:] == ["--dry-run", "--json"]
    assert calls(fakebin, "claude-session", "stop") == []


def test_a_dry_run_works_while_another_reconcile_holds_the_lease(
    env: Env, fakebin: Fakebin
) -> None:
    old = tombstone(env, fakebin, 40.0)
    with Store.open(env.state_dir) as store:
        lease = store.acquire_lease("reconcile")
        try:
            body = reconcile(env, "--dry-run")
        finally:
            store.release_lease(lease)
    assert kinds(body) == [(old, "tombstone_deleted")]


def test_the_real_pass_does_what_the_dry_run_reported_and_a_second_one_is_empty(
    env: Env, fakebin: Fakebin, proc_tracker: ProcTracker
) -> None:
    rich_state(env, fakebin, proc_tracker)
    planned = reconcile(env, "--dry-run")
    done = reconcile(env)
    assert [c for c in done["changes"] if c["kind"] != "events_delivered"] == [
        c for c in planned["changes"] if c["kind"] != "events_delivered"
    ]
    quiet = reconcile(env)
    assert [c["kind"] for c in quiet["changes"]] == [
        "teardown_incomplete"
    ]  # a report, not a change
    assert quiet["warnings"] == [] and not (env.state_dir / "prompts" / "ag-aaaaaaaa").exists()


def test_a_step_that_raises_is_a_warning_and_the_later_steps_still_run(
    env: Env,
    fakebin: Fakebin,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from agent_spawn import reconcile as reconcile_module
    from tests.inspect_support import run_inprocess

    old = tombstone(env, fakebin, 40.0)

    def boom(self: Any) -> None:
        raise RuntimeError("secret detail that must not be printed")

    monkeypatch.setattr(reconcile_module._Pass, "liveness", boom)  # pyright: ignore[reportPrivateUsage]
    code, out, _ = run_inprocess(env, monkeypatch, capsys, "reconcile", "--json")
    body = json.loads(out)
    assert code == 0
    assert body["warnings"] == ["liveness failed (RuntimeError)"]
    assert kinds(body) == [(old, "tombstone_deleted")]


def test_reconcile_never_signals_a_process(
    env: Env, fakebin: Fakebin, proc_tracker: ProcTracker
) -> None:
    survivors = [proc_tracker.popen(["sleep", "300"], stdin=subprocess.DEVNULL) for _ in range(2)]
    identities = [procs.read_identity(p.pid) for p in survivors]
    seed(
        env,
        fakebin,
        status="live",
        park_reason=None,
        supervisor=dead_identity(),
        exec_proc=identities[0],
    )
    launching_dead(env, fakebin, proc_tracker)
    tearing_down(env, fakebin, "started", identities[1])
    script_live_run(fakebin)
    reconcile(env)
    assert all(p.poll() is None for p in survivors)
    assert [a for a in fakebin.argvs("claude-session") if a[0] in ("stop", "forget")] == []
    assert log_events(env, "signal_sent") == []


# --- golden --json bodies -------------------------------------------------------------------------


def normalized(body: Any, ids: dict[str, str]) -> Any:
    """The body with the generated agent ids replaced by the golden's fixed ones."""
    text = json.dumps(body)
    fixed = {
        "launching": "ag-aaaaaaaa",
        "old": "ag-bbbbbbbb",
        "labelled": "ag-cccccccc",
        "tearing": "ag-dddddddd",
    }
    for name, real in ids.items():
        text = text.replace(real, fixed[name])
    body = json.loads(text)
    for change in body["changes"]:  # the one time-dependent detail
        if change["kind"] == "tombstone_deleted":
            change["details"]["torn_down_at"] = "2026-01-01T00:00:00.000Z"
    return body


@pytest.mark.parametrize(
    ("flags", "golden"),
    [((), "reconcile_result_live.json"), (("--dry-run",), "reconcile_result_dry_run.json")],
    ids=["live", "dry"],
)
def test_json_body_matches_the_golden_and_validates(
    env: Env, fakebin: Fakebin, proc_tracker: ProcTracker, flags: tuple[str, ...], golden: str
) -> None:
    ids = rich_state(env, fakebin, proc_tracker)
    body = reconcile(env, *flags)
    path = GOLDEN / golden
    expected = json.loads(path.read_text())
    schema_registry.validate(expected, "agent-spawn/cli/reconcile.json")
    assert normalized(body, ids) == expected


def test_the_already_running_body_validates_against_the_schema(env: Env, fakebin: Fakebin) -> None:
    seed(env, fakebin)
    with Store.open(env.state_dir) as store:
        lease = store.acquire_lease("reconcile")
        try:
            body = reconcile(env)
        finally:
            store.release_lease(lease)
    assert body["already_running"] is True
    bad = {**body, "changes": [{"agent_id": "ag-aaaaaaaa", "kind": "parked", "details": {}}]}
    with pytest.raises(Exception):  # noqa: B017, PT011 - the schema forbids changes here
        schema_registry.validate(bad, "agent-spawn/cli/reconcile.json")
