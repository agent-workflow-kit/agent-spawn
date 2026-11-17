"""``prune``: age, protections, continuation of interrupted teardowns, the dry run, isolation of
one agent's failure from the others, and the fresh-state guard under the lease (#24)."""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

import pytest

from agent_spawn import labels as labellib
from agent_spawn import prune as prune_module
from agent_spawn.config import Config
from agent_spawn.lifecycle import LEASE_WAIT, lease_key
from agent_spawn.output import CliError
from agent_spawn.store import CasLost, LeaseHeld, Store
from agent_spawn.teardown import Teardown
from tests import schema_registry
from tests.conftest import Isolation
from tests.fakebin.harness import Fakebin
from tests.launch_support import Env, result_json, script_live_run
from tests.test_teardown import dead, log_text, me, seed
from tests.test_teardown_worktrees import DIRTY, PARTIAL
from tests.test_worktrees_launch import records, wtctl_calls
from tests.test_worktrees_resume import launched_in_worktree, set_state

OLD = time.time() - 40 * 86400
GOLDEN = Path(__file__).parent / "golden"


@pytest.fixture
def env(fakebin: Fakebin, isolation_env: Isolation) -> Env:
    return Env(fakebin, isolation_env)


def prune(env: Env, *flags: str) -> Any:
    done = env.run("prune", "--json", *flags)
    assert done.returncode == 0, (done.stdout, done.stderr)
    body = result_json(done)
    schema_registry.validate(body, "agent-spawn/cli/prune.json")
    return body


def backdate(env: Env, agent_id: str, created: float = OLD) -> None:
    """``created`` is immutable through the store: age an agent in the database directly."""
    db = sqlite3.connect(env.state_dir / "state.db")
    try:
        with db:
            db.execute(
                "UPDATE agents SET created = ?, last_live_at = CASE WHEN last_live_at IS NULL "
                "THEN NULL ELSE ? END WHERE agent_id = ?",
                (created, created, agent_id),
            )
    finally:
        db.close()


def old(env: Env, fakebin: Fakebin, **fields: Any) -> str:
    agent_id = seed(env, fakebin, session=False, **fields)
    backdate(env, agent_id)
    return agent_id


def by_id(body: Any) -> dict[str, dict[str, Any]]:
    return {r["agent_id"]: r for r in body["results"]}


def add_label(env: Env, agent_id: str, text: str) -> None:
    with env.store() as store:
        with store.lease(lease_key(agent_id), wait=LEASE_WAIT) as lease:
            store.edit_labels(lease, agent_id, "add", [labellib.parse_label(text)])


def status_of(env: Env, agent_id: str) -> str:
    return str(env.agent(agent_id)["status"])


def test_without_a_threshold_nothing_happens_and_it_warns(env: Env, fakebin: Fakebin) -> None:
    agent_id = old(env, fakebin)
    body = prune(env)
    assert body["results"] == [] and body["dry_run"] is False
    assert any("nothing pruned" in w for w in body["warnings"])
    assert status_of(env, agent_id) == "parked"


def test_no_state_dir_is_an_empty_result(env: Env) -> None:
    assert prune(env, "--older-than", "1d")["results"] == []


def test_old_parked_agents_are_torn_down_and_young_ones_are_left(
    env: Env, fakebin: Fakebin
) -> None:
    agent_id = old(env, fakebin)
    young = seed(env, fakebin, session=False)
    body = prune(env, "--older-than", "7d")
    assert by_id(body) == {
        agent_id: {
            "agent_id": agent_id,
            "result": "torn_down",
            "reason": None,
            "tool": None,
            "details": None,
        }
    }
    assert status_of(env, agent_id) == "torn_down" and status_of(env, young) == "parked"


def test_age_comes_from_last_live_at_else_created(env: Env, fakebin: Fakebin) -> None:
    recent_run = seed(env, fakebin, session=False, last_live_at=time.time() - 3600)
    old_run = seed(env, fakebin, session=False, last_live_at=OLD)
    backdate(env, old_run)
    db = sqlite3.connect(env.state_dir / "state.db")
    try:
        with db:
            db.execute("UPDATE agents SET created = ? WHERE agent_id = ?", (OLD, recent_run))
    finally:
        db.close()
    body = prune(env, "--older-than", "7d")
    assert list(by_id(body)) == [old_run]
    assert status_of(env, recent_run) == "parked"


def test_config_older_than_is_the_default_and_the_flag_wins(
    fakebin: Fakebin, isolation_env: Isolation
) -> None:
    env = Env(fakebin, isolation_env, "prune:\n  older_than: 3650d\n")
    agent_id = old(env, fakebin)
    assert prune(env)["results"] == []
    assert list(by_id(prune(env, "--older-than", "7d"))) == [agent_id]


def test_config_older_than_applies_when_no_flag_is_given(
    fakebin: Fakebin, isolation_env: Isolation
) -> None:
    env = Env(fakebin, isolation_env, "prune:\n  older_than: 7d\n")
    agent_id = old(env, fakebin)
    young = seed(env, fakebin, session=False)
    body = prune(env)
    assert list(by_id(body)) == [agent_id] and body["results"][0]["result"] == "torn_down"
    assert status_of(env, young) == "parked"


def test_owner_label_protects_until_include_owned(env: Env, fakebin: Fakebin) -> None:
    agent_id = old(env, fakebin)
    add_label(env, agent_id, "owner=flow")
    entry = by_id(prune(env, "--older-than", "7d"))[agent_id]
    assert (entry["result"], entry["reason"]) == ("skipped", "protected")
    assert status_of(env, agent_id) == "parked"
    assert by_id(prune(env, "--older-than", "7d", "--include-owned"))[agent_id]["result"] == (
        "torn_down"
    )


def test_a_key_value_protect_label_matches_only_that_pair(
    fakebin: Fakebin, isolation_env: Isolation
) -> None:
    env = Env(fakebin, isolation_env, "prune:\n  protect_labels: [keep=yes]\n")
    kept, gone = old(env, fakebin), old(env, fakebin)
    add_label(env, kept, "keep=yes")
    add_label(env, gone, "keep=no")
    results = by_id(prune(env, "--older-than", "7d"))
    assert results[kept]["reason"] == "protected" and results[gone]["result"] == "torn_down"


def test_needs_attention_is_skipped_until_include_attention(env: Env, fakebin: Fakebin) -> None:
    agent_id = old(env, fakebin, needs_attention="failed")
    entry = by_id(prune(env, "--older-than", "7d"))[agent_id]
    assert (entry["result"], entry["reason"]) == ("skipped", "needs_attention")
    assert by_id(prune(env, "--older-than", "7d", "--include-attention"))[agent_id]["result"] == (
        "torn_down"
    )


def test_include_flags_never_imply_force_and_the_actor_is_internal(
    env: Env, fakebin: Fakebin
) -> None:
    agent_id = launched_in_worktree(env)
    add_label(env, agent_id, "owner=flow")
    with env.store() as store:
        with store.lease(lease_key(agent_id), wait=LEASE_WAIT) as lease:
            store.update_agent(lease, agent_id, {"needs_attention": "failed"})
    backdate(env, agent_id)
    fakebin.script("wtctl", "remove", times=1, **DIRTY)
    body = prune(
        env,
        "--older-than",
        "7d",
        "--include-owned",
        "--include-attention",
        "--actor",
        "human:someone",
        "--decision-id",
        "mine",
    )
    entry = by_id(body)[agent_id]
    assert (entry["result"], entry["reason"], entry["tool"]) == ("refused", "dirty", "wtctl")
    assert entry["details"]["code"] == "refused"
    (call,) = wtctl_calls(fakebin, "remove")
    assert "--force" not in call and "--actor=auto:agent-spawn" in call
    assert not any(arg.startswith("--merged-head") for arg in call)
    assert "--decision-id=mine" not in call
    assert any("--actor and --decision-id are ignored" in w for w in body["warnings"])
    assert status_of(env, agent_id) == "parked"


def test_a_worktree_agent_is_removed_under_the_internal_actor(env: Env, fakebin: Fakebin) -> None:
    agent_id = launched_in_worktree(env)
    backdate(env, agent_id)
    record = next(iter(records(fakebin).values()))
    assert by_id(prune(env, "--older-than", "7d"))[agent_id]["result"] == "torn_down"
    assert records(fakebin)[record["id"]]["state"] == "removed"
    (call,) = wtctl_calls(fakebin, "remove")
    assert "--actor=auto:agent-spawn" in call and "--force" not in call


def test_dry_run_mutates_nothing_and_asks_wtctl_status_only(env: Env, fakebin: Fakebin) -> None:
    agent_id = launched_in_worktree(env)
    backdate(env, agent_id)
    plain = old(env, fakebin)
    before = {a["agent_id"]: a for a in env.agents()}
    body = prune(env, "--older-than", "7d", "--dry-run")
    assert body["dry_run"] is True
    assert {r["result"] for r in body["results"]} == {"torn_down"} and len(body["results"]) == 2
    assert wtctl_calls(fakebin, "remove") == []
    assert {a["agent_id"]: a for a in env.agents()} == before
    assert plain in by_id(body)


def test_dry_run_reports_a_predicted_refusal(env: Env, fakebin: Fakebin) -> None:
    agent_id = launched_in_worktree(env)
    backdate(env, agent_id)
    record_id = env.agent(agent_id)["wt_record_id"]
    set_state(fakebin, record_id, "active")
    fakebin.script("wtctl", "status", times=1, merge={"would_remove": False, "reason": "dirty"})
    entry = by_id(prune(env, "--older-than", "7d", "--dry-run"))[agent_id]
    assert (entry["result"], entry["reason"]) == ("refused", "dirty")
    assert status_of(env, agent_id) == "parked"


def test_a_tearing_down_agent_with_a_dead_process_is_continued(env: Env, fakebin: Fakebin) -> None:
    agent_id = old(
        env, fakebin, status="tearing_down", teardown_step="hooks_done", teardown_proc=dead()
    )
    assert by_id(prune(env, "--older-than", "7d"))[agent_id]["result"] == "torn_down"
    assert status_of(env, agent_id) == "torn_down"


def test_a_tearing_down_agent_with_a_live_process_is_left_alone(env: Env, fakebin: Fakebin) -> None:
    agent_id = old(env, fakebin, status="tearing_down", teardown_step="started", teardown_proc=me())
    entry = by_id(prune(env, "--older-than", "7d"))[agent_id]
    assert (entry["result"], entry["reason"]) == ("skipped", "tearing_down")
    assert env.agent(agent_id)["teardown_proc"] == me()


def test_a_worktree_partial_agent_is_skipped_until_the_record_is_a_tombstone(
    env: Env, fakebin: Fakebin
) -> None:
    agent_id = launched_in_worktree(env)
    record_id = env.agent(agent_id)["wt_record_id"]
    fakebin.script("wtctl", "remove", times=1, **PARTIAL)
    assert env.run("teardown", agent_id, "--json").returncode == 12
    backdate(env, agent_id)
    with env.store() as store:
        with store.lease(lease_key(agent_id), wait=LEASE_WAIT) as lease:
            store.update_agent(lease, agent_id, {"teardown_proc": dead()})
    before = env.agent(agent_id)
    for state in ("partial", "active"):  # active would be removed again by a teardown retry
        set_state(fakebin, record_id, state)
        entry = by_id(prune(env, "--older-than", "7d"))[agent_id]
        assert (entry["result"], entry["reason"]) == ("skipped", "worktree_partial")
        assert status_of(env, agent_id) == "tearing_down"
        assert len(wtctl_calls(fakebin, "remove")) == 1
        assert env.agent(agent_id) == before  # skipped before entering: the row is untouched
        assert '"teardown_retry"' not in log_text(env)
        dry = by_id(prune(env, "--older-than", "7d", "--dry-run"))[agent_id]
        assert dry["reason"] == "worktree_partial"
    set_state(fakebin, record_id, "removed")
    assert by_id(prune(env, "--older-than", "7d"))[agent_id]["result"] == "torn_down"
    assert len(wtctl_calls(fakebin, "remove")) == 1


def test_a_parked_agent_whose_session_has_a_live_run_is_refused_live(
    env: Env, fakebin: Fakebin
) -> None:
    agent_id = seed(env, fakebin)
    backdate(env, agent_id)
    script_live_run(fakebin)
    entry = by_id(prune(env, "--older-than", "7d"))[agent_id]
    assert (entry["result"], entry["reason"], entry["tool"]) == ("refused", "live", "agent-spawn")
    assert status_of(env, agent_id) == "parked"


def test_live_and_launching_and_torn_down_agents_are_not_candidates(
    env: Env, fakebin: Fakebin
) -> None:
    ids = [
        old(env, fakebin, status="live", park_reason=None),
        old(env, fakebin, status="launching", park_reason=None),
    ]
    done = old(env, fakebin)
    assert env.run("teardown", done, "--json").returncode == 0
    assert prune(env, "--older-than", "7d")["results"] == []
    assert [status_of(env, i) for i in ids] == ["live", "launching"]


def test_an_agent_with_a_window_is_torn_down_with_its_dead_panes_closed(
    env: Env, fakebin: Fakebin
) -> None:
    agent_id = old(env, fakebin)
    with env.store() as store:
        with store.lease(lease_key(agent_id), wait=LEASE_WAIT) as lease:
            store.put_run(lease, agent_id, 1, {"term_id": "tc-aaaaaaaa", "socket_path": "/tmp/s"})
    body = prune(env, "--older-than", "7d")
    assert by_id(body)[agent_id]["result"] == "torn_down"
    assert status_of(env, agent_id) == "torn_down"
    listing = [a for a in fakebin.argvs("termctl") if a[0] == "list"]
    assert len(listing) == 1 and "--dead" in listing[0]  # step 5 ran


def test_a_pre_teardown_hook_runs_for_each_agent_and_a_veto_refuses_only_that_one(
    fakebin: Fakebin, isolation_env: Isolation, tmp_path: Path
) -> None:
    gate = tmp_path / "veto-this"
    cmd = f'[ "$AGENT_SPAWN_ID" = "$(cat {gate})" ] && exit 1; exit 0'
    hooked = Env(fakebin, isolation_env, f'hooks:\n  pre-teardown:\n    - cmd: {json.dumps(cmd)}\n')
    vetoed, passed = old(hooked, fakebin), old(hooked, fakebin)
    gate.write_text(vetoed)
    body = prune(hooked, "--older-than", "7d")
    results = by_id(body)
    assert results[vetoed]["result"] == "refused" and results[vetoed]["reason"] == "vetoed"
    assert results[vetoed]["tool"] == "agent-spawn"
    assert results[passed]["result"] == "torn_down"
    assert status_of(hooked, vetoed) == "parked" and status_of(hooked, passed) == "torn_down"
    assert not any("#32" in w for w in body["warnings"])


def test_one_agents_refusal_does_not_stop_the_others(env: Env, fakebin: Fakebin) -> None:
    first = launched_in_worktree(env)
    second = old(env, fakebin)
    backdate(env, first, OLD - 1000)
    fakebin.script("wtctl", "remove", times=1, **DIRTY)
    results = by_id(prune(env, "--older-than", "7d"))
    assert results[first]["result"] == "refused" and results[second]["result"] == "torn_down"


def test_an_unexpected_exception_for_one_agent_is_a_failure_for_that_agent_only(
    env: Env, fakebin: Fakebin, monkeypatch: pytest.MonkeyPatch
) -> None:
    first, second = old(env, fakebin), old(env, fakebin)
    real = prune_module._tear_down  # pyright: ignore[reportPrivateUsage]

    def flaky(td: Any, agent_id: str) -> Any:
        if agent_id == min(first, second):
            raise RuntimeError("boom")
        return real(td, agent_id)

    monkeypatch.setattr(prune_module, "_tear_down", flaky)
    with Store.open(env.state_dir) as store:
        judge = prune_module.Judge(
            store, Config(), older_than=7 * 86400.0, include_owned=False, include_attention=False
        )
        results = [
            prune_module._one(  # pyright: ignore[reportPrivateUsage]
                store, Config(), judge, row, str(env.state_dir), False, []
            )
            for row in sorted(store.list_agents(), key=lambda r: r["agent_id"])
        ]
    assert sorted(r["result"] for r in results) == ["failed", "torn_down"]
    failed = next(r for r in results if r["result"] == "failed")
    assert failed["details"] == {"error": "RuntimeError"}


def test_protections_are_rechecked_on_fresh_state_under_the_entry_lease(
    env: Env, fakebin: Fakebin
) -> None:
    attention = old(env, fakebin)
    owned = old(env, fakebin)
    with Store.open(env.state_dir) as store:
        stale = {a["agent_id"]: a for a in store.list_agents()}  # what the listing saw
        judge = prune_module.Judge(
            store, Config(), older_than=7 * 86400.0, include_owned=False, include_attention=False
        )
        with store.lease(lease_key(attention), wait=LEASE_WAIT) as lease:
            store.update_agent(lease, attention, {"needs_attention": "failed"})
        with store.lease(lease_key(owned), wait=LEASE_WAIT) as lease:
            store.edit_labels(lease, owned, "add", [labellib.parse_label("owner=flow")])
        results = {
            agent_id: prune_module._one(  # pyright: ignore[reportPrivateUsage]
                store, Config(), judge, stale[agent_id], str(env.state_dir), False, []
            )
            for agent_id in (attention, owned)
        }
    assert results[attention]["reason"] == "needs_attention"
    assert results[owned]["reason"] == "protected"
    assert status_of(env, attention) == status_of(env, owned) == "parked"


def corrupt_partial_marker(env: Env, agent_id: str) -> None:
    """``worktree_partial`` without the record it should name (a dead teardown process)."""
    with env.store() as store:
        with store.lease(lease_key(agent_id), wait=LEASE_WAIT) as lease:
            store.compare_and_set(
                lease,
                agent_id,
                expect_status="parked",
                expect={},
                changes={
                    "status": "tearing_down",
                    "teardown_step": "worktree_partial",
                    "teardown_partial_record": None,
                    "teardown_proc": dead(),
                },
            )


@pytest.mark.parametrize("flags", [(), ("--dry-run",)], ids=["real", "dry"])
def test_a_partial_marker_without_a_record_is_skipped_and_removes_nothing(
    env: Env, fakebin: Fakebin, flags: tuple[str, ...]
) -> None:
    agent_id = launched_in_worktree(env)
    backdate(env, agent_id)
    corrupt_partial_marker(env, agent_id)
    before = env.agent(agent_id)
    entry = by_id(prune(env, "--older-than", "7d", *flags))[agent_id]
    assert (entry["result"], entry["reason"]) == ("skipped", "worktree_partial")
    assert wtctl_calls(fakebin, "remove") == []
    assert env.agent(agent_id) == before


# --- golden --json bodies -----------------------------------------------------------------------


def normalized(body: Any, ids: list[str]) -> Any:
    """The body with the generated agent ids replaced by the golden's fixed ones."""
    text = json.dumps(body)
    for real, fixed in zip(ids, ("ag-aaaaaaaa", "ag-bbbbbbbb"), strict=True):
        text = text.replace(real, fixed)
    return json.loads(text)


@pytest.mark.parametrize(
    ("flags", "golden"),
    [((), "prune_result_live.json"), (("--dry-run",), "prune_result_dry_run.json")],
    ids=["live", "dry"],
)
def test_json_body_matches_the_golden_and_validates(
    env: Env, fakebin: Fakebin, flags: tuple[str, ...], golden: str
) -> None:
    gone = old(env, fakebin)
    kept = old(env, fakebin)
    backdate(env, gone, OLD)
    backdate(env, kept, OLD + 100)
    add_label(env, kept, "owner=flow")
    body = prune(env, "--older-than", "7d", *flags)
    expected = json.loads((GOLDEN / golden).read_text())
    schema_registry.validate(expected, "agent-spawn/cli/prune.json")
    assert normalized(body, [gone, kept]) == expected


# --- the protections under the entry lease -------------------------------------------------------


def run_one(env: Env, agent_id: str, judge_wrap: Any = None, dry_run: bool = False) -> Any:
    with Store.open(env.state_dir) as store:
        judge = prune_module.Judge(
            store, Config(), older_than=7 * 86400.0, include_owned=False, include_attention=False
        )
        if judge_wrap is not None:
            judge_wrap(store, judge)
        row = store.get_agent(agent_id)
        assert row is not None
        return prune_module._one(  # pyright: ignore[reportPrivateUsage]
            store, Config(), judge, row, str(env.state_dir), dry_run, []
        )


@pytest.mark.parametrize("what", ["owner", "attention"])
def test_the_entry_guard_alone_catches_a_change_after_the_pre_check(
    env: Env, fakebin: Fakebin, what: str
) -> None:
    """The pre-check passes (the stale listing), the change lands before the entry; only the
    guard under the lease can stop the teardown."""
    agent_id = old(env, fakebin)
    calls: list[int] = []

    def wrap(store: Store, judge: prune_module.Judge) -> None:
        real = judge.skip_reason

        def skip_reason(row: Any) -> str | None:
            calls.append(1)
            if len(calls) > 1:
                return real(row)
            with store.lease(lease_key(agent_id), wait=LEASE_WAIT) as lease:
                if what == "owner":
                    store.edit_labels(lease, agent_id, "add", [labellib.parse_label("owner=x")])
                else:
                    store.update_agent(lease, agent_id, {"needs_attention": "failed"})
            return None

        judge.skip_reason = skip_reason  # type: ignore[method-assign]

    entry = run_one(env, agent_id, wrap)
    assert len(calls) == 2  # the pre-check, then the guard under the lease
    expected = "protected" if what == "owner" else "needs_attention"
    assert (entry["result"], entry["reason"]) == ("skipped", expected)
    assert status_of(env, agent_id) == "parked"
    assert env.agent(agent_id)["teardown_step"] is None


# --- first-time unknown outcomes, tearing_down candidates, records, mappings --------------------


def test_a_first_time_partial_remove_fails_for_that_agent_and_stays_tearing_down(
    env: Env, fakebin: Fakebin
) -> None:
    agent_id = launched_in_worktree(env)
    backdate(env, agent_id)
    fakebin.script("wtctl", "remove", times=1, **PARTIAL)
    entry = by_id(prune(env, "--older-than", "7d"))[agent_id]
    assert (entry["result"], entry["reason"], entry["tool"]) == ("failed", "partial", "wtctl")
    assert status_of(env, agent_id) == "tearing_down"
    assert env.agent(agent_id)["teardown_step"] == "worktree_partial"


def test_protections_apply_to_a_tearing_down_candidate(env: Env, fakebin: Fakebin) -> None:
    agent_id = old(
        env, fakebin, status="tearing_down", teardown_step="hooks_done", teardown_proc=dead()
    )
    add_label(env, agent_id, "owner=flow")
    entry = by_id(prune(env, "--older-than", "7d"))[agent_id]
    assert (entry["result"], entry["reason"]) == ("skipped", "protected")
    assert status_of(env, agent_id) == "tearing_down"
    assert env.agent(agent_id)["teardown_proc"] == dead()


def test_a_prune_log_event_holds_the_result_without_a_message(env: Env, fakebin: Fakebin) -> None:
    agent_id = old(env, fakebin)
    kept = old(env, fakebin)
    add_label(env, kept, "owner=flow")
    prune(env, "--older-than", "7d")
    records = [json.loads(line) for line in log_text(env).splitlines()]
    events = {r["agent_id"]: r for r in records if r["event"] == "prune"}
    for record in events.values():
        schema_registry.validate(record, "agent-spawn/log.json")
        assert "message" not in record and record["actor"] == "auto:agent-spawn"
    assert events[agent_id]["details"] == {"result": "torn_down", "reason": None}
    assert events[kept]["details"] == {"result": "skipped", "reason": "protected"}


def test_a_dry_run_writes_no_prune_log_event(env: Env, fakebin: Fakebin) -> None:
    old(env, fakebin)
    prune(env, "--older-than", "7d", "--dry-run")
    assert '"prune"' not in log_text(env)


def test_the_hook_outbox_is_delivered_after_a_prune_but_not_a_dry_run(
    fakebin: Fakebin, isolation_env: Isolation, tmp_path: Path
) -> None:
    sink = tmp_path / "sink"
    env = Env(fakebin, isolation_env, f'hooks:\n  post-state-change:\n    - cmd: "cat >> {sink}"\n')
    old(env, fakebin)
    prune(env, "--older-than", "7d", "--dry-run")
    assert not sink.exists()
    prune(env, "--older-than", "7d")
    assert sink.exists() and sink.read_text() != ""


def test_a_log_write_failure_is_a_warning_in_the_result(
    env: Env, fakebin: Fakebin, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent_id = old(env, fakebin)

    def broken(*args: Any, **kwargs: Any) -> Any:
        raise OSError("disk full")

    monkeypatch.setattr("agent_spawn.teardown.append_log", broken)
    warnings: list[str] = []
    with Store.open(env.state_dir) as store:
        judge = prune_module.Judge(
            store, Config(), older_than=7 * 86400.0, include_owned=False, include_attention=False
        )
        row = store.get_agent(agent_id)
        assert row is not None
        prune_module._one(  # pyright: ignore[reportPrivateUsage]
            store, Config(), judge, row, str(env.state_dir), False, warnings
        )
    assert any("could not write the log" in w for w in warnings)


@pytest.mark.parametrize(
    ("raised", "result", "reason"),
    [
        (CliError("not_found", "gone"), "skipped", "gone"),
        (CasLost("ag-aaaaaaaa", ("parked",), None), "skipped", "changed"),
        (LeaseHeld("agent:x", me()), "skipped", "lease_held"),
    ],
    ids=["not_found", "cas_lost", "lease_held"],
)
def test_entry_failures_map_to_skips(
    env: Env,
    fakebin: Fakebin,
    monkeypatch: pytest.MonkeyPatch,
    raised: Exception,
    result: str,
    reason: str,
) -> None:
    agent_id = old(env, fakebin)

    def enter(self: Teardown, agent_id: str) -> Any:
        raise raised

    monkeypatch.setattr(Teardown, "enter", enter)
    entry = run_one(env, agent_id)
    assert (entry["result"], entry["reason"], entry["tool"]) == (result, reason, "agent-spawn")
    assert status_of(env, agent_id) == "parked"


def test_a_corrupted_created_value_never_aborts_the_prune(env: Env, fakebin: Fakebin) -> None:
    healthy = old(env, fakebin)
    broken = seed(env, fakebin, session=False)
    db = sqlite3.connect(env.state_dir / "state.db")
    try:
        with db:
            db.execute(
                "UPDATE agents SET created = 'junk', last_live_at = NULL WHERE agent_id = ?",
                (broken,),
            )
    finally:
        db.close()
    body = prune(env, "--older-than", "7d")
    assert list(by_id(body)) == [healthy]
    assert status_of(env, broken) == "parked" and status_of(env, healthy) == "torn_down"
