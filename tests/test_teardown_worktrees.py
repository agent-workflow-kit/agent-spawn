"""``teardown`` step 4 against the fake wtctl: the exact argv, held records released in order and
the owned record removed last, the abort and partial rules, retries, the dry run, the audit
record, the ephemeral teardown and the fork end to end (#23, and #22's third criterion)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

from tests import schema_registry
from tests.conftest import Isolation
from tests.fakebin.harness import Fakebin
from tests.launch_support import Env, result_json
from tests.test_fork_holds import holders, nested_source
from tests.test_teardown import dead, events, seed
from tests.test_worktrees_launch import LAUNCH, REPO, records, wtctl_calls
from tests.test_worktrees_resume import launched_in_worktree, records_path, set_state

SHA = "a" * 40
HUMAN = ("--actor", "human:tester")
PARTIAL = {
    "error": {"code": "partial", "message": "x", "details": {"remaining_path": "/tmp/x"}},
    "exit": 1,
}
DIRTY = {
    "error": {"code": "refused", "reason": "dirty", "message": "x", "checks": {}},
    "exit": 20,
}


@pytest.fixture
def env(fakebin: Fakebin, isolation_env: Isolation) -> Env:
    return Env(fakebin, isolation_env)


def td(env: Env, agent_id: str, *flags: str) -> Any:
    return env.run("teardown", agent_id, "--json", *flags)


def edit_record(fakebin: Fakebin, record_id: str, **changes: Any) -> None:
    data = records(fakebin)
    data[record_id].update(changes)
    records_path(fakebin).write_text(json.dumps(data))


def td_with_short_remove_timeout(env: Env, agent_id: str) -> Any:
    """``teardown`` in a child whose wtctl ``remove`` times out after 2 seconds."""
    import subprocess

    code = (
        "import agent_spawn.worktrees as w; w.REMOVE_TIMEOUT = 2.0; "
        "from agent_spawn.cli import main; raise SystemExit(main())"
    )
    return subprocess.run(
        [sys.executable, "-c", code, "teardown", agent_id, "--json"],
        capture_output=True,
        text=True,
        env=env.env(),
        stdin=subprocess.DEVNULL,
        timeout=60,
        check=False,
    )


def audit_records(env: Env) -> list[dict[str, Any]]:
    path = env.state_dir / "audit.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines()]


def hold_rows(env: Env, agent_id: str) -> list[dict[str, Any]]:
    with env.store() as store:
        return store.get_holds(agent_id, include_released=True)


# --- the owned record --------------------------------------------------------------------------


def test_the_owned_record_is_removed_with_the_exact_argv(env: Env, fakebin: Fakebin) -> None:
    agent_id = launched_in_worktree(env)
    record = next(iter(records(fakebin).values()))
    done = td(env, agent_id, "--merged-head", SHA, "--decision-id", "d1")
    body = result_json(done)
    assert done.returncode == 0, (done.stdout, done.stderr)
    schema_registry.validate(body, "agent-spawn/cli/teardown.json")
    assert wtctl_calls(fakebin, "remove") == [
        [
            "remove",
            record["id"],
            "--yes",
            f"--holder={agent_id}",
            f"--merged-head={SHA}",
            "--actor=auto:unknown",
            "--decision-id=d1",
            "--json",
        ]
    ]
    (entry,) = body["worktrees"]
    assert entry["record_id"] == record["id"] and entry["owned"] is True
    assert entry["action"] == "removed" and entry["wtctl"]["record"]["state"] == "removed"
    assert records(fakebin)[record["id"]]["state"] == "removed"
    assert [r["released"] for r in hold_rows(env, agent_id)] == [True]
    assert env.agent(agent_id)["status"] == "torn_down"
    assert wtctl_calls(fakebin, "forget") == []
    assert audit_records(env)[0]["inputs"]["worktree"] == {
        "record_id": record["id"],
        "path": record["path"],
        "owned": True,
    }


def test_the_post_teardown_event_lists_the_worktrees(env: Env, fakebin: Fakebin) -> None:
    agent_id = launched_in_worktree(env)
    record_id = env.agent(agent_id)["wt_record_id"]
    assert td(env, agent_id).returncode == 0
    post = dict(events(env, agent_id))["post-teardown"]
    assert [(w["record_id"], w["action"], w["owned"]) for w in post["worktrees"]] == [
        (record_id, "removed", True)
    ]
    assert all("wtctl" not in w for w in post["worktrees"])


def test_no_merged_head_and_no_force_flag_unless_given(env: Env, fakebin: Fakebin) -> None:
    agent_id = launched_in_worktree(env)
    assert td(env, agent_id).returncode == 0
    (argv,) = wtctl_calls(fakebin, "remove")
    assert not any(a.startswith("--merged-head") for a in argv)
    assert "--force" not in argv and "--override-escalation" not in argv
    assert audit_records(env)[0]["inputs"]["force"] is False  # the removal itself is audited


def test_force_from_a_human_reaches_only_wtctl_remove_and_is_audited(
    env: Env, fakebin: Fakebin
) -> None:
    agent_id = launched_in_worktree(env)
    done = td(env, agent_id, *HUMAN, "--force")
    assert done.returncode == 0, (done.stdout, done.stderr)
    (argv,) = wtctl_calls(fakebin, "remove")
    assert "--force" in argv and "--actor=human:tester" in argv
    assert [r["inputs"]["force"] for r in audit_records(env)] == [True]  # one record, not two


@pytest.mark.parametrize("flag", ["--force", "--override-escalation"])
def test_human_only_flags_from_automation_exit_2_and_change_nothing(
    env: Env, fakebin: Fakebin, flag: str
) -> None:
    agent_id = launched_in_worktree(env)
    for actor in ("auto:bot", "human-ish:x"):
        done = td(env, agent_id, "--actor", actor, flag)
        assert done.returncode == 2, (done.stdout, done.stderr)
    assert wtctl_calls(fakebin, "remove") == []
    assert env.agent(agent_id)["status"] == "parked"


@pytest.mark.parametrize(
    "sha", ["abc", "A" * 40, "a" * 39, "a" * 41, "--force", "g" * 40, "a" * 40 + "\n"]
)
def test_a_merged_head_that_is_not_a_full_lowercase_sha_exits_2(
    env: Env, fakebin: Fakebin, sha: str
) -> None:
    agent_id = launched_in_worktree(env)
    done = td(env, agent_id, f"--merged-head={sha}")
    assert done.returncode == 2
    assert wtctl_calls(fakebin, "remove") == []
    assert env.agent(agent_id)["status"] == "parked"


def test_override_escalation_from_a_human_is_passed_to_remove(env: Env, fakebin: Fakebin) -> None:
    agent_id = launched_in_worktree(env)
    assert td(env, agent_id, *HUMAN, "--override-escalation").returncode == 0
    (argv,) = wtctl_calls(fakebin, "remove")
    assert "--override-escalation" in argv and "--force" not in argv


# --- held records --------------------------------------------------------------------------------


def test_an_inner_and_an_outer_record_are_released_in_order_and_both_listed(
    env: Env, fakebin: Fakebin, tmp_path: Path
) -> None:
    source, _, _ = nested_source(env, fakebin, tmp_path)
    stored = [h["record_id"] for h in hold_rows(env, source)]
    assert len(stored) == 2
    done = td(env, source, *HUMAN)
    body = result_json(done)
    assert done.returncode == 0, (done.stdout, done.stderr)
    schema_registry.validate(body, "agent-spawn/cli/teardown.json")
    assert [c[1] for c in wtctl_calls(fakebin, "forget")] == stored
    assert wtctl_calls(fakebin, "remove") == []
    assert [(w["record_id"], w["owned"]) for w in body["worktrees"]] == [(r, False) for r in stored]
    assert {w["action"] for w in body["worktrees"]} == {
        "released"
    }  # the test records keep another holder
    for argv in wtctl_calls(fakebin, "forget"):
        assert f"--holder={source}" in argv and "--force" not in argv and "--yes" not in argv
    assert all(r["released"] for r in hold_rows(env, source))
    assert audit_records(env) == []  # nothing was removed, nothing forced


def test_held_records_come_before_the_owned_one(env: Env, fakebin: Fakebin, tmp_path: Path) -> None:
    agent_id = launched_in_worktree(env)
    owned_id = env.agent(agent_id)["wt_record_id"]
    other = tmp_path / "other"
    other.mkdir()
    held = json.loads(
        fakebin.run("wtctl", "adopt", str(other), "--holder", agent_id, "--json").stdout
    )["record"]
    with env.store() as store:
        from agent_spawn.lifecycle import lease_key

        with store.lease(lease_key(agent_id)) as lease:
            store.put_hold(lease, agent_id, held["id"], held["path"], owned=False)
    body = result_json(td(env, agent_id))
    assert [(w["record_id"], w["action"]) for w in body["worktrees"]] == [
        (held["id"], "forgotten"),
        (owned_id, "removed"),
    ]
    order = [(c[0], c[1]) for c in wtctl_calls(fakebin) if c[0] in ("forget", "remove")]
    assert order == [("forget", held["id"]), ("remove", owned_id)]


def test_a_failure_on_the_second_hold_parks_with_the_first_released_and_resume_re_holds(
    env: Env, fakebin: Fakebin, tmp_path: Path
) -> None:
    source, _, _ = nested_source(env, fakebin, tmp_path)
    first, second = (h["record_id"] for h in hold_rows(env, source))
    fakebin.script("wtctl", "forget", argv_contains=[second], exit=12,
                   error={"code": "busy", "message": "later"})  # fmt: skip
    done = td(env, source)
    body = result_json(done)
    schema_registry.validate(body, "agent-spawn/cli/error.json")
    assert done.returncode == 12 and body["error"]["tool"] == "wtctl"
    assert body["error"]["lower_error"]["code"] == "busy"
    row = env.agent(source)
    assert row["status"] == "parked" and row["teardown_step"] is None
    assert row["teardown_proc"] is None
    assert [(h["record_id"], h["released"]) for h in hold_rows(env, source)] == [
        (first, True),
        (second, False),
    ]
    assert source not in holders(fakebin, first) and source in holders(fakebin, second)
    # The next run re-holds the released record.
    resumed = env.run("resume", source, "--json", "--wait")
    assert resumed.returncode == 0, (resumed.stdout, resumed.stderr)
    assert source in holders(fakebin, first)
    assert sorted(h["record_id"] for h in result_json(resumed)["agent"]["holds"]) == sorted(
        [first, second]
    )


def test_a_hold_row_released_earlier_is_not_released_again(
    env: Env, fakebin: Fakebin, tmp_path: Path
) -> None:
    source, _, _ = nested_source(env, fakebin, tmp_path)
    first, second = (h["record_id"] for h in hold_rows(env, source))
    fakebin.script("wtctl", "forget", argv_contains=[second], times=1, exit=12,
                   error={"code": "busy", "message": "later"})  # fmt: skip
    assert td(env, source).returncode == 12
    body = result_json(td(env, source))
    assert [(w["record_id"], w["action"]) for w in body["worktrees"]] == [
        (first, "none"),
        (second, "released"),
    ]
    assert body["worktrees"][0]["wtctl"] is None
    assert [c[1] for c in wtctl_calls(fakebin, "forget")] == [first, second, second]


def test_a_refusal_passes_through_with_lower_error_and_parks(env: Env, fakebin: Fakebin) -> None:
    agent_id = launched_in_worktree(env)
    record_id = env.agent(agent_id)["wt_record_id"]
    fakebin.script("wtctl", "remove", times=1, **DIRTY)
    done = td(env, agent_id)
    body = result_json(done)
    schema_registry.validate(body, "agent-spawn/cli/error.json")
    assert done.returncode == 20
    err = body["error"]
    assert err["reason"] == "dirty" and err["tool"] == "wtctl"
    assert err["lower_error"]["reason"] == "dirty"
    row = env.agent(agent_id)
    assert row["status"] == "parked" and row["teardown_step"] is None
    assert records(fakebin)[record_id]["state"] == "active"
    assert [r["released"] for r in hold_rows(env, agent_id)] == [False]
    assert td(env, agent_id).returncode == 0  # nothing was lost: the next teardown works


@pytest.mark.parametrize(
    ("script", "code"),
    [
        ({"exit": 13, "error": {"code": "confirmation_required", "message": "x"}}, 13),
        ({"exit": 11, "error": {"code": "vetoed", "message": "x"}}, 11),
        ({"exit": 12, "error": {"code": "fetch_failed", "message": "x"}}, 12),
        ({"exit": 21, "error": {"code": "ambiguous", "message": "x", "candidates": []}}, 21),
    ],
)
def test_other_refusals_return_the_agent_to_parked_with_wtctls_exit(
    env: Env, fakebin: Fakebin, script: dict[str, Any], code: int
) -> None:
    agent_id = launched_in_worktree(env)
    fakebin.script("wtctl", "remove", times=1, **script)
    done = td(env, agent_id)
    assert done.returncode == code, (done.stdout, done.stderr)
    assert result_json(done)["error"]["tool"] == "wtctl"
    assert env.agent(agent_id)["status"] == "parked"


# --- unknown outcomes and partial removals -----------------------------------------------------


def test_an_exit_1_from_remove_leaves_the_agent_tearing_down_and_a_retry_finishes(
    env: Env, fakebin: Fakebin
) -> None:
    agent_id = launched_in_worktree(env)
    fakebin.script("wtctl", "remove", times=1, exit=1, error={"code": "internal", "message": "x"})
    done = td(env, agent_id)
    assert done.returncode == 1, (done.stdout, done.stderr)
    row = env.agent(agent_id)
    assert row["status"] == "tearing_down" and row["teardown_step"] == "worktree_partial"
    assert row["teardown_partial_record"] == env.agent(agent_id)["wt_record_id"]
    again = td(env, agent_id)
    assert again.returncode == 0, (again.stdout, again.stderr)
    assert env.agent(agent_id)["status"] == "torn_down"


def test_partial_keeps_tearing_down_and_a_retry_checks_the_stored_record(
    env: Env, fakebin: Fakebin
) -> None:
    agent_id = launched_in_worktree(env)
    record_id = env.agent(agent_id)["wt_record_id"]
    fakebin.script("wtctl", "remove", times=1, **PARTIAL)
    done = td(env, agent_id)
    set_state(fakebin, record_id, "partial")  # what wtctl leaves behind
    body = result_json(done)
    schema_registry.validate(body, "agent-spawn/cli/error.json")
    assert done.returncode == 12 and body["error"]["code"] == "partial"
    assert body["error"]["tool"] == "wtctl" and body["error"]["lower_error"]["code"] == "partial"
    row = env.agent(agent_id)
    assert row["status"] == "tearing_down" and row["teardown_step"] == "worktree_partial"
    assert row["teardown_partial_record"] == record_id
    # The record is still partial: the retry shows it, never removes it again, stays partial.
    again = td(env, agent_id)
    assert again.returncode == 12, (again.stdout, again.stderr)
    assert len(wtctl_calls(fakebin, "remove")) == 1
    assert wtctl_calls(fakebin, "show")[-1][1] == record_id
    assert env.agent(agent_id)["status"] == "tearing_down"
    # A human finishes it; the next retry continues past the worktree step.
    set_state(fakebin, record_id, "removed")
    last = td(env, agent_id)
    body = result_json(last)
    assert last.returncode == 0, (last.stdout, last.stderr)
    assert len(wtctl_calls(fakebin, "remove")) == 1
    assert [(w["record_id"], w["action"]) for w in body["worktrees"]] == [(record_id, "removed")]
    assert env.agent(agent_id)["status"] == "torn_down"
    assert [r["released"] for r in hold_rows(env, agent_id)] == [True]
    assert len(audit_records(env)) == 1  # the removal is audited once


def test_a_partial_marker_naming_a_record_that_is_not_the_agents_stops_the_teardown(
    env: Env, fakebin: Fakebin
) -> None:
    foreign = json.loads(
        fakebin.run(
            "wtctl", "create", "--repo", REPO, "--holder", "ag-zzzzzzzz", "--name", "x", "--json"
        ).stdout
    )["record"]["id"]
    agent_id = seed(
        env,
        fakebin,
        status="tearing_down",
        teardown_step="worktree_partial",
        teardown_partial_record=foreign,
        teardown_proc=dead(),
    )
    done = td(env, agent_id)
    assert done.returncode != 0
    message = result_json(done)["error"]["message"]
    assert foreign in message and "human" in message  # names the record, asks for a look
    assert wtctl_calls(fakebin, "remove") == [] and wtctl_calls(fakebin, "forget") == []
    assert records(fakebin)[foreign]["state"] == "active"
    assert env.agent(agent_id)["status"] == "tearing_down"


# --- an unknown remove outcome is sticky (R1) ---------------------------------------------------

UNKNOWN = {"error": {"code": "internal", "message": "x"}, "exit": 1}


def tearing_down_at(env: Env, agent_id: str, record_id: str) -> None:
    row = env.agent(agent_id)
    assert row["status"] == "tearing_down", row["status"]
    assert row["teardown_step"] == "worktree_partial"
    assert row["teardown_partial_record"] == record_id


def test_an_unknown_remove_then_a_partial_record_stays_tearing_down(
    env: Env, fakebin: Fakebin
) -> None:
    agent_id = launched_in_worktree(env)
    record_id = env.agent(agent_id)["wt_record_id"]
    fakebin.script("wtctl", "remove", times=1, **UNKNOWN)
    assert td(env, agent_id).returncode == 1
    tearing_down_at(env, agent_id, record_id)
    set_state(fakebin, record_id, "partial")
    again = td(env, agent_id)
    assert again.returncode == 12 and result_json(again)["error"]["code"] == "partial"
    tearing_down_at(env, agent_id, record_id)
    assert len(wtctl_calls(fakebin, "remove")) == 1


def test_an_unknown_remove_then_a_retry_refused_13_stays_tearing_down(
    env: Env, fakebin: Fakebin
) -> None:
    agent_id = launched_in_worktree(env)
    record_id = env.agent(agent_id)["wt_record_id"]
    fakebin.script("wtctl", "remove", times=1, **UNKNOWN)
    assert td(env, agent_id).returncode == 1
    fakebin.script(
        "wtctl", "remove", times=1,
        exit=13, error={"code": "confirmation_required", "message": "x"},
    )  # fmt: skip
    again = td(env, agent_id)
    body = result_json(again)
    assert again.returncode == 13, (again.stdout, again.stderr)
    assert body["error"]["tool"] == "wtctl" and body["error"]["lower_error"]["code"]
    tearing_down_at(env, agent_id, record_id)
    assert len(wtctl_calls(fakebin, "remove")) == 2


def test_a_timed_out_remove_leaves_tearing_down_and_a_dirty_retry_does_too(
    env: Env, fakebin: Fakebin
) -> None:
    agent_id = launched_in_worktree(env)
    record_id = env.agent(agent_id)["wt_record_id"]
    fakebin.script("wtctl", "remove", times=1, sleep=30, **UNKNOWN)
    done = td_with_short_remove_timeout(env, agent_id)
    assert done.returncode != 0, (done.stdout, done.stderr)
    tearing_down_at(env, agent_id, record_id)
    fakebin.script("wtctl", "remove", times=1, **DIRTY)
    again = td(env, agent_id)
    assert again.returncode == 20, (again.stdout, again.stderr)
    assert result_json(again)["error"]["reason"] == "dirty"
    tearing_down_at(env, agent_id, record_id)  # never back to parked
    assert len(wtctl_calls(fakebin, "remove")) == 2  # the retry called remove again
    last = td(env, agent_id)  # remove works now
    assert last.returncode == 0, (last.stdout, last.stderr)
    assert env.agent(agent_id)["status"] == "torn_down"


def test_a_timed_out_remove_that_deleted_everything_is_a_tombstone_on_retry(
    env: Env, fakebin: Fakebin
) -> None:
    agent_id = launched_in_worktree(env)
    record_id = env.agent(agent_id)["wt_record_id"]
    fakebin.script("wtctl", "remove", times=1, sleep=30, **UNKNOWN)
    assert td_with_short_remove_timeout(env, agent_id).returncode != 0
    set_state(fakebin, record_id, "removed")
    again = td(env, agent_id)
    assert again.returncode == 0, (again.stdout, again.stderr)
    assert env.agent(agent_id)["status"] == "torn_down"
    assert len(wtctl_calls(fakebin, "remove")) == 1
    assert len(audit_records(env)) == 1  # R4: the removal is audited


def test_a_retry_over_an_unknown_outcome_with_an_exit_10_tombstone_completes(
    env: Env, fakebin: Fakebin
) -> None:
    agent_id = launched_in_worktree(env)
    fakebin.script("wtctl", "remove", times=1, **UNKNOWN)
    assert td(env, agent_id).returncode == 1
    fakebin.script(
        "wtctl", "remove", times=1,
        exit=10, error={"code": "not_found", "message": "x", "details": {"state": "removed"}},
    )  # fmt: skip
    again = td(env, agent_id)
    assert again.returncode == 0, (again.stdout, again.stderr)
    assert env.agent(agent_id)["status"] == "torn_down"


def test_an_unknown_outcome_audits_the_removal_once_when_the_record_turns_out_removed(
    env: Env, fakebin: Fakebin
) -> None:
    agent_id = launched_in_worktree(env)
    record_id = env.agent(agent_id)["wt_record_id"]
    fakebin.script("wtctl", "remove", times=1, **UNKNOWN)
    assert td(env, agent_id).returncode == 1
    set_state(fakebin, record_id, "removed")
    assert td(env, agent_id).returncode == 0
    (audit,) = audit_records(env)
    assert audit["action"] == "teardown" and audit["inputs"]["worktree"]["record_id"] == record_id
    schema_registry.validate(audit, "agent-spawn/audit.json")
    assert td(env, agent_id).returncode == 10  # torn down already: no second record
    assert len(audit_records(env)) == 1


def test_an_owned_row_whose_record_is_a_tombstone_is_audited(env: Env, fakebin: Fakebin) -> None:
    agent_id = launched_in_worktree(env)
    record_id = env.agent(agent_id)["wt_record_id"]
    set_state(fakebin, record_id, "removed")
    done = td(env, agent_id)
    assert done.returncode == 0, (done.stdout, done.stderr)
    assert [a["inputs"]["worktree"]["record_id"] for a in audit_records(env)] == [record_id]


def test_an_owned_row_whose_record_is_forgotten_or_gone_is_not_audited_as_a_removal(
    env: Env, fakebin: Fakebin
) -> None:
    for state in ("forgotten", "gone"):
        agent_id = launched_in_worktree(env)
        record_id = env.agent(agent_id)["wt_record_id"]
        set_state(fakebin, record_id, state)
        done = td(env, agent_id)
        assert done.returncode == 0, (done.stdout, done.stderr)
        assert env.agent(agent_id)["status"] == "torn_down"
    assert audit_records(env) == []


def test_a_retry_over_a_forgotten_record_is_not_audited_as_a_removal(
    env: Env, fakebin: Fakebin
) -> None:
    agent_id = launched_in_worktree(env)
    record_id = env.agent(agent_id)["wt_record_id"]
    fakebin.script("wtctl", "remove", times=1, **UNKNOWN)
    assert td(env, agent_id).returncode == 1
    set_state(fakebin, record_id, "forgotten")
    assert td(env, agent_id).returncode == 0
    assert audit_records(env) == []


# --- the hold is checked before remove (R2) ------------------------------------------------------


def test_a_record_wtctl_says_is_not_owned_is_forgotten_not_removed(
    env: Env, fakebin: Fakebin
) -> None:
    agent_id = launched_in_worktree(env)
    record_id = env.agent(agent_id)["wt_record_id"]
    edit_record(fakebin, record_id, owned=False)
    done = td(env, agent_id)
    body = result_json(done)
    assert done.returncode == 0, (done.stdout, done.stderr)
    assert wtctl_calls(fakebin, "remove") == []
    assert [c[1] for c in wtctl_calls(fakebin, "forget")] == [record_id]
    assert [(w["action"], w["owned"]) for w in body["worktrees"]] == [("forgotten", False)]
    assert records(fakebin)[record_id]["state"] == "forgotten"


def test_an_owned_row_for_a_record_the_agent_does_not_hold_is_refused_not_held(
    env: Env, fakebin: Fakebin
) -> None:
    agent_id = launched_in_worktree(env)
    record_id = env.agent(agent_id)["wt_record_id"]
    edit_record(fakebin, record_id, holders=["ag-zzzzzzzz"])
    done = td(env, agent_id)
    assert done.returncode == 20 and result_json(done)["error"]["reason"] == "not_held"
    assert wtctl_calls(fakebin, "remove") == [] and wtctl_calls(fakebin, "forget") == []
    assert records(fakebin)[record_id]["state"] == "active"
    row = env.agent(agent_id)
    assert row["status"] == "parked" and row["teardown_step"] is None
    assert [r["released"] for r in hold_rows(env, agent_id)] == [False]


# --- the override is audited (R5) -----------------------------------------------------------------


def test_override_escalation_is_audited_with_the_real_result(env: Env, fakebin: Fakebin) -> None:
    agent_id = launched_in_worktree(env)
    assert td(env, agent_id, *HUMAN, "--override-escalation").returncode == 0
    audit = audit_records(env)
    assert sorted(a["action"] for a in audit) == ["teardown", "teardown_escalation_override"]
    override = next(a for a in audit if a["action"] == "teardown_escalation_override")
    assert override["result"] == "ok" and override["target"] == agent_id
    assert override["actor"] == "human:tester"
    schema_registry.validate(override, "agent-spawn/audit.json")


def test_a_refused_override_is_audited_as_refused(env: Env, fakebin: Fakebin) -> None:
    agent_id = launched_in_worktree(env)
    fakebin.script("wtctl", "remove", times=1, **DIRTY)
    assert td(env, agent_id, *HUMAN, "--override-escalation").returncode == 20
    (override,) = audit_records(env)
    assert override["action"] == "teardown_escalation_override" and override["result"] == "refused"


def test_no_override_audit_without_the_flag_or_without_a_remove(env: Env, fakebin: Fakebin) -> None:
    agent_id = launched_in_worktree(env)
    assert td(env, agent_id, *HUMAN).returncode == 0
    assert [a["action"] for a in audit_records(env)] == ["teardown"]


# --- exit 2 changed nothing (R7) ---------------------------------------------------------


def test_a_wtctl_usage_error_exit_2_returns_the_agent_to_parked(env: Env, fakebin: Fakebin) -> None:
    agent_id = launched_in_worktree(env)
    record_id = env.agent(agent_id)["wt_record_id"]
    fakebin.script("wtctl", "remove", times=1, exit=2, error={"code": "usage", "message": "x"})
    done = td(env, agent_id)
    assert done.returncode != 0
    assert result_json(done)["error"]["tool"] == "wtctl"
    row = env.agent(agent_id)
    assert row["status"] == "parked" and row["teardown_step"] is None
    assert records(fakebin)[record_id]["state"] == "active"


@pytest.mark.parametrize("code", [3, 137, 255])
def test_an_unmapped_wtctl_exit_stays_tearing_down(env: Env, fakebin: Fakebin, code: int) -> None:
    agent_id = launched_in_worktree(env)
    record_id = env.agent(agent_id)["wt_record_id"]
    fakebin.script("wtctl", "remove", times=1, exit=code, error={"code": "x", "message": "x"})
    assert td(env, agent_id).returncode != 0
    tearing_down_at(env, agent_id, record_id)


# --- the dry run's prediction (R6, R8, R9, R10) ------------------------------------------------


@pytest.mark.parametrize(
    ("flag", "reason"), [("fake_dirty", "dirty"), ("fake_unpushed", "unpushed")]
)
def test_dry_run_with_force_overrides_dirty_and_unpushed(
    env: Env, fakebin: Fakebin, flag: str, reason: str
) -> None:
    agent_id = launched_in_worktree(env)
    edit_record(fakebin, env.agent(agent_id)["wt_record_id"], **{flag: True})
    plain = td(env, agent_id, "--dry-run")
    assert plain.returncode == 20 and result_json(plain)["would_refuse"] == reason
    forced = td(env, agent_id, *HUMAN, "--force", "--dry-run")
    body = result_json(forced)
    assert forced.returncode == 0, (forced.stdout, forced.stderr)
    assert body["would_refuse"] is None
    schema_registry.validate(body, "agent-spawn/cli/teardown-dry-run.json")


def test_dry_run_with_force_does_not_override_other_reasons(env: Env, fakebin: Fakebin) -> None:
    agent_id = launched_in_worktree(env)
    fakebin.script(
        "wtctl", "status",
        merge={"would_remove": False, "reason": "in_use", "force_overrides": []},
    )  # fmt: skip
    done = td(env, agent_id, *HUMAN, "--force", "--dry-run")
    assert done.returncode == 20 and result_json(done)["would_refuse"] == "in_use"


def test_dry_run_predicts_confirmation_required_for_a_non_human_actor(
    env: Env, fakebin: Fakebin
) -> None:
    agent_id = launched_in_worktree(env)
    fakebin.script("wtctl", "status", merge={"human_required": True})
    robot = td(env, agent_id, "--dry-run")
    assert robot.returncode == 20
    assert result_json(robot)["would_refuse"] == "confirmation_required"
    human = td(env, agent_id, *HUMAN, "--dry-run")
    assert human.returncode == 0 and result_json(human)["would_refuse"] is None


@pytest.mark.parametrize("would_remove", [None, "yes"])
def test_dry_run_does_not_fail_open_on_an_undetermined_status(
    env: Env, fakebin: Fakebin, would_remove: object
) -> None:
    agent_id = launched_in_worktree(env)
    fakebin.script("wtctl", "status", merge={"would_remove": would_remove, "reason": None})
    done = td(env, agent_id, "--dry-run")
    body = result_json(done)
    assert done.returncode == 20 and body["would_refuse"] == "unknown"
    schema_registry.validate(body, "agent-spawn/cli/teardown-dry-run.json")


def test_dry_run_does_not_fail_open_on_a_status_without_would_remove(
    env: Env, fakebin: Fakebin
) -> None:
    agent_id = launched_in_worktree(env)
    record_id = env.agent(agent_id)["wt_record_id"]
    fakebin.script(
        "wtctl", "status",
        stdout={"id": record_id, "object_type": "worktree", "reason": None},
    )  # fmt: skip
    done = td(env, agent_id, "--dry-run")
    assert done.returncode == 20 and result_json(done)["would_refuse"] == "unknown"


def test_dry_run_reports_not_held_as_would_refuse(env: Env, fakebin: Fakebin) -> None:
    agent_id = seed(env, fakebin, wt_owned=True)
    foreign = json.loads(
        fakebin.run(
            "wtctl", "create", "--repo", REPO, "--holder", "ag-zzzzzzzz", "--name", "x", "--json"
        ).stdout
    )["record"]
    from agent_spawn.lifecycle import lease_key

    with env.store() as store:
        with store.lease(lease_key(agent_id)) as lease:
            store.update_agent(lease, agent_id, {"wt_record_id": foreign["id"]})
    done = td(env, agent_id, "--dry-run")
    body = result_json(done)
    assert done.returncode == 20, (done.stdout, done.stderr)
    assert body["would_refuse"] == "not_held" and body["dry_run"] is True
    schema_registry.validate(body, "agent-spawn/cli/teardown-dry-run.json")


def test_force_with_no_worktree_still_writes_the_audit_record(env: Env, fakebin: Fakebin) -> None:
    agent_id = seed(env, fakebin)
    assert td(env, agent_id, *HUMAN, "--force").returncode == 0
    (audit,) = audit_records(env)
    assert audit["inputs"]["worktree"] is None and audit["inputs"]["force"] is True


# --- a bootstrap_failed launch: a record id without a path or a hold row -----------------------


def test_a_record_id_without_a_hold_row_is_shown_then_removed(env: Env, fakebin: Fakebin) -> None:
    agent_id = seed(env, fakebin, wt_owned=True)
    created = json.loads(
        fakebin.run(
            "wtctl", "create", "--repo", REPO, "--holder", agent_id, "--name", agent_id, "--json"
        ).stdout
    )["record"]
    from agent_spawn.lifecycle import lease_key

    with env.store() as store:
        with store.lease(lease_key(agent_id)) as lease:
            store.update_agent(lease, agent_id, {"wt_record_id": created["id"]})
    assert env.agent(agent_id)["wt_path"] is None
    done = td(env, agent_id)
    body = result_json(done)
    assert done.returncode == 0, (done.stdout, done.stderr)
    assert wtctl_calls(fakebin, "show")[0][1] == created["id"]
    assert [(w["record_id"], w["path"], w["action"]) for w in body["worktrees"]] == [
        (created["id"], created["path"], "removed")
    ]
    assert records(fakebin)[created["id"]]["state"] == "removed"
    assert env.agent(agent_id)["status"] == "torn_down"


def test_a_stored_record_that_the_agent_does_not_hold_is_never_removed(
    env: Env, fakebin: Fakebin, tmp_path: Path
) -> None:
    agent_id = seed(env, fakebin, wt_owned=True)
    foreign = json.loads(
        fakebin.run(
            "wtctl", "create", "--repo", REPO, "--holder", "ag-zzzzzzzz", "--name", "x", "--json"
        ).stdout
    )["record"]
    from agent_spawn.lifecycle import lease_key

    with env.store() as store:
        with store.lease(lease_key(agent_id)) as lease:
            store.update_agent(lease, agent_id, {"wt_record_id": foreign["id"]})
    done = td(env, agent_id)
    assert done.returncode == 20 and result_json(done)["error"]["reason"] == "not_held"
    assert wtctl_calls(fakebin, "remove") == []
    assert records(fakebin)[foreign["id"]]["state"] == "active"
    assert env.agent(agent_id)["status"] == "parked"


def test_a_record_wtctl_does_not_know_is_skipped_with_a_warning(env: Env, fakebin: Fakebin) -> None:
    agent_id = seed(env, fakebin, wt_record_id="wt-abcdefgh", wt_owned=True)
    done = td(env, agent_id)
    body = result_json(done)
    assert done.returncode == 0, (done.stdout, done.stderr)
    assert body["worktrees"] == [] and body["warnings"]
    assert wtctl_calls(fakebin, "remove") == []


# --- the fork end to end (#22, third criterion) -----------------------------------------------


def test_teardown_of_a_source_is_refused_held_while_a_fork_holds_it_then_succeeds(
    env: Env, fakebin: Fakebin
) -> None:
    source = launched_in_worktree(env)
    record_id = env.agent(source)["wt_record_id"]
    forked = env.run("resume", source, "--fork", "--json", "--wait")
    assert forked.returncode == 0, (forked.stdout, forked.stderr)
    fork = result_json(forked)["agent"]["agent_id"]
    assert [h["owned"] for h in hold_rows(env, fork)] == [False]  # adopted, never removed by it
    done = td(env, source)
    body = result_json(done)
    schema_registry.validate(body, "agent-spawn/cli/error.json")
    assert done.returncode == 20
    err = body["error"]
    assert err["reason"] == "held" and err["tool"] == "wtctl"
    assert fork in err["lower_error"]["details"]["holders"]
    row = env.agent(source)
    assert row["status"] == "parked" and row["teardown_step"] is None
    assert records(fakebin)[record_id]["state"] == "active"
    assert [r["released"] for r in hold_rows(env, source)] == [False]
    # The fork is torn down first (a release, not a removal); then the source's teardown works.
    gone = td(env, fork)
    assert gone.returncode == 0, (gone.stdout, gone.stderr)
    assert [w["action"] for w in result_json(gone)["worktrees"]] == ["released"]
    assert records(fakebin)[record_id]["state"] == "active"
    last = td(env, source)
    assert last.returncode == 0, (last.stdout, last.stderr)
    assert records(fakebin)[record_id]["state"] == "removed"
    assert env.agent(source)["status"] == "torn_down"


def test_an_adopted_source_record_is_released_not_removed(
    env: Env, fakebin: Fakebin, tmp_path: Path
) -> None:
    from tests.test_worktrees_launch import tree

    root = tree(tmp_path)
    done = env.run(*LAUNCH, "--worktree", str(root), "--repo", REPO)
    assert done.returncode == 0, (done.stdout, done.stderr)
    agent_id = result_json(done)["agent"]["agent_id"]
    body = result_json(td(env, agent_id))
    assert [w["action"] for w in body["worktrees"]] == ["forgotten"]
    assert wtctl_calls(fakebin, "remove") == []


# --- the dry run -------------------------------------------------------------------------------


def test_dry_run_asks_wtctl_status_for_the_owned_record_and_changes_nothing(
    env: Env, fakebin: Fakebin
) -> None:
    agent_id = launched_in_worktree(env)
    record_id = env.agent(agent_id)["wt_record_id"]
    done = td(env, agent_id, "--dry-run", "--merged-head", SHA)
    body = result_json(done)
    assert done.returncode == 0, (done.stdout, done.stderr)
    schema_registry.validate(body, "agent-spawn/cli/teardown-dry-run.json")
    assert wtctl_calls(fakebin, "status") == [
        ["status", record_id, f"--holder={agent_id}", f"--merged-head={SHA}", "--json"]
    ]
    (entry,) = body["worktrees"]
    assert entry["action"] == "removed" and entry["wtctl_status"]["would_remove"] is True
    assert body["would_refuse"] is None
    assert wtctl_calls(fakebin, "remove") == [] and wtctl_calls(fakebin, "forget") == []
    assert env.agent(agent_id)["status"] == "parked"


def test_dry_run_predicts_held_with_exit_20(env: Env, fakebin: Fakebin) -> None:
    source = launched_in_worktree(env)
    assert env.run("resume", source, "--fork", "--json", "--wait").returncode == 0
    done = td(env, source, "--dry-run")
    body = result_json(done)
    schema_registry.validate(body, "agent-spawn/cli/teardown-dry-run.json")
    assert done.returncode == 20 and body["would_refuse"] == "held"
    assert body["worktrees"][0]["wtctl_status"]["reason"] == "held"
    assert env.agent(source)["status"] == "parked"


def test_dry_run_lists_held_records_without_asking_wtctl(
    env: Env, fakebin: Fakebin, tmp_path: Path
) -> None:
    source, _, _ = nested_source(env, fakebin, tmp_path)
    done = td(env, source, "--dry-run")
    body = result_json(done)
    assert done.returncode == 0, (done.stdout, done.stderr)
    schema_registry.validate(body, "agent-spawn/cli/teardown-dry-run.json")
    assert [(w["action"], w["wtctl_status"]) for w in body["worktrees"]] == [
        ("released", None),
        ("released", None),
    ]
    assert wtctl_calls(fakebin, "status") == []


def test_dry_run_of_a_live_agent_reports_live_without_calling_wtctl(
    env: Env, fakebin: Fakebin
) -> None:
    agent_id = seed(
        env, fakebin, status="live", park_reason=None, wt_record_id="wt-abcdefgh", wt_owned=True
    )
    done = td(env, agent_id, "--dry-run")
    body = result_json(done)
    assert done.returncode == 20 and body["would_refuse"] == "live"
    assert wtctl_calls(fakebin, "status") == []


# --- the ephemeral teardown --------------------------------------------------------------------


def test_an_ephemeral_agent_with_a_worktree_is_torn_down_by_its_supervisor(
    env: Env, fakebin: Fakebin
) -> None:
    done = env.run(*LAUNCH, "--repo", REPO, "--worktree", "new", "--ephemeral")
    assert done.returncode == 0, (done.stdout, done.stderr)
    agent_id = result_json(done)["agent"]["agent_id"]
    row = env.agent(agent_id)
    assert row["status"] == "torn_down"
    (record,) = records(fakebin).values()
    assert record["state"] == "removed"
    (argv,) = wtctl_calls(fakebin, "remove")
    assert "--force" not in argv and not any(a.startswith("--merged-head") for a in argv)
    assert "--actor=auto:agent-spawn" in argv
    post = dict(events(env, agent_id))["post-teardown"]
    assert [w["action"] for w in post["worktrees"]] == ["removed"]


def test_an_ephemeral_teardown_refused_by_wtctl_parks_done_with_teardown_refused(
    env: Env, fakebin: Fakebin
) -> None:
    fakebin.script("wtctl", "remove", **DIRTY)
    done = env.run(*LAUNCH, "--repo", REPO, "--worktree", "new", "--ephemeral")
    assert done.returncode == 0, (done.stdout, done.stderr)
    agent_id = result_json(done)["agent"]["agent_id"]
    row = env.agent(agent_id)
    assert row["status"] == "parked" and row["park_reason"] == "done"
    assert row["teardown_step"] is None
    (record,) = records(fakebin).values()
    assert record["state"] == "active"
    park = [p for n, p in events(env, agent_id) if n == "post-park"]
    assert park and park[-1]["teardown_refused"] is True


# --- windows -----------------------------------------------------------------------------------


def test_a_run_with_a_window_is_still_refused_before_the_worktree_is_touched(
    env: Env, fakebin: Fakebin
) -> None:
    # TODO(#27): the windows step replaces this refusal.
    agent_id = launched_in_worktree(env)
    with env.store() as store:
        from agent_spawn.lifecycle import lease_key

        with store.lease(lease_key(agent_id)) as lease:
            store.put_run(lease, agent_id, 1, {"term_id": "t1", "socket_path": "/tmp/s"})
    done = td(env, agent_id)
    assert done.returncode == 1
    assert wtctl_calls(fakebin, "remove") == []
    assert env.agent(agent_id)["status"] == "parked"


def test_a_record_without_a_wtctl_on_the_machine_is_refused_and_nothing_changes(
    fakebin: Fakebin, isolation_env: Isolation
) -> None:
    env = Env(fakebin, isolation_env, "wtctl_cmd: null\n")
    agent_id = seed(env, fakebin, wt_record_id="wt-abcdefgh", wt_path="/x", wt_owned=False)
    from agent_spawn.lifecycle import lease_key

    with env.store() as store:
        with store.lease(lease_key(agent_id)) as lease:
            store.put_hold(lease, agent_id, "wt-abcdefgh", "/x", owned=False)
    done = td(env, agent_id)
    assert done.returncode == 12, (done.stdout, done.stderr)
    assert result_json(done)["error"]["code"] == "tool_missing"
    assert env.agent(agent_id)["status"] == "parked"
    assert [h["released"] for h in hold_rows(env, agent_id)] == [False]


def test_resume_adopting_a_wtctl_owned_record_never_has_it_removed_at_teardown(
    env: Env, fakebin: Fakebin
) -> None:
    created = json.loads(
        fakebin.run(
            "wtctl", "create", "--repo", REPO, "--holder", "ag-zzzzzzzz", "--name", "own", "--json"
        ).stdout
    )["record"]
    assert created["owned"] is True
    fakebin.script("wtctl", "adopt", times=1, exit=12, error={"code": "busy", "message": "x"})
    failed = env.run(*LAUNCH, "--worktree", created["path"], "--prompt", "P")
    assert failed.returncode == 12, (failed.stdout, failed.stderr)
    (agent,) = env.agents()
    agent_id = str(agent["agent_id"])
    resumed = env.run("resume", agent_id, "--json", "--wait")
    assert resumed.returncode == 0, (resumed.stdout, resumed.stderr)
    assert [h["owned"] for h in hold_rows(env, agent_id)] == [False]
    assert env.agent(agent_id)["wt_owned"] is False
    done = td(env, agent_id)
    assert done.returncode == 0, (done.stdout, done.stderr)
    assert wtctl_calls(fakebin, "remove") == []
    assert [w["action"] for w in result_json(done)["worktrees"]] == ["released"]
    assert records(fakebin)[created["id"]]["state"] == "active"
