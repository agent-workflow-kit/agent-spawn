"""Forks and their holds (interface.md, Resume order and Records of a cwd): the fork's own id
holds the source's records, every failure path releases exactly the holds this fork took, a fork
into a throwaway runs holding nothing, and an in-place fork holds the source's record under its
own id (what makes wtctl refuse the original's `remove` with `held`; checked end to end in #23)."""

from __future__ import annotations

import json
import signal
import subprocess
from pathlib import Path
from typing import Any

import pytest

from tests.conftest import Isolation
from tests.fakebin.harness import Fakebin
from tests.launch_support import Env, result_json
from tests.procguard import ProcTracker
from tests.test_resume import launched, starts
from tests.test_worktrees_launch import fake_adopted, fake_throwaway, tree, wtctl_calls
from tests.test_worktrees_resume import BUSY, launched_in_worktree, records

REFUSE_LIVE = {"error": {"code": "refused", "message": "no", "reason": "live"}, "exit": 20}


@pytest.fixture
def env(fakebin: Fakebin, isolation_env: Isolation) -> Env:
    return Env(fakebin, isolation_env)


def third_party_dry_run(fakebin: Fakebin, cwd: str) -> None:
    fakebin.script(
        "claude-session",
        "start",
        argv_contains=["--dry-run"],
        merge={"profile": {"reads_third_party": True}, "cwd": cwd},
    )


def holders(fakebin: Fakebin, record_id: str) -> list[str]:
    """The record's holders as a sorted list (wtctl's `holders` is sorted; never rely on order)."""
    return sorted(records(fakebin)[record_id]["holders"])


def real_start_calls(fakebin: Fakebin) -> list[Any]:
    return [
        i
        for i in fakebin.invocations("claude-session")
        if i.argv[0] == "start" and "--dry-run" not in i.argv
    ]


def forks_of(env: Env, source: str) -> list[dict[str, Any]]:
    return [a for a in env.agents() if a["agent_id"] != source]


# --- the cwd rule for a fork with --cwd -------------------------------------------------------


def test_a_third_party_fork_cwd_at_a_throwaway_root_runs_and_holds_nothing(
    env: Env, fakebin: Fakebin
) -> None:
    throwaway = fake_throwaway(fakebin)
    source = launched(env)
    third_party_dry_run(fakebin, throwaway["path"])
    done = env.run("resume", source, "--fork", "--cwd", throwaway["path"], "--json", "--wait")
    assert done.returncode == 0, (done.stdout, done.stderr)
    assert result_json(done)["agent"]["holds"] == []
    assert wtctl_calls(fakebin, "adopt") == []
    assert f"--cwd={throwaway['path']}" in starts(fakebin)[-1]


def test_a_third_party_fork_cwd_in_a_throwaway_subdirectory_is_refused_with_no_trace(
    env: Env, fakebin: Fakebin
) -> None:
    throwaway = fake_throwaway(fakebin)
    source = launched(env)
    before = env.agent(source)
    third_party_dry_run(fakebin, throwaway["path"] + "/pkg")
    done = env.run("resume", source, "--fork", "--cwd", throwaway["path"] + "/pkg", "--json")
    assert done.returncode == 20, (done.stdout, done.stderr)
    error = json.loads(done.stdout)["error"]
    assert (error["reason"], error["tool"]) == ("third_party_cwd", "agent-spawn")
    assert forks_of(env, source) == []
    assert env.agent(source) == before
    assert wtctl_calls(fakebin, "adopt") == []
    assert len(starts(fakebin)) == 1


def test_a_third_party_fork_cwd_in_a_worktree_is_refused_before_any_hold(
    env: Env, fakebin: Fakebin, tmp_path: Path
) -> None:
    root = tree(tmp_path)
    fake_adopted(fakebin, root)
    adopts = len(wtctl_calls(fakebin, "adopt"))
    source = launched(env)
    third_party_dry_run(fakebin, str(root))
    done = env.run("resume", source, "--fork", "--cwd", str(root), "--json")
    assert done.returncode == 20, (done.stdout, done.stderr)
    assert json.loads(done.stdout)["error"]["reason"] == "third_party_cwd"
    assert forks_of(env, source) == []
    assert len(wtctl_calls(fakebin, "adopt")) == adopts


# --- in place: the fork holds the source's record ----------------------------------------------


def test_an_in_place_fork_holds_the_sources_record_under_its_own_id(
    env: Env, fakebin: Fakebin
) -> None:
    """The precondition of acceptance 3: the fork holds the source's record under its own id,
    distinct from the source's, and the store's `agent_holds` row and wtctl's holder list agree.
    The end-to-end check (`agent-spawn teardown <source>` -> 20 `held`, the source still parked)
    belongs to #23."""
    source = launched_in_worktree(env)
    done = env.run("resume", source, "--fork", "--json", "--wait")
    assert done.returncode == 0, (done.stdout, done.stderr)
    fork = result_json(done)["agent"]["agent_id"]
    record_id = env.agent(source)["wt_record_id"]
    with env.store() as store:
        assert [h["record_id"] for h in store.get_holds(fork)] == [record_id]  # its own row
        assert [h["record_id"] for h in store.get_holds(source)] == [record_id]
    assert holders(fakebin, record_id) == sorted([source, fork])


def test_the_fork_row_and_hold_exist_and_are_adopted_before_start_runs(
    env: Env, fakebin: Fakebin, proc_tracker: ProcTracker
) -> None:
    source = launched_in_worktree(env)
    record_id = env.agent(source)["wt_record_id"]
    fakebin.script("claude-session", "start", argv_contains=["--dry-run"])
    fakebin.script("claude-session", "start", hang=True, on_signal={"SIGTERM": {"exit": 143}})
    launches = len(real_start_calls(fakebin))  # the source's own start
    forker = proc_tracker.popen(
        env.argv("resume", source, "--fork", "--json"),
        env=env.env(),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        env.wait_for(lambda: len(real_start_calls(fakebin)) > launches, "the fork's start")
        start = real_start_calls(fakebin)[-1]
        fakebin.wait_for_hang(start.pid)
        (fork,) = forks_of(env, source)  # while `start` runs
        assert fork["status"] == "launching"
        with env.store() as store:
            assert [h["record_id"] for h in store.get_holds(fork["agent_id"])] == [record_id]
        order = [(i.tool, i.argv[0], i) for i in fakebin.invocations()]
        adopt_at = next(
            n
            for n, (tool, command, i) in enumerate(order)
            if tool == "wtctl" and command == "adopt" and f"--holder={fork['agent_id']}" in i.argv
        )
        start_at = next(n for n, (_, _, i) in enumerate(order) if i.pid == start.pid)
        assert adopt_at < start_at
    finally:
        forker.send_signal(signal.SIGTERM)
        forker.communicate(timeout=30)


def test_a_fork_whose_start_exit_12_stays_parked_and_keeps_its_holds(
    env: Env, fakebin: Fakebin
) -> None:
    source = launched_in_worktree(env)
    record_id = env.agent(source)["wt_record_id"]
    fakebin.script("claude-session", "start", argv_contains=["--dry-run"])
    fakebin.script("claude-session", "start", exit=12,
                   error={"code": "launch_failed", "message": "x", "csession_id": "cs-bbbbbbbb",
                          "run": 1, "end_reason": "launch_failed"})  # fmt: skip
    done = env.run("resume", source, "--fork", "--json")
    assert done.returncode == 12, (done.stdout, done.stderr)
    (fork,) = forks_of(env, source)
    assert (fork["status"], fork["csession_id"]) == ("parked", "cs-bbbbbbbb")
    with env.store() as store:
        assert [h["record_id"] for h in store.get_holds(fork["agent_id"])] == [record_id]
    assert holders(fakebin, record_id) == sorted([source, fork["agent_id"]])  # the session exists
    assert wtctl_calls(fakebin, "forget") == []


# --- partial holds -----------------------------------------------------------------------------


def nested_source(env: Env, fakebin: Fakebin, tmp_path: Path) -> tuple[str, str, str]:
    """A source whose cwd lies in two nested records: returns (source, outer id, inner id)."""
    outer = tree(tmp_path)
    inner = outer / "pkg"
    outer_id = fake_adopted(fakebin, outer)["id"]
    inner_id = fake_adopted(fakebin, inner)["id"]
    source = launched(env, "--cwd", str(inner))
    assert source in holders(fakebin, outer_id) and source in holders(fakebin, inner_id)
    return source, outer_id, inner_id


def test_a_second_adopt_that_fails_releases_the_first_and_removes_the_fork(
    env: Env, fakebin: Fakebin, tmp_path: Path
) -> None:
    source, outer_id, inner_id = nested_source(env, fakebin, tmp_path)
    before = {rid: holders(fakebin, rid) for rid in (outer_id, inner_id)}
    first = len(wtctl_calls(fakebin, "adopt"))
    # An empty rule answers the canned adopt and uses up one call: the first hold succeeds.
    fakebin.script("wtctl", "adopt", times=1)
    fakebin.script("wtctl", "adopt", times=1, **BUSY)  # the second is busy
    done = env.run("resume", source, "--fork", "--json")
    assert done.returncode == 12, (done.stdout, done.stderr)
    assert len(wtctl_calls(fakebin, "adopt")) == first + 2
    assert forks_of(env, source) == []
    assert json.loads(done.stdout)["error"]["agent_id"] is None
    for record_id, expected in before.items():  # nobody else's hold was touched
        assert holders(fakebin, record_id) == expected
    forgets = wtctl_calls(fakebin, "forget")
    assert len(forgets) == 1 and forgets[0][1] in (outer_id, inner_id)
    assert not forgets[0][2].endswith(source)
    with env.store() as store:
        rows = store._conn.execute("SELECT agent_id FROM agent_holds").fetchall()
        assert {r["agent_id"] for r in rows} == {source}


def test_a_signal_during_the_second_adopt_releases_the_first_hold(
    env: Env, fakebin: Fakebin, tmp_path: Path, proc_tracker: ProcTracker
) -> None:
    """The fake adopt takes the hold (the holder is listed) and only then sleeps, so the signal
    lands while a hold exists that the fork has not recorded yet. The signal is held until the
    hold reaches the row: the second adopt runs to its end (exit 0) and the fork then undoes both
    holds by their rows. Without that guard the adopt is cut short and only the cwd lookup of the
    release finds the hold; the completed adopt and the sorted forgets below pin the guard."""
    source, outer_id, inner_id = nested_source(env, fakebin, tmp_path)
    before = {rid: holders(fakebin, rid) for rid in (outer_id, inner_id)}
    first = len(wtctl_calls(fakebin, "adopt"))
    # An empty rule answers the canned adopt and uses up one call: the first hold succeeds.
    fakebin.script("wtctl", "adopt", times=1)
    # The sleep (longer than any stall of this test) comes after the fake took the hold.
    fakebin.script("wtctl", "adopt", times=1, sleep=10)
    forker = proc_tracker.popen(
        env.argv("resume", source, "--fork", "--json"),
        env=env.env(),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    env.wait_for(lambda: len(wtctl_calls(fakebin, "adopt")) == first + 2, "the second adopt")
    forker.send_signal(signal.SIGTERM)
    forker.communicate(timeout=60)
    assert forker.returncode == 143
    assert forks_of(env, source) == []
    for record_id, expected in before.items():
        assert holders(fakebin, record_id) == expected
    adopts = [i for i in fakebin.invocations("wtctl") if i.argv[0] == "adopt"]
    assert [i.exit_code for i in adopts[first:]] == [0, 0]  # the held signal let adopt finish
    forgets = wtctl_calls(fakebin, "forget")
    assert sorted(c[1] for c in forgets) == sorted([outer_id, inner_id])
    assert all(c[2] != f"--holder={source}" for c in forgets)
    with env.store() as store:
        rows = store._conn.execute("SELECT agent_id FROM agent_holds").fetchall()
        assert {r["agent_id"] for r in rows} == {source}
    assert len(starts(fakebin)) == 1


def test_a_fork_start_that_fails_without_a_session_releases_its_holds(
    env: Env, fakebin: Fakebin
) -> None:
    source = launched_in_worktree(env)
    record_id = env.agent(source)["wt_record_id"]
    fakebin.script("claude-session", "start", argv_contains=["--dry-run"])
    fakebin.script("claude-session", "start", exit=12,
                   error={"code": "launch_failed", "message": "x"})  # fmt: skip
    fakebin.script("claude-session", "list", stdout={"sessions": []})
    done = env.run("resume", source, "--fork", "--json")
    assert done.returncode == 12, (done.stdout, done.stderr)
    assert forks_of(env, source) == []
    assert holders(fakebin, record_id) == [source]
    forgets = wtctl_calls(fakebin, "forget")
    assert len(forgets) == 1 and forgets[0][2].startswith("--holder=ag-")
    assert forgets[0][2] != f"--holder={source}"
