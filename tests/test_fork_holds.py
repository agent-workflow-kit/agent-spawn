"""Forks and their holds (interface.md, Resume order and Records of a cwd): the fork's own id
holds the source's records, every failure path releases exactly the holds this fork took, a fork
into a throwaway runs holding nothing, and a record the fork holds cannot be removed under the
original (wtctl's `held` refusal, against the fake)."""

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
    return list(records(fakebin)[record_id]["holders"])


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


def test_the_original_cannot_remove_a_record_its_fork_holds(env: Env, fakebin: Fakebin) -> None:
    """Acceptance 3 at the wtctl level: wtctl refuses `remove` with `held` while the fork
    (the only holder besides the original) holds the record; the teardown of the original
    (#59/#23) names its own holder and relies on this refusal."""
    source = launched_in_worktree(env)
    done = env.run("resume", source, "--fork", "--json", "--wait")
    assert done.returncode == 0, (done.stdout, done.stderr)
    fork = result_json(done)["agent"]["agent_id"]
    record_id = env.agent(source)["wt_record_id"]
    with env.store() as store:
        assert [h["record_id"] for h in store.get_holds(fork)] == [record_id]  # its own row
    refused = fakebin.run("wtctl", "remove", record_id, "--holder", source, "--yes", "--json")
    assert refused.returncode == 20, (refused.stdout, refused.stderr)
    error = json.loads(refused.stdout)["error"]
    assert error["reason"] == "held" and fork in json.dumps(error)
    assert record_id in records(fakebin) and holders(fakebin, record_id) == [source, fork]
    # The fork's hold goes (its teardown forgets it): the original may now remove the record.
    assert fakebin.run("wtctl", "forget", record_id, "--holder", fork, "--json").returncode == 0
    allowed = fakebin.run("wtctl", "remove", record_id, "--holder", source, "--yes", "--json")
    assert allowed.returncode == 0, (allowed.stdout, allowed.stderr)


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
    assert holders(fakebin, record_id) == [source, fork["agent_id"]]  # the session exists
    assert wtctl_calls(fakebin, "forget") == []


# --- partial holds -----------------------------------------------------------------------------


def nested_source(env: Env, fakebin: Fakebin, tmp_path: Path) -> tuple[str, str, str]:
    """A source whose cwd lies in two nested records: returns (source, outer id, inner id)."""
    outer = tree(tmp_path)
    inner = outer / "pkg"
    outer_id = fake_adopted(fakebin, outer)["id"]
    inner_id = fake_adopted(fakebin, inner)["id"]
    source = launched(env, "--cwd", str(inner))
    assert holders(fakebin, outer_id)[-1] == source and holders(fakebin, inner_id)[-1] == source
    return source, outer_id, inner_id


def test_a_second_adopt_that_fails_releases_the_first_and_removes_the_fork(
    env: Env, fakebin: Fakebin, tmp_path: Path
) -> None:
    source, outer_id, inner_id = nested_source(env, fakebin, tmp_path)
    before = {rid: holders(fakebin, rid) for rid in (outer_id, inner_id)}
    first = len(wtctl_calls(fakebin, "adopt"))
    fakebin.script("wtctl", "adopt", times=1)  # the first hold succeeds
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
    source, outer_id, inner_id = nested_source(env, fakebin, tmp_path)
    before = {rid: holders(fakebin, rid) for rid in (outer_id, inner_id)}
    first = len(wtctl_calls(fakebin, "adopt"))
    fakebin.script("wtctl", "adopt", times=1)
    # The signal arrives while the second adopt runs: it is held until that hold reaches the row
    # (an interrupt never loses a hold wtctl took), then the fork undoes both.
    fakebin.script("wtctl", "adopt", times=1, sleep=2)
    forker = proc_tracker.popen(
        env.argv("resume", source, "--fork", "--json"),
        env=env.env(),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    env.wait_for(lambda: len(wtctl_calls(fakebin, "adopt")) == first + 2, "the second adopt")
    forker.send_signal(signal.SIGTERM)
    forker.communicate(timeout=30)
    assert forker.returncode == 143
    assert forks_of(env, source) == []
    for record_id, expected in before.items():
        assert holders(fakebin, record_id) == expected
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
