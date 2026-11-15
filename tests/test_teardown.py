"""``teardown`` for agents without worktrees: the command against the fakes, the markers and
retries, the file deletion confinement and the supervisor's ephemeral teardown."""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from agent_spawn import procs, teardown
from agent_spawn.config import Config
from agent_spawn.ids import new_agent_id
from agent_spawn.session_backend import SessionBackend
from agent_spawn.store import Store
from tests import schema_registry
from tests.conftest import Isolation
from tests.fakebin.harness import Fakebin
from tests.launch_support import Env, result_json, script_live_run
from tests.session_backend_support import FakebinGateway

SESSION = "cs-abcdefgh"


@pytest.fixture
def env(fakebin: Fakebin, isolation_env: Isolation) -> Env:
    return Env(fakebin, isolation_env)


def seed(env: Env, fakebin: Fakebin, *, session: bool = True, **fields: Any) -> str:
    """A settled agent (parked ``done``) with prompt copies and an outcome file on disk."""
    state = env.state_dir
    with Store.open(state) as store:
        agent_id = new_agent_id(store)
        row: dict[str, Any] = {
            "status": "parked",
            "park_reason": "done",
            "name": "agent test",
            "mode": "headless",
            "claude_session_path": str(fakebin.bin_dir / "claude-session"),
            "state_dir": str(state),
            "run": 1,
            "last_outcome": None,
        }
        if session:
            row["csession_id"] = fakebin.seed_session(SESSION)
        row.update(fields)
        store.insert_agent(agent_id, row)
    for sub, name in (("prompts", "prompt"), ("outcomes", "tok.json")):
        directory = state / sub / agent_id
        directory.mkdir(parents=True, mode=0o700)
        (directory / name).write_text("x")
    return agent_id


def events(env: Env, agent_id: str) -> list[tuple[str, dict[str, Any]]]:
    db = sqlite3.connect(f"file:{env.state_dir / 'state.db'}?mode=ro", uri=True)
    try:
        rows = db.execute(
            "SELECT event, payload FROM events WHERE agent_id = ? ORDER BY event_id", (agent_id,)
        ).fetchall()
    finally:
        db.close()
    return [(name, json.loads(payload)) for name, payload in rows]


def forget_calls(fakebin: Fakebin) -> list[list[str]]:
    return [a for a in fakebin.argvs("claude-session") if a[0] == "forget"]


def td(env: Env, agent_id: str, *flags: str) -> Any:
    return env.run("teardown", agent_id, "--json", *flags)


# --- the happy path ---------------------------------------------------------------------------


def test_teardown_tombstones_forgets_the_session_and_deletes_the_files(
    env: Env, fakebin: Fakebin
) -> None:
    agent_id = seed(env, fakebin)
    outside = env.state_dir / "keep"
    outside.mkdir()
    (outside / "secret").write_text("keep me")
    os.symlink(outside, env.state_dir / "prompts" / agent_id / "link")
    done = td(env, agent_id, "--merged-head", "a" * 40)
    body = result_json(done)
    assert done.returncode == 0, (done.stdout, done.stderr)
    schema_registry.validate(body, "agent-spawn/cli/teardown.json")
    assert body["agent"]["status"] == "torn_down" and body["agent"]["torn_down_at"]
    assert body["worktrees"] == [] and body["windows_closed"] == []
    assert body["session_forgotten"] is True
    assert [a[1] for a in forget_calls(fakebin)] == [SESSION]
    assert not (env.state_dir / "prompts" / agent_id).exists()
    assert not (env.state_dir / "outcomes" / agent_id).exists()
    assert (outside / "secret").read_text() == "keep me"  # the symlink was not followed
    row = env.agent(agent_id)
    assert row["status"] == "torn_down" and row["teardown_step"] is None
    assert row["needs_attention"] is None
    names = [n for n, _ in events(env, agent_id)]
    assert names == ["post-state-change", "post-state-change", "post-teardown"]
    post = dict(events(env, agent_id))["post-teardown"]
    assert post["merged_head"] == "a" * 40 and post["force"] is False and post["worktrees"] == []
    schema_registry.validate(
        {
            **post,
            "tool": "agent-spawn",
            "hook": "post-teardown",
            "object_type": "agent",
            "object_id": agent_id,
            "event_id": 1,
            "ts": "2026-11-14T00:00:00.000Z",
        },
        "agent-spawn/hooks/post-teardown.json",
    )


def test_teardown_without_a_session_skips_claude_session(env: Env, fakebin: Fakebin) -> None:
    agent_id = seed(env, fakebin, session=False)
    done = td(env, agent_id)
    body = result_json(done)
    assert done.returncode == 0 and body["session_forgotten"] is False
    assert forget_calls(fakebin) == []


def test_a_second_teardown_of_a_tombstone_exits_10_torn_down(env: Env, fakebin: Fakebin) -> None:
    agent_id = seed(env, fakebin)
    assert td(env, agent_id).returncode == 0
    done = td(env, agent_id)
    body = result_json(done)
    schema_registry.validate(body, "agent-spawn/cli/error.json")
    assert done.returncode == 10 and body["error"]["details"] == {"status": "torn_down"}
    assert td(env, agent_id[:7]).returncode == 10  # a prefix never matches a tombstone


def test_forget_exit_10_counts_as_done(env: Env, fakebin: Fakebin) -> None:
    agent_id = seed(env, fakebin)
    fakebin.script("claude-session", "forget", exit=10, error={"code": "not_found", "message": "x"})
    done = td(env, agent_id)
    body = result_json(done)
    assert done.returncode == 0 and body["session_forgotten"] is False
    assert env.agent(agent_id)["status"] == "torn_down"


def test_force_is_audited(env: Env, fakebin: Fakebin) -> None:
    agent_id = seed(env, fakebin)
    done = td(env, agent_id, "--force", "--actor", "human:tester")
    assert done.returncode == 0, done.stderr
    audit = [json.loads(line) for line in (env.state_dir / "audit.jsonl").read_text().splitlines()]
    assert [(a["action"], a["inputs"]["force"], a["result"]) for a in audit] == [
        ("teardown", True, "ok")
    ]
    assert dict(events(env, agent_id))["post-teardown"]["force"] is True


def test_a_post_teardown_hook_is_delivered(env: Env, fakebin: Fakebin, tmp_path: Path) -> None:
    agent_id = seed(env, fakebin)
    marker = tmp_path / "hook-out"
    done = td(env, agent_id, "--hook", f"post-teardown=cat > {marker}")
    assert done.returncode == 0, done.stderr
    payload = json.loads(marker.read_text())
    assert payload["hook"] == "post-teardown" and payload["object_id"] == agent_id


# --- usage and refusals ------------------------------------------------------------------------


@pytest.mark.parametrize(
    "flags",
    [("--hook", "pre-launch=true"), ("--hook", "nonsense"), ("--merged-head", "ABC"), ()],
)
def test_usage_errors_exit_2_and_change_nothing(
    env: Env, fakebin: Fakebin, flags: tuple[str, ...]
) -> None:
    agent_id = seed(env, fakebin)
    done = env.run("teardown", agent_id if flags else "ag", "--json", *flags)
    assert done.returncode == 2
    assert env.agent(agent_id)["status"] == "parked"
    fakebin.assert_not_called("claude-session")


@pytest.mark.parametrize("status", ["live", "launching"])
def test_a_live_or_launching_agent_is_refused_even_with_force(
    env: Env, fakebin: Fakebin, status: str
) -> None:
    agent_id = seed(env, fakebin, status=status, park_reason=None)
    done = td(env, agent_id, "--force", "--actor", "human:tester")
    body = result_json(done)
    schema_registry.validate(body, "agent-spawn/cli/error.json")
    assert done.returncode == 20 and body["error"]["reason"] == "live"
    assert env.agent(agent_id)["status"] == status
    assert (env.state_dir / "prompts" / agent_id).exists()
    fakebin.assert_not_called("claude-session")


def test_a_pending_or_live_run_of_the_session_is_refused(env: Env, fakebin: Fakebin) -> None:
    agent_id = seed(env, fakebin)
    script_live_run(fakebin)
    done = td(env, agent_id, "--force", "--actor", "human:tester")
    body = result_json(done)
    assert done.returncode == 20 and body["error"]["reason"] == "live"
    row = env.agent(agent_id)
    assert row["status"] == "parked" and row["teardown_step"] is None
    assert forget_calls(fakebin) == []


def test_a_session_reported_missing_is_skipped(env: Env, fakebin: Fakebin) -> None:
    agent_id = seed(env, fakebin)
    fakebin.script("claude-session", "status", exit=10, error={"code": "not_found", "message": "x"})
    done = td(env, agent_id)
    body = result_json(done)
    assert done.returncode == 0 and body["session_forgotten"] is False
    assert forget_calls(fakebin) == []


def test_agents_with_worktrees_are_refused_before_any_change(env: Env, fakebin: Fakebin) -> None:
    # TODO(#23): worktree teardown replaces this refusal.
    agent_id = seed(env, fakebin, wt_record_id="wt-abcdefgh", wt_path="/x", wt_owned=True)
    done = td(env, agent_id)
    assert done.returncode == 1
    assert env.agent(agent_id)["status"] == "parked"
    assert (env.state_dir / "prompts" / agent_id).exists()


def test_a_configured_pre_teardown_hook_is_refused_not_skipped(env: Env, fakebin: Fakebin) -> None:
    # TODO(#32): the hook runs instead of refusing.
    agent_id = seed(env, fakebin)
    done = td(env, agent_id, "--hook", "pre-teardown=true")
    assert done.returncode == 1
    assert env.agent(agent_id)["status"] == "parked"


# --- dry run ----------------------------------------------------------------------------------


def test_dry_run_changes_nothing(env: Env, fakebin: Fakebin) -> None:
    agent_id = seed(env, fakebin)
    before = env.agent(agent_id)
    done = td(env, agent_id, "--dry-run")
    body = result_json(done)
    schema_registry.validate(body, "agent-spawn/cli/teardown-dry-run.json")
    assert done.returncode == 0 and body["would_refuse"] is None
    assert env.agent(agent_id) == before
    assert events(env, agent_id) == []
    assert forget_calls(fakebin) == []
    assert (env.state_dir / "prompts" / agent_id).exists()


def test_dry_run_reports_live_with_exit_20(env: Env, fakebin: Fakebin) -> None:
    agent_id = seed(env, fakebin, status="live", park_reason=None)
    done = td(env, agent_id, "--dry-run")
    body = result_json(done)
    schema_registry.validate(body, "agent-spawn/cli/teardown-dry-run.json")
    assert done.returncode == 20 and body["would_refuse"] == "live"
    assert env.agent(agent_id)["status"] == "live"


# --- markers and retries ------------------------------------------------------------------------


def me() -> procs.ProcessIdentity:
    identity = procs.read_identity(os.getpid())
    assert identity is not None
    return identity


def dead() -> procs.ProcessIdentity:
    """A process identity that is not alive: this boot, a pid in use, another start time."""
    mine = me()
    return procs.ProcessIdentity(mine.pid, mine.start_time + 1, mine.boot_id)


def test_a_forget_failure_keeps_tearing_down_and_a_retry_continues(
    env: Env, fakebin: Fakebin
) -> None:
    agent_id = seed(env, fakebin)
    fakebin.script(
        "claude-session",
        "forget",
        times=1,
        exit=20,
        error={"code": "refused", "reason": "live", "message": "x"},
    )
    done = td(env, agent_id)
    body = result_json(done)
    schema_registry.validate(body, "agent-spawn/cli/error.json")
    assert done.returncode == 20 and body["error"]["reason"] == "live"
    assert body["error"]["details"]["hint"] == f"agent-spawn stop {agent_id}"
    row = env.agent(agent_id)
    assert row["status"] == "tearing_down" and row["teardown_step"] == "hooks_done"
    assert (env.state_dir / "prompts" / agent_id).exists()  # the tombstone step did not run
    # The teardown process of that run exited: the retry enters over it and continues.
    again = td(env, agent_id)
    assert again.returncode == 0, (again.stdout, again.stderr)
    assert env.agent(agent_id)["status"] == "torn_down"
    assert len(forget_calls(fakebin)) == 2
    log = (env.state_dir / "log.jsonl").read_text()
    assert '"teardown_retry"' in log


def test_a_live_teardown_process_refuses_with_tearing_down(env: Env, fakebin: Fakebin) -> None:
    agent_id = seed(
        env, fakebin, status="tearing_down", teardown_step="started", teardown_proc=me()
    )
    done = td(env, agent_id)
    body = result_json(done)
    assert done.returncode == 20 and body["error"]["reason"] == "tearing_down"
    assert env.agent(agent_id)["teardown_proc"] == me()
    dry = result_json(td(env, agent_id, "--dry-run"))
    assert dry["would_refuse"] == "tearing_down"
    fakebin.assert_not_called("claude-session")


def test_a_retry_after_session_done_does_not_forget_again(env: Env, fakebin: Fakebin) -> None:
    agent_id = seed(
        env, fakebin, status="tearing_down", teardown_step="session_done", teardown_proc=dead()
    )
    done = td(env, agent_id)
    assert done.returncode == 0, (done.stdout, done.stderr)
    assert forget_calls(fakebin) == []
    assert env.agent(agent_id)["status"] == "torn_down"


def test_a_retry_of_a_tearing_down_agent_with_a_live_run_is_refused_unchanged(
    env: Env, fakebin: Fakebin
) -> None:
    agent_id = seed(
        env, fakebin, status="tearing_down", teardown_step="hooks_done", teardown_proc=dead()
    )
    script_live_run(fakebin)
    done = td(env, agent_id)
    assert done.returncode == 20
    row = env.agent(agent_id)
    assert row["status"] == "tearing_down" and row["teardown_proc"] == dead()


def test_two_concurrent_entries_one_wins(env: Env, fakebin: Fakebin) -> None:
    agent_id = seed(env, fakebin)
    gateway = FakebinGateway(fakebin)
    backend = SessionBackend([str(fakebin.bin_dir / "claude-session")], gateway=gateway)
    with Store.open(env.state_dir) as store:
        first = teardown.Teardown(
            store, cli_config(env), state_dir=str(env.state_dir), actor="human:t", backend=backend
        )
        row, retry = first.enter(agent_id)
        assert not retry and row["status"] == "tearing_down"
        second = teardown.Teardown(
            store, cli_config(env), state_dir=str(env.state_dir), actor="human:t", backend=backend
        )
        # the recorded teardown process (this one) is alive: refused, nothing changed
        with pytest.raises(teardown.CliError) as raised:
            second.enter(agent_id)
        assert raised.value.reason == "tearing_down"


def cli_config(env: Env) -> Any:
    return Config()


# --- deleting files ---------------------------------------------------------------------------


def test_removal_never_follows_symlinks(tmp_path: Path) -> None:
    state = tmp_path / "state"
    victim = tmp_path / "victim"
    victim.mkdir()
    (victim / "file").write_text("keep")
    agent_id = "ag-abcdefgh"
    (state / "prompts").mkdir(parents=True)
    os.symlink(victim, state / "prompts" / agent_id)  # the agent's dir is a symlink
    (state / "outcomes" / agent_id / "nested").mkdir(parents=True)
    os.symlink(victim, state / "outcomes" / agent_id / "nested" / "l")
    assert teardown.remove_agent_files(str(state), agent_id) == []
    assert (victim / "file").read_text() == "keep"
    assert not os.path.lexists(state / "prompts" / agent_id)
    assert not os.path.lexists(state / "outcomes" / agent_id)


def test_removal_does_not_traverse_a_symlinked_parent(tmp_path: Path) -> None:
    state = tmp_path / "state"
    state.mkdir()
    elsewhere = tmp_path / "elsewhere"
    (elsewhere / "ag-abcdefgh").mkdir(parents=True)
    (elsewhere / "ag-abcdefgh" / "file").write_text("keep")
    os.symlink(elsewhere, state / "prompts")
    problems = teardown.remove_agent_files(str(state), "ag-abcdefgh")
    assert problems and (elsewhere / "ag-abcdefgh" / "file").exists()


@pytest.mark.parametrize("bad", ["../x", "ag-abcdefgh/../..", "", "ag-ABCDEFGH", "/etc"])
def test_removal_refuses_a_non_id(tmp_path: Path, bad: str) -> None:
    with pytest.raises(ValueError):
        teardown.remove_agent_files(str(tmp_path), bad)


# --- the supervisor's ephemeral teardown --------------------------------------------------------


def test_ephemeral_agent_is_torn_down_by_its_supervisor(env: Env, fakebin: Fakebin) -> None:
    done = env.run(
        "launch", "--mode", "headless", "--skill", "do-it", "--json", "--ephemeral", "--wait"
    )
    body = result_json(done)
    assert done.returncode == 0, (done.stdout, done.stderr)
    agent_id = body["agent"]["agent_id"]
    row = env.agent(agent_id)
    assert row["status"] == "torn_down" and row["teardown_step"] is None
    assert [a[1] for a in forget_calls(fakebin)] == [row["csession_id"]]
    assert not (env.state_dir / "outcomes" / agent_id).exists()
    names = [n for n, _ in events(env, agent_id)]
    assert names.count("post-teardown") == 1 and "post-park" not in names


def test_ephemeral_forget_failure_leaves_tearing_down_for_a_retry(
    env: Env, fakebin: Fakebin
) -> None:
    fakebin.script(
        "claude-session",
        "forget",
        times=1,
        exit=20,
        error={"code": "refused", "reason": "live", "message": "x"},
    )
    done = env.run(
        "launch", "--mode", "headless", "--skill", "do-it", "--json", "--ephemeral", "--wait"
    )
    agent_id = result_json(done)["agent"]["agent_id"]
    row = env.agent(agent_id)
    assert row["status"] == "tearing_down" and row["teardown_step"] == "hooks_done"
    names = [n for n, _ in events(env, agent_id)]
    assert "post-park" not in names and "post-teardown" not in names
    assert td(env, agent_id).returncode == 0  # the supervisor is dead: a retry finishes it
    assert env.agent(agent_id)["status"] == "torn_down"


def test_a_concurrent_teardown_during_an_ephemeral_teardown_is_refused(
    env: Env, fakebin: Fakebin
) -> None:
    # Slow forget: while the supervisor tears down, a second teardown gets 20 tearing_down.
    fakebin.script("claude-session", "forget", times=1, sleep=3)
    import subprocess

    proc = subprocess.Popen(
        env.argv(
            "launch", "--mode", "headless", "--skill", "do-it", "--json", "--ephemeral", "--wait"
        ),
        env=env.env(),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        env.wait_for(
            lambda: any(a["status"] == "tearing_down" for a in env.agents()), "tearing_down"
        )
        (row,) = env.agents()
        other = td(env, row["agent_id"])
        body = result_json(other)
        assert other.returncode == 20 and body["error"]["reason"] == "tearing_down"
        proc.wait(timeout=60)
    finally:
        if proc.poll() is None:
            proc.wait(timeout=60)
    assert env.agent(row["agent_id"])["status"] == "torn_down"


def test_ephemeral_live_run_is_refused_with_a_reason_and_nothing_changes(
    env: Env, fakebin: Fakebin
) -> None:
    me_id = me()
    agent_id = seed(
        env, fakebin, status="tearing_down", teardown_step="started", teardown_proc=me_id
    )
    script_live_run(fakebin)
    with Store.open(env.state_dir) as store:
        reason = teardown.ephemeral(
            store,
            agent_id,
            config=cli_config(env),
            state_dir=str(env.state_dir),
            backend=SessionBackend(
                [str(fakebin.bin_dir / "claude-session")], gateway=FakebinGateway(fakebin)
            ),
            log=lambda *a, **k: None,
            gateway=FakebinGateway(fakebin),
        )
    assert reason == "live"
    row = env.agent(agent_id)
    assert row["status"] == "tearing_down" and row["teardown_step"] == "started"
    assert (env.state_dir / "prompts" / agent_id).exists()
    assert forget_calls(fakebin) == []
