"""``teardown`` for agents without worktrees: the command against the fakes, the markers and
retries, the file deletion confinement and the supervisor's ephemeral teardown."""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from agent_spawn import lifecycle, procs, teardown
from agent_spawn.agent_view import agent_object
from agent_spawn.config import Config
from agent_spawn.hooks import agent_event
from agent_spawn.ids import new_agent_id
from agent_spawn.lifecycle import LEASE_WAIT, lease_key
from agent_spawn.session_backend import SessionBackend
from agent_spawn.store import CasLost, Store
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
    agent_id = seed(env, fakebin, needs_attention="failed")
    assert env.agent(agent_id)["needs_attention"] == "failed"
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
    assert row["teardown_proc"] is None  # null when not tearing down
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
    for record in audit:
        schema_registry.validate(record, "agent-spawn/audit.json")


def test_a_teardown_without_force_writes_no_audit_record(env: Env, fakebin: Fakebin) -> None:
    agent_id = seed(env, fakebin)
    assert td(env, agent_id).returncode == 0
    audit = env.state_dir / "audit.jsonl"
    assert not audit.exists() or audit.read_text() == ""


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
    body = result_json(done)
    assert done.returncode == 1
    assert body["error"]["code"] == "internal" and "not implemented" in body["error"]["message"]
    assert env.agent(agent_id)["status"] == "parked"
    assert (env.state_dir / "prompts" / agent_id).exists()


def test_a_configured_pre_teardown_hook_is_refused_not_skipped(env: Env, fakebin: Fakebin) -> None:
    # TODO(#32): the hook runs instead of refusing.
    agent_id = seed(env, fakebin)
    done = td(env, agent_id, "--hook", "pre-teardown=true")
    body = result_json(done)
    assert done.returncode == 1
    assert body["error"]["code"] == "internal" and "not implemented" in body["error"]["message"]
    assert env.agent(agent_id)["status"] == "parked"


def test_a_config_defined_pre_teardown_hook_is_refused_not_skipped(
    fakebin: Fakebin, isolation_env: Isolation
) -> None:
    config = 'hooks:\n  pre-teardown:\n    - cmd: "true"\n'
    cfg_env = Env(fakebin, isolation_env, config)
    agent_id = seed(cfg_env, fakebin)
    done = td(cfg_env, agent_id)
    body = result_json(done)
    assert done.returncode == 1
    assert body["error"]["code"] == "internal" and "not implemented" in body["error"]["message"]
    assert cfg_env.agent(agent_id)["status"] == "parked"
    assert (cfg_env.state_dir / "prompts" / agent_id).exists()


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
    assert body["error"]["tool"] == "claude-session"
    assert body["error"]["lower_error"]["code"] == "refused"
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


def test_an_entry_over_a_live_teardown_process_is_refused_unchanged(
    env: Env, fakebin: Fakebin
) -> None:
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


# --- review findings ----------------------------------------------------------------------------


def pass_status_once(fakebin: Fakebin) -> None:
    """The next ``status`` call answers as usual; the ones after it report a live run."""
    fakebin.script("claude-session", "status", times=1)
    script_live_run(fakebin)


def fail_forget_once(fakebin: Fakebin) -> None:
    fakebin.script(
        "claude-session",
        "forget",
        times=1,
        exit=20,
        error={"code": "refused", "reason": "live", "message": "x"},
    )


def log_text(env: Env) -> str:
    path = env.state_dir / "log.jsonl"
    return path.read_text() if path.exists() else ""


def test_a_retry_without_hook_flags_keeps_the_stored_post_teardown_hook(
    env: Env, fakebin: Fakebin, tmp_path: Path
) -> None:
    agent_id = seed(env, fakebin)
    marker = tmp_path / "hook-out"
    fail_forget_once(fakebin)
    first = td(env, agent_id, "--hook", f"post-teardown=cat > {marker}")
    assert first.returncode == 20
    assert env.agent(agent_id)["hooks"] == [{"hook": "post-teardown", "cmd": f"cat > {marker}"}]
    again = td(env, agent_id)  # no --hook: the stored one must survive the retry's entry
    assert again.returncode == 0, (again.stdout, again.stderr)
    assert json.loads(marker.read_text())["hook"] == "post-teardown"


def test_an_aborted_call_carrying_a_hook_leaves_the_stored_hooks_as_before(
    env: Env, fakebin: Fakebin
) -> None:
    old = [{"hook": "post-teardown", "cmd": "echo old"}]
    agent_id = seed(env, fakebin, hooks=old)
    pass_status_once(fakebin)  # live between step 2 and step 3: the call aborts to parked
    done = td(env, agent_id, "--hook", "post-teardown=echo new")
    assert done.returncode == 20 and result_json(done)["error"]["reason"] == "live"
    assert env.agent(agent_id)["hooks"] == old


def test_an_aborted_call_without_stored_hooks_leaves_them_null(env: Env, fakebin: Fakebin) -> None:
    agent_id = seed(env, fakebin)
    pass_status_once(fakebin)
    done = td(env, agent_id, "--hook", "post-teardown=echo new")
    assert done.returncode == 20
    assert env.agent(agent_id)["hooks"] is None


def test_a_refused_call_carrying_a_hook_leaves_the_stored_hooks_as_before(
    env: Env, fakebin: Fakebin
) -> None:
    old = [{"hook": "post-teardown", "cmd": "echo old"}]
    agent_id = seed(env, fakebin, status="live", park_reason=None, hooks=old)
    done = td(env, agent_id, "--hook", "post-teardown=echo new")
    assert done.returncode == 20
    assert env.agent(agent_id)["hooks"] == old


def test_the_abort_path_returns_to_parked_with_the_markers_cleared(
    env: Env, fakebin: Fakebin
) -> None:
    agent_id = seed(env, fakebin)
    pass_status_once(fakebin)
    done = td(env, agent_id)
    body = result_json(done)
    schema_registry.validate(body, "agent-spawn/cli/error.json")
    assert done.returncode == 20 and body["error"]["reason"] == "live"
    row = env.agent(agent_id)
    assert row["status"] == "parked" and row["park_reason"] == "done"
    assert row["teardown_step"] is None and row["teardown_proc"] is None
    assert (env.state_dir / "prompts" / agent_id).exists()
    assert forget_calls(fakebin) == []
    emitted = events(env, agent_id)
    assert [n for n, _ in emitted] == [
        "post-state-change",
        "post-park",
        "post-state-change",
    ]
    park = dict(emitted)["post-park"]
    assert park["reason"] == "done" and park["teardown_refused"] is False
    assert emitted[-1][1]["previous_status"] == "tearing_down"


def test_a_dry_run_runs_no_hooks(
    fakebin: Fakebin, isolation_env: Isolation, tmp_path: Path
) -> None:
    sink = tmp_path / "sink"
    config = f'hooks:\n  post-state-change:\n    - cmd: "cat >> {sink}"\n'
    cfg_env = Env(fakebin, isolation_env, config)
    agent_id = seed(cfg_env, fakebin)
    with Store.open(cfg_env.state_dir) as store:
        row = store.get_agent(agent_id)
        assert row is not None
        event = agent_event(
            "post-state-change",
            agent_object(store, row),
            reads_third_party=None,
            extra={"change": "agent", "previous_status": "live", "previous_needs_attention": None},
        )
        with store.lease(lease_key(agent_id), wait=LEASE_WAIT) as lease:
            store.update_agent(lease, agent_id, {}, events=[event])
    dry = td(cfg_env, agent_id, "--dry-run")
    assert dry.returncode == 0, dry.stderr
    assert not sink.exists()  # the pending event was not delivered
    assert td(cfg_env, agent_id).returncode == 0
    assert sink.exists()  # a real run delivers it


@pytest.mark.parametrize(
    ("step", "logged"), [("hooks_done", True), ("started", False)], ids=["skipped", "reruns"]
)
def test_teardown_retry_is_logged_only_when_the_hooks_step_is_skipped(
    env: Env, fakebin: Fakebin, step: str, logged: bool
) -> None:
    agent_id = seed(env, fakebin, status="tearing_down", teardown_step=step, teardown_proc=dead())
    assert td(env, agent_id).returncode == 0
    assert ('"teardown_retry"' in log_text(env)) is logged


def test_dry_run_of_a_live_agent_holding_a_worktree_reports_live(
    env: Env, fakebin: Fakebin
) -> None:
    agent_id = seed(
        env,
        fakebin,
        status="live",
        park_reason=None,
        wt_record_id="wt-abcdefgh",
        wt_path="/x",
        wt_owned=True,
    )
    done = td(env, agent_id, "--dry-run")
    body = result_json(done)
    assert done.returncode == 20 and body["would_refuse"] == "live"


def test_the_tombstone_checks_ownership_before_deleting_any_file(
    env: Env, fakebin: Fakebin
) -> None:
    agent_id = seed(
        env, fakebin, status="tearing_down", teardown_step="session_done", teardown_proc=dead()
    )
    with Store.open(env.state_dir) as store:
        engine = teardown.Teardown(store, Config(), state_dir=str(env.state_dir), actor="human:t")
        with pytest.raises(teardown.CliError) as raised:
            engine.step_tombstone(agent_id)
    assert "no longer owns" in raised.value.message
    assert (env.state_dir / "prompts" / agent_id).exists()
    assert (env.state_dir / "outcomes" / agent_id).exists()
    assert env.agent(agent_id)["status"] == "tearing_down"


def test_a_lost_entry_compare_and_set_is_judged_again(env: Env, fakebin: Fakebin) -> None:
    agent_id = seed(env, fakebin)
    backend = SessionBackend(
        [str(fakebin.bin_dir / "claude-session")], gateway=FakebinGateway(fakebin)
    )
    with Store.open(env.state_dir) as store:

        class Racing(teardown.Teardown):
            raced = 0

            def live_reason(self, row: Any) -> str | None:
                result = super().live_reason(row)
                if not self.raced:  # a promotion starts between the check and the CAS
                    self.raced += 1
                    with store.lease(lease_key(agent_id), wait=LEASE_WAIT) as lease:
                        store.update_agent(lease, agent_id, {"promotion_pending": True})
                return result

        engine = Racing(
            store, Config(), state_dir=str(env.state_dir), actor="human:t", backend=backend
        )
        with pytest.raises(teardown.CliError) as raised:
            engine.enter(agent_id)
        assert raised.value.reason == "live" and engine.raced == 1
        row = store.get_agent(agent_id)
        assert row is not None and row["status"] == "parked" and row["teardown_step"] is None


class _Inv:
    """The pieces of an invocation ``teardown_command`` reads."""

    def __init__(self, env: Env, agent_id: str) -> None:
        import argparse

        self.args = argparse.Namespace(
            agent_id=agent_id, hook=[], dry_run=False, force=False, merged_head=None
        )
        self.state_dir = str(env.state_dir)
        self.config = Config()
        self.actor = "human:t"
        self.decision_ids: list[str] = []
        self.warnings: list[str] = []


def test_a_lost_compare_and_set_reports_that_the_teardown_no_longer_owns_the_agent(
    env: Env, fakebin: Fakebin, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent_id = seed(env, fakebin, session=False)

    def lose(self: teardown.Teardown, agent: str, marker: str) -> None:
        raise CasLost(agent, ("tearing_down",), {"status": "parked"})

    monkeypatch.setattr(teardown.Teardown, "set_step", lose)
    with pytest.raises(teardown.CliError) as raised:
        teardown.teardown_command(_Inv(env, agent_id))
    assert raised.value.code == "failed"
    assert "no longer owns the agent" in raised.value.message


# --- the ephemeral refusal's compare-and-set -----------------------------------------------------


def test_the_ephemeral_refusal_needs_this_supervisor_as_the_teardown_process(
    env: Env, fakebin: Fakebin
) -> None:
    agent_id = seed(
        env, fakebin, status="tearing_down", teardown_step="started", teardown_proc=dead()
    )
    with Store.open(env.state_dir) as store:
        assert lifecycle.refuse_ephemeral_teardown(store, agent_id, outcome=None) is None
    row = env.agent(agent_id)
    assert row["status"] == "tearing_down" and row["teardown_proc"] == dead()


def test_the_ephemeral_refusal_parks_from_hooks_done(env: Env, fakebin: Fakebin) -> None:
    agent_id = seed(
        env, fakebin, status="tearing_down", teardown_step="hooks_done", teardown_proc=me()
    )
    with Store.open(env.state_dir) as store:
        moved = lifecycle.refuse_ephemeral_teardown(store, agent_id, outcome=None)
    assert moved is not None and moved["status"] == "parked"
    row = env.agent(agent_id)
    assert row["teardown_step"] is None and row["teardown_proc"] is None


@pytest.mark.parametrize("marker", ["worktree_partial", "session_done"])
def test_the_ephemeral_refusal_is_refused_from_later_markers(
    env: Env, fakebin: Fakebin, marker: str
) -> None:
    agent_id = seed(env, fakebin, status="tearing_down", teardown_step=marker, teardown_proc=me())
    with Store.open(env.state_dir) as store:
        assert lifecycle.refuse_ephemeral_teardown(store, agent_id, outcome=None) is None
    assert env.agent(agent_id)["status"] == "tearing_down"


def test_the_ephemeral_refusal_parks_the_agent_it_owns(env: Env, fakebin: Fakebin) -> None:
    agent_id = seed(
        env, fakebin, status="tearing_down", teardown_step="started", teardown_proc=me()
    )
    with Store.open(env.state_dir) as store:
        moved = lifecycle.refuse_ephemeral_teardown(store, agent_id, outcome=None)
    assert moved is not None and moved["status"] == "parked"
    row = env.agent(agent_id)
    assert row["teardown_step"] is None and row["teardown_proc"] is None


def test_an_ephemeral_agent_with_a_pre_teardown_hook_parks_done_with_teardown_refused(
    fakebin: Fakebin, isolation_env: Isolation
) -> None:
    # TODO(#32): the hook runs instead of refusing.
    config = 'hooks:\n  pre-teardown:\n    - cmd: "true"\n'
    cfg_env = Env(fakebin, isolation_env, config)
    done = cfg_env.run(
        "launch", "--mode", "headless", "--skill", "do-it", "--json", "--ephemeral", "--wait"
    )
    agent_id = result_json(done)["agent"]["agent_id"]
    row = cfg_env.agent(agent_id)
    assert row["status"] == "parked" and row["park_reason"] == "done"
    assert row["teardown_step"] is None and row["teardown_proc"] is None
    emitted = events(cfg_env, agent_id)
    park = next(p for n, p in emitted if n == "post-park")
    assert park["teardown_refused"] is True
    assert "post-teardown" not in [n for n, _ in emitted]
    assert forget_calls(fakebin) == []
