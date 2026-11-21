"""``list --no-liveness`` and ``show --no-liveness``: the stored state, read with no side effect.

Nothing is probed (no process entry, no pidfd, no lower tool) and nothing is written (state
database, log, audit records, hook outbox); the output says so with ``liveness_checked: false``.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from agent_spawn import cli
from agent_spawn import gateway as gw
from agent_spawn.procs import ProcessIdentity
from agent_spawn.store import Store
from tests import schema_registry
from tests.conftest import Isolation
from tests.fakebin.harness import Fakebin
from tests.inspect_support import A1, A2, SESSION, seed_agent
from tests.launch_support import Env

LIST = "agent-spawn/cli/list.json"
SHOW = "agent-spawn/cli/show.json"
ERROR = "agent-spawn/cli/error.json"

# Every gateway call that touches a process, a pid, a descriptor or a lower tool.
PROBES = {
    "run", "spawn", "run_on_terminal", "exec_replace", "read_file", "pidfd_open",
    "pidfd_send_signal", "kill", "getpgid", "killpg", "wait_pid_exit", "waitpid", "fork",
    "poll_readable",
}  # fmt: skip


@pytest.fixture
def env(fakebin: Fakebin, isolation_env: Isolation) -> Env:
    return Env(fakebin, isolation_env)


def snapshot(env: Env) -> dict[str, str]:
    """The digest of every file under the state dir (database, sidecars, log, audit)."""
    found: dict[str, str] = {}
    for root, _dirs, files in os.walk(env.state_dir):
        for name in files:
            path = Path(root, name)
            found[str(path.relative_to(env.state_dir))] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
    return found


def dead_identity(env: Env) -> ProcessIdentity:
    with Store.open(env.state_dir) as store:
        me = store.holder_identity()
    return ProcessIdentity(pid=2**22 - 3, start_time=1, boot_id=me.boot_id)


def seed_dead(env: Env, fakebin: Fakebin, agent_id: str = A1, **fields: Any) -> None:
    dead = dead_identity(env)
    seed_agent(
        env, agent_id, status="live", csession_id=SESSION, session_state="working",
        supervisor=dead, exec_proc=dead, cwd="/work/x", wt_path="/work/x",
        wt_record_id="wt-aaaaaaaa",
        claude_session_path=str(fakebin.bin_dir / "claude-session"), **fields,
    )  # fmt: skip


def body_of(done: Any, schema: str) -> Any:
    assert done.returncode == 0, (done.stdout, done.stderr)
    body = json.loads(done.stdout)
    schema_registry.validate(body, schema)
    return body


def no_tool_called(fakebin: Fakebin) -> None:
    for tool in ("claude-session", "wtctl", "termctl", "tmux"):
        assert fakebin.argvs(tool) == [], tool


# --- the guarantee: no side effect on a dead supervisor ----------------------------------------


def test_list_no_liveness_leaves_every_file_byte_identical(env: Env, fakebin: Fakebin) -> None:
    seed_dead(env, fakebin)
    before = snapshot(env)
    body = body_of(env.run("list", "--no-liveness", "--json"), LIST)
    assert snapshot(env) == before
    assert body["liveness_checked"] is False
    agent = body["agents"][0]
    assert agent["status"] == "live" and agent["needs_attention"] is None  # as stored
    no_tool_called(fakebin)


def test_list_without_the_flag_still_checks_and_parks(env: Env, fakebin: Fakebin) -> None:
    fakebin.seed_session(SESSION, adopted=True, final=True, outcome="done")
    seed_dead(env, fakebin)
    body = body_of(env.run("list", "--json"), LIST)
    assert "liveness_checked" not in body  # without the flag nothing changes
    assert body["agents"][0]["status"] == "parked"


def test_show_no_liveness_leaves_every_file_byte_identical(env: Env, fakebin: Fakebin) -> None:
    seed_dead(env, fakebin, runs=[{"mode": "headless"}], run=1)
    before = snapshot(env)
    body = body_of(env.run("show", A1, "--no-liveness", "--json"), SHOW)
    assert snapshot(env) == before
    assert body["liveness_checked"] is False
    assert body["agent"]["status"] == "live"
    assert body["agent"]["park_reason"] is None
    assert [r["run"] for r in body["runs"]] == [1]
    no_tool_called(fakebin)


def test_show_without_the_flag_has_no_member_and_still_checks(env: Env, fakebin: Fakebin) -> None:
    fakebin.seed_session(SESSION, adopted=True, final=True, outcome="done")
    seed_dead(env, fakebin)
    body = body_of(env.run("show", A1, "--json"), SHOW)
    assert "liveness_checked" not in body
    assert body["supervisor"]["alive"] is False
    assert body["agent"]["status"] == "parked"


def test_show_no_liveness_does_not_park_a_missing_session(env: Env, fakebin: Fakebin) -> None:
    seed_agent(
        env, A1, status="live", csession_id="cs-zzzzzzzz",
        claude_session_path=str(fakebin.bin_dir / "claude-session"),
    )  # fmt: skip
    before = snapshot(env)
    body = body_of(env.run("show", A1, "--no-liveness", "--json"), SHOW)
    assert body["agent"]["status"] == "live"
    assert body["session"] is None
    assert snapshot(env) == before
    no_tool_called(fakebin)


def test_show_no_liveness_names_the_stored_supervisor_without_probing(
    env: Env, fakebin: Fakebin
) -> None:
    dead = dead_identity(env)
    seed_dead(env, fakebin)
    body = body_of(env.run("show", A1, "--no-liveness", "--json"), SHOW)
    # `alive` is not evaluated: null, never a guess
    assert body["supervisor"] == {"pid": dead.pid, "alive": None, "exec_pid": dead.pid}
    assert body["session"] is None
    assert body["third_party_run"] is None
    assert body["repo_config_changed"] is None


def test_show_no_liveness_of_a_tombstone_and_a_supervisorless_agent(
    env: Env, fakebin: Fakebin
) -> None:
    seed_agent(env, A1, status="torn_down", csession_id=SESSION, torn_down_at=10.0)
    seed_agent(env, A2, status="parked")
    body = body_of(env.run("show", A1, "--no-liveness", "--json"), SHOW)
    assert body["supervisor"] is None and body["liveness_checked"] is False
    assert body_of(env.run("show", A2, "--no-liveness", "--json"), SHOW)["supervisor"] is None
    no_tool_called(fakebin)


def test_no_liveness_never_migrates_an_older_database(env: Env) -> None:
    seed_agent(env, A1)
    import sqlite3

    db = sqlite3.connect(env.state_dir / "state.db")
    with db:
        db.execute("UPDATE meta SET value = '0' WHERE key = 'schema_version'")
    db.close()
    before = snapshot(env)
    for args in (("list",), ("show", A1)):
        done = env.run(*args, "--no-liveness", "--json")
        assert done.returncode == 12, (done.stdout, done.stderr)
        schema_registry.validate(json.loads(done.stdout), ERROR)
    assert snapshot(env) == before


def test_no_liveness_without_a_database_creates_nothing(env: Env) -> None:
    assert body_of(env.run("list", "--no-liveness", "--json"), LIST) == {
        "agents": [],
        "liveness_checked": False,
    }
    done = env.run("show", A1, "--no-liveness", "--json")
    assert done.returncode == 10
    assert not env.state_dir.exists()


def test_human_output_says_the_check_did_not_run(env: Env, fakebin: Fakebin) -> None:
    seed_dead(env, fakebin)
    assert "liveness not checked" in env.run("show", A1, "--no-liveness").stdout
    assert "liveness not checked" in env.run("list", "--no-liveness").stdout
    assert "liveness not checked" not in env.run("show", A1).stdout


def test_list_no_liveness_keeps_filters_and_unreadable_rows_unlogged(
    env: Env, fakebin: Fakebin
) -> None:
    seed_dead(env, fakebin, A1)
    seed_agent(env, A2, needs_attention="blocked")
    body = body_of(env.run("list", "--no-liveness", "--needs-attention", "--json"), LIST)
    assert [a["agent_id"] for a in body["agents"]] == [A2]


# --- no gateway probe -------------------------------------------------------------------------


class SpyGateway:
    """The real gateway, recording which method was called."""

    def __init__(self) -> None:
        self.real = gw.SystemGateway()
        self.calls: list[str] = []

    def __getattr__(self, name: str) -> Callable[..., Any]:
        target = getattr(self.real, name)
        if not callable(target):
            return target  # type: ignore[no-any-return]

        def call(*args: Any, **kwargs: Any) -> Any:
            self.calls.append(name)
            return target(*args, **kwargs)

        return call


@pytest.mark.parametrize("command", [("list",), ("show", A1)])
def test_no_gateway_probe_runs_with_the_flag_and_does_with_the_other(
    env: Env,
    fakebin: Fakebin,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    command: tuple[str, ...],
) -> None:
    fakebin.seed_session(SESSION, adopted=True, final=True, outcome="done")
    seed_dead(env, fakebin)
    for name, value in env.env().items():
        monkeypatch.setenv(name, value)
    try:
        spy = SpyGateway()
        previous = gw.set_gateway(spy)  # type: ignore[arg-type]
        try:
            assert cli.main([*command, "--no-liveness", "--json"]) == 0
        finally:
            gw.set_gateway(previous)
        assert [c for c in spy.calls if c in PROBES] == []
        assert fakebin.argvs("claude-session") == []

        # control: the same call without the flag does probe (the spy can see probes)
        spy = SpyGateway()
        previous = gw.set_gateway(spy)  # type: ignore[arg-type]
        try:
            assert cli.main([*command, "--json"]) == 0
        finally:
            gw.set_gateway(previous)
        assert [c for c in spy.calls if c in PROBES] != []
    finally:
        for name in [n for n in os.environ if n.startswith(("FAKEBIN_", "CSESSION_"))]:
            monkeypatch.delenv(name)
    capsys.readouterr()
