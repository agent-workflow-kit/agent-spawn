"""Dead panes that teardown step 5 did not reach before, and ``reconcile`` step 6 (#79): the
ownership rule (label ``agent=<id>``), the default server, a pane kept for reading a failure, the
age threshold, ``--dry-run`` and the audit log."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pytest

from agent_spawn.lifecycle import LEASE_WAIT, lease_key
from agent_spawn.store import Store
from tests import schema_registry
from tests.conftest import Isolation
from tests.fakebin.harness import Fakebin
from tests.launch_support import Env, result_json
from tests.test_teardown import seed

HOUR = 3600.0
CONFIG = "reconcile:\n  close_dead_panes_after: 1h\n"


@pytest.fixture
def env(fakebin: Fakebin, isolation_env: Isolation) -> Env:
    return Env(fakebin, isolation_env)


def open_pane(
    env: Env,
    fakebin: Fakebin,
    agent_id: str,
    *,
    dead: bool = True,
    socket: str | None = None,
) -> str:
    """A pane of the fake termctl labeled ``agent=<id>`` (on the default server unless
    ``socket``), dead or running."""
    args = ["open", "--name", "p", "--label", f"agent={agent_id}", "--keep-on-exit", "failed"]
    if socket is not None:
        args += ["--socket", socket]
    done = fakebin.run("termctl", *args, "--json", "--", "/bin/true", env=env.env())
    assert done.returncode == 0, (done.stdout, done.stderr)
    opened = json.loads(done.stdout)
    term_id, sock = opened["id"], opened["socket_path"]
    if dead:
        path = Path(str(fakebin.log) + ".panes.json")
        data = json.loads(path.read_text())
        data[sock][term_id]["state"] = "dead"
        path.write_text(json.dumps(data))
    return term_id


def pane_states(env: Env, fakebin: Fakebin) -> dict[str, str]:
    done = fakebin.run("termctl", "list", "--json", env=env.env())
    return {p["id"]: p["state"] for p in json.loads(done.stdout)["panes"]}


def put_run(env: Env, agent_id: str, **fields: Any) -> None:
    with Store.open(env.state_dir) as store:
        with store.lease(lease_key(agent_id), wait=LEASE_WAIT) as lease:
            store.put_run(lease, agent_id, 1, {"run_kind": "implement", **fields})


def closes(fakebin: Fakebin) -> list[list[str]]:
    return [a for a in fakebin.argvs("termctl") if a and a[0] == "close"]


def reconcile(env: Env, *flags: str) -> Any:
    done = env.run("reconcile", "--json", *flags)
    assert done.returncode == 0, (done.stdout, done.stderr)
    body = result_json(done)
    schema_registry.validate(body, "agent-spawn/cli/reconcile.json")
    return body


def panes_of(body: Any) -> list[dict[str, Any]]:
    return [c for c in body["changes"] if c["kind"] == "dead_pane"]


def window_agent(env: Env, fakebin: Fakebin, age: float = 2 * HOUR, **fields: Any) -> str:
    agent_id = seed(env, fakebin, updated=time.time() - age, **fields)
    put_run(env, agent_id, placement="window")  # opened, never recorded
    return agent_id


# --- teardown --------------------------------------------------------------------------------


def test_teardown_closes_a_dead_pane_on_a_socket_no_run_records(env: Env, fakebin: Fakebin) -> None:
    agent_id = window_agent(env, fakebin)
    other = seed(env, fakebin, session=False)
    mine = open_pane(env, fakebin, agent_id)
    theirs = open_pane(env, fakebin, other)
    running = open_pane(env, fakebin, agent_id, dead=False)
    done = env.run("teardown", agent_id, "--json")
    assert done.returncode == 0, (done.stdout, done.stderr)
    assert result_json(done)["windows_closed"] == [mine]
    assert set(pane_states(env, fakebin)) == {theirs, running}  # only the dead, owned pane


def test_teardown_without_a_window_run_never_looks_for_panes(env: Env, fakebin: Fakebin) -> None:
    agent_id = seed(env, fakebin)
    put_run(env, agent_id, placement=None)
    pane = open_pane(env, fakebin, agent_id)
    done = env.run("teardown", agent_id, "--json")
    assert done.returncode == 0, (done.stdout, done.stderr)
    assert result_json(done)["windows_closed"] == []
    assert pane in pane_states(env, fakebin)
    assert not [a for a in fakebin.argvs("termctl") if a and a[0] == "list" and "--dead" in a]


def test_teardown_closes_a_dead_promotion_pane_with_a_term_id_and_no_pane_proc(
    env: Env, fakebin: Fakebin
) -> None:
    agent_id = seed(env, fakebin)
    pane = open_pane(env, fakebin, agent_id)
    socket = next(iter(json.loads(Path(str(fakebin.log) + ".panes.json").read_text())))
    put_run(env, agent_id, placement="window", term_id=pane, socket_path=socket, pane_proc=None)
    done = env.run("teardown", agent_id, "--json")
    assert result_json(done)["windows_closed"] == [pane]


# --- reconcile -------------------------------------------------------------------------------


def test_reconcile_only_reports_a_dead_pane_by_default(env: Env, fakebin: Fakebin) -> None:
    agent_id = window_agent(env, fakebin)
    pane = open_pane(env, fakebin, agent_id)
    body = reconcile(env)
    (change,) = panes_of(body)
    assert change["agent_id"] == agent_id
    assert change["details"]["term_id"] == pane
    assert (change["details"]["action"], change["details"]["reason"]) == ("kept", "report_only")
    assert closes(fakebin) == [] and pane in pane_states(env, fakebin)


def test_reconcile_closes_an_aged_dead_pane_of_an_agent_without_attention(
    fakebin: Fakebin, isolation_env: Isolation
) -> None:
    env = Env(fakebin, isolation_env, CONFIG)
    agent_id = window_agent(env, fakebin)
    pane = open_pane(env, fakebin, agent_id)
    body = reconcile(env)
    (change,) = panes_of(body)
    assert change["details"]["action"] == "closed" and change["details"]["term_id"] == pane
    assert pane not in pane_states(env, fakebin)
    (close,) = closes(fakebin)
    assert "--force" not in close and "--actor=auto:agent-spawn" in close
    log = [json.loads(x) for x in (env.state_dir / "log.jsonl").read_text().splitlines()]
    (entry,) = [
        x for x in log if x["event"] == "reconcile_change" and x["details"]["kind"] == "pane_closed"
    ]
    assert entry["agent_id"] == agent_id and entry["details"]["term_id"] == pane
    assert panes_of(reconcile(env)) == []  # idempotent: nothing left


def test_reconcile_keeps_a_pane_of_an_agent_changed_recently(
    fakebin: Fakebin, isolation_env: Isolation
) -> None:
    env = Env(fakebin, isolation_env, CONFIG)
    agent_id = window_agent(env, fakebin, age=60.0)
    pane = open_pane(env, fakebin, agent_id)
    (change,) = panes_of(reconcile(env))
    assert (change["details"]["action"], change["details"]["reason"]) == ("kept", "too_new")
    assert pane in pane_states(env, fakebin)


def test_reconcile_keeps_a_pane_kept_for_reading_a_failure_however_old(
    fakebin: Fakebin, isolation_env: Isolation
) -> None:
    env = Env(fakebin, isolation_env, CONFIG)
    agent_id = window_agent(env, fakebin, age=900 * HOUR, needs_attention="failed")
    pane = open_pane(env, fakebin, agent_id)
    (change,) = panes_of(reconcile(env))
    assert (change["details"]["action"], change["details"]["reason"]) == ("kept", "attention")
    assert pane in pane_states(env, fakebin) and closes(fakebin) == []
    done = env.run("ack", agent_id, "--json")  # acknowledged: the failure was read
    assert done.returncode == 0, (done.stdout, done.stderr)
    with Store.open(env.state_dir) as store:  # the ack is old enough for the threshold
        with store.lease(lease_key(agent_id), wait=LEASE_WAIT) as lease:
            store.update_agent(lease, agent_id, {"updated": time.time() - 2 * HOUR}, touch=False)
    (change,) = panes_of(reconcile(env))
    assert change["details"]["action"] == "closed"


def test_reconcile_dry_run_closes_nothing(fakebin: Fakebin, isolation_env: Isolation) -> None:
    env = Env(fakebin, isolation_env, CONFIG)
    agent_id = window_agent(env, fakebin)
    pane = open_pane(env, fakebin, agent_id)
    (change,) = panes_of(reconcile(env, "--dry-run"))
    assert change["details"]["action"] == "would_close"
    assert closes(fakebin) == [] and pane in pane_states(env, fakebin)
    assert (
        not (env.state_dir / "log.jsonl").exists()
        or "pane_closed" not in (env.state_dir / "log.jsonl").read_text()
    )


def test_reconcile_never_touches_panes_of_live_agents_or_running_panes(
    fakebin: Fakebin, isolation_env: Isolation
) -> None:
    env = Env(fakebin, isolation_env, CONFIG)
    live = window_agent(env, fakebin, status="live", park_reason=None)
    parked = window_agent(env, fakebin)
    p_live = open_pane(env, fakebin, live)
    p_running = open_pane(env, fakebin, parked, dead=False)
    body = reconcile(env)
    assert [c for c in panes_of(body)] == []
    assert {p_live, p_running} <= set(pane_states(env, fakebin))
    assert closes(fakebin) == []


def test_reconcile_without_a_window_run_makes_no_termctl_call(
    fakebin: Fakebin, isolation_env: Isolation
) -> None:
    env = Env(fakebin, isolation_env, CONFIG)
    seed(env, fakebin, updated=time.time() - 9 * HOUR)
    assert panes_of(reconcile(env)) == []
    assert fakebin.argvs("termctl") == []


def test_a_failing_close_is_a_warning_and_the_pass_goes_on(
    fakebin: Fakebin, isolation_env: Isolation
) -> None:
    env = Env(fakebin, isolation_env, CONFIG)
    agent_id = window_agent(env, fakebin)
    open_pane(env, fakebin, agent_id)
    fakebin.script("termctl", "close", exit=12, error={"code": "failed", "message": "x"})
    body = reconcile(env)
    assert panes_of(body) == []
    assert any("termctl close failed" in w for w in body["warnings"])
