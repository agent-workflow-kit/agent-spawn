"""Dead panes that teardown step 5 did not reach before, and ``reconcile`` step 6 (#79): the
ownership rule (label ``agent=<id>``), the default server, a pane kept for reading a failure, the
age threshold, ``--dry-run`` and the audit log."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from agent_spawn.lifecycle import LEASE_WAIT, lease_key
from agent_spawn.output import CliError
from agent_spawn.panes import scan_dead, termctl_for
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


def result_json_text(out: str) -> Any:
    body = json.loads(out)
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
    (change,) = panes_of(reconcile(env))  # the ack restarted the clock: not old enough yet
    assert (change["details"]["action"], change["details"]["reason"]) == ("kept", "too_new")
    assert closes(fakebin) == []
    with Store.open(env.state_dir) as store:  # now the ack is old enough for the threshold
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


@pytest.mark.parametrize("status", ["launching", "live", "tearing_down"])
def test_reconcile_never_touches_panes_of_unparked_agents_or_running_panes(
    fakebin: Fakebin, isolation_env: Isolation, status: str
) -> None:
    env = Env(fakebin, isolation_env, CONFIG)
    other = window_agent(env, fakebin, status=status, park_reason=None)
    parked = window_agent(env, fakebin)
    p_other = open_pane(env, fakebin, other)
    p_running = open_pane(env, fakebin, parked, dead=False)
    body = reconcile(env)
    assert [c for c in panes_of(body)] == []
    assert {p_other, p_running} <= set(pane_states(env, fakebin))
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


# --- review round 1 -----------------------------------------------------------------------------


def only_default_socket(fakebin: Fakebin) -> str:
    return next(iter(json.loads(Path(str(fakebin.log) + ".panes.json").read_text())))


def test_a_busy_agent_lease_skips_all_its_panes_with_one_warning(
    fakebin: Fakebin,
    isolation_env: Isolation,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """F1/F2: a held lease skips the close with a warning, once per agent (not per pane, so the
    wait is paid once); a later pass looks again."""
    from tests.inspect_support import run_inprocess

    env = Env(fakebin, isolation_env, CONFIG)
    agent_id = window_agent(env, fakebin)
    panes = [open_pane(env, fakebin, agent_id), open_pane(env, fakebin, agent_id)]
    monkeypatch.setattr("agent_spawn.reconcile.LEASE_WAIT", 0.2)
    waits: list[float] = []
    original = Store.lease

    def counting(self: Store, key: str, **kwargs: Any) -> Any:
        if key == lease_key(agent_id):
            waits.append(kwargs["wait"])
        return original(self, key, **kwargs)

    with Store.open(env.state_dir) as store:
        held = store.acquire_lease(lease_key(agent_id))
        try:
            monkeypatch.setattr(Store, "lease", counting)
            code, out, _ = run_inprocess(env, monkeypatch, capsys, "reconcile", "--json")
            monkeypatch.setattr(Store, "lease", original)
        finally:
            store.release_lease(held)
    assert code == 0
    body = result_json_text(out)
    assert body["warnings"] == [f"{agent_id}: agent busy: dead pane not closed this pass"]
    assert len(waits) == 1 and panes_of(body) == []
    assert closes(fakebin) == [] and all(p in pane_states(env, fakebin) for p in panes)
    again = reconcile(env)
    assert {c["details"]["action"] for c in panes_of(again)} == {"closed"}


def test_a_race_between_the_listing_and_the_close_keeps_the_pane(
    fakebin: Fakebin, isolation_env: Isolation, tmp_path: Path
) -> None:
    """R1: the agent gains attention after the dead pane was listed: nothing is closed."""
    env = Env(fakebin, isolation_env, CONFIG)
    agent_id = window_agent(env, fakebin)
    pane = open_pane(env, fakebin, agent_id)
    gate = tmp_path / "list-gate"
    fakebin.script("termctl", "list", gate=str(gate))
    result: list[Any] = []
    runner = threading.Thread(target=lambda: result.append(env.run("reconcile", "--json")))
    runner.start()
    try:
        fakebin.wait_for_gate(gate, timeout=60)
        with Store.open(env.state_dir) as store:
            with store.lease(lease_key(agent_id), wait=LEASE_WAIT) as lease:
                store.update_agent(lease, agent_id, {"needs_attention": "failed"}, touch=False)
    finally:
        gate.touch()
        runner.join(60)
    (done,) = result
    assert done.returncode == 0, (done.stdout, done.stderr)
    assert closes(fakebin) == [] and pane in pane_states(env, fakebin)
    (change,) = panes_of(result_json(done))
    assert (change["details"]["action"], change["details"]["reason"]) == ("kept", "attention")


def test_a_resume_between_the_listing_and_the_close_keeps_the_pane(
    fakebin: Fakebin, isolation_env: Isolation, tmp_path: Path
) -> None:
    env = Env(fakebin, isolation_env, CONFIG)
    agent_id = window_agent(env, fakebin)
    pane = open_pane(env, fakebin, agent_id)
    gate = tmp_path / "list-gate"
    fakebin.script("termctl", "list", gate=str(gate))
    result: list[Any] = []
    runner = threading.Thread(target=lambda: result.append(env.run("reconcile", "--json")))
    runner.start()
    try:
        fakebin.wait_for_gate(gate, timeout=60)
        with Store.open(env.state_dir) as store:
            with store.lease(lease_key(agent_id), wait=LEASE_WAIT) as lease:
                store.compare_and_set(
                    lease,
                    agent_id,
                    expect_status="parked",
                    changes={"status": "launching"},
                    touch=False,
                )
    finally:
        gate.touch()
        runner.join(60)
    (done,) = result
    assert done.returncode == 0, (done.stdout, done.stderr)
    assert closes(fakebin) == [] and pane in pane_states(env, fakebin)


def test_a_recorded_socket_that_is_also_the_default_server_lists_a_pane_once(
    fakebin: Fakebin, isolation_env: Isolation
) -> None:
    """R2: report-only, dry-run and real runs each see the pane once."""
    env = Env(fakebin, isolation_env)
    agent_id = window_agent(env, fakebin)
    pane = open_pane(env, fakebin, agent_id)
    put_run(env, agent_id, placement="window", socket_path=only_default_socket(fakebin))
    (change,) = panes_of(reconcile(env))
    assert change["details"]["term_id"] == pane and change["details"]["action"] == "kept"


def test_a_pane_on_two_servers_is_closed_once(fakebin: Fakebin, isolation_env: Isolation) -> None:
    env = Env(fakebin, isolation_env, CONFIG)
    agent_id = window_agent(env, fakebin)
    pane = open_pane(env, fakebin, agent_id)
    put_run(env, agent_id, placement="window", socket_path=only_default_socket(fakebin))
    (would,) = panes_of(reconcile(env, "--dry-run"))
    assert would["details"]["action"] == "would_close"
    (change,) = panes_of(reconcile(env))
    assert change["details"]["action"] == "closed" and change["details"]["term_id"] == pane
    assert len(closes(fakebin)) == 1


class _Term:
    """A stand-in for ``TermBackend.find_dead`` with canned answers per target."""

    def __init__(self, answers: dict[str | None, Any]) -> None:
        self.answers = answers

    def find_dead(self, agent_id: str, socket_path: str | None) -> Any:
        answer = self.answers[socket_path]
        if isinstance(answer, CliError):
            raise answer
        return answer


def test_a_failed_default_server_sweep_only_warns() -> None:
    """R4: the recorded servers decide completeness; the default sweep is best effort."""
    from agent_spawn.term import DeadPanes

    failing = CliError("failed", "x", tool="termctl", exit_code=12)
    term = _Term({"/s": DeadPanes([("tc-aaaaaaaa", "/s")]), None: failing})
    scan = scan_dead(term, "ag-aaaaaaaa", ["/s", None])  # type: ignore[arg-type]
    assert scan.complete and scan.panes == [("tc-aaaaaaaa", "/s")]
    assert scan.warnings == ["termctl list failed (exit 12)"]


def test_a_failed_recorded_server_list_is_incomplete() -> None:
    failing = CliError("failed", "x", tool="termctl", exit_code=12)
    scan = scan_dead(_Term({"/s": failing}), "ag-aaaaaaaa", ["/s"])  # type: ignore[arg-type]
    assert not scan.complete


def test_a_pane_without_a_socket_path_is_incomplete_and_warns() -> None:
    """R5: never silently dropped."""
    from agent_spawn.term import DeadPanes

    scan = scan_dead(_Term({None: DeadPanes([], 1)}), "ag-aaaaaaaa", [None])  # type: ignore[arg-type]
    assert not scan.complete and scan.panes == []
    assert any("without a socket path" in w for w in scan.warnings)


def test_a_pane_takes_the_replys_socket_path_when_it_has_none(
    fakebin: Fakebin, isolation_env: Isolation
) -> None:
    env = Env(fakebin, isolation_env)
    agent_id = window_agent(env, fakebin)
    fakebin.script(
        "termctl",
        "list",
        stdout={"socket_path": "/run/top.sock", "panes": [{"id": "tc-aaaaaaaa", "state": "dead"}]},
    )
    (change,) = panes_of(reconcile(env))
    assert change["agent_id"] == agent_id
    assert change["details"]["socket_path"] == "/run/top.sock"


def test_a_pane_with_no_socket_path_anywhere_warns_and_is_not_reported(
    fakebin: Fakebin, isolation_env: Isolation
) -> None:
    env = Env(fakebin, isolation_env)
    window_agent(env, fakebin)
    fakebin.script("termctl", "list", stdout={"panes": [{"id": "tc-aaaaaaaa", "state": "dead"}]})
    body = reconcile(env)
    assert panes_of(body) == []
    assert any("without a socket path" in w for w in body["warnings"])


def test_teardown_closes_a_dead_pane_of_an_agent_with_attention_and_a_fresh_update(
    env: Env, fakebin: Fakebin
) -> None:
    """R11: teardown is the explicit decision; neither attention nor age matters."""
    agent_id = window_agent(env, fakebin, age=0.0, needs_attention="failed")
    pane = open_pane(env, fakebin, agent_id)
    done = env.run("teardown", agent_id, "--json")
    assert done.returncode == 0, (done.stdout, done.stderr)
    assert result_json(done)["windows_closed"] == [pane]
    assert pane not in pane_states(env, fakebin)


def test_a_failing_list_is_a_warning_and_the_other_agents_are_still_processed(
    fakebin: Fakebin, isolation_env: Isolation
) -> None:
    env = Env(fakebin, isolation_env)
    first = window_agent(env, fakebin)
    second = window_agent(env, fakebin)
    open_pane(env, fakebin, first)
    open_pane(env, fakebin, second)
    fakebin.script("termctl", "list", times=1, error={"code": "failed", "message": "x"}, exit=12)
    body = reconcile(env)
    assert len(panes_of(body)) == 1
    assert any("termctl list failed" in w for w in body["warnings"])


def test_a_missing_termctl_is_a_warning_and_no_pane_is_examined(
    fakebin: Fakebin, isolation_env: Isolation, tmp_path: Path
) -> None:
    env = Env(fakebin, isolation_env, f"termctl_cmd: [{tmp_path}/no-such-termctl]\n")
    agent_id = window_agent(env, fakebin)
    body = reconcile(env)
    assert panes_of(body) == []
    assert any(w.startswith(f"{agent_id}: termctl could not be found") for w in body["warnings"])


def test_termctl_is_resolved_once_per_pass(
    monkeypatch: pytest.MonkeyPatch, fakebin: Fakebin
) -> None:
    """R16: the cache answers every agent after the first."""
    import agent_spawn.launch as launch
    from agent_spawn import gateway as gw
    from agent_spawn.config import Config

    calls: list[str | None] = []

    def resolve(command: Any, name: str, stored: str | None) -> list[str]:
        calls.append(stored)
        return ["/bin/true"]

    monkeypatch.setattr(launch, "resolve_stored_tool", resolve)
    cache: dict[str | None, list[str] | None] = {}
    config = Config()
    for _ in range(3):
        backend = termctl_for(
            config, {}, actor=None, decision_ids=[], gateway=gw.get_gateway(), cache=cache
        )
        assert backend is not None
    assert calls == [None]
