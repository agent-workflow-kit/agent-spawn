"""Liveness: the findings of review round 1, each pinned by a test that fails without its fix.

No test depends on timing or on the fake's default ``done`` status: a polling loop runs on a manual
clock (the gateway's), a race is a barrier or a scripted mutation inside a ``status`` call, and an
outcome is scripted explicitly.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable, Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from agent_spawn import cli, lifecycle, procs
from agent_spawn import gateway as gw
from agent_spawn import teardown as teardown_code
from agent_spawn.launch import Launch, SignalState
from agent_spawn.liveness import Checked, Liveness, row_moved
from agent_spawn.output import CliError
from agent_spawn.procs import ProcessIdentity
from agent_spawn.session_backend import SessionBackend
from agent_spawn.store import Store
from agent_spawn.worktrees import WtctlBackend
from tests.conftest import Isolation
from tests.fakebin.fakelib.claude_session import outcome_object
from tests.fakebin.harness import Fakebin
from tests.launch_support import Env, script_live_run, track_agent
from tests.procguard import ProcTracker
from tests.test_liveness import (
    AGENT,
    OTHER,
    PRE_RESUME,
    SESSION,
    calls,
    dead,
    events,
    liveness_parks,
    me,
    put_run,
    script_launch_phase,
    script_pending,
    seed,
    show,
    start_together,
)

THIRD = "ag-cdefghij"


@pytest.fixture
def env(fakebin: Fakebin, isolation_env: Isolation) -> Iterator[Env]:
    fakebin.seed_session(SESSION)
    made = Env(fakebin, isolation_env)
    # checks and commands run inside the test process use the fakes too; the environment is
    # restored before the isolation guard looks at it
    with pytest.MonkeyPatch.context() as inside:
        live_env(made, inside)
        yield made


def log_lines(env: Env) -> list[dict[str, Any]]:
    path = env.state_dir / "log.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def logged(env: Env, event: str, **match: Any) -> list[dict[str, Any]]:
    found = [line for line in log_lines(env) if line["event"] == event]
    for key, value in match.items():
        found = [
            line
            for line in found
            if line.get(key) == value or (line.get("details") or {}).get(key) == value
        ]
    return found


def live_env(env: Env, monkeypatch: pytest.MonkeyPatch) -> None:
    """The fakes and the state of this test, for commands run inside the test process."""
    for key, value in env.env().items():
        monkeypatch.setenv(key, value)


def checker(env: Env, store: Store, **kwargs: Any) -> Liveness:
    return Liveness(
        store,
        state_dir=str(env.state_dir),
        fallback_command=[str(env.fakebin.bin_dir / "claude-session")],
        **kwargs,
    )


def check(env: Env, agent_id: str = AGENT, **kwargs: Any) -> Checked:
    with Store.open(env.state_dir) as store:
        return checker(env, store, **kwargs).check(agent_id)


def kinds(checked: Checked) -> list[str]:
    return [change.kind for change in checked.changes]


class ClockGateway(gw.SystemGateway):
    """The real gateway with a manual clock: ``sleep`` advances it and nothing really waits. A
    loop that never sleeps or never looks at its deadline runs the clock dry and fails the test
    instead of hanging it."""

    def __init__(self) -> None:
        super().__init__()
        self.now = 1000.0
        self.reads = 0
        self.runs = 0

    def run(self, *args: Any, **kwargs: Any) -> Any:
        self.runs += 1
        assert self.runs < 200, "a polling loop is spinning on lower-tool calls"
        return super().run(*args, **kwargs)

    def monotonic(self) -> float:
        self.reads += 1
        assert self.reads < 20000, "a polling loop is spinning (it neither sleeps nor times out)"
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def clock() -> Any:
    gateway = ClockGateway()
    previous = gw.set_gateway(gateway)
    yield gateway
    gw.set_gateway(previous)


# -- R1: a report-only change never spins a polling loop ---------------------------------------


def pending_with_dead_supervisor(env: Env, fakebin: Fakebin, proc_tracker: ProcTracker) -> None:
    fakebin.seed_session(SESSION, adopted=False, final=False)
    script_pending(fakebin)
    seed(env, supervisor=dead(proc_tracker))


def test_wait_times_out_with_a_bounded_number_of_status_calls_on_a_pending_run(
    env: Env,
    fakebin: Fakebin,
    proc_tracker: ProcTracker,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    clock: ClockGateway,
) -> None:
    pending_with_dead_supervisor(env, fakebin, proc_tracker)
    assert cli.main(["wait", AGENT, "--timeout", "30s", "--json"]) == 22
    assert json.loads(capsys.readouterr().out)["timed_out"] is True
    statuses = len(calls(fakebin, "status"))
    assert 1 <= statuses <= 32  # about one per second of the manual clock (rate limit), not a spin
    assert clock.now >= 1030.0
    # reported by the check, logged once for the run, not on every poll
    assert len(logged(env, "reconcile_change", kind="pending_run")) == 1
    assert env.agent(AGENT)["status"] == "live"


def test_launch_wait_times_out_with_a_bounded_number_of_status_calls_on_a_pending_run(
    env: Env,
    fakebin: Fakebin,
    proc_tracker: ProcTracker,
    monkeypatch: pytest.MonkeyPatch,
    clock: ClockGateway,
) -> None:
    pending_with_dead_supervisor(env, fakebin, proc_tracker)
    launch = object.__new__(Launch)
    with Store.open(env.state_dir) as store:
        launch.store = store
        launch.agent_id = AGENT
        launch.args = SimpleNamespace(wait_timeout="30s", wait=True)
        launch.sig = SignalState()
        launch.gateway = clock
        launch.state_dir = str(env.state_dir)
        launch.backend = None
        launch.warnings = []
        launch.decision_ids = []
        launch.started = False
        outcome = launch.wait()
    assert outcome.exit_code == 22
    assert 1 <= len(calls(fakebin, "status")) <= 32
    assert len(logged(env, "reconcile_change", kind="pending_run")) == 1


def test_a_pending_run_is_logged_once_however_often_it_is_looked_at(
    env: Env, fakebin: Fakebin, proc_tracker: ProcTracker
) -> None:
    pending_with_dead_supervisor(env, fakebin, proc_tracker)
    for _ in range(3):
        assert show(env)["agent"]["status"] == "live"
    assert len(logged(env, "reconcile_change", kind="pending_run")) == 1


def test_row_moved_looks_only_at_the_row() -> None:
    row = {"status": "live", "updated": 1.0, "supervisor_lost": False, "csession_id": "cs-a"}
    assert row_moved(row, dict(row)) is False
    for field, value in (
        ("status", "parked"),
        ("updated", 2.0),
        ("supervisor_lost", True),
        ("csession_id", "cs-b"),
    ):
        assert row_moved(row, {**row, field: value}) is True


# -- R2: a resume in flight is not parked by the previous run's dead supervisor -------------------


def test_enter_resume_clears_the_previous_runs_processes(
    env: Env, proc_tracker: ProcTracker
) -> None:
    gone = dead(proc_tracker)
    seed(env, status="parked", supervisor=gone, exec_proc=gone, supervisor_lost=True)
    with Store.open(env.state_dir) as store:
        with store.lease(lifecycle.lease_key(AGENT), wait=5) as lease:
            row = store.enter_resume(lease, AGENT, launcher=me(env), expect_run_after=1)
    assert row["status"] == "launching" and row["launcher"] == me(env)
    assert row["supervisor"] is None and row["exec_proc"] is None
    assert row["supervisor_lost"] is False


@pytest.mark.parametrize("command", ["show", "list", "wait", "stop", "teardown"])
def test_a_resume_in_flight_survives_every_check_with_a_stale_dead_supervisor(
    env: Env, fakebin: Fakebin, proc_tracker: ProcTracker, command: str
) -> None:
    gone = dead(proc_tracker)
    seed(env, status="parked", supervisor=gone, exec_proc=gone, supervisor_lost=True)
    fakebin.script("claude-session", "status", outcome="done")  # a park would be visible
    with Store.open(env.state_dir) as store:
        with store.lease(lifecycle.lease_key(AGENT), wait=5) as lease:
            store.enter_resume(lease, AGENT, launcher=me(env), expect_run_after=1)
    argv = {
        "show": ["show", AGENT, "--json"],
        "list": ["list", "--json"],
        "wait": ["wait", AGENT, "--timeout", "1s", "--json"],
        "stop": ["stop", AGENT, "--dry-run", "--json"],
        "teardown": ["teardown", AGENT, "--json"],
    }[command]
    env.run(*argv)
    row = env.agent(AGENT)
    assert row["status"] == "launching" and row["pre_resume"] is not None
    if command in ("list", "wait", "teardown"):  # (`show` and `stop` read the status themselves)
        assert calls(fakebin, "status") == []  # the live launcher decides: nothing is read
    assert events(env) == []
    assert logged(env, "status_change") == []


# -- R3: a session found by label is adopted only when it is surely this agent's ------------------


def launching_without_session(env: Env, proc_tracker: ProcTracker, **fields: Any) -> None:
    seed(
        env,
        status="launching",
        launcher=dead(proc_tracker),
        csession_id=None,
        run=None,
        **fields,
    )


def test_a_session_recorded_on_another_agent_is_not_adopted(
    env: Env, fakebin: Fakebin, proc_tracker: ProcTracker
) -> None:
    fakebin.seed_session(SESSION, labels=[f"agent={AGENT}"])
    seed(env, OTHER, supervisor=me(env))  # OTHER already records SESSION
    launching_without_session(env, proc_tracker)
    checked = check(env)
    assert kinds(checked) == ["ambiguous_session", "launch_failed"]
    assert checked.changes[1].details["cause"] == "ambiguous_session"
    assert any("was not adopted" in w for w in checked.warnings)
    row = env.agent(AGENT)
    assert row["csession_id"] is None and row["park_reason"] == "launch_failed"
    assert calls(fakebin, "stop") == []


def test_a_session_labeled_for_another_agent_too_is_not_adopted(
    env: Env, fakebin: Fakebin, proc_tracker: ProcTracker
) -> None:
    fakebin.seed_session(SESSION, labels=[f"agent={AGENT}", f"agent={OTHER}"])
    launching_without_session(env, proc_tracker)
    checked = check(env)
    assert kinds(checked) == ["ambiguous_session", "launch_failed"]
    assert env.agent(AGENT)["csession_id"] is None


def test_the_one_session_of_this_agent_is_adopted(
    env: Env, fakebin: Fakebin, proc_tracker: ProcTracker
) -> None:
    fakebin.seed_session(SESSION, adopted=False, final=False, labels=[f"agent={AGENT}", "job=j-1"])
    script_pending(fakebin)
    launching_without_session(env, proc_tracker)
    checked = check(env)
    assert kinds(checked) == ["pending_run"]
    assert env.agent(AGENT)["csession_id"] == SESSION


def test_two_labeled_sessions_report_ambiguous_session(
    env: Env, fakebin: Fakebin, proc_tracker: ProcTracker
) -> None:
    fakebin.seed_session(SESSION, labels=[f"agent={AGENT}"])
    fakebin.seed_session("cs-bcdefghi", labels=[f"agent={AGENT}"])
    launching_without_session(env, proc_tracker)
    checked = check(env)
    assert kinds(checked) == ["ambiguous_session", "launch_failed"]
    assert checked.changes[0].details["sessions"] == sorted([SESSION, "cs-bcdefghi"])
    assert len(logged(env, "reconcile_change", kind="ambiguous_session")) == 1


# -- R4 and R7: an abandoned fork ------------------------------------------------------------------


def wtctl_for(env: Env) -> Callable[[Any], WtctlBackend]:
    return lambda _row: WtctlBackend([str(env.fakebin.bin_dir / "wtctl")])


def seed_fork(env: Env, proc_tracker: ProcTracker, *, hold: bool = True) -> str:
    """A fork whose launcher died before its session existed; with ``hold``, one wtctl record
    that lists the fork as its holder (and the fork's hold row). Returns the record id."""
    seed(env, OTHER, supervisor=me(env))
    launching_without_session(env, proc_tracker, forked_from=OTHER)
    if not hold:
        return ""
    worktree = env.state_dir.parent / "fork-worktree"
    worktree.mkdir(parents=True, exist_ok=True)
    adopted = subprocess.run(
        [str(env.fakebin.bin_dir / "wtctl"), "adopt", str(worktree), f"--holder={AGENT}", "--json"],
        env=env.env(),
        capture_output=True,
        text=True,
        check=True,
    )
    record = json.loads(adopted.stdout)["record"]["id"]
    with Store.open(env.state_dir) as store:
        with store.lease(lifecycle.lease_key(AGENT), wait=5) as lease:
            store.put_hold(lease, AGENT, record, str(worktree), owned=False)
    return record


def forgets(fakebin: Fakebin) -> list[list[str]]:
    return [a for a in fakebin.argvs("wtctl") if a[0] == "forget"]


def test_a_session_that_appears_before_the_holds_are_released_aborts_the_abandonment(
    env: Env, fakebin: Fakebin, proc_tracker: ProcTracker
) -> None:
    seed_fork(env, proc_tracker)
    fakebin.seed_session(SESSION, labels=[f"agent={AGENT}"])  # appears after the first look
    fakebin.script("claude-session", "list", times=1, stdout={"sessions": []})
    checked = check(env, wtctl_for=wtctl_for(env))
    assert "fork_abandoned" not in kinds(checked) and not checked.removed
    assert forgets(fakebin) == []  # nothing was released
    with Store.open(env.state_dir) as store:
        assert store.get_agent(AGENT) is not None and len(store.get_holds(AGENT)) == 1
    assert logged(env, "fork_abandoned") == []


def test_a_fork_whose_row_moved_before_the_release_keeps_everything(
    env: Env, fakebin: Fakebin, proc_tracker: ProcTracker, monkeypatch: pytest.MonkeyPatch
) -> None:
    seed_fork(env, proc_tracker)
    real = SessionBackend.list_by_label
    moves = {"n": 0}

    def moving(self: SessionBackend, pair: str) -> list[str]:
        found = real(self, pair)
        moves["n"] += 1
        if moves["n"] == 1:  # between the first look and the release: the launcher is back
            set_fields(env, AGENT, launcher=me(env))
        return found

    monkeypatch.setattr(SessionBackend, "list_by_label", moving)
    checked = check(env, wtctl_for=wtctl_for(env))
    assert not checked.removed and forgets(fakebin) == []
    assert env.agent(AGENT)["status"] == "launching"


def test_a_hold_that_cannot_be_released_keeps_the_fork_record(
    env: Env, fakebin: Fakebin, proc_tracker: ProcTracker
) -> None:
    seed_fork(env, proc_tracker)
    fakebin.script("wtctl", "forget", error={"code": "failed", "message": "x"}, exit=12)
    checked = check(env, wtctl_for=wtctl_for(env))
    assert not checked.removed and "fork_abandoned" not in kinds(checked)
    assert any("could not release" in w for w in checked.warnings)
    assert forgets(fakebin)
    assert env.agent(AGENT)["status"] == "launching"


def test_a_fork_with_holds_and_no_wtctl_keeps_its_record(
    env: Env, fakebin: Fakebin, proc_tracker: ProcTracker
) -> None:
    seed_fork(env, proc_tracker)
    checked = check(env, wtctl_for=lambda _row: None)
    assert not checked.removed and any("wtctl is not available" in w for w in checked.warnings)
    assert env.agent(AGENT)["status"] == "launching"


def test_a_fork_without_holds_is_removed_even_without_wtctl(
    env: Env, fakebin: Fakebin, proc_tracker: ProcTracker
) -> None:
    seed_fork(env, proc_tracker, hold=False)
    checked = check(env, wtctl_for=lambda _row: None)
    assert checked.removed and kinds(checked) == ["fork_abandoned"]


def test_stop_of_an_abandoned_fork_releases_its_holds_through_wtctl(
    env: Env, fakebin: Fakebin, proc_tracker: ProcTracker
) -> None:
    record = seed_fork(env, proc_tracker)
    done = env.run("stop", AGENT, "--json")
    assert done.returncode == 0, (done.stdout, done.stderr)
    assert [f[1] for f in forgets(fakebin)] == [record]
    with Store.open(env.state_dir) as store:
        assert store.get_agent(AGENT) is None and store.is_reserved(AGENT)


def test_the_delete_guard_keeps_a_fork_whose_launcher_came_back_at_the_last_moment(
    env: Env, fakebin: Fakebin, proc_tracker: ProcTracker, monkeypatch: pytest.MonkeyPatch
) -> None:
    seed_fork(env, proc_tracker, hold=False)
    real = teardown_code.remove_agent_files

    def moving(state_dir: str, agent_id: str) -> list[str]:
        set_fields(env, agent_id, launcher=me(env))
        return real(state_dir, agent_id)

    monkeypatch.setattr(teardown_code, "remove_agent_files", moving)
    checked = check(env, wtctl_for=wtctl_for(env))
    assert not checked.removed and "fork_abandoned" not in kinds(checked)
    assert env.agent(AGENT)["status"] == "launching"
    assert logged(env, "fork_abandoned") == []


# -- R5: stop does not report "already parked" for a row nobody parked -----------------------------
# (test_stop_command.py::test_stop_exits_12_when_the_session_missing_park_did_not_happen)


# -- R6 and R8: an unreadable process entry --------------------------------------------------------

UNREADABLE_PID = 999999


@pytest.fixture
def unreadable(monkeypatch: pytest.MonkeyPatch) -> ProcessIdentity:
    bad = ProcessIdentity(UNREADABLE_PID, 1, "boot")
    real = procs.is_alive

    def is_alive(identity: ProcessIdentity, *args: Any, **kwargs: Any) -> bool:
        if identity.pid == UNREADABLE_PID:
            raise procs.ProcessIdentityError("unreadable")
        return real(identity, *args, **kwargs)

    monkeypatch.setattr(procs, "is_alive", is_alive)
    return bad


@pytest.mark.parametrize(
    "argv",
    [
        ["wait", AGENT, "--timeout", "5s", "--json"],
        ["show", AGENT, "--json"],
        ["stop", AGENT, "--json"],
        ["teardown", AGENT, "--json"],
        ["resume", AGENT, "--json"],
    ],
    ids=["wait", "show", "stop", "teardown", "resume"],
)
def test_an_unreadable_process_entry_is_exit_12_for_wait_show_and_mutating_commands(
    env: Env,
    unreadable: ProcessIdentity,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    clock: ClockGateway,
    argv: list[str],
) -> None:
    seed(env, supervisor=unreadable)
    assert cli.main(argv) == 12
    assert json.loads(capsys.readouterr().out)["error"]["code"] == "failed"
    assert env.agent(AGENT)["status"] == "live"


def test_launch_wait_exits_12_on_an_unreadable_supervisor_entry(
    env: Env, unreadable: ProcessIdentity, monkeypatch: pytest.MonkeyPatch, clock: ClockGateway
) -> None:
    seed(env, supervisor=unreadable)
    launch = object.__new__(Launch)
    with Store.open(env.state_dir) as store:
        launch.store = store
        launch.agent_id = AGENT
        launch.args = SimpleNamespace(wait_timeout=None, wait=True)
        launch.sig = SignalState()
        launch.gateway = clock
        launch.state_dir = str(env.state_dir)
        launch.backend = None
        launch.started = False
        with pytest.raises(CliError) as caught:
            launch.wait()
    assert caught.value.exit_code == 12


def test_list_lists_a_row_with_an_unreadable_entry_unchanged_and_logs_it(
    env: Env,
    unreadable: ProcessIdentity,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    seed(env, supervisor=unreadable)
    assert cli.main(["list", "--json"]) == 0
    agents = json.loads(capsys.readouterr().out)["agents"]
    assert [(a["agent_id"], a["status"]) for a in agents] == [(AGENT, "live")]
    unreadable_lines = logged(env, "warning", cause="liveness_unreadable")
    assert [line["agent_id"] for line in unreadable_lines] == [AGENT]


# -- R10: behaviors the first round left untested --------------------------------------------------


def test_a_live_row_without_a_supervisor_counts_as_a_dead_supervisor(
    env: Env, fakebin: Fakebin
) -> None:
    script_live_run(fakebin)
    seed(env)  # live, no supervisor recorded
    agent = show(env)["agent"]
    assert agent["status"] == "live" and agent["supervisor_lost"] is True


def test_a_live_row_without_a_supervisor_and_a_final_run_is_parked(
    env: Env, fakebin: Fakebin
) -> None:
    fakebin.script("claude-session", "status", outcome="failed")
    seed(env)
    assert show(env)["agent"]["park_reason"] == "failed"


def test_a_live_supervisor_decides_even_when_the_launcher_is_dead(
    env: Env, fakebin: Fakebin, proc_tracker: ProcTracker
) -> None:
    seed(env, status="launching", supervisor=me(env), launcher=dead(proc_tracker))
    checked = check(env)
    assert checked.changes == [] and fakebin.argvs("claude-session") == []
    assert env.agent(AGENT)["status"] == "launching"


def test_a_dead_supervisor_decides_even_when_the_launcher_is_alive(
    env: Env, fakebin: Fakebin, proc_tracker: ProcTracker
) -> None:
    script_live_run(fakebin)
    seed(env, status="launching", supervisor=dead(proc_tracker), launcher=me(env))
    checked = check(env)
    assert kinds(checked) == ["supervisor_lost"]
    assert env.agent(AGENT)["status"] == "live"


def test_a_plain_launch_whose_launcher_died_is_launch_failed_with_its_cause(
    env: Env, proc_tracker: ProcTracker
) -> None:
    launching_without_session(env, proc_tracker)
    checked = check(env)
    assert kinds(checked) == ["launch_failed"]
    assert checked.changes[0].details == {"cause": "launcher_died"}
    assert env.agent(AGENT)["park_reason"] == "launch_failed"


def test_a_resume_whose_launcher_died_before_a_session_run_is_restored_not_failed(
    env: Env, proc_tracker: ProcTracker
) -> None:
    launching_without_session(env, proc_tracker, pre_resume=PRE_RESUME, expect_run_after=1)
    checked = check(env)
    assert kinds(checked) == ["parked"]
    assert checked.changes[0].details["restored"] is True
    row = env.agent(AGENT)
    assert (row["status"], row["park_reason"]) == ("parked", "done")
    assert row["pre_resume"] is None and row["launcher"] is None
    assert events(env) == []
    assert len(logged(env, "interrupted")) == 1


def test_a_reused_pid_is_dead_whichever_way_the_start_time_differs(
    env: Env, fakebin: Fakebin
) -> None:
    mine = me(env)
    fakebin.script("claude-session", "status", outcome="done")
    seed(env, supervisor=ProcessIdentity(mine.pid, mine.start_time - 1, mine.boot_id))
    seed(env, OTHER, supervisor=ProcessIdentity(mine.pid, mine.start_time + 1, mine.boot_id))
    assert kinds(check(env, AGENT)) == ["parked"]
    assert kinds(check(env, OTHER)) == ["parked"]
    seed(env, THIRD, supervisor=mine)  # the same start time: the very process
    assert check(env, THIRD).changes == []


def write_outcome_file(env: Env, token: str, name: str, *, text: str | None = None) -> None:
    folder = env.state_dir / "outcomes" / AGENT
    folder.mkdir(parents=True, mode=0o700, exist_ok=True)
    path = folder / f"{token}.json"
    path.write_text(text if text is not None else json.dumps(outcome_object(name, SESSION, 1)))
    path.chmod(0o600)


def test_the_outcome_file_of_the_run_decides_over_status(
    env: Env, fakebin: Fakebin, proc_tracker: ProcTracker
) -> None:
    fakebin.script("claude-session", "status", outcome="done")
    seed(env, supervisor=dead(proc_tracker))
    put_run(env, AGENT, 1, outcome_token="tok-file")
    write_outcome_file(env, "tok-file", "failed")
    assert kinds(check(env)) == ["parked"]
    assert env.agent(AGENT)["last_outcome"]["outcome"] == "failed"


@pytest.mark.parametrize("damage", ["missing", "corrupt"])
def test_a_missing_or_corrupt_outcome_file_falls_back_to_status(
    env: Env, fakebin: Fakebin, proc_tracker: ProcTracker, damage: str
) -> None:
    fakebin.script("claude-session", "status", outcome="blocked")
    seed(env, supervisor=dead(proc_tracker))
    put_run(env, AGENT, 1, outcome_token="tok-file")
    if damage == "corrupt":
        write_outcome_file(env, "tok-file", "failed", text="{not json")
    assert kinds(check(env)) == ["parked"]
    assert env.agent(AGENT)["last_outcome"]["outcome"] == "blocked"


def test_a_dry_run_of_supervisor_lost_reports_and_changes_nothing(
    env: Env, fakebin: Fakebin, proc_tracker: ProcTracker
) -> None:
    script_live_run(fakebin)
    seed(env, supervisor=dead(proc_tracker))
    checked = check(env, dry_run=True)
    assert kinds(checked) == ["supervisor_lost"]
    assert checked.row is not None
    assert checked.row["supervisor_lost"] is True and checked.row["needs_attention"] == "lost"
    row = env.agent(AGENT)
    assert row["supervisor_lost"] is False and row["needs_attention"] is None
    assert log_lines(env) == [] and events(env) == []


# -- R11: every recovery is a compare-and-set on what was judged -----------------------------------


def mutate_during_status(monkeypatch: pytest.MonkeyPatch, change: Callable[[], None]) -> None:
    """The row moves between the judgment (a ``status`` call) and the apply."""
    real = SessionBackend.status
    done = {"once": False}

    def status(self: SessionBackend, *args: Any, **kwargs: Any) -> Any:
        answer = real(self, *args, **kwargs)
        if not done["once"]:
            done["once"] = True
            change()
        return answer

    monkeypatch.setattr(SessionBackend, "status", status)


def set_fields(env: Env, agent_id: str, **fields: Any) -> None:
    with Store.open(env.state_dir) as store:
        with store.lease(lifecycle.lease_key(agent_id), wait=5) as lease:
            store.update_agent(lease, agent_id, fields)


@pytest.mark.parametrize("field", ["supervisor", "launcher", "csession_id", "run"])
def test_a_park_is_not_applied_when_the_judged_row_moved(
    env: Env,
    fakebin: Fakebin,
    proc_tracker: ProcTracker,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
) -> None:
    fakebin.script("claude-session", "status", outcome="done")
    seed(env, supervisor=dead(proc_tracker), launcher=dead(proc_tracker))
    moved = {
        "supervisor": me(env),
        "launcher": me(env),
        "csession_id": "cs-bcdefghi",
        "run": 2,
    }[field]
    mutate_during_status(monkeypatch, lambda: set_fields(env, AGENT, **{field: moved}))
    checked = check(env)
    assert checked.changes == []
    row = env.agent(AGENT)
    assert row["status"] == "live" and row[field] == moved
    assert events(env) == [] and logged(env, "status_change") == []


@pytest.mark.parametrize("field", ["supervisor", "csession_id"])
def test_supervisor_lost_is_not_set_when_the_judged_row_moved(
    env: Env,
    fakebin: Fakebin,
    proc_tracker: ProcTracker,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
) -> None:
    script_live_run(fakebin)
    seed(env, supervisor=dead(proc_tracker))
    moved: Any = dead(proc_tracker) if field == "supervisor" else "cs-bcdefghi"
    mutate_during_status(monkeypatch, lambda: set_fields(env, AGENT, **{field: moved}))
    checked = check(env)
    assert checked.changes == []
    assert env.agent(AGENT)["supervisor_lost"] is False
    assert logged(env, "attention_set") == [] and events(env) == []


def test_supervisor_lost_is_flagged_once_and_a_second_check_writes_nothing(
    env: Env, fakebin: Fakebin, proc_tracker: ProcTracker
) -> None:
    script_live_run(fakebin)
    seed(env, supervisor=dead(proc_tracker))
    assert kinds(check(env)) == ["supervisor_lost"]
    again = check(env)
    assert again.changes == []
    assert len(logged(env, "attention_set")) == 1
    assert len(logged(env, "reconcile_change", kind="supervisor_lost")) == 1


@pytest.mark.parametrize("outcome", ["done", "supervisor_lost"])
def test_concurrent_checks_apply_the_recovery_once(
    env: Env,
    fakebin: Fakebin,
    proc_tracker: ProcTracker,
    tmp_path: Path,
    outcome: str,
) -> None:
    if outcome == "done":
        fakebin.script("claude-session", "status", outcome="done")
    else:
        script_live_run(fakebin)
    seed(env, supervisor=dead(proc_tracker))
    started = start_together(env, proc_tracker, tmp_path, [["show", AGENT, "--json"]] * 6)
    for proc in started:
        proc.communicate(timeout=120)
        assert proc.returncode == 0
    if outcome == "done":
        assert len(logged(env, "status_change", to="parked", cause="liveness")) == 1
        assert len(logged(env, "reconcile_change", kind="parked")) == 1
        assert [e["event"] for e in events(env)].count("post-park") == 1
    else:
        assert len(logged(env, "attention_set")) == 1
        assert len(logged(env, "reconcile_change", kind="supervisor_lost")) == 1
        assert [e["event"] for e in events(env)].count("post-escalate") == 1


# -- R12: the hook-free reads, with a hook that would show a delivery ------------------------------


def test_wait_parks_a_fresh_agent_and_delivers_no_hook(
    env: Env, fakebin: Fakebin, proc_tracker: ProcTracker, tmp_path: Path
) -> None:
    marker = tmp_path / "hook-ran"
    env.config.parent.mkdir(parents=True, exist_ok=True)
    env.config.write_text(
        f"hooks:\n  post-park:\n    - cmd: 'touch {marker}'\n"
        f"  post-state-change:\n    - cmd: 'touch {marker}'\n"
    )
    fakebin.script("claude-session", "status", outcome="done")
    seed(env, THIRD, supervisor=dead(proc_tracker))  # nothing looked at it before `wait`
    assert env.run("wait", THIRD, "--json").returncode == 0
    assert env.agent(THIRD)["status"] == "parked"
    assert len(logged(env, "status_change", to="parked", cause="liveness")) == 1
    assert "post-park" in [e["event"] for e in events(env, THIRD)]  # queued, not delivered
    assert not marker.exists()


def test_resume_and_a_real_stop_race_for_a_dead_supervisors_agent(
    env: Env, fakebin: Fakebin, proc_tracker: ProcTracker, tmp_path: Path
) -> None:
    marker = tmp_path / "post-park-hooks"
    env.config.parent.mkdir(parents=True, exist_ok=True)
    env.config.write_text(f"hooks:\n  post-park:\n    - cmd: 'echo parked >> {marker}'\n")
    fakebin.script("claude-session", "status", outcome="done")  # the orphaned run has ended
    fakebin.script("claude-session", "exec", outcome="done")
    seed(env, supervisor=dead(proc_tracker))
    started = start_together(
        env, proc_tracker, tmp_path, [["resume", AGENT, "--json"], ["stop", AGENT, "--json"]]
    )
    for proc in started:
        proc.communicate(timeout=120)
    track_agent(env, proc_tracker, AGENT)
    # whichever command checked first parked the agent, once: one status change, one reported
    # change, one park event (delivered at most once, by one of the mutating commands)
    assert len(liveness_parks(env)) == 1
    assert len(logged(env, "reconcile_change", kind="parked")) == 1
    # one snapshot of every post-park event (delivered or not): reading the undelivered ones and
    # the hook marker at two moments would count an event twice while a supervisor delivers it
    with Store.open(env.state_dir) as store:
        parks = [e for e in store.all_events(AGENT) if e["event"] == "post-park"]
    delivered = marker.read_text().count("parked") if marker.exists() else 0
    assert 1 <= len(parks) <= 2  # the liveness park, then possibly the resumed run's own
    assert delivered <= len(parks)  # a hook runs at most once per event


def test_a_launching_agent_with_a_dead_recorded_supervisor_and_a_pending_run_is_left_to_expiry(
    env: Env, fakebin: Fakebin, proc_tracker: ProcTracker
) -> None:
    fakebin.seed_session(SESSION, adopted=False, final=False)
    script_pending(fakebin)
    seed(env, status="launching", supervisor=dead(proc_tracker), launcher=me(env))
    checked = check(env)
    assert kinds(checked) == ["pending_run"]
    row = env.agent(AGENT)
    assert row["status"] == "launching" and row["supervisor_lost"] is False
    assert row["needs_attention"] is None and calls(fakebin, "stop") == []


def test_a_launching_agent_with_a_dead_supervisor_in_its_launch_phase_stays_launching(
    env: Env, fakebin: Fakebin, proc_tracker: ProcTracker
) -> None:
    script_launch_phase(fakebin)
    seed(env, status="launching", supervisor=dead(proc_tracker))
    assert check(env).changes == []
    assert env.agent(AGENT)["status"] == "launching"
