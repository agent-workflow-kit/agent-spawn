"""The supervisor driven in process against the fakes: bookkeeping that is hard to time from
outside (first_working_at, the ephemeral hand-over, when a report resume is allowed)."""

from __future__ import annotations

import json
import signal
import sqlite3
import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pytest

from agent_spawn import chain, cli
from agent_spawn.agent_view import ts
from agent_spawn.gateway import SystemGateway
from agent_spawn.ids import new_agent_id
from agent_spawn.session_backend import RunOutcome
from agent_spawn.store import Store
from agent_spawn.supervise import Final, RunCtx, Signals, Supervisor
from tests.conftest import Isolation
from tests.fakebin.harness import Fakebin
from tests.session_backend_support import FakebinGateway


class SpawnGateway(FakebinGateway):
    """The fakes' environment for `exec` too."""

    def spawn(self, argv: Sequence[str], **kwargs: Any) -> subprocess.Popen[bytes]:
        kwargs["env"] = self._fakebin.env()
        return SystemGateway.spawn(self, argv, **kwargs)


class Rig:
    """A seeded agent with a pending run, and a supervisor for it."""

    def __init__(
        self,
        fakebin: Fakebin,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        **agent: object,
    ) -> None:
        monkeypatch.setattr("agent_spawn.gateway._default", SpawnGateway(fakebin))
        self.fakebin = fakebin
        self.state = tmp_path / "state"
        outcome_dir = self.state / "outcomes"
        self.session = fakebin.seed_session("cs-abcdefgh", adopted=False, final=False, labels=[])
        with Store.open(self.state) as store:
            self.me = store.holder_identity()
            self.agent_id = new_agent_id(store)
            fields: dict[str, Any] = {
                "status": "launching",
                "name": "agent test",
                "mode": "headless",
                "supervisor": self.me,
                "csession_id": self.session,
                "run": 1,
                "claude_session_path": str(fakebin.bin_dir / "claude-session"),
                "state_dir": str(self.state),
                "agent_spawn_path": "/opt/agent-spawn",
                "config_path": str(tmp_path / "config.yaml"),
            }
            fields.update(agent)
            store.insert_agent(self.agent_id, fields)
            with store.lease(f"agent:{self.agent_id}") as lease:
                store.put_run(lease, self.agent_id, 1, {"outcome_token": "tok", "mode": "headless"})
        (outcome_dir / self.agent_id).mkdir(parents=True, mode=0o700)
        self.outcome_file = str(outcome_dir / self.agent_id / "tok.json")
        fakebin.seed_session(
            self.session, adopted=False, final=False, labels=[], outcome_file=self.outcome_file
        )
        inv = cli.parse_invocation(
            [
                "supervise",
                self.agent_id,
                "--state-dir",
                str(self.state),
                "--config",
                str(tmp_path / "config.yaml"),
            ]
        )
        self.supervisor = Supervisor(inv, Signals())

    def agent(self) -> dict[str, Any]:
        with Store.open(self.state, create=False) as store:
            row = store.get_agent(self.agent_id)
        assert row is not None
        return row

    def put_run(self, run: int, **fields: object) -> None:
        with Store.open(self.state, create=False) as store:
            with store.lease(f"agent:{self.agent_id}") as lease:
                store.put_run(lease, self.agent_id, run, fields)


@pytest.fixture
def rig_for(
    fakebin: Fakebin, isolation_env: Isolation, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> Callable[..., Rig]:
    def make(**agent: object) -> Rig:
        return Rig(fakebin, monkeypatch, tmp_path, **agent)

    return make


# --- adopted_at (agent_runs.started_at) ---------------------------------------------------------


def test_the_supervisor_records_the_adoption_time_of_its_run_once(
    rig_for: Callable[..., Rig],
) -> None:
    rig = rig_for()
    with Store.open(rig.state, create=False) as store:
        assert (store.get_run(rig.agent_id, 1) or {})["started_at"] is None
    rig.supervisor.run()
    with Store.open(rig.state, create=False) as store:
        run = store.get_run(rig.agent_id, 1)
        assert run is not None and isinstance(run["started_at"], float)
        first = run["started_at"]
        # a second observation of the same run (a later poll) never rewrites it
        with store.lease(f"agent:{rig.agent_id}") as lease:
            assert store.mark_run_adopted(lease, rig.agent_id, 1, first + 50) is False
        assert (store.get_run(rig.agent_id, 1) or {})["started_at"] == first
    # and `show` maps it to adopted_at
    from agent_spawn import showing

    with Store.open(rig.state, create=False) as store:
        shown = showing.run_objects(store, rig.agent())
    assert shown[0]["adopted_at"] == ts(first)


def test_a_run_that_never_reached_its_session_phase_has_no_adoption_time(
    rig_for: Callable[..., Rig],
) -> None:
    rig = rig_for()
    with Store.open(rig.state, create=False) as store:
        rig.supervisor.store = store
        assert rig.supervisor.mark_started(RunCtx(1, rig.outcome_file), "launch") is False
        assert (store.get_run(rig.agent_id, 1) or {})["started_at"] is None


# --- first_working_at ---------------------------------------------------------------------------

EVENT_TS = "2026-11-02T10:00:05.000Z"


def test_first_working_at_from_the_relay_wins(
    rig_for: Callable[..., Rig], fakebin: Fakebin
) -> None:
    rig = rig_for()
    rig.put_run(1, first_working_at=1_700_000_000.0)
    fakebin.script(
        "claude-session",
        "status",
        merge={
            "recent_events": [
                {"event_id": 1, "run": 1, "event_ts": EVENT_TS, "to_activity": "working"}
            ]
        },
    )
    rig.supervisor.run()
    last = rig.agent()["last_outcome"]
    assert last["first_working_at"] == ts(1_700_000_000.0)


def test_first_working_at_from_recent_events_when_the_relay_has_none(
    rig_for: Callable[..., Rig], fakebin: Fakebin
) -> None:
    rig = rig_for()
    fakebin.script(
        "claude-session",
        "status",
        merge={
            "recent_events": [
                {"event_id": 1, "run": 1, "event_ts": EVENT_TS, "to_activity": "working"}
            ]
        },
    )
    rig.supervisor.run()
    assert rig.agent()["last_outcome"]["first_working_at"] == EVENT_TS


def test_first_working_at_null_when_nothing_says(rig_for: Callable[..., Rig]) -> None:
    rig = rig_for()
    rig.supervisor.run()
    assert rig.agent()["last_outcome"]["first_working_at"] is None


# --- the ephemeral hand-over ----------------------------------------------------------------------


def test_ephemeral_done_enters_tearing_down_before_teardown_runs(
    rig_for: Callable[..., Rig],
) -> None:
    rig = rig_for(ephemeral=True)
    seen: list[dict[str, Any]] = []

    def teardown(store: Store, agent_id: str) -> str | None:
        seen.append(store.get_agent(agent_id) or {})
        return None  # teardown completed: it owns the rest

    rig.supervisor.teardown = teardown
    rig.supervisor.run()
    (during,) = seen
    assert during["status"] == "tearing_down" and during["teardown_step"] == "started"
    assert during["teardown_proc"] == rig.me and during["last_outcome"]["outcome"] == "done"
    assert rig.agent()["status"] == "tearing_down"  # nothing parked it


def test_ephemeral_refused_teardown_parks_done_with_teardown_refused(
    rig_for: Callable[..., Rig],
) -> None:
    rig = rig_for(ephemeral=True)
    rig.supervisor.teardown = lambda store, agent_id: "live"
    rig.supervisor.run()
    row = rig.agent()
    assert row["status"] == "parked" and row["park_reason"] == "done"
    assert row["teardown_step"] is None and row["teardown_proc"] is None
    db = sqlite3.connect(f"file:{rig.state / 'state.db'}?mode=ro", uri=True)
    try:
        rows = db.execute(
            "SELECT payload FROM events WHERE agent_id = ? AND event = 'post-park'",
            (rig.agent_id,),
        ).fetchall()
    finally:
        db.close()
    assert [json.loads(r[0])["teardown_refused"] for r in rows] == [True]


def test_ephemeral_is_ignored_unless_the_run_is_done(
    rig_for: Callable[..., Rig], fakebin: Fakebin
) -> None:
    rig = rig_for(ephemeral=True)
    fakebin.script("claude-session", "exec", outcome="failed")
    called: list[str] = []
    rig.supervisor.teardown = lambda store, agent_id: called.append(agent_id)
    rig.supervisor.run()
    row = rig.agent()
    assert row["status"] == "parked" and row["park_reason"] == "failed" and not called


# --- when a report resume is allowed ---------------------------------------------------------


def final_of(outcome: str = "exited-unreported", end_reason: str | None = "normal") -> Final:
    result = RunOutcome("cs-abcdefgh", 1, "session", outcome, end_reason, 14, None, None, "file")
    return Final(RunCtx(1, "/x/o.json"), result, {})


@pytest.fixture
def supervisor(tmp_path: Path, isolation_env: Isolation) -> Supervisor:
    inv = cli.parse_invocation(
        [
            "supervise",
            "ag-abcdefgh",
            "--state-dir",
            str(tmp_path / "state"),
            "--config",
            str(tmp_path / "config.yaml"),
        ]
    )
    return Supervisor(inv, Signals())


ROW = {"mode": "headless"}


def test_report_resume_is_allowed_for_a_plain_unreported_headless_run(
    supervisor: Supervisor,
) -> None:
    assert supervisor.wants_report_resume(ROW, final_of(), None) is True


@pytest.mark.parametrize(
    "end_reason",
    ["no_session_start", "stopped", "forwarded_term", "forwarded_hup", "forwarded_int"],
)
def test_report_resume_skipped_for_these_end_reasons(
    supervisor: Supervisor, end_reason: str
) -> None:
    assert supervisor.wants_report_resume(ROW, final_of(end_reason=end_reason), None) is False


@pytest.mark.parametrize("outcome", ["done", "blocked", "failed", "killed", "timeout", "lost"])
def test_report_resume_only_after_exited_unreported(supervisor: Supervisor, outcome: str) -> None:
    assert supervisor.wants_report_resume(ROW, final_of(outcome), None) is False


def test_report_resume_once_per_chain(supervisor: Supervisor) -> None:
    assert supervisor.wants_report_resume(ROW, final_of(), final_of()) is False
    again = final_of()
    again.ctx.auto_resumed = True
    assert supervisor.wants_report_resume(ROW, again, None) is False


def test_report_resume_skipped_for_interactive_runs(supervisor: Supervisor) -> None:
    assert supervisor.wants_report_resume({"mode": "interactive"}, final_of(), None) is False


def test_report_resume_skipped_when_disabled(supervisor: Supervisor) -> None:
    supervisor.config = supervisor.config.model_copy(update={"auto_report_resume": False})
    assert supervisor.wants_report_resume(ROW, final_of(), None) is False


@pytest.mark.parametrize("received", [signal.SIGTERM, signal.SIGHUP, signal.SIGINT])
def test_report_resume_skipped_after_a_termination_signal(
    supervisor: Supervisor, received: int
) -> None:
    supervisor.note_signal(received)
    assert supervisor.wants_report_resume(ROW, final_of(), None) is False


def test_other_signals_do_not_end_the_chain(supervisor: Supervisor) -> None:
    supervisor.note_signal(signal.SIGQUIT)
    assert supervisor.terminated is False


def test_a_signal_before_the_resume_starts_nothing(
    rig_for: Callable[..., Rig], fakebin: Fakebin
) -> None:
    rig = rig_for()
    sup = rig.supervisor
    sup.store = Store.open(rig.state, create=False)
    try:
        sup.signals.received.append(signal.SIGTERM)
        backend = sup.backend(rig.agent())
        with pytest.raises(chain.ChainCanceled):
            chain.start_report_resume(sup, backend, rig.agent(), rig.session, final_of())
    finally:
        sup.store.close()
    fakebin.assert_not_called("claude-session")


def test_chain_failure_words() -> None:
    assert chain.reason_word("not-started") == "not_started"
    assert chain.reason_word("Third_Party_Cwd") == "third_party_cwd"
    assert chain.reason_word("9bad") == "failed" and chain.reason_word(None) == "failed"
