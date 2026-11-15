"""Review fixes of the supervisor chain (round 1): signals, bookkeeping and failure paths."""

from __future__ import annotations

import json
import os
import signal
import sqlite3
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pytest

from agent_spawn import chain, lifecycle, procs, session_backend, supervise
from agent_spawn.chain import Final, RunCtx
from agent_spawn.session_backend import RunOutcome, StartError
from agent_spawn.store import Store, StoreError
from agent_spawn.supervise import Signals
from tests.fakebin.harness import Fakebin
from tests.launch_support import Env, result_json, script_live_run, track_agent
from tests.procguard import ProcTracker
from tests.test_supervise_chain import (
    LAUNCH,
    chain_failure_of,
    real_starts,
    runs,
    script_unreported_first,
)
from tests.test_supervise_chain import env as env  # noqa: F401 - the fixture
from tests.test_supervise_inprocess import Rig, SpawnGateway
from tests.test_supervise_inprocess import rig_for as rig_for  # noqa: F401 - the fixture


def log_records(state: Path) -> list[dict[str, Any]]:
    path = state / "log.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def rig_events(rig: Rig) -> list[tuple[str, dict[str, Any]]]:
    db = sqlite3.connect(f"file:{rig.state / 'state.db'}?mode=ro", uri=True)
    try:
        rows = db.execute(
            "SELECT event, payload FROM events WHERE agent_id = ? ORDER BY event_id",
            (rig.agent_id,),
        ).fetchall()
    finally:
        db.close()
    return [(name, json.loads(payload)) for name, payload in rows]


def rig_runs(rig: Rig) -> list[dict[str, Any]]:
    with Store.open(rig.state, create=False) as store:
        return store.get_runs(rig.agent_id)


def final_of(run: int = 1, outcome: str = "exited-unreported") -> Final:
    result = RunOutcome("cs-abcdefgh", run, "session", outcome, "no_report", 14, None, None, "file")
    return Final(RunCtx(run, "/x/o.json"), result, {})


# --- 1: the signal handler forwards itself ----


def test_the_handler_forwards_to_exec_without_waiting_for_the_loop(
    proc_tracker: ProcTracker,
) -> None:
    child = proc_tracker.popen(
        [sys.executable, "-c", "import time; time.sleep(60)"], stdin=subprocess.DEVNULL
    )
    identity = procs.read_identity(child.pid)
    assert identity is not None
    signals = Signals()
    signals.forwarder = procs.SignalForwarder(identity)
    signals.handle(signal.SIGTERM, None)  # no loop runs: the handler alone must deliver
    assert child.wait(timeout=10) == -signal.SIGTERM
    assert signals.received == [signal.SIGTERM]  # still recorded for the bookkeeping


def test_a_handler_interleaving_with_a_loop_forward_sends_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = procs.ProcessIdentity(pid=1, start_time=1, boot_id="b")
    forwarder = procs.SignalForwarder(identity)
    sent: list[int] = []

    def fake_forward(ident: object, sig: int, *, gateway: object = None) -> None:
        sent.append(sig)
        forwarder.forward(sig)  # the handler interleaves while the loop's send is in flight

    monkeypatch.setattr(procs, "forward_signal", fake_forward)
    forwarder.forward(signal.SIGHUP)
    assert sent == [signal.SIGHUP]


def test_a_failed_forward_may_be_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    forwarder = procs.SignalForwarder(procs.ProcessIdentity(pid=1, start_time=1, boot_id="b"))
    calls: list[int] = []

    def flaky(ident: object, sig: int, *, gateway: object = None) -> None:
        calls.append(sig)
        if len(calls) == 1:
            raise OSError("boom")

    monkeypatch.setattr(procs, "forward_signal", flaky)
    with pytest.raises(OSError):
        forwarder.forward(signal.SIGTERM)
    forwarder.forward(signal.SIGTERM)
    assert calls == [signal.SIGTERM, signal.SIGTERM]


def test_the_watch_loop_status_poll_has_a_short_timeout(
    rig_for: Callable[..., Rig], fakebin: Fakebin
) -> None:
    rig = rig_for()
    assert supervise.STATUS_TIMEOUT <= 10
    fakebin.script("claude-session", "exec", sleep=2)
    fakebin.script("claude-session", "status", times=1, sleep=3)  # the first poll hangs
    original = supervise.STATUS_TIMEOUT
    supervise.STATUS_TIMEOUT = 0.5
    try:
        rig.supervisor.run()
    finally:
        supervise.STATUS_TIMEOUT = original
    messages = [r.get("message") for r in log_records(rig.state)]
    assert "status failed: failed" in messages


# --- 2: first_working_at is the chain's first run ----


def test_first_working_at_is_taken_from_the_first_run_of_the_chain(
    env: Env, fakebin: Fakebin
) -> None:
    script_unreported_first(fakebin)
    stamp = "2026-11-02T10:00:05.000Z"
    fakebin.script(
        "claude-session",
        "status",
        merge={
            "recent_events": [
                {"event_id": 1, "run": 1, "event_ts": stamp, "to_activity": "working"}
            ]
        },
    )
    done = env.run(*LAUNCH, "--wait")
    body = result_json(done)
    assert done.returncode == 0, (done.stdout, done.stderr)
    assert env.agent(body["agent"]["agent_id"])["run"] == 2
    assert body["first_working_at"] == stamp


# --- 5: forward() without a forwarder ----


def test_forward_without_a_forwarder_signals_the_child_itself(
    rig_for: Callable[..., Rig], proc_tracker: ProcTracker
) -> None:
    sup = rig_for().supervisor
    child = proc_tracker.popen(
        [sys.executable, "-c", "import time; time.sleep(60)"], stdin=subprocess.DEVNULL
    )
    sup.forwarder = None
    sup.child = child
    sup.forward(signal.SIGTERM)
    assert child.wait(timeout=10) == -signal.SIGTERM


# --- 8: signals after the pre-start checks ----


def test_a_signal_during_the_pre_start_checks_starts_nothing(
    rig_for: Callable[..., Rig], fakebin: Fakebin, monkeypatch: pytest.MonkeyPatch
) -> None:
    rig = rig_for()
    sup = rig.supervisor
    sup.store = Store.open(rig.state, create=False)
    original = chain._pre_start_checks  # pyright: ignore[reportPrivateUsage]

    def checks(*args: Any, **kwargs: Any) -> Any:
        result = original(*args, **kwargs)
        sup.signals.received.append(signal.SIGHUP)
        return result

    monkeypatch.setattr(chain, "_pre_start_checks", checks)
    try:
        with pytest.raises(chain.ChainCanceled):
            chain.start_report_resume(
                sup, sup.backend(rig.agent()), rig.agent(), rig.session, final_of()
            )
    finally:
        sup.store.close()
    assert not [
        a for a in fakebin.argvs("claude-session") if a[0] == "start" and "--dry-run" not in a
    ]


# --- 9: a user cancellation of a chain run is not a chain failure ----


def test_a_canceled_chain_run_after_a_user_sigterm_parks_killed_without_attention(
    rig_for: Callable[..., Rig], fakebin: Fakebin
) -> None:
    rig = rig_for()
    sup = rig.supervisor
    script_unreported_first(fakebin)
    fakebin.script("claude-session", "exec", times=1, outcome="not-started")
    original = sup.spawn_exec
    spawned: list[int] = []

    def spawn(row: dict[str, Any], session: str) -> subprocess.Popen[bytes]:
        child = original(row, session)
        spawned.append(child.pid)
        if len(spawned) == 2:
            sup.terminated = True  # the user's SIGTERM, forwarded to the chain run
        return child

    sup.spawn_exec = spawn  # type: ignore[method-assign]
    sup.run()
    row = rig.agent()
    assert row["status"] == "parked" and row["park_reason"] == "killed"
    assert row["needs_attention"] is None and row["last_outcome"]["run"] == 2
    parks = [p for n, p in rig_events(rig) if n == "post-park"]
    assert [p["chain_failure"] for p in parks] == [None]


# --- 10, 11: record_run ----


def test_record_run_clears_exec_proc_and_uses_the_gateway_clock(
    rig_for: Callable[..., Rig], fakebin: Fakebin
) -> None:
    rig = rig_for()
    sup = rig.supervisor

    class Clock(SpawnGateway):
        def wall_time(self) -> float:
            return 1_234_567.0

    sup.gateway = Clock(fakebin)
    sup.store = Store.open(rig.state, create=False)
    try:
        with sup.store.lease(lifecycle.lease_key(rig.agent_id)) as lease:
            sup.store.update_agent(lease, rig.agent_id, {"exec_proc": rig.me})
        final = final_of(outcome="done")
        sup.record_run(final.ctx, final.result, {"first_working_at": None})
        assert (sup.store.get_agent(rig.agent_id) or {})["exec_proc"] is None
        assert (sup.store.get_run(rig.agent_id, 1) or {})["ended_at"] == 1_234_567.0
    finally:
        sup.store.close()


# --- 12: failures after a run exists ----


class FailingSpawn(SpawnGateway):
    """``exec`` can be started only ``allow`` times."""

    allow = 0

    def spawn(self, argv: Sequence[str], **kwargs: Any) -> subprocess.Popen[bytes]:
        if self.allow <= 0:
            raise OSError(2, "no such file")
        self.allow -= 1
        return super().spawn(argv, **kwargs)


def test_a_failing_first_exec_cancels_the_pending_run_and_parks(
    rig_for: Callable[..., Rig], fakebin: Fakebin
) -> None:
    rig = rig_for()
    rig.supervisor.gateway = FailingSpawn(fakebin)
    rig.supervisor.run()
    row = rig.agent()
    assert row["status"] == "parked" and row["park_reason"] == "killed"
    assert row["needs_attention"] is None
    assert [a for a in fakebin.argvs("claude-session") if a[0] == "stop"]


def test_a_failing_exec_of_a_chain_run_is_a_chain_failure(
    rig_for: Callable[..., Rig], fakebin: Fakebin
) -> None:
    rig = rig_for()
    script_unreported_first(fakebin)
    gateway = FailingSpawn(fakebin)
    gateway.allow = 1
    rig.supervisor.gateway = gateway
    rig.supervisor.run()
    row = rig.agent()
    assert row["status"] == "parked" and row["park_reason"] == "unreported"
    assert row["last_outcome"]["run"] == 1
    (park,) = [p for n, p in rig_events(rig) if n == "post-park"]
    assert park["chain_failure"]["tool"] == "agent-spawn"
    assert park["chain_failure"]["reason"] == "failed" and park["chain_failure"]["run"] == 2
    assert [a for a in fakebin.argvs("claude-session") if a[0] == "stop"]
    assert rig_runs(rig)[-1]["ended_at"] is not None


def test_a_store_error_after_start_is_a_chain_failure_not_a_crash(
    rig_for: Callable[..., Rig], fakebin: Fakebin, monkeypatch: pytest.MonkeyPatch
) -> None:
    rig = rig_for()
    script_unreported_first(fakebin)

    def broken(*args: Any, **kwargs: Any) -> None:
        raise StoreError("disk full")

    monkeypatch.setattr(chain, "_classification_check", broken)
    rig.supervisor.run()
    (park,) = [p for n, p in rig_events(rig) if n == "post-park"]
    assert park["chain_failure"]["tool"] == "agent-spawn" and park["chain_failure"]["run"] == 2
    assert [a for a in fakebin.argvs("claude-session") if a[0] == "stop"]


# --- 13: a stop that did not end the run ----


@pytest.mark.parametrize("failure", ["live", "error"])
def test_cancel_pending_does_not_fabricate_an_ended_run(
    rig_for: Callable[..., Rig], fakebin: Fakebin, failure: str
) -> None:
    rig = rig_for()
    sup = rig.supervisor
    if failure == "live":
        fakebin.script("claude-session", "stop", exit=12, merge={"ended": False})
    else:
        fakebin.script("claude-session", "stop", exit=1, error={"code": "failed", "message": "x"})
    sup.store = Store.open(rig.state, create=False)
    try:
        chain.cancel_pending(
            sup, sup.backend(rig.agent()), rig.session, 2, hooks=None, token="tok2"
        )
    finally:
        sup.store.close()
    second = rig_runs(rig)[-1]
    assert second["run"] == 2 and second["outcome"] is None and second["end_reason"] is None
    assert second["ended_at"] is not None
    assert any(r["level"] == "warn" for r in log_records(rig.state))


def test_a_classification_cancel_that_cannot_stop_still_records_the_run(
    env: Env, fakebin: Fakebin
) -> None:
    script_unreported_first(fakebin)
    fakebin.script("claude-session", "status", times=1)
    fakebin.script(
        "claude-session",
        "status",
        merge={"run": {"reads_third_party": True}, "tainted_by_run": 2},
    )
    fakebin.script("claude-session", "stop", exit=12, merge={"ended": False})
    done = env.run(*LAUNCH, "--wait")
    agent_id = result_json(done)["agent"]["agent_id"]
    failure = chain_failure_of(env, agent_id)
    assert failure is not None and failure["run"] == 2
    second = runs(env, agent_id)[-1]
    assert second["run"] == 2 and second["outcome"] is None and second["ended_at"] is not None


# --- 14: _recover_run ----


def recover(rig: Rig, error: StartError) -> tuple[int | None, str | None]:
    sup = rig.supervisor
    sup.store = Store.open(rig.state, create=False)
    try:
        return chain._recover_run(  # pyright: ignore[reportPrivateUsage]
            sup, sup.backend(rig.agent()), rig.session, final_of(), error, None, "tok2"
        )
    finally:
        sup.store.close()


def test_recover_run_leaves_another_actors_live_run_alone(
    rig_for: Callable[..., Rig], fakebin: Fakebin
) -> None:
    rig = rig_for()
    fakebin.script(
        "claude-session",
        "status",
        merge={"current_run": 5, "pending": False, "run": {"run": 5, "phase": "session"}},
    )
    error = StartError("failed", "x", maybe_allocated=True)
    assert recover(rig, error) == (None, None)
    assert not [a for a in fakebin.argvs("claude-session") if a[0] == "stop"]
    assert [r["run"] for r in rig_runs(rig)] == [1]


def test_recover_run_cancels_the_pending_run_it_left_behind(
    rig_for: Callable[..., Rig], fakebin: Fakebin
) -> None:
    rig = rig_for()
    fakebin.script(
        "claude-session",
        "status",
        merge={"current_run": 2, "pending": True, "run": {"run": 2, "pending": True}},
    )
    error = StartError("failed", "x", maybe_allocated=True)
    assert recover(rig, error) == (2, "canceled")
    assert [a for a in fakebin.argvs("claude-session") if a[0] == "stop"]
    second = rig_runs(rig)[-1]
    assert second["end_reason"] == "canceled" and second["outcome"]["outcome"] == "not-started"


def test_recover_run_retries_status_once_and_then_warns_of_an_orphan(
    rig_for: Callable[..., Rig], fakebin: Fakebin
) -> None:
    rig = rig_for()
    fakebin.script("claude-session", "status", exit=1, error={"code": "failed", "message": "x"})
    assert recover(rig, StartError("failed", "x", maybe_allocated=True)) == (None, None)
    assert len([a for a in fakebin.argvs("claude-session") if a[0] == "status"]) == 2
    assert any("orphan" in str(r.get("message")) for r in log_records(rig.state))


# --- 15: a start that failed after its run ended in the launch phase ---


def test_start_failure_with_an_ended_run_records_the_real_end_reason(
    rig_for: Callable[..., Rig], fakebin: Fakebin
) -> None:
    rig = rig_for()
    bump_run(fakebin, rig.session, 2)  # the failed start allocated run 2, which then ended
    fakebin.script("claude-session", "status", argv_contains=["--run=2"], outcome="vetoed")
    script_unreported_first(fakebin)
    fakebin.script("claude-session", "start", argv_contains=["--dry-run"])
    fakebin.script(
        "claude-session",
        "start",
        exit=12,
        error={
            "code": "launch_failed",
            "message": "vetoed",
            "csession_id": rig.session,
            "run": 2,
            "end_reason": "hook_veto",
        },
    )
    rig.supervisor.run()
    (park,) = [p for n, p in rig_events(rig) if n == "post-park"]
    assert park["chain_failure"]["run"] == 2
    assert park["chain_failure"]["end_reason"] == "hook_veto"
    assert rig_runs(rig)[-1]["end_reason"] == "hook_veto"
    assert not [a for a in fakebin.argvs("claude-session") if a[0] == "stop"]


# --- 16: the prompt file ----


def test_a_loose_prompts_directory_is_tightened(tmp_path: Path) -> None:
    base = tmp_path / "prompts"
    directory = base / "ag-abcdefgh"
    directory.mkdir(parents=True)
    os.chmod(base, 0o777)
    os.chmod(directory, 0o777)
    path = chain.write_prompt(str(tmp_path), "ag-abcdefgh")
    assert os.stat(base).st_mode & 0o077 == 0
    assert os.stat(directory).st_mode & 0o077 == 0
    assert os.stat(path).st_mode & 0o777 == 0o600


def test_a_failed_prompt_write_leaves_no_file_and_no_descriptor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(fd: int, *args: Any, **kwargs: Any) -> Any:
        raise OSError("no space")

    before = len(os.listdir("/proc/self/fd"))
    with monkeypatch.context() as patch:
        patch.setattr(os, "fdopen", broken)
        with pytest.raises(OSError):
            chain.write_prompt(str(tmp_path), "ag-abcdefgh")
    assert os.listdir(tmp_path / "prompts" / "ag-abcdefgh") == []
    assert len(os.listdir("/proc/self/fd")) == before


# --- 17: one read of the outcome file ----


def test_the_outcome_file_is_read_once_per_run(
    rig_for: Callable[..., Rig], monkeypatch: pytest.MonkeyPatch
) -> None:
    rig = rig_for()
    reads: list[str] = []
    real = session_backend.read_outcome_file

    def counting(path: str, *args: Any) -> Any:
        reads.append(path)
        return real(path, *args)

    monkeypatch.setattr(session_backend, "read_outcome_file", counting)
    monkeypatch.setattr(supervise, "read_outcome_file", counting, raising=False)
    rig.supervisor.run()
    assert len(reads) == 1


# --- 18: a refusal that was not applied ----


def test_a_lost_teardown_refusal_is_logged(
    rig_for: Callable[..., Rig], monkeypatch: pytest.MonkeyPatch
) -> None:
    rig = rig_for(ephemeral=True)
    rig.supervisor.teardown = lambda store, agent_id: "live"
    monkeypatch.setattr(lifecycle, "refuse_ephemeral_teardown", lambda *a, **k: None)
    rig.supervisor.run()
    assert any("not applied" in str(r.get("message")) for r in log_records(rig.state))


# --- 21: more chain coverage ----


def test_a_third_party_run_in_its_session_dir_gets_the_report_resume(
    env: Env, fakebin: Fakebin
) -> None:
    script_unreported_first(fakebin)
    done = env.run(*LAUNCH, "--reads-third-party", "--cwd", "session-dir", "--wait")
    assert done.returncode == 0, (done.stdout, done.stderr)
    assert len(real_starts(fakebin)) == 2


@pytest.mark.parametrize("cwd_kind", ["real", "none"])
def test_a_third_party_resume_outside_its_session_dir_is_refused(
    env: Env, fakebin: Fakebin, tmp_path: Path, cwd_kind: str
) -> None:
    script_unreported_first(fakebin)
    fakebin.script("claude-session", "start", argv_contains=["--dry-run"], times=1)
    fakebin.script(
        "claude-session",
        "start",
        argv_contains=["--dry-run"],
        merge={"profile": {"reads_third_party": True}},
    )
    args = [*LAUNCH, "--wait"]
    if cwd_kind == "real":
        args += ["--cwd", str(tmp_path)]
    done = env.run(*args)
    agent_id = result_json(done)["agent"]["agent_id"]
    failure = chain_failure_of(env, agent_id)
    assert failure is not None
    assert failure["tool"] == "agent-spawn" and failure["reason"] == "third_party_cwd"
    assert len(real_starts(fakebin)) == 1  # no real start was made
    log = [json.loads(x) for x in (env.state_dir / "log.jsonl").read_text().splitlines()]
    assert [r for r in log if r["event"] == "third_party_refused"]


@pytest.mark.parametrize("outcome", ["not-started", "vetoed", "escalated"])
def test_a_chain_run_that_ends_in_its_launch_phase_parks_per_run_one(
    env: Env, fakebin: Fakebin, outcome: str
) -> None:
    fakebin.script("claude-session", "exec", times=1, outcome="exited-unreported")
    fakebin.script("claude-session", "exec", times=1, outcome=outcome)
    done = env.run(*LAUNCH, "--wait")
    agent_id = result_json(done)["agent"]["agent_id"]
    row = env.agent(agent_id)
    assert row["park_reason"] == "unreported" and row["needs_attention"] == "unreported"
    assert row["escalated_by"] is None and row["last_outcome"]["run"] == 1
    failure = chain_failure_of(env, agent_id)
    assert failure is not None and failure["run"] == 2 and failure["tool"] == "claude-session"
    assert runs(env, agent_id)[1]["outcome"]["outcome"] == outcome


@pytest.mark.parametrize(
    ("sig", "reason", "code"),
    [(signal.SIGHUP, "forwarded_hup", 15), (signal.SIGINT, "forwarded_int", 15)],
)
def test_sighup_and_sigint_reach_exec_of_a_headless_run(
    env: Env,
    fakebin: Fakebin,
    proc_tracker: ProcTracker,
    sig: signal.Signals,
    reason: str,
    code: int,
) -> None:
    script_live_run(fakebin)
    fakebin.script("claude-session", "exec", hang=True)
    body = result_json(env.run(*LAUNCH))
    agent_id = body["agent"]["agent_id"]
    track_agent(env, proc_tracker, agent_id)
    env.wait_for(lambda: bool(fakebin.records("hang_ready")), "exec to be waiting")
    supervisor = env.agent(agent_id)["supervisor"]
    handle = procs.open_verified(supervisor)
    assert handle is not None
    with handle:
        assert handle.send(sig)
    env.wait_for(lambda: env.agent(agent_id)["status"] == "parked", "the park", timeout=30)
    row = env.agent(agent_id)
    assert row["park_reason"] == "killed" and row["needs_attention"] is None
    assert row["last_end_reason"] == reason
    execs = [i for i in fakebin.invocations("claude-session") if i.argv[:1] == ["exec"]]
    assert len(execs) == 1 and execs[0].signals == [sig]
    assert len(real_starts(fakebin)) == 1  # no report resume after a forwarded signal


def test_a_repeated_signal_is_forwarded_once(
    env: Env, fakebin: Fakebin, proc_tracker: ProcTracker
) -> None:
    script_live_run(fakebin)
    fakebin.script("claude-session", "exec", hang=True)
    body = result_json(env.run(*LAUNCH))
    agent_id = body["agent"]["agent_id"]
    track_agent(env, proc_tracker, agent_id)
    env.wait_for(lambda: bool(fakebin.records("hang_ready")), "exec to be waiting")
    handle = procs.open_verified(env.agent(agent_id)["supervisor"])
    assert handle is not None
    with handle:
        assert handle.send(signal.SIGTERM)
        handle.send(signal.SIGTERM)  # may find the supervisor busy or gone: either is fine
    env.wait_for(lambda: env.agent(agent_id)["status"] == "parked", "the park", timeout=30)
    log = [json.loads(x) for x in (env.state_dir / "log.jsonl").read_text().splitlines()]
    sent = [r for r in log if r["event"] == "signal_sent" and r.get("signal") == "SIGTERM"]
    assert len(sent) == 1


def test_recover_run_after_a_start_that_allocated_a_run(
    rig_for: Callable[..., Rig], fakebin: Fakebin
) -> None:
    rig = rig_for()
    bump_run(fakebin, rig.session, 2)
    fakebin.script("claude-session", "status", outcome="vetoed")
    error = StartError("launch_failed", "x", csession_id=rig.session, run=2)
    assert recover(rig, error) == (2, "hook_veto")  # the run ended itself: no stop
    assert not [a for a in fakebin.argvs("claude-session") if a[0] == "stop"]
    second = rig_runs(rig)[-1]
    assert second["run"] == 2 and second["end_reason"] == "hook_veto"


def test_sigterm_after_exec_exited_skips_the_resume(
    rig_for: Callable[..., Rig], fakebin: Fakebin
) -> None:
    rig = rig_for()
    sup = rig.supervisor
    script_unreported_first(fakebin)
    original = sup.end_of_run

    def end_of_run(*args: Any, **kwargs: Any) -> Final:
        final = original(*args, **kwargs)
        sup.signals.received.append(signal.SIGTERM)  # exec is gone, the resume not yet started
        return final

    sup.end_of_run = end_of_run  # type: ignore[method-assign]
    sup.run()
    row = rig.agent()
    assert row["status"] == "parked" and row["park_reason"] == "unreported"
    assert not [a for a in fakebin.argvs("claude-session") if a[0] == "start"]
    assert [r["run"] for r in rig_runs(rig)] == [1]


# --- 16: hardening of signal forwarding and run recovery ----


def test_a_cancel_pending_failure_still_parks_the_chain_failure(
    rig_for: Callable[..., Rig], fakebin: Fakebin, monkeypatch: pytest.MonkeyPatch
) -> None:
    rig = rig_for()
    sup = rig.supervisor
    script_unreported_first(fakebin)
    original = sup.spawn_exec
    spawned: list[int] = []

    def spawn(row: dict[str, Any], session: str) -> subprocess.Popen[bytes]:
        spawned.append(1)
        if len(spawned) == 2:
            raise OSError("no exec")
        return original(row, session)

    def boom(*args: Any, **kwargs: Any) -> None:
        raise StoreError("store down")

    sup.spawn_exec = spawn  # type: ignore[method-assign]
    monkeypatch.setattr(chain, "cancel_pending", boom)
    sup.run()
    assert rig.agent()["status"] == "parked"
    (park,) = [p for n, p in rig_events(rig) if n == "post-park"]
    assert park["chain_failure"]["tool"] == "agent-spawn"


def test_the_forwarder_blocks_the_signals_around_its_check_and_mark(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    forwarder = procs.SignalForwarder(procs.ProcessIdentity(pid=1, start_time=1, boot_id="b"))
    calls: list[tuple[int, set[int]]] = []
    real = signal.pthread_sigmask

    def spy(how: int, mask: Any) -> Any:
        calls.append((how, set(mask)))
        return real(how, mask)

    monkeypatch.setattr(signal, "pthread_sigmask", spy)
    monkeypatch.setattr(procs, "forward_signal", lambda *a, **k: None)
    forwarder.forward(signal.SIGTERM)
    assert calls[0][0] == signal.SIG_BLOCK
    assert {signal.SIGTERM, signal.SIGHUP, signal.SIGINT} <= calls[0][1]
    assert calls[-1][0] in (signal.SIG_SETMASK, signal.SIG_UNBLOCK)


def test_an_error_after_the_main_signal_was_delivered_counts_as_sent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = procs.ProcessIdentity(pid=1, start_time=1, boot_id="b")
    forwarder = procs.SignalForwarder(identity)
    sent: list[int] = []

    class Handle:
        def __enter__(self) -> Handle:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def send(self, sig: int) -> bool:
            sent.append(sig)
            if sig == signal.SIGCONT:
                raise OSError("cont failed")
            return True

    monkeypatch.setattr(procs, "open_verified", lambda *a, **k: Handle())
    outcome = forwarder.forward(signal.SIGTERM)
    assert outcome is not None and outcome.sent
    assert forwarder.forward(signal.SIGTERM) is None  # marked: not sent twice
    assert sent == [signal.SIGTERM, signal.SIGCONT]


def test_a_failed_main_send_is_unmarked(monkeypatch: pytest.MonkeyPatch) -> None:
    forwarder = procs.SignalForwarder(procs.ProcessIdentity(pid=1, start_time=1, boot_id="b"))

    def open_fails(*args: Any, **kwargs: Any) -> None:
        raise OSError("open failed")

    monkeypatch.setattr(procs, "open_verified", open_fails)
    with pytest.raises(OSError):
        forwarder.forward(signal.SIGTERM)
    monkeypatch.setattr(procs, "open_verified", lambda *a, **k: None)
    outcome = forwarder.forward(signal.SIGTERM)
    assert outcome is not None and outcome.reason == "gone"


def test_the_handler_swallows_any_forward_error_and_keeps_the_signal_queued() -> None:
    signals = Signals()

    class Broken:
        def forward(self, sig: int) -> None:
            raise ValueError("bad")

    signals.forwarder = Broken()  # type: ignore[assignment]
    signals.handle(signal.SIGTERM, None)
    assert signals.received == [signal.SIGTERM] and signals.sent == []


def bump_run(fakebin: Fakebin, session: str, run: int) -> None:
    """Make the fake claude-session know ``run`` (as if a start had allocated it)."""
    path = Path(str(fakebin.log) + ".sessions.json")
    sessions = json.loads(path.read_text())
    sessions[session]["run"] = run
    path.write_text(json.dumps(sessions))


def _ended_run_error(rig: Rig) -> StartError:
    return StartError("launch_failed", "x", csession_id=rig.session, run=2)


@pytest.mark.parametrize(
    ("outcome", "reason", "code"), [("vetoed", "hook_veto", 11), ("escalated", "hook_escalate", 13)]
)
def test_recover_run_records_the_real_outcome_of_an_ended_run(
    rig_for: Callable[..., Rig], fakebin: Fakebin, outcome: str, reason: str, code: int
) -> None:
    rig = rig_for()
    bump_run(fakebin, rig.session, 2)
    fakebin.script("claude-session", "status", outcome=outcome)
    assert recover(rig, _ended_run_error(rig)) == (2, reason)
    assert not [a for a in fakebin.argvs("claude-session") if a[0] == "stop"]
    row = rig_runs(rig)[-1]
    assert row["end_reason"] == reason and row["exit_code"] == code
    assert row["outcome"]["outcome"] == outcome


def test_recover_run_with_unreadable_status_records_no_outcome_and_does_not_stop(
    rig_for: Callable[..., Rig], fakebin: Fakebin
) -> None:
    rig = rig_for()
    bump_run(fakebin, rig.session, 2)
    fakebin.script("claude-session", "status", exit=1, error={"code": "failed", "message": "x"})
    assert recover(rig, _ended_run_error(rig)) == (2, None)
    assert not [a for a in fakebin.argvs("claude-session") if a[0] == "stop"]
    row = rig_runs(rig)[-1]
    assert row["run"] == 2 and row["outcome"] is None and row["end_reason"] is None
    assert any("without an outcome" in str(r.get("message")) for r in log_records(rig.state))
