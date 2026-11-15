"""The supervisor's run chain: the automatic report resume, the chain rule, started polling,
ephemeral hand-over and the signal chain with a real child (process tests against the fakes)."""

from __future__ import annotations

import json
import os
import signal
import sqlite3
import subprocess
from pathlib import Path
from typing import Any

import pytest

from agent_spawn import procs
from tests import schema_registry
from tests.conftest import Isolation
from tests.fakebin.harness import Fakebin
from tests.launch_support import (
    Env,
    result_json,
    script_hung_exec,
    script_live_run,
    track_agent,
)
from tests.procguard import ProcTracker

LAUNCH = ("launch", "--mode", "headless", "--skill", "do-it", "--json")


@pytest.fixture
def env(fakebin: Fakebin, isolation_env: Isolation) -> Env:
    return Env(fakebin, isolation_env)


def events(env: Env, agent_id: str) -> list[tuple[str, dict[str, Any]]]:
    """The agent's outbox events, oldest first (read-only)."""
    db = sqlite3.connect(f"file:{env.state_dir / 'state.db'}?mode=ro", uri=True)
    try:
        rows = db.execute(
            "SELECT event, payload FROM events WHERE agent_id = ? ORDER BY event_id", (agent_id,)
        ).fetchall()
    finally:
        db.close()
    return [(name, json.loads(payload)) for name, payload in rows]


def validate_event(name: str, payload: dict[str, Any], agent_id: str) -> None:
    """The stored body plus the fields delivery adds, against the hook's schema."""
    full = {
        **payload,
        "tool": "agent-spawn",
        "hook": name,
        "object_type": "agent",
        "object_id": agent_id,
        "event_id": 1,
        "ts": "2026-11-14T00:00:00.000Z",
    }
    schema_registry.validate(full, f"agent-spawn/hooks/{name}.json")


def runs(env: Env, agent_id: str) -> list[dict[str, Any]]:
    with env.store() as store:
        return store.get_runs(agent_id)


def script_unreported_first(fakebin: Fakebin, *, times: int = 1) -> None:
    """The first `exec`s end `exited-unreported`; later ones end `done`."""
    fakebin.script("claude-session", "exec", times=times, outcome="exited-unreported")


def real_starts(fakebin: Fakebin) -> list[list[str]]:
    return [a for a in fakebin.argvs("claude-session") if a[0] == "start" and "--dry-run" not in a]


def dry_runs(fakebin: Fakebin) -> list[list[str]]:
    return [a for a in fakebin.argvs("claude-session") if a[0] == "start" and "--dry-run" in a]


# --- the automatic report resume ------------------------------------------------------------


def test_report_resume_runs_a_second_run_and_its_outcome_applies(
    env: Env, fakebin: Fakebin
) -> None:
    script_unreported_first(fakebin)
    done = env.run(*LAUNCH, "--wait")
    body = result_json(done)
    assert done.returncode == 0, (done.stdout, done.stderr)
    schema_registry.validate(body, "agent-spawn/cli/launch.json")
    agent_id = body["agent"]["agent_id"]
    assert body["outcome"] == "done"
    row = env.agent(agent_id)
    assert row["status"] == "parked" and row["park_reason"] == "done"
    assert row["needs_attention"] is None and row["run"] == 2
    first, second = runs(env, agent_id)
    assert first["outcome"]["outcome"] == "exited-unreported" and not first["auto_resumed"]
    assert second["auto_resumed"] is True and second["outcome"]["auto_resumed"] is True
    assert second["outcome"]["outcome"] == "done"
    # start --resume with the fixed prompt, a classification dry run before it
    resume = real_starts(fakebin)[1]
    session = row["csession_id"]
    assert f"--resume={session}" in resume and "--mode=headless" in resume
    prompt = next(a for a in resume if a.startswith("--prompt-file=")).split("=", 1)[1]
    assert prompt.startswith(f"{env.state_dir}/prompts/{agent_id}/")
    assert not os.path.exists(prompt)  # the private copy is deleted once start returned
    assert len(dry_runs(fakebin)) == 2
    assert [a for a in resume if a.startswith("--label=agent=")] == [f"--label=agent={agent_id}"]
    # post-launch for the resumed run says auto_resumed
    launches = [p for name, p in events(env, agent_id) if name == "post-launch"]
    assert [(p["run"], p["auto_resumed"]) for p in launches] == [(1, False), (2, True)]
    for name, payload in events(env, agent_id):
        validate_event(name, payload, agent_id)


def test_report_prompt_file_is_private_and_fixed(tmp_path: Path) -> None:
    from agent_spawn import chain

    path = chain.write_prompt(str(tmp_path), "ag-abcdefgh")
    assert Path(path).read_text() == (
        "Your last run ended without a status report. "
        "File your status report now with claude-session report.\n"
    )
    assert os.stat(path).st_mode & 0o777 == 0o600
    assert os.stat(os.path.dirname(path)).st_mode & 0o777 == 0o700
    chain.remove_prompt(path)
    assert not os.path.exists(path) and not os.path.exists(os.path.dirname(path))


def test_report_prompt_refuses_a_symlinked_prompts_directory(tmp_path: Path) -> None:
    from agent_spawn import chain

    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (tmp_path / "prompts").symlink_to(elsewhere)
    with pytest.raises(OSError):
        chain.write_prompt(str(tmp_path), "ag-abcdefgh")
    assert list(elsewhere.iterdir()) == []


def test_second_unreported_run_parks_unreported_and_resumes_only_once(
    env: Env, fakebin: Fakebin
) -> None:
    script_unreported_first(fakebin, times=2)
    done = env.run(*LAUNCH, "--wait")
    body = result_json(done)
    assert done.returncode == 14, (done.stdout, done.stderr)
    agent_id = body["agent"]["agent_id"]
    row = env.agent(agent_id)
    assert row["status"] == "parked" and row["park_reason"] == "unreported"
    assert row["needs_attention"] == "unreported"
    assert len(real_starts(fakebin)) == 2  # the launch and exactly one resume
    assert [r["run"] for r in runs(env, agent_id)] == [1, 2]


def test_report_resume_disabled_by_config_parks_unreported(
    fakebin: Fakebin, isolation_env: Isolation
) -> None:
    env = Env(fakebin, isolation_env, "auto_report_resume: false\n")
    script_unreported_first(fakebin)
    done = env.run(*LAUNCH, "--wait")
    body = result_json(done)
    assert done.returncode == 14
    row = env.agent(body["agent"]["agent_id"])
    assert row["park_reason"] == "unreported" and row["needs_attention"] == "unreported"
    assert len(real_starts(fakebin)) == 1


def test_report_resume_not_offered_to_a_done_run(env: Env, fakebin: Fakebin) -> None:
    done = env.run(*LAUNCH, "--wait")
    assert done.returncode == 0
    assert len(real_starts(fakebin)) == 1


# --- the chain rule ---------------------------------------------------------------------------


def chain_failure_of(env: Env, agent_id: str) -> dict[str, Any] | None:
    parks = [p for name, p in events(env, agent_id) if name == "post-park"]
    assert len(parks) == 1
    validate_event("post-park", parks[0], agent_id)
    return parks[0]["chain_failure"]


def test_vetoed_report_resume_parks_per_the_triggering_run(env: Env, fakebin: Fakebin) -> None:
    script_unreported_first(fakebin)
    fakebin.script("claude-session", "start", argv_contains=["--dry-run"])
    fakebin.script("claude-session", "start", times=1)  # the launch itself
    fakebin.script("claude-session", "start", error={"code": "vetoed", "message": "no"}, exit=11)
    done = env.run(*LAUNCH, "--wait")
    body = result_json(done)
    # The waiter's result is the triggering run's.
    assert done.returncode == 14, (done.stdout, done.stderr)
    assert body["outcome"] == "exited-unreported"
    agent_id = body["agent"]["agent_id"]
    row = env.agent(agent_id)
    assert row["status"] == "parked" and row["park_reason"] == "unreported"
    assert row["needs_attention"] == "unreported" and row["escalated_by"] is None
    assert row["last_outcome"]["outcome"] == "exited-unreported" and row["last_outcome"]["run"] == 1
    assert chain_failure_of(env, agent_id) == {
        "tool": "claude-session",
        "reason": "vetoed",
        "run": None,
        "end_reason": None,
    }
    escalations = [p for name, p in events(env, agent_id) if name == "post-escalate"]
    assert [p["chain_failure"]["reason"] for p in escalations] == ["vetoed"]
    assert [r["run"] for r in runs(env, agent_id)] == [1]


def test_refused_classification_of_the_resume_is_a_chain_failure(
    env: Env, fakebin: Fakebin
) -> None:
    script_unreported_first(fakebin)
    fakebin.script("claude-session", "start", argv_contains=["--dry-run"], times=1)
    fakebin.script(
        "claude-session",
        "start",
        argv_contains=["--dry-run"],
        merge={"would_refuse": "live"},
        exit=20,
    )
    done = env.run(*LAUNCH, "--wait")
    body = result_json(done)
    assert done.returncode == 14, (done.stdout, done.stderr)
    agent_id = body["agent"]["agent_id"]
    failure = chain_failure_of(env, agent_id)
    assert failure is not None and failure["tool"] == "claude-session"
    assert failure["run"] is None
    assert len(real_starts(fakebin)) == 1  # nothing was started
    assert env.agent(agent_id)["park_reason"] == "unreported"


def test_resume_run_that_ends_in_its_launch_phase_stops_the_chain(
    env: Env, fakebin: Fakebin
) -> None:
    fakebin.script("claude-session", "exec", times=1, outcome="exited-unreported")
    fakebin.script("claude-session", "exec", times=1, outcome="not-started")
    done = env.run(*LAUNCH, "--wait")
    body = result_json(done)
    assert done.returncode == 14, (done.stdout, done.stderr)
    agent_id = body["agent"]["agent_id"]
    row = env.agent(agent_id)
    assert row["park_reason"] == "unreported" and row["needs_attention"] == "unreported"
    assert row["last_outcome"]["run"] == 1  # the triggering run's
    failure = chain_failure_of(env, agent_id)
    assert failure is not None
    assert failure["tool"] == "claude-session" and failure["run"] == 2
    assert failure["reason"] == "not_started" and failure["end_reason"] is not None
    first, second = runs(env, agent_id)
    assert second["outcome"]["outcome"] == "not-started"  # recorded with its own outcome


def test_classification_mismatch_of_the_resume_cancels_it_and_parks(
    env: Env, fakebin: Fakebin
) -> None:
    script_unreported_first(fakebin)
    fakebin.script("claude-session", "status", times=1)  # the launcher's own check
    fakebin.script(
        "claude-session",
        "status",
        merge={"run": {"reads_third_party": True}, "tainted_by_run": 2},
    )
    done = env.run(*LAUNCH, "--wait")
    body = result_json(done)
    assert done.returncode == 14, (done.stdout, done.stderr)
    agent_id = body["agent"]["agent_id"]
    failure = chain_failure_of(env, agent_id)
    assert failure is not None and failure["tool"] == "agent-spawn" and failure["run"] == 2
    assert failure["reason"] in ("classification_mismatch", "classification_unknown")
    stops = [a for a in fakebin.argvs("claude-session") if a[0] == "stop"]
    assert stops, "the pending run was not canceled"
    execs = [a for a in fakebin.argvs("claude-session") if a[0] == "exec"]
    assert len(execs) == 1  # the canceled run was never adopted
    assert runs(env, agent_id)[-1]["outcome"]["outcome"] == "not-started"


def test_sigterm_during_the_resume_start_cancels_the_pending_run(
    env: Env, fakebin: Fakebin, proc_tracker: ProcTracker
) -> None:
    script_unreported_first(fakebin)
    fakebin.script("claude-session", "start", argv_contains=["--dry-run"])
    fakebin.script("claude-session", "start", times=1)
    fakebin.script("claude-session", "start", sleep=4)  # the resume's start, in flight
    done = env.run(*LAUNCH)  # without --wait: returns once the run is live
    body = result_json(done)
    agent_id = body["agent"]["agent_id"]
    track_agent(env, proc_tracker, agent_id)
    env.wait_for(lambda: len(real_starts(fakebin)) == 2, "the resume's start")
    supervisor = env.agent(agent_id)["supervisor"]
    handle = procs.open_verified(supervisor)
    assert handle is not None
    with handle:
        assert handle.send(signal.SIGTERM)  # no `exec` child right now
    env.wait_for(lambda: env.agent(agent_id)["status"] == "parked", "the park", timeout=30)
    row = env.agent(agent_id)
    assert row["park_reason"] == "unreported" and row["needs_attention"] == "unreported"
    assert [a for a in fakebin.argvs("claude-session") if a[0] == "stop"]
    assert len([a for a in fakebin.argvs("claude-session") if a[0] == "exec"]) == 1
    assert [p["chain_failure"] for n, p in events(env, agent_id) if n == "post-park"] == [None]
    env.wait_for(lambda: not procs.is_alive(supervisor), "the supervisor to exit")


# --- started polling --------------------------------------------------------------------------


def test_run_at_the_trust_dialog_goes_live_by_polling_alone(
    env: Env, fakebin: Fakebin, proc_tracker: ProcTracker
) -> None:
    script_live_run(fakebin)
    script_hung_exec(fakebin)
    fakebin.script(
        "claude-session",
        "status",
        merge={"state": "needs-input", "run": {"state": "needs-input", "phase": "session"}},
    )
    done = env.run(*LAUNCH)
    body = result_json(done)
    agent_id = body["agent"]["agent_id"]
    track_agent(env, proc_tracker, agent_id)
    assert done.returncode == 0 and body["agent"]["status"] == "live"
    with env.store() as store:
        row = store.get_agent(agent_id)
        count = store._conn.execute("SELECT COUNT(*) FROM relay_events").fetchone()[0]  # noqa: SLF001
    assert row is not None and row["needs_attention"] is None
    assert count == 0  # no relayed event was involved
    started = [p for n, p in events(env, agent_id) if n == "post-launch"]
    assert [p["run"] for p in started] == [1]


# --- ephemeral --------------------------------------------------------------------------------


def test_ephemeral_done_hands_over_and_refusal_parks_done(env: Env, fakebin: Fakebin) -> None:
    done = env.run(*LAUNCH, "--ephemeral", "--wait")
    body = result_json(done)
    assert done.returncode == 0, (done.stdout, done.stderr)
    agent_id = body["agent"]["agent_id"]
    row = env.agent(agent_id)
    # TODO(#18): teardown itself is not implemented yet: the hand-over is refused and logged.
    assert row["status"] == "parked" and row["park_reason"] == "done"
    assert row["teardown_step"] is None and row["teardown_proc"] is None
    names = [n for n, _ in events(env, agent_id)]
    assert names.count("post-park") == 1
    park = next(p for n, p in events(env, agent_id) if n == "post-park")
    assert park["teardown_refused"] is True
    validate_event("post-park", park, agent_id)
    log = (env.state_dir / "supervise" / f"{agent_id}.log").read_text()
    assert "Traceback" not in log


def test_a_non_ephemeral_done_park_does_not_refuse_teardown(env: Env) -> None:
    body = result_json(env.run(*LAUNCH, "--wait"))
    park = next(p for n, p in events(env, body["agent"]["agent_id"]) if n == "post-park")
    assert park["teardown_refused"] is False


# --- the signal chain with a real child --------------------------------------------------------


def test_sigterm_chain_reaches_exec_and_its_child_even_when_exec_is_stopped(
    env: Env, fakebin: Fakebin, proc_tracker: ProcTracker
) -> None:
    script_live_run(fakebin)
    fakebin.script("claude-session", "exec", hang=True, child=True)
    waiter = proc_tracker.popen(
        env.argv(*LAUNCH, "--wait"),
        env=env.env(),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    env.wait_for(lambda: bool(fakebin.records("child_started")), "the child of exec")
    env.wait_for(lambda: any(a["status"] == "live" for a in _agents(env)), "a live agent")
    agent_id = env.agents()[0]["agent_id"]
    track_agent(env, proc_tracker, agent_id)
    started = fakebin.records("child_started")[0]
    child = procs.read_identity(started["pid"])
    assert child is not None
    proc_tracker.record(child)
    env.wait_for(lambda: bool(fakebin.records("hang_ready")), "exec to be waiting")
    # A stopped exec never acts on SIGTERM unless SIGCONT follows it (Signals).
    exec_identity = env.agent(agent_id)["exec_proc"]
    handle = procs.open_verified(exec_identity)
    assert handle is not None
    with handle:
        assert handle.send(signal.SIGSTOP)
    waiter.send_signal(signal.SIGTERM)  # the exact Popen child
    out, _err = waiter.communicate(timeout=60)
    body = json.loads(out)
    assert waiter.returncode == 15
    assert body["outcome"] == "killed" and body["end_reason"] == "forwarded_term"
    row = env.agent(agent_id)
    assert row["status"] == "parked" and row["park_reason"] == "killed"
    assert row["needs_attention"] is None
    got = Path(started["signals_file"]).read_text().split()
    assert "SIGTERM" in got  # exec's child got the signal at the end of the chain
    env.wait_for(lambda: not procs.is_alive(child), "the child to end")
    execs = [i for i in fakebin.invocations("claude-session") if i.argv[:1] == ["exec"]]
    assert len(execs) == 1 and signal.SIGTERM in execs[0].signals


def _agents(env: Env) -> list[dict[str, Any]]:
    try:
        return env.agents()
    except Exception:  # noqa: BLE001 - the database may not exist yet
        return []
