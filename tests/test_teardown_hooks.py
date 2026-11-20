"""The ``pre-teardown`` hook (#32; interface.md, Teardown order step 3 and Escalation overrides):
veto 11, escalation 13 and the override, the payload, the return to ``parked`` unchanged, the
retry rule, the live re-check after a slow hook, and the supervisor's ephemeral teardown."""

from __future__ import annotations

import json
import os
import signal
import subprocess
from pathlib import Path
from typing import Any

import pytest

from tests import schema_registry
from tests.conftest import Isolation
from tests.fakebin.harness import Fakebin
from tests.launch_support import Env, result_json
from tests.test_teardown import dead, events, forget_calls, seed, td
from tests.test_teardown_worktrees import audit_records
from tests.test_worktrees_launch import wtctl_calls
from tests.test_worktrees_resume import launched_in_worktree

HUMAN = ("--actor", "human:tester")
VETO = "pre-teardown=echo nope >&2; exit 1"
ESCALATE = "pre-teardown=echo careful >&2; exit 13"


@pytest.fixture
def env(fakebin: Fakebin, isolation_env: Isolation) -> Env:
    return Env(fakebin, isolation_env)


def unchanged(env: Env, agent_id: str, before: dict[str, Any], files: bool = True) -> None:
    """The agent is ``parked`` exactly as before the attempt (but for the bookkeeping fields)."""
    after = env.agent(agent_id)
    for key in ("updated",):
        after.pop(key, None)
        before = {k: v for k, v in before.items() if k != key}
    assert after["status"] == "parked"
    assert after["teardown_step"] is None and after["teardown_proc"] is None
    assert after == before
    if files:
        assert (env.state_dir / "prompts" / agent_id).exists()
        assert (env.state_dir / "outcomes" / agent_id).exists()


# --- veto and escalation ------------------------------------------------------------------------


def test_a_veto_exits_11_and_leaves_the_agent_parked_unchanged(env: Env, fakebin: Fakebin) -> None:
    agent_id = seed(env, fakebin)
    before = env.agent(agent_id)
    done = td(env, agent_id, "--hook", VETO)
    body = result_json(done)
    assert done.returncode == 11, (done.stdout, done.stderr)
    schema_registry.validate(body, "agent-spawn/cli/error.json")
    assert body["error"]["code"] == "vetoed" and body["error"]["hook"] == "pre-teardown"
    assert "nope" in body["error"]["hook_stderr"] and body["error"]["agent_id"] == agent_id
    unchanged(env, agent_id, before)
    assert env.agent(agent_id)["hooks"] == before["hooks"]  # this call's --hook goes with it
    assert forget_calls(fakebin) == []


def test_an_escalation_exits_13_and_leaves_the_agent_parked_unchanged(
    env: Env, fakebin: Fakebin
) -> None:
    agent_id = seed(env, fakebin)
    before = env.agent(agent_id)
    done = td(env, agent_id, "--hook", ESCALATE)
    body = result_json(done)
    assert done.returncode == 13, (done.stdout, done.stderr)
    schema_registry.validate(body, "agent-spawn/cli/error.json")
    assert body["error"]["code"] == "escalated" and "careful" in body["error"]["hook_stderr"]
    unchanged(env, agent_id, before)
    assert forget_calls(fakebin) == []


def test_a_hook_timeout_vetoes(fakebin: Fakebin, isolation_env: Isolation) -> None:
    config = 'hooks:\n  pre-teardown:\n    - cmd: "sleep 30"\n      timeout: 1\n'
    cfg_env = Env(fakebin, isolation_env, config)
    agent_id = seed(cfg_env, fakebin)
    done = td(cfg_env, agent_id)
    assert done.returncode == 11, (done.stdout, done.stderr)
    assert cfg_env.agent(agent_id)["status"] == "parked"
    assert forget_calls(fakebin) == []


def test_a_config_defined_hook_vetoes(fakebin: Fakebin, isolation_env: Isolation) -> None:
    cfg_env = Env(fakebin, isolation_env, 'hooks:\n  pre-teardown:\n    - cmd: "exit 3"\n')
    agent_id = seed(cfg_env, fakebin)
    before = cfg_env.agent(agent_id)
    done = td(cfg_env, agent_id)
    assert done.returncode == 11
    unchanged(cfg_env, agent_id, before)


def test_a_veto_with_a_worktree_makes_no_wtctl_removal_call(env: Env, fakebin: Fakebin) -> None:
    agent_id = launched_in_worktree(env)
    before = env.agent(agent_id)
    mark = len(fakebin.argvs("wtctl"))
    done = td(env, agent_id, "--hook", VETO)
    assert done.returncode == 11
    assert fakebin.argvs("wtctl")[mark:] == []  # not even a `show`
    unchanged(env, agent_id, before, files=False)
    assert forget_calls(fakebin) == [] and wtctl_calls(fakebin, "remove") == []
    assert audit_records(env) == []


def test_a_passing_hook_lets_the_teardown_finish(env: Env, fakebin: Fakebin) -> None:
    agent_id = seed(env, fakebin)
    done = td(env, agent_id, "--hook", "pre-teardown=true")
    assert done.returncode == 0, (done.stdout, done.stderr)
    assert env.agent(agent_id)["status"] == "torn_down"


def test_a_later_veto_overrides_an_earlier_escalation(
    fakebin: Fakebin, isolation_env: Isolation
) -> None:
    config = 'hooks:\n  pre-teardown:\n    - cmd: "exit 13"\n    - cmd: "exit 1"\n'
    cfg_env = Env(fakebin, isolation_env, config)
    agent_id = seed(cfg_env, fakebin)
    done = td(cfg_env, agent_id, *HUMAN, "--override-escalation")
    assert done.returncode == 11
    assert cfg_env.agent(agent_id)["status"] == "parked"


def test_decision_ids_a_hook_prints_are_logged(env: Env, fakebin: Fakebin) -> None:
    agent_id = seed(env, fakebin)
    cmd = """pre-teardown=echo '{"decision_id": "d-9"}'; exit 1"""
    done = td(env, agent_id, "--hook", cmd)
    assert done.returncode == 11
    body = result_json(done)
    schema_registry.validate(body, "agent-spawn/cli/error.json")
    assert body["error"]["decision_ids"] == ["d-9"]
    logged = [json.loads(line) for line in (env.state_dir / "log.jsonl").read_text().splitlines()]
    (veto,) = [r for r in logged if r["event"] == "hook_veto"]
    assert veto["decision_ids"] == ["d-9"] and veto["hook"] == "pre-teardown"


def test_decision_ids_of_an_escalating_hook_are_logged(env: Env, fakebin: Fakebin) -> None:
    agent_id = seed(env, fakebin)
    cmd = """pre-teardown=echo '{"decision_id": "d-8"}'; exit 13"""
    done = td(env, agent_id, "--hook", cmd)
    assert done.returncode == 13
    assert result_json(done)["error"]["decision_ids"] == ["d-8"]
    logged = [json.loads(line) for line in (env.state_dir / "log.jsonl").read_text().splitlines()]
    (escalate,) = [r for r in logged if r["event"] == "hook_escalate"]
    assert escalate["decision_ids"] == ["d-8"]


def test_the_decision_ids_of_an_error_are_capped_at_32(env: Env, fakebin: Fakebin) -> None:
    agent_id = seed(env, fakebin)
    flags = [f for n in range(32) for f in ("--decision-id", f"c-{n}")]
    cmd = """pre-teardown=echo '{"decision_id": "extra"}'; exit 1"""
    done = td(env, agent_id, "--hook", cmd, *flags)
    body = result_json(done)
    assert done.returncode == 11, (done.stdout, done.stderr)
    schema_registry.validate(body, "agent-spawn/cli/error.json")
    assert body["error"]["decision_ids"] == [f"c-{n}" for n in range(32)]


def test_a_veto_record_after_several_hooks_caps_its_decision_ids_at_32(
    fakebin: Fakebin, isolation_env: Isolation
) -> None:
    def hook(prefix: str, code: int) -> str:
        payload = json.dumps({"decision_id": [f"{prefix}-{n}" for n in range(20)]})
        loop = f"echo '{payload}'"
        return f"    - cmd: |\n        {loop}; exit {code}\n"

    cfg = "hooks:\n  pre-teardown:\n" + hook("a", 0) + hook("b", 11)
    cfg_env = Env(fakebin, isolation_env, cfg)
    agent_id = seed(cfg_env, fakebin)
    done = td(cfg_env, agent_id)
    body = result_json(done)
    assert done.returncode == 11, (done.stdout, done.stderr)
    assert len(body["error"]["decision_ids"]) == 32
    logged = [
        json.loads(line) for line in (cfg_env.state_dir / "log.jsonl").read_text().splitlines()
    ]
    (veto,) = [r for r in logged if r["event"] == "hook_veto"]
    assert len(veto["decision_ids"]) == 32


# --- the override --------------------------------------------------------------------------------


def test_the_override_lets_an_escalation_through(env: Env, fakebin: Fakebin) -> None:
    agent_id = seed(env, fakebin)
    done = td(env, agent_id, "--hook", ESCALATE, *HUMAN, "--override-escalation")
    assert done.returncode == 0, (done.stdout, done.stderr)
    assert env.agent(agent_id)["status"] == "torn_down"
    assert [a[1] for a in forget_calls(fakebin)]


def test_the_override_does_not_override_a_veto(env: Env, fakebin: Fakebin) -> None:
    agent_id = seed(env, fakebin)
    done = td(env, agent_id, "--hook", VETO, *HUMAN, "--override-escalation")
    assert done.returncode == 11
    assert env.agent(agent_id)["status"] == "parked"


def test_the_override_flag_needs_a_human_actor(env: Env, fakebin: Fakebin) -> None:
    agent_id = seed(env, fakebin)
    ran = env.fakebin.bin_dir / "ran"
    done = td(env, agent_id, "--hook", f"pre-teardown=touch {ran}", "--override-escalation")
    assert done.returncode == 2
    done = td(env, agent_id, "--actor", "auto:bot", "--override-escalation")
    assert done.returncode == 2
    assert not ran.exists() and env.agent(agent_id)["status"] == "parked"


def test_the_override_reaches_the_hook_and_wtctl_remove_and_nothing_else(
    env: Env, fakebin: Fakebin, tmp_path: Path
) -> None:
    agent_id = launched_in_worktree(env)
    sink = tmp_path / "payload.json"
    done = td(env, agent_id, "--hook", f"pre-teardown=cat > {sink}; exit 13",
              *HUMAN, "--override-escalation")  # fmt: skip
    assert done.returncode == 0, (done.stdout, done.stderr)
    assert result_json(done).get("warnings", []) == []
    payload = json.loads(sink.read_text())
    schema_registry.validate(payload, "agent-spawn/hooks/pre-teardown.json")
    assert payload["override_escalation"] is True
    (remove,) = wtctl_calls(fakebin, "remove")
    assert "--override-escalation" in remove
    others = [a for a in fakebin.argvs("wtctl") if a[0] != "remove"]
    assert all("--override-escalation" not in a for a in others)
    for argv in fakebin.argvs("claude-session") + fakebin.argvs("termctl"):
        assert "--override-escalation" not in argv
    # one audit record per wtctl remove call that carried the flag, and one for the hook
    actions = sorted(a["action"] for a in audit_records(env))
    assert actions == ["teardown", "teardown_escalation_override", "teardown_escalation_override"]


def test_an_overridden_hook_escalation_is_audited_once_without_a_worktree(
    env: Env, fakebin: Fakebin
) -> None:
    agent_id = seed(env, fakebin)
    done = td(env, agent_id, "--hook", ESCALATE, *HUMAN, "--override-escalation",
              "--decision-id", "d-5")  # fmt: skip
    assert done.returncode == 0, (done.stdout, done.stderr)
    assert result_json(done).get("warnings", []) == []
    (record,) = [a for a in audit_records(env) if a["action"] == "teardown_escalation_override"]
    assert record["result"] == "ok" and record["target"] == agent_id
    assert record["decision_id"] == ["d-5"] and record["inputs"]["worktree"] is None
    assert wtctl_calls(fakebin, "remove") == []
    assert [a["action"] for a in audit_records(env)] == ["teardown_escalation_override"]


def test_an_override_that_meets_no_escalation_is_not_audited(env: Env, fakebin: Fakebin) -> None:
    agent_id = seed(env, fakebin)
    done = td(env, agent_id, "--hook", "pre-teardown=true", *HUMAN, "--override-escalation")
    assert done.returncode == 0, (done.stdout, done.stderr)
    assert audit_records(env) == []


# --- the payload ---------------------------------------------------------------------------------


def test_the_payload_matches_its_schema_and_travels_on_stdin(
    env: Env, fakebin: Fakebin, tmp_path: Path
) -> None:
    agent_id = seed(env, fakebin)
    sink = tmp_path / "payload.json"
    envdump = tmp_path / "env"
    cmd = f"pre-teardown=cat > {sink}; env > {envdump}"
    done = td(env, agent_id, "--hook", cmd, "--merged-head", "b" * 40, "--decision-id", "d-1")
    assert done.returncode == 0, (done.stdout, done.stderr)
    payload = json.loads(sink.read_text())
    schema_registry.validate(payload, "agent-spawn/hooks/pre-teardown.json")
    assert payload["object_id"] == agent_id and payload["agent"]["agent_id"] == agent_id
    assert payload["merged_head"] == "b" * 40 and payload["force"] is False
    assert payload["override_escalation"] is False
    assert payload["decision_ids"] == ["d-1"] and payload["actor"].startswith("auto:")
    assert payload["status"] == "tearing_down"
    variables = dict(line.split("=", 1) for line in envdump.read_text().splitlines() if "=" in line)
    assert variables["AGENT_SPAWN_HOOK"] == "pre-teardown"
    assert variables["AGENT_SPAWN_ID"] == agent_id


def test_a_selector_that_does_not_match_skips_the_entry(
    fakebin: Fakebin, isolation_env: Isolation
) -> None:
    config = 'hooks:\n  pre-teardown:\n    - cmd: "exit 1"\n      labels: ["only=this"]\n'
    cfg_env = Env(fakebin, isolation_env, config)
    agent_id = seed(cfg_env, fakebin)
    assert td(cfg_env, agent_id).returncode == 0


def test_veto_and_escalation_payloads_match_their_schema(
    env: Env, fakebin: Fakebin, tmp_path: Path
) -> None:
    agent_id = seed(env, fakebin)
    for code in (1, 13):
        sink = tmp_path / f"payload-{code}.json"
        done = td(env, agent_id, "--hook", f"pre-teardown=cat > {sink}; exit {code}",
                  "--merged-head", "c" * 40, *HUMAN, "--force")  # fmt: skip
        assert done.returncode == (11 if code == 1 else 13), (done.stdout, done.stderr)
        payload = json.loads(sink.read_text())
        schema_registry.validate(payload, "agent-spawn/hooks/pre-teardown.json")
        assert payload["merged_head"] == "c" * 40 and payload["force"] is True
        assert payload["actor"] == "human:tester" and payload["override_escalation"] is False


# --- retries ---


def test_a_retry_that_never_completed_the_hooks_runs_them(env: Env, fakebin: Fakebin) -> None:
    agent_id = seed(
        env, fakebin, status="tearing_down", teardown_step="started", teardown_proc=dead()
    )
    done = td(env, agent_id, "--hook", VETO)
    assert done.returncode == 11
    row = env.agent(agent_id)
    assert row["status"] == "parked" and row["teardown_step"] is None


def test_a_retry_past_the_hooks_step_skips_them_and_logs_it(
    env: Env, fakebin: Fakebin, tmp_path: Path
) -> None:
    agent_id = seed(
        env, fakebin, status="tearing_down", teardown_step="hooks_done", teardown_proc=dead()
    )
    ran = tmp_path / "ran"
    done = td(env, agent_id, "--hook", f"pre-teardown=touch {ran}; exit 1")
    assert done.returncode == 0, (done.stdout, done.stderr)
    assert not ran.exists()
    assert '"teardown_retry"' in (env.state_dir / "log.jsonl").read_text()


def test_a_retry_without_hook_flags_runs_the_stored_hook(env: Env, fakebin: Fakebin) -> None:
    agent_id = seed(
        env,
        fakebin,
        status="tearing_down",
        teardown_step="started",
        teardown_proc=dead(),
        hooks=[{"hook": "pre-teardown", "cmd": "exit 1"}],
    )
    assert td(env, agent_id).returncode == 11


# --- liveness after a slow hook ---


def test_a_run_that_went_live_during_the_hook_returns_the_agent_to_parked(
    env: Env, fakebin: Fakebin, tmp_path: Path
) -> None:
    from tests.launch_support import script_live_run

    agent_id = seed(env, fakebin)
    before = env.agent(agent_id)
    marker = tmp_path / "hook-ran"
    # The first `status` (step 2's live check) answers like the fake's default, every later one
    # (step 3's re-check, after the hook) reports a live run.
    fakebin.script("claude-session", "status", times=1)
    script_live_run(fakebin)
    done = td(env, agent_id, "--hook", f"pre-teardown=touch {marker}")
    body = result_json(done)
    assert done.returncode == 20, (done.stdout, done.stderr)
    assert marker.exists()  # the hook ran, so the refusal is the re-check after it
    assert [a for a in fakebin.argvs("claude-session") if a[0] == "status"][1:]  # re-checked
    assert body["error"]["reason"] == "live"
    unchanged(env, agent_id, before)
    assert forget_calls(fakebin) == []


# --- concurrency, dry runs and selectors ---


def test_a_teardown_and_a_resume_during_a_running_hook_are_refused(
    env: Env, fakebin: Fakebin, tmp_path: Path
) -> None:
    import subprocess

    agent_id = seed(env, fakebin)
    started, gate = tmp_path / "started", tmp_path / "gate"
    hook = f"pre-teardown=touch {started}; while [ ! -e {gate} ]; do sleep 0.1; done"
    proc = subprocess.Popen(
        env.argv("teardown", agent_id, "--json", "--hook", hook),
        env=env.env(),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        env.wait_for(started.exists, "the hook to start")
        other = td(env, agent_id)
        assert other.returncode == 20, (other.stdout, other.stderr)
        assert result_json(other)["error"]["reason"] == "tearing_down"
        resumed = env.run("resume", agent_id, "--json")
        assert resumed.returncode == 20, (resumed.stdout, resumed.stderr)
    finally:
        gate.write_text("")
        out, _ = proc.communicate(timeout=60)
    assert proc.returncode == 0, out
    assert env.agent(agent_id)["status"] == "torn_down"


def test_a_dry_run_does_not_run_the_hook(env: Env, fakebin: Fakebin, tmp_path: Path) -> None:
    agent_id = seed(env, fakebin)
    ran = tmp_path / "ran"
    done = td(env, agent_id, "--dry-run", "--hook", f"pre-teardown=touch {ran}; exit 1")
    assert done.returncode == 0, (done.stdout, done.stderr)
    assert not ran.exists() and env.agent(agent_id)["status"] == "parked"


def test_a_selector_that_matches_the_agent_vetoes(
    fakebin: Fakebin, isolation_env: Isolation
) -> None:
    config = 'hooks:\n  pre-teardown:\n    - cmd: "exit 1"\n      labels: ["only=this"]\n'
    cfg_env = Env(fakebin, isolation_env, config)
    agent_id = seed(cfg_env, fakebin)
    added = cfg_env.run("label", "add", agent_id, "only=this", "--json")
    assert added.returncode == 0, (added.stdout, added.stderr)
    assert td(cfg_env, agent_id).returncode == 11
    assert cfg_env.agent(agent_id)["status"] == "parked"


# --- events ---


def test_a_veto_queues_state_changes_but_no_post_teardown(env: Env, fakebin: Fakebin) -> None:
    agent_id = seed(env, fakebin)
    assert td(env, agent_id, "--hook", VETO).returncode == 11
    names = [n for n, _ in events(env, agent_id)]
    assert "post-teardown" not in names
    states = [p["agent"]["status"] for n, p in events(env, agent_id) if n == "post-state-change"]
    assert states == ["tearing_down", "parked"]


# --- a signal during a slow hook (#33) ---


@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGHUP])
def test_a_signal_during_a_hook_terminates_the_hook_and_leaves_the_agent_retryable(
    env: Env, fakebin: Fakebin, tmp_path: Path, sig: signal.Signals
) -> None:
    from agent_spawn import procs

    agent_id = seed(env, fakebin)
    pidfile = tmp_path / "hook-pid"
    gate = tmp_path / "gate"
    hook = (
        f"pre-teardown=[ -e {gate} ] || {{ sleep 300 & echo $! > {pidfile}.child; }}; "
        f"echo $$ > {pidfile}.tmp; mv {pidfile}.tmp {pidfile}; "
        f"while [ ! -e {gate} ]; do wait; done"
    )
    proc = subprocess.Popen(
        env.argv("teardown", agent_id, "--json", "--hook", hook),
        env=env.env(),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    hook_identity = None
    try:
        env.wait_for(pidfile.exists, "the hook to start")
        hook_identity = procs.read_identity(int(pidfile.read_text()))
        assert hook_identity is not None
        os.kill(proc.pid, sig)  # exactly the process this test started
        proc.communicate(timeout=60)
        assert proc.returncode == -sig  # it still ended by the signal
        env.wait_for(lambda: not procs.is_alive(hook_identity), "the hook's group to be stopped")
        child = procs.read_identity(int((tmp_path / "hook-pid.child").read_text()))
        if child is not None:  # the hook's children went with it (the group's kill is not instant)
            env.wait_for(
                lambda: not procs.is_alive(child), "the hook's child to be gone", timeout=60
            )
    finally:
        if hook_identity is not None and procs.is_alive(hook_identity):
            os.kill(hook_identity.pid, signal.SIGKILL)
        if proc.poll() is None:
            proc.kill()
            proc.communicate()
    row = env.agent(agent_id)
    assert (row["status"], row["teardown_step"]) == ("tearing_down", "started")
    gate.write_text("")
    again = td(env, agent_id)  # the retry: the stored hook runs again and passes
    assert again.returncode == 0, (again.stdout, again.stderr)


# --- the signal-to-exception context, in process (#33) ---


class _SignalStub:
    """The signal-handler part of the gateway: records installs, serves canned previous handlers
    (``unsettable`` signals raise ``ValueError``, as off the main thread)."""

    def __init__(self, previous: dict[int, Any], *, main_thread: bool = True) -> None:
        self.current = dict(previous)
        self.main_thread = main_thread
        self.installs: list[tuple[int, Any]] = []

    def install_signal_handler(self, sig: int, handler: Any) -> Any:
        if not self.main_thread:
            raise ValueError("signal only works in main thread of the main interpreter")
        if handler is None:
            raise TypeError("signal handler must be signal.SIG_IGN, signal.SIG_DFL, or a callable")
        self.installs.append((sig, handler))
        previous = self.current[sig]
        self.current[sig] = handler
        return previous


def _context(stub: _SignalStub) -> Any:
    from agent_spawn import teardown

    return teardown._terminate_hook_on_signal(stub)  # pyright: ignore[reportPrivateUsage,reportArgumentType]


def test_a_foreign_signal_handler_is_left_alone_and_restored() -> None:
    def foreign(signum: int, frame: object) -> None:  # the supervisor's own
        pass

    stub = _SignalStub({signal.SIGTERM: foreign, signal.SIGHUP: signal.SIG_DFL})
    with _context(stub):
        assert stub.current[signal.SIGTERM] is foreign  # put back at once, never replaced
        assert stub.current[signal.SIGHUP] is not signal.SIG_DFL  # ours
    assert stub.current[signal.SIGTERM] is foreign and stub.current[signal.SIGHUP] is signal.SIG_DFL


def test_a_normal_exit_restores_the_previous_handlers() -> None:
    stub = _SignalStub({signal.SIGTERM: signal.SIG_DFL, signal.SIGHUP: signal.SIG_DFL})
    with _context(stub):
        assert callable(stub.current[signal.SIGTERM]) and callable(stub.current[signal.SIGHUP])
    assert stub.current == {signal.SIGTERM: signal.SIG_DFL, signal.SIGHUP: signal.SIG_DFL}


def test_off_the_main_thread_the_body_still_runs() -> None:
    stub = _SignalStub({signal.SIGTERM: signal.SIG_DFL, signal.SIGHUP: signal.SIG_DFL})
    stub.main_thread = False
    ran: list[bool] = []
    with _context(stub):
        ran.append(True)
    assert ran == [True] and stub.installs == []


def test_a_handler_the_interpreter_cannot_report_is_not_reinstalled() -> None:
    """``signal.signal`` returns ``None`` for a handler installed from C: it cannot be put back,
    so it is treated as foreign, and the context neither fails nor loses the other signal."""
    stub = _SignalStub({signal.SIGTERM: None, signal.SIGHUP: signal.SIG_DFL})
    with _context(stub):
        pass
    assert stub.current[signal.SIGHUP] is signal.SIG_DFL
    assert all(handler is not None for _, handler in stub.installs)


def test_a_failure_while_installing_still_restores_what_was_installed() -> None:
    class Failing(_SignalStub):
        def install_signal_handler(self, sig: int, handler: Any) -> Any:
            if sig == signal.SIGHUP and callable(handler):
                raise ValueError("no")
            return super().install_signal_handler(sig, handler)

    stub = Failing({signal.SIGTERM: signal.SIG_DFL, signal.SIGHUP: signal.SIG_DFL})
    with _context(stub):
        pass
    assert stub.current[signal.SIGTERM] is signal.SIG_DFL


def test_a_second_signal_during_the_shutdown_is_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    from agent_spawn import teardown

    stub = _SignalStub({signal.SIGTERM: signal.SIG_DFL, signal.SIGHUP: signal.SIG_DFL})
    resent: list[tuple[int, int]] = []
    monkeypatch.setattr(teardown.os, "kill", lambda pid, sig: resent.append((pid, sig)))
    with pytest.raises(teardown._Terminated):  # pyright: ignore[reportPrivateUsage]
        with _context(stub):
            stub.current[signal.SIGTERM](signal.SIGTERM, None)  # the first signal
    # between the raise and the cleanup both signals were ignored (the shutdown is not
    # interrupted by a second one); the default action is back before the signal is re-sent
    ignored = [(s, h) for s, h in stub.installs if h is signal.SIG_IGN]
    assert {s for s, _ in ignored} == {signal.SIGTERM, signal.SIGHUP}
    assert stub.current == {signal.SIGTERM: signal.SIG_DFL, signal.SIGHUP: signal.SIG_DFL}
    assert resent == [(os.getpid(), signal.SIGTERM)]


def test_a_failure_between_installing_the_two_signals_restores_the_first() -> None:
    class Failing(_SignalStub):
        def install_signal_handler(self, sig: int, handler: Any) -> Any:
            if sig == signal.SIGHUP and callable(handler):
                raise RuntimeError("boom")
            return super().install_signal_handler(sig, handler)

    stub = Failing({signal.SIGTERM: signal.SIG_DFL, signal.SIGHUP: signal.SIG_DFL})
    with pytest.raises(RuntimeError):
        with _context(stub):
            pass
    assert stub.current == {signal.SIGTERM: signal.SIG_DFL, signal.SIGHUP: signal.SIG_DFL}
