"""Round-1 review findings of ``resume`` (definite refusals, stale settings, orphans, audits,
cwd rules) and the coverage the first round found missing."""

from __future__ import annotations

import json
import signal
import subprocess
from pathlib import Path
from typing import Any

import pytest

from agent_spawn.lifecycle import lease_key
from tests.conftest import Isolation
from tests.fakebin.harness import Fakebin
from tests.launch_support import Env, result_json
from tests.procguard import ProcTracker
from tests.test_launch_review import LIVE_LAUNCH_PHASE, stops
from tests.test_resume import LAUNCH, events, launched, pass_dry_run, starts
from tests.test_resume_hardening import popen, refuse_start


@pytest.fixture
def env(fakebin: Fakebin, isolation_env: Isolation) -> Env:
    return Env(fakebin, isolation_env)


def pending_run(fakebin: Fakebin, run: int) -> None:
    """`status` reports run ``run`` pending in its launch phase."""
    fakebin.script(
        "claude-session",
        "status",
        merge={"pending": True, "current_run": run, "run": {**LIVE_LAUNCH_PHASE, "run": run}},
    )


def sessionless(env: Env, fakebin: Fakebin, *extra: str) -> str:
    refuse_start(fakebin)
    assert env.run(*LAUNCH, *extra).returncode == 20
    (row,) = [a for a in env.agents() if a["csession_id"] is None]
    return str(row["agent_id"])


def error_of(done: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    return dict(json.loads(done.stdout)["error"])


def audit_results(env: Env) -> list[tuple[str, str]]:
    lines = (env.state_dir / "audit.jsonl").read_text().splitlines()
    return [(json.loads(x)["action"], json.loads(x)["result"]) for x in lines]


# --- R1: a definite refusal never touches another caller's run ---------------------------


def test_a_definite_refusal_does_not_stop_another_callers_pending_run(
    env: Env, fakebin: Fakebin
) -> None:
    agent_id = launched(env)
    before = env.agent(agent_id)
    pass_dry_run(fakebin)
    pending_run(fakebin, 9)
    fakebin.script("claude-session", "start", exit=20,
                   error={"code": "refused", "message": "no", "reason": "live"})  # fmt: skip
    done = env.run("resume", agent_id, "--json")
    assert done.returncode == 20
    assert stops(fakebin) == []
    after = env.agent(agent_id)
    assert (after["status"], after["updated"], after["run"]) == (
        "parked", before["updated"], before["run"])  # fmt: skip


def test_a_definite_refusal_of_a_fork_stops_nothing(env: Env, fakebin: Fakebin) -> None:
    source = launched(env)
    pass_dry_run(fakebin)
    pending_run(fakebin, 9)
    fakebin.script("claude-session", "start", exit=20,
                   error={"code": "refused", "message": "no", "reason": "live"})  # fmt: skip
    done = env.run("resume", source, "--fork", "--cwd", "session-dir", "--json")
    assert done.returncode == 20
    assert stops(fakebin) == []
    assert [a["agent_id"] for a in env.agents()] == [source]


# --- R14: a failed status after start still cancels the allocated run --------------------


def test_a_failed_status_after_an_uncertain_start_stops_the_session(
    env: Env, fakebin: Fakebin
) -> None:
    agent_id = launched(env)
    session = env.agent(agent_id)["csession_id"]
    before = env.agent(agent_id)
    pass_dry_run(fakebin)
    fakebin.script("claude-session", "status", exit=1, error={"code": "failed", "message": "x"})
    fakebin.script("claude-session", "start", exit=1, error={"code": "failed", "message": "x"})
    done = env.run("resume", agent_id, "--json")
    assert done.returncode == 1
    assert [s[1] for s in stops(fakebin)] == [session]
    assert env.agent(agent_id)["status"] == "parked"
    assert env.agent(agent_id)["updated"] == before["updated"]


# --- R15: copies go once a session is recorded for an orphan -----------------------------


def test_an_orphan_settled_after_an_unknown_start_removes_the_prompt_copies(
    env: Env, fakebin: Fakebin
) -> None:
    other = launched(env)
    session = env.agent(other)["csession_id"]
    agent_id = sessionless(env, fakebin, "--prompt", "first words")
    assert (env.state_dir / "prompts" / agent_id / "prompt").exists()
    pass_dry_run(fakebin)
    fakebin.script("claude-session", "list", stdout={"sessions": [{"csession_id": session}]})
    pending_run(fakebin, 1)
    fakebin.script("claude-session", "start", exit=1, error={"code": "failed", "message": "x"})
    done = env.run("resume", agent_id, "--json")
    assert done.returncode == 1, (done.stdout, done.stderr)
    row = env.agent(agent_id)
    assert (row["status"], row["park_reason"], row["csession_id"]) == (
        "parked", "launch_failed", session)  # fmt: skip
    assert not (env.state_dir / "prompts" / agent_id).exists()


# --- R2: a fork always passes its resolved cwd -------------------------------------------


def test_a_fork_passes_a_nonexistent_cwd_for_claude_session_to_refuse(
    env: Env, fakebin: Fakebin, tmp_path: Path
) -> None:
    source = launched(env)
    missing = str(tmp_path / "does" / "not" / "exist")
    done = env.run("resume", source, "--fork", "--cwd", missing, "--json", "--wait")
    assert done.returncode == 1, (done.stdout, done.stderr)  # claude-session exits 2
    assert f"--cwd={missing}" in starts(fakebin)[-1]
    error = error_of(done)  # claude-session's own refusal, passed through
    assert (error["tool"], error["lower_code"]) == ("claude-session", "usage")
    assert missing in error["lower_error"]["message"]


# --- R9: a relative --cwd is resolved, not refused ---------------------------------------


def test_a_fork_with_a_relative_cwd_uses_its_realpath(env: Env) -> None:
    source = launched(env)
    done = env.run("resume", source, "--fork", "--cwd", ".", "--dry-run", "--json")
    assert done.returncode == 0, (done.stdout, done.stderr)
    argv = result_json(done)["session_start_argv"]
    assert f"--cwd={Path('.').resolve()}" in argv


# --- R3: a dry run reports a missing session without parking ------------------------------


@pytest.mark.parametrize("fork", [False, True])
def test_dry_run_of_a_missing_session_exits_10_without_parking(
    env: Env, fakebin: Fakebin, fork: bool
) -> None:
    agent_id = launched(env)
    before = env.agent(agent_id)
    fakebin.script("claude-session", "start", argv_contains=["--dry-run"],
                   exit=10, error={"code": "not_found", "message": "gone"})  # fmt: skip
    args = ["--fork", "--cwd", "session-dir"] if fork else []
    done = env.run("resume", agent_id, *args, "--dry-run", "--json")
    assert done.returncode == 10, (done.stdout, done.stderr)
    assert error_of(done)["details"]["status"] == "session_missing"
    assert env.agent(agent_id) == before
    assert len(env.agents()) == 1


# --- R4: --placement here needs a terminal for an interactive resume -----------------------
# (a headless resume ignores --placement with a warning: tests/test_placement_here.py)


def test_placement_here_without_a_terminal_exits_2_for_an_interactive_resume(env: Env) -> None:
    agent_id = launched(env)
    before = env.agent(agent_id)
    done = env.run("resume", agent_id, "--mode", "interactive", "--placement", "here", "--json")
    assert done.returncode == 2, (done.stdout, done.stderr)
    assert env.agent(agent_id) == before


# --- R7: labels of an unstarted run are stored -------------------------------------------


def test_labels_given_to_a_resume_whose_start_exit_12_are_stored_mirrored(
    env: Env, fakebin: Fakebin
) -> None:
    agent_id = launched(env, "--label", "a=1")
    session = env.agent(agent_id)["csession_id"]
    pass_dry_run(fakebin)
    fakebin.script("claude-session", "start", exit=12,
                   error={"code": "launch_failed", "message": "x", "csession_id": session,
                          "run": 2, "end_reason": "launch_failed"})  # fmt: skip
    done = env.run("resume", agent_id, "--label", "b=2", "--json")
    assert done.returncode == 12
    with env.store() as store:
        assert store.get_labels(agent_id).texts() == ["a=1", "b=2"]
        assert store.mirror_pending(agent_id) == []


# --- R8: a tearing_down fork source without a session is a 20, not a 12 -------------------


@pytest.mark.parametrize("dry", [False, True])
def test_fork_of_a_tearing_down_agent_without_a_session_exits_20(
    env: Env, fakebin: Fakebin, dry: bool
) -> None:
    agent_id = sessionless(env, fakebin)
    with env.store() as store:
        with store.lease(lease_key(agent_id)) as lease:
            store.enter_teardown(lease, agent_id, teardown_proc=store.holder_identity())
    done = env.run("resume", agent_id, "--fork", "--json", *(["--dry-run"] if dry else []))
    assert done.returncode == 20, (done.stdout, done.stderr)
    body = json.loads(done.stdout)
    reason = body["would_refuse"] if dry else body["error"]["reason"]
    assert reason == "tearing_down"


# --- R10: replaced copies and the stored settings change together ------------------------


def test_a_failed_sessionless_resume_with_a_new_prompt_keeps_it_for_the_next_resume(
    env: Env, fakebin: Fakebin, tmp_path: Path
) -> None:
    agent_id = sessionless(env, fakebin)  # launched without a prompt
    new = tmp_path / "new.txt"
    new.write_text("second words")
    refuse_start(fakebin)
    assert env.run("resume", agent_id, "--prompt-file", str(new), "--json").returncode == 20
    stored = env.state_dir / "prompts" / agent_id / "prompt"
    assert stored.read_text() == "second words"
    done = env.run("resume", agent_id, "--json", "--wait")
    assert done.returncode == 0, (done.stdout, done.stderr)
    assert f"--prompt-file={stored}" in starts(fakebin)[-1]


def test_a_failed_sessionless_resume_with_new_inputs_replaces_the_stored_set(
    env: Env, fakebin: Fakebin, tmp_path: Path
) -> None:
    files = {name: tmp_path / name for name in ("a.txt", "b.txt", "c.txt")}
    for name, path in files.items():
        path.write_text(name)
    agent_id = sessionless(
        env, fakebin, "--input-file", str(files["a.txt"]), "--input-file", str(files["b.txt"])
    )
    refuse_start(fakebin)
    assert env.run("resume", agent_id, "--input-file", str(files["c.txt"]),
                   "--json").returncode == 20  # fmt: skip
    base = env.state_dir / "prompts" / agent_id
    assert sorted(p.name for p in base.iterdir()) == ["inputs"]
    assert sorted(p.name for p in (base / "inputs").iterdir()) == ["0"]
    done = env.run("resume", agent_id, "--json", "--wait")
    assert done.returncode == 0, (done.stdout, done.stderr)
    passed = [a for a in starts(fakebin)[-1] if a.startswith("--input-file=")]
    assert len(passed) == 1 and passed[0].endswith("/0/c.txt")
    assert not base.exists()


# --- R11: damaged stored settings fail closed --------------------------------------------


@pytest.mark.parametrize(
    "damaged",
    [
        "not a dict",
        {"settings": "x", "pass_throughs": {}},
        {"settings": {}, "pass_throughs": ["x"]},
        {"settings": {}},
    ],
)
def test_damaged_stored_settings_fail_a_sessionless_resume_closed(
    env: Env, fakebin: Fakebin, damaged: object
) -> None:
    agent_id = sessionless(env, fakebin)
    with env.store() as store:
        with store.lease(lease_key(agent_id)) as lease:
            store.update_agent(lease, agent_id, {"settings": damaged}, touch=False)
    before = env.agent(agent_id)
    done = env.run("resume", agent_id, "--json")
    assert done.returncode == 12, (done.stdout, done.stderr)
    assert "hint" in error_of(done)["details"]
    assert env.agent(agent_id) == before
    assert len(starts(fakebin)) == 1


# --- R13: a failed copy leaves no empty prompt directory ---------------------------------


def test_a_failed_copy_leaves_no_prompt_directory(env: Env, tmp_path: Path) -> None:
    agent_id = launched(env)
    done = env.run("resume", agent_id, "--input-file", str(tmp_path / "missing.txt"), "--json")
    assert done.returncode != 0
    assert not (env.state_dir / "prompts" / agent_id).exists()
    assert env.agent(agent_id)["status"] == "parked"


# --- R16: the plan's cwd must satisfy the third-party rule --------------------------------


def test_a_third_party_resume_whose_plan_cwd_is_not_the_session_dir_is_refused(
    env: Env, fakebin: Fakebin
) -> None:
    agent_id = launched(env)
    with env.store() as store:
        with store.lease(lease_key(agent_id)) as lease:
            store.update_agent(lease, agent_id, {"cwd": "session-dir"}, touch=False)
    before = env.agent(agent_id)
    fakebin.script(
        "claude-session",
        "start",
        argv_contains=["--dry-run"],
        merge={"profile": {"reads_third_party": True}, "taints": True, "cwd": "/srv/elsewhere"},
    )
    done = env.run("resume", agent_id, "--json")
    assert done.returncode == 20, (done.stdout, done.stderr)
    assert error_of(done)["reason"] == "third_party_cwd"
    assert env.agent(agent_id)["status"] == "parked"
    assert env.agent(agent_id)["updated"] == before["updated"]
    assert len(starts(fakebin)) == 1


# --- R17: the override audit precedes start ----------------------------------------------


def test_an_interrupt_during_start_keeps_the_audit_record_of_allow_tainted(
    env: Env, fakebin: Fakebin, proc_tracker: ProcTracker
) -> None:
    agent_id = launched(env)
    pass_dry_run(fakebin)
    fakebin.script("claude-session", "start", sleep=30)
    child = popen(env, proc_tracker, "resume", agent_id, "--allow-tainted", "--actor",
                  "human:me", "--json")  # fmt: skip
    env.wait_for(lambda: len(starts(fakebin)) == 2, "the resume's start")
    child.send_signal(signal.SIGTERM)
    child.communicate(timeout=30)
    assert child.returncode == 143
    assert [action for action, _ in audit_results(env)] == ["resume_tainted"]


def test_a_refused_start_adds_the_error_record_after_the_audit(env: Env, fakebin: Fakebin) -> None:
    agent_id = launched(env)
    refuse_start(fakebin)
    done = env.run("resume", agent_id, "--allow-tainted", "--actor", "human:me", "--json")
    assert done.returncode == 20
    assert audit_results(env) == [("resume_tainted", "ok"), ("resume_tainted", "refused")]


# --- R6: coverage the first round lacked -------------------------------------------------


def mark_launch_phase(env: Env, agent_id: str) -> None:
    with env.store() as store:
        with store.lease(lease_key(agent_id)) as lease:
            store.put_run(lease, agent_id, 1, {"outcome": {"phase": "launch"}})


def test_a_never_started_session_passes_the_stored_skill_once(env: Env, fakebin: Fakebin) -> None:
    agent_id = launched(env)
    mark_launch_phase(env, agent_id)
    assert env.run("resume", agent_id, "--json", "--wait").returncode == 0
    assert "--skill=do-it" in starts(fakebin)[-1]
    assert env.run("resume", agent_id, "--json", "--wait").returncode == 0
    assert not any(a.startswith("--skill") for a in starts(fakebin)[-1])


def test_a_fork_never_passes_the_stored_skill(env: Env, fakebin: Fakebin) -> None:
    source = launched(env)
    mark_launch_phase(env, source)
    done = env.run("resume", source, "--fork", "--cwd", "session-dir", "--json", "--wait")
    assert done.returncode == 0, (done.stdout, done.stderr)
    assert not any(a.startswith("--skill") for a in starts(fakebin)[-1])


def test_a_fork_whose_start_exit_12_stays_parked_with_its_session(
    env: Env, fakebin: Fakebin
) -> None:
    source = launched(env)
    before = env.agent(source)
    pass_dry_run(fakebin)
    fakebin.script("claude-session", "start", exit=12,
                   error={"code": "launch_failed", "message": "x", "csession_id": "cs-bbbbbbbb",
                          "run": 1, "end_reason": "launch_failed"})  # fmt: skip
    done = env.run("resume", source, "--fork", "--cwd", "session-dir", "--json",
                   "--prompt", "words")  # fmt: skip
    assert done.returncode == 12
    fork = error_of(done)["agent_id"]
    row = env.agent(fork)
    assert (row["status"], row["park_reason"], row["csession_id"], row["forked_from"]) == (
        "parked", "launch_failed", "cs-bbbbbbbb", source)  # fmt: skip
    assert env.agent(source) == before
    assert not (env.state_dir / "prompts" / fork).exists()


def test_a_fork_defaults_to_session_dir_when_the_source_ran_in_its_work_dir(
    env: Env, fakebin: Fakebin
) -> None:
    source = launched(env)
    session = env.agent(source)["csession_id"]
    status = fakebin.run("claude-session", "status", session, "--json", env=env.env())
    work = Path(json.loads(status.stdout)["csession_dir"]) / "work"
    with env.store() as store:
        with store.lease(lease_key(source)) as lease:
            store.update_agent(lease, source, {"cwd": str(work)}, touch=False)
    done = env.run("resume", source, "--fork", "--json", "--wait")
    assert done.returncode == 0, (done.stdout, done.stderr)
    assert "--cwd=session-dir" in starts(fakebin)[-1]


def test_the_mode_skips_promoted_runs(env: Env, fakebin: Fakebin) -> None:
    agent_id = launched(env)
    with env.store() as store:
        with store.lease(lease_key(agent_id)) as lease:
            store.put_run(lease, agent_id, 2, {"mode": "interactive", "promoted": True})
    done = env.run("resume", agent_id, "--json", "--wait")
    assert done.returncode == 0, (done.stdout, done.stderr)
    assert "--mode=headless" in starts(fakebin)[-1]


def test_the_mode_follows_the_last_unpromoted_run(env: Env, fakebin: Fakebin) -> None:
    agent_id = launched(env)
    with env.store() as store:
        with store.lease(lease_key(agent_id)) as lease:
            store.put_run(lease, agent_id, 2, {"mode": "interactive"})
    # interactive: the start is interactive and a window is wanted. The fake termctl has no tmux
    # here; it is scripted to report no pane PID, so the handoff fails whatever processes the
    # host has (tests/test_windows.py has the real windows).
    fakebin.script("termctl", "open", times=1)  # the preflight
    fakebin.script("termctl", "open", merge={"pid": None})
    done = env.run("resume", agent_id, "--json")
    assert "--mode=interactive" in starts(fakebin)[-1]
    assert done.returncode == 12, (done.stdout, done.stderr)
    error = json.loads(done.stdout)["error"]
    assert (error["code"], error["details"]["cause"]) == ("handoff_failed", "no_pane_pid")
    assert env.agent(agent_id)["status"] == "parked"


def test_reset_and_kind_reach_start(env: Env, fakebin: Fakebin) -> None:
    agent_id = launched(env)
    done = env.run("resume", agent_id, "--reset", "model", "--kind", "review", "--json", "--wait")
    assert done.returncode == 0, (done.stdout, done.stderr)
    argv = starts(fakebin)[-1]
    assert "--reset=model" in argv and "--kind=review" in argv


def test_name_term_placement_and_promote_are_stored_on_success(env: Env) -> None:
    agent_id = launched(env)
    done = env.run("resume", agent_id, "--name", "renamed", "--term-placement", "main",
                   "--promote", "--json", "--wait")  # fmt: skip
    assert done.returncode == 0, (done.stdout, done.stderr)
    row = env.agent(agent_id)
    assert (row["name"], row["term_placement"], row["promote"]) == ("renamed", "main", True)
    assert events(env, agent_id) is not None


# --- install_copies keeps the disk and the stored settings consistent ----------------------


def _copy_tree(root: Path, prompt: str, inputs: dict[str, str]) -> Any:
    from agent_spawn.launch import _Copies

    root.mkdir(parents=True)
    copies = _Copies(str(root))
    if prompt:
        (root / "prompt").write_text(prompt)
        copies.prompt = str(root / "prompt")
    for index, (name, text) in enumerate(inputs.items()):
        (root / "inputs" / str(index)).mkdir(parents=True, exist_ok=True)
        (root / "inputs" / str(index) / name).write_text(text)
        copies.inputs.append(str(root / "inputs" / str(index) / name))
        copies.input_names.append(name)
    return copies


def _resume_with_stored(stored: dict[str, Any]) -> Any:
    from agent_spawn.resume import Resume

    resume = Resume.__new__(Resume)
    resume.stored = dict(stored)
    resume.new_copies = None
    return resume


def _snapshot(root: Path) -> dict[str, str]:
    return {str(p.relative_to(root)): p.read_text() for p in sorted(root.rglob("*")) if p.is_file()}


def test_a_failed_settings_write_puts_the_previous_copies_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stored = _copy_tree(tmp_path / "base", "old prompt", {"a.txt": "old a"})
    fresh = _copy_tree(tmp_path / "fresh", "new prompt", {"b.txt": "new b"})
    resume = _resume_with_stored({"has_prompt": True, "input_names": ["a.txt"]})
    before = _snapshot(tmp_path / "base")

    def boom(*_args: Any) -> None:
        raise RuntimeError("lease timeout")

    monkeypatch.setattr(resume, "store_copy_settings", boom)
    with pytest.raises(RuntimeError):
        resume.install_copies(stored, fresh, True, True)
    assert _snapshot(tmp_path / "base") == before
    assert resume.stored == {"has_prompt": True, "input_names": ["a.txt"]}


def test_a_failed_inputs_rename_leaves_the_stored_prompt_and_inputs_untouched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import os

    stored = _copy_tree(tmp_path / "base", "old prompt", {"a.txt": "old a"})
    fresh = _copy_tree(tmp_path / "fresh", "new prompt", {"b.txt": "new b"})
    resume = _resume_with_stored({"has_prompt": True, "input_names": ["a.txt"]})
    before = _snapshot(tmp_path / "base")
    real_rename = os.rename

    def flaky(src: Any, dst: Any) -> None:
        if str(src) == str(tmp_path / "fresh" / "inputs"):
            raise OSError("rename failed")
        real_rename(src, dst)

    monkeypatch.setattr(os, "rename", flaky)
    called: list[bool] = []
    monkeypatch.setattr(resume, "store_copy_settings", lambda *_a: called.append(True))
    with pytest.raises(OSError):
        resume.install_copies(stored, fresh, True, True)
    assert _snapshot(tmp_path / "base") == before
    assert not called


def test_a_successful_swap_replaces_both_and_leaves_no_old_files(tmp_path: Path) -> None:
    stored = _copy_tree(tmp_path / "base", "old prompt", {"a.txt": "old a"})
    fresh = _copy_tree(tmp_path / "fresh", "new prompt", {"b.txt": "new b"})
    resume = _resume_with_stored({})
    resume.store_copy_settings = lambda *_a: None
    result = resume.install_copies(stored, fresh, True, True)
    assert _snapshot(tmp_path / "base") == {"prompt": "new prompt", "inputs/0/b.txt": "new b"}
    assert result.input_names == ["b.txt"]


# --- the cwd rule compares canonical paths -----------------------------------------------


def test_a_symlinked_recorded_cwd_equal_to_the_plan_cwd_needs_no_status_call(
    tmp_path: Path,
) -> None:
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real)
    from agent_spawn.resume import Resume

    resume = Resume.__new__(Resume)
    resume.fork = False
    resume.source_session = "s"
    resume.log = lambda *_a, **_k: None  # type: ignore[method-assign]

    class Backend:
        def status(self, _session: str) -> Any:
            raise AssertionError("status must not be called")

    resume.backend = Backend()  # type: ignore[assignment]
    from agent_spawn import third_party

    classification = third_party.Classification(
        reads_third_party=True,
        run_kind="x",
        repo=None,
        cwd=str(real),
        would_refuse=None,
        would_fail=None,
    )
    resume.check_plan_cwd(classification, str(link))  # recorded cwd under a symlink
