"""The session backend against the fake claude-session: exact argv, classification, the
post-start check, outcome files and the small commands."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
from pathlib import Path
from typing import Any

import pytest

from agent_spawn import session_backend as sb
from agent_spawn.gateway import SystemGateway
from agent_spawn.output import CliError
from agent_spawn.session_backend import SessionBackend, StartError
from tests import sibling_schemas
from tests.fakebin.fakelib.claude_session import outcome_object
from tests.fakebin.harness import Fakebin
from tests.session_backend_support import AGENT, RELAY, SESSION, fake_backend, spec

DRY = ["--dry-run"]


def starts(fakebin: Fakebin) -> list[list[str]]:
    return [a for a in fakebin.argvs("claude-session") if a[:1] == ["start"]]


def real_starts(fakebin: Fakebin) -> list[list[str]]:
    return [a for a in starts(fakebin) if "--dry-run" not in a]


def files(tmp_path: Path) -> tuple[Path, Path, Path]:
    prompt = tmp_path / "prompt.md"
    prompt.write_text("do the thing")
    note = tmp_path / "a.txt"
    note.write_text("note")
    return prompt, note, tmp_path / "outcome.json"


# --- argv ---------------------------------------------------------------------------------------


def test_launch_argv_is_exact(fakebin: Fakebin, tmp_path: Path) -> None:
    prompt, note, outcome = files(tmp_path)
    backend = fake_backend(fakebin, actor="human:me", decision_ids=["d1", "d2"])
    run = spec(
        kind="implement",
        plugin_dirs=("/p1",),
        model="haiku",
        timeout="30m",
        max_turns=5,
        permission_profile="strict",
        sandbox="required",
        setting_sources="",
        reads_third_party=True,
        env_allow=("A", "B"),
        env_allowlist_only=("X",),
        auto_exit=False,
        repo="host/o/r",
        cwd="session-dir",
        labels=("job=j1", "owner=me"),
        prompt_file=str(prompt),
        input_files=(str(note),),
    )
    backend.start(run, outcome_file=str(outcome), hook=RELAY)
    fakebin.assert_argv(
        "claude-session",
        [
            "start",
            "--mode=headless",
            "--kind=implement",
            "--skill=implement-issue",
            "--model=haiku",
            "--timeout=30m",
            "--permission-profile=strict",
            "--sandbox=required",
            "--setting-sources=",
            "--plugin-dir=/p1",
            "--max-turns=5",
            "--reads-third-party",
            "--env-allow=A",
            "--env-allow=B",
            "--env-allowlist-only=X",
            "--no-auto-exit",
            "--repo=host/o/r",
            "--cwd=session-dir",
            "--label=job=j1",
            "--label=owner=me",
            f"--label=agent={AGENT}",
            f"--prompt-file={prompt}",
            f"--input-file={note}",
            f"--hook={RELAY}",
            f"--outcome-file={outcome}",
            "--actor=human:me",
            "--decision-id=d1",
            "--decision-id=d2",
            "--json",
        ],
    )


def test_mode_is_always_explicit(fakebin: Fakebin, tmp_path: Path) -> None:
    backend = fake_backend(fakebin)
    prompt, _, outcome = files(tmp_path)
    backend.start(
        spec(mode="interactive", skill=None, prompt_file=str(prompt)),
        outcome_file=str(outcome),
        hook=RELAY,
    )
    assert "--mode=interactive" in real_starts(fakebin)[0]


def test_resume_and_fork_argv(fakebin: Fakebin, tmp_path: Path) -> None:
    fakebin.seed_session(SESSION)
    backend = fake_backend(fakebin)
    _, _, outcome = files(tmp_path)
    backend.start(spec(resume=SESSION, skill=None), outcome_file=str(outcome), hook=RELAY)
    fake_backend(fakebin, actor="human:me").start(
        spec(resume=SESSION, fork=True, skill=None, cwd="session-dir", allow_tainted=True),
        outcome_file=str(outcome),
        hook=RELAY,
    )
    resume, fork = real_starts(fakebin)
    assert resume[:3] == ["start", f"--resume={SESSION}", "--mode=headless"]
    assert fork[:4] == ["start", f"--resume={SESSION}", "--fork", "--mode=headless"]
    assert "--allow-tainted" in fork and "--allow-tainted" not in resume
    assert f"--label=agent={AGENT}" in resume


def test_the_dry_run_is_the_same_command_without_hook_and_outcome_file(
    fakebin: Fakebin, tmp_path: Path
) -> None:
    prompt, _, outcome = files(tmp_path)
    backend = fake_backend(fakebin, actor="auto:x", decision_ids=["d1"])
    run = spec(labels=("job=j",), prompt_file=str(prompt), reads_third_party=True)
    backend.classify(run)
    backend.start(run, outcome_file=str(outcome), hook=RELAY)
    dry, real = starts(fakebin)
    stripped = [a for a in real if not a.startswith(("--hook=", "--outcome-file="))]
    assert dry == [*stripped[:-1], "--dry-run", "--json"]
    assert not any(a.startswith(("--hook", "--outcome-file")) for a in dry)


def test_a_flag_value_starting_with_a_dash_stays_one_token(fakebin: Fakebin) -> None:
    argv = fake_backend(fakebin).dry_run_argv(spec(skill="--actor=human:root", kind="-x"))
    assert "--skill=--actor=human:root" in argv and "--kind=-x" in argv
    assert "--actor=human:root" not in argv


def test_prompt_and_inputs_travel_as_files_never_as_text(fakebin: Fakebin, tmp_path: Path) -> None:
    prompt, note, outcome = files(tmp_path)
    prompt.write_text("SECRET-PROMPT-TEXT")
    fake_backend(fakebin).start(
        spec(prompt_file=str(prompt), input_files=(str(note),)),
        outcome_file=str(outcome),
        hook=RELAY,
    )
    call = fakebin.invocations("claude-session")[-1]
    assert not any("SECRET-PROMPT-TEXT" in a for a in call.argv)
    assert not any("SECRET-PROMPT-TEXT" in v for v in call.env.values())
    assert f"--prompt-file={prompt}" in call.argv


def test_every_start_carries_the_agent_label_and_the_relay_hook(
    fakebin: Fakebin, tmp_path: Path
) -> None:
    _, _, outcome = files(tmp_path)
    fake_backend(fakebin).start(spec(), outcome_file=str(outcome), hook=RELAY)
    argv = real_starts(fakebin)[0]
    assert f"--label=agent={AGENT}" in argv
    assert f"--hook={RELAY}" in argv
    assert f"--outcome-file={outcome}" in argv


@pytest.mark.parametrize(
    "agent, state, config",
    [
        ("/opt/agent spawn/bin/agent-spawn", "/state dir", "/cfg/c.yaml"),
        ("/opt/as", "/it's/state", '/c"fg $HOME `id`; rm -rf x/c.yaml'),
        ("/opt/as", "/state\nnewline", "/cfg/é中.yaml"),
        ("/a", "/b*?[x]", "/c#d&e|f>g<h(i)"),
    ],
)
def test_the_relay_hook_survives_sh_c(agent: str, state: str, config: str) -> None:
    hook = sb.relay_hook(agent, state, config)
    name, _, command = hook.partition("=")
    assert name == "post-state-change"
    done = subprocess.run(
        ["sh", "-c", "printf '%s\\0' " + command],
        capture_output=True,
        env={"PATH": "/usr/bin:/bin"},
        check=True,
    )
    assert done.stdout.decode().split("\0")[:-1] == [
        agent,
        "--state-dir",
        state,
        "--config",
        config,
        "_session-event",
    ]
    assert shlex.split(command) == [
        agent,
        "--state-dir",
        state,
        "--config",
        config,
        "_session-event",
    ]


def test_the_relay_hook_needs_absolute_paths() -> None:
    with pytest.raises(ValueError):
        sb.relay_hook("agent-spawn", "/s", "/c")
    with pytest.raises(ValueError):
        sb.relay_hook("/a", "s", "/c")
    with pytest.raises(CliError):
        sb.relay_hook("/a\0", "/s", "/c")


# --- classification -----------------------------------------------------------------------------


def dry_script(fakebin: Fakebin, **respond: Any) -> None:
    fakebin.script("claude-session", "start", argv_contains=DRY, **respond)


def test_a_plain_launch_is_trusted(fakebin: Fakebin) -> None:
    result = fake_backend(fakebin).classify(spec(kind="implement"))
    assert (result.reads_third_party, result.run_kind) == (False, "implement")


def test_a_carried_reads_third_party_flag_is_third_party_reading(fakebin: Fakebin) -> None:
    assert fake_backend(fakebin).classify(spec(reads_third_party=True)).reads_third_party


def test_resume_without_kind_is_classified_by_the_plan_alone(fakebin: Fakebin) -> None:
    fakebin.seed_session(SESSION)
    dry_script(fakebin, merge={"profile": {"reads_third_party": True}, "run_kind": "implement"})
    run = spec(resume=SESSION, skill=None)
    assert fake_backend(fakebin).classify(run).reads_third_party
    assert not any(a.startswith("--kind") for a in starts(fakebin)[0])


def test_a_fork_of_a_respond_source_is_third_party_reading(fakebin: Fakebin) -> None:
    fakebin.seed_session(SESSION, kind="respond")
    dry_script(fakebin, merge={"taints": True, "run_kind": "respond"})
    result = fake_backend(fakebin).classify(
        spec(resume=SESSION, fork=True, skill=None, cwd="session-dir")
    )
    assert (result.reads_third_party, result.run_kind) == (True, "respond")
    assert "--fork" in starts(fakebin)[0]


def test_classification_reads_repo_and_the_kinds_default_cwd(fakebin: Fakebin) -> None:
    dry_script(
        fakebin,
        merge={"repo": "host/o/r", "profile": {"cwd": "session-dir", "reads_third_party": True}},
    )
    result = fake_backend(fakebin).classify(spec())
    assert (result.repo, result.cwd) == ("host/o/r", "session-dir")


def test_a_dry_run_failure_starts_nothing(fakebin: Fakebin) -> None:
    dry_script(fakebin, exit=2, error={"code": "usage", "message": "bad flag"})
    with pytest.raises(CliError) as caught:
        fake_backend(fakebin).classify(spec())
    assert caught.value.tool == "claude-session"
    assert real_starts(fakebin) == []


@pytest.mark.parametrize("code, name", [(10, "not_found"), (21, "ambiguous"), (1, "internal")])
def test_other_dry_run_exits_pass_through(fakebin: Fakebin, code: int, name: str) -> None:
    dry_script(fakebin, exit=code, error={"code": name, "message": "x"})
    with pytest.raises(CliError) as caught:
        fake_backend(fakebin).classify(spec())
    assert caught.value.exit_code == (1 if code == 2 else code)
    assert caught.value.lower_code == name


def test_would_refuse_refuses_before_anything_changes(fakebin: Fakebin) -> None:
    dry_script(fakebin, exit=20, merge={"would_refuse": "tainted"})
    with pytest.raises(CliError) as caught:
        fake_backend(fakebin).classify(spec())
    error = caught.value
    assert (error.exit_code, error.code, error.reason, error.tool) == (
        20,
        "refused",
        "tainted",
        "claude-session",
    )
    assert real_starts(fakebin) == []


def test_would_refuse_can_be_reported_instead_of_raised(fakebin: Fakebin) -> None:
    dry_script(fakebin, exit=20, merge={"would_refuse": "live"})
    result = fake_backend(fakebin).classify(spec(), refuse=False)
    assert result.would_refuse == "live"


def test_would_fail_is_not_acted_on(fakebin: Fakebin) -> None:
    dry_script(fakebin, exit=12, merge={"would_fail": "claude_not_found"})
    result = fake_backend(fakebin).classify(spec())
    assert result.would_fail == "claude_not_found"


# --- start --------------------------------------------------------------------------------------


def test_start_returns_the_allocated_run(fakebin: Fakebin, tmp_path: Path) -> None:
    _, _, outcome = files(tmp_path)
    result = fake_backend(fakebin).start(spec(), outcome_file=str(outcome), hook=RELAY)
    assert SESSION.startswith("cs-") and result.csession_id.startswith("cs-")
    assert result.run == 1 and result.forked_from is None
    assert result.argv[1:4] == ("exec", "--session-id", result.csession_id)
    assert os.path.isabs(result.csession_dir)


def test_a_fork_start_names_its_source(fakebin: Fakebin, tmp_path: Path) -> None:
    fakebin.seed_session(SESSION)
    _, _, outcome = files(tmp_path)
    result = fake_backend(fakebin).start(
        spec(resume=SESSION, fork=True, skill=None, cwd="session-dir"),
        outcome_file=str(outcome),
        hook=RELAY,
    )
    assert result.forked_from == SESSION and result.csession_id != SESSION


def test_launch_failed_carries_the_allocated_session_and_run(
    fakebin: Fakebin, tmp_path: Path
) -> None:
    _, _, outcome = files(tmp_path)
    envelope = {
        "code": "launch_failed",
        "message": "no claude",
        "end_reason": "claude_not_found",
        "hint": "install claude",
        "csession_id": SESSION,
        "run": 2,
    }
    fakebin.script("claude-session", "start", exit=12, error=envelope)
    with pytest.raises(StartError) as caught:
        fake_backend(fakebin).start(spec(), outcome_file=str(outcome), hook=RELAY)
    error = caught.value
    assert (error.csession_id, error.run) == (SESSION, 2)
    assert (error.exit_code, error.code, error.tool) == (12, "launch_failed", "claude-session")
    assert error.lower_error == envelope
    assert error.lower_code == "launch_failed"


def test_start_failed_without_an_allocation_names_none(fakebin: Fakebin, tmp_path: Path) -> None:
    _, _, outcome = files(tmp_path)
    fakebin.script("claude-session", "start", exit=12, error={"code": "failed", "message": "db"})
    with pytest.raises(StartError) as caught:
        fake_backend(fakebin).start(spec(), outcome_file=str(outcome), hook=RELAY)
    assert (caught.value.csession_id, caught.value.run, caught.value.code) == (None, None, "failed")


@pytest.mark.parametrize(
    "code, name, expected_code, expected_exit",
    [
        (1, "internal", "internal", 1),
        (2, "usage", "internal", 1),
        (10, "not_found", "not_found", 10),
        (20, "refused", "refused", 20),
        (21, "ambiguous", "ambiguous", 21),
    ],
)
def test_other_start_exits_pass_through(
    fakebin: Fakebin,
    tmp_path: Path,
    code: int,
    name: str,
    expected_code: str,
    expected_exit: int,
) -> None:
    _, _, outcome = files(tmp_path)
    error: dict[str, Any] = {"code": name, "message": "m"}
    if name == "refused":
        error["reason"] = "live"
    fakebin.script("claude-session", "start", exit=code, error=error)
    with pytest.raises(StartError) as caught:
        fake_backend(fakebin).start(spec(), outcome_file=str(outcome), hook=RELAY)
    assert (caught.value.code, caught.value.exit_code) == (expected_code, expected_exit)
    assert caught.value.csession_id is None


def test_start_needs_an_absolute_outcome_file(fakebin: Fakebin) -> None:
    with pytest.raises(ValueError):
        fake_backend(fakebin).start(spec(), outcome_file="out.json", hook=RELAY)
    assert fakebin.invocations("claude-session") == []


# --- the post-start classification check --------------------------------------------------------


def started(fakebin: Fakebin, tmp_path: Path, backend: SessionBackend) -> sb.StartResult:
    _, _, outcome = files(tmp_path)
    return backend.start(spec(), outcome_file=str(outcome), hook=RELAY)


def expected(reads: bool):
    from agent_spawn.third_party import Classification

    return Classification(reads, "implement", None, None, None, None)


def stops(fakebin: Fakebin) -> list[list[str]]:
    return [a for a in fakebin.argvs("claude-session") if a[:1] == ["stop"]]


def test_agreement_is_accepted_and_nothing_is_stopped(fakebin: Fakebin, tmp_path: Path) -> None:
    backend = fake_backend(fakebin)
    run = started(fakebin, tmp_path, backend)
    check = backend.check_classification(run, expected(False), rule_allows=lambda _: False)
    assert check.verdict == "accept" and check.status is not None
    assert check.status.csession_id == run.csession_id
    assert stops(fakebin) == []


def test_a_mismatch_the_cwd_rule_allows_warns_and_continues(
    fakebin: Fakebin, tmp_path: Path
) -> None:
    backend = fake_backend(fakebin)
    run = started(fakebin, tmp_path, backend)
    fakebin.script("claude-session", "status", merge={"run": {"reads_third_party": True}})
    check = backend.check_classification(run, expected(False), rule_allows=lambda _: True)
    assert check.verdict == "warn"
    assert (check.decision.expected, check.decision.actual) == (False, True)
    assert stops(fakebin) == []


def test_a_mismatch_the_rule_forbids_cancels_the_pending_run(
    fakebin: Fakebin, tmp_path: Path
) -> None:
    backend = fake_backend(fakebin, actor="auto:t")
    run = started(fakebin, tmp_path, backend)
    fakebin.script("claude-session", "status", merge={"run": {"reads_third_party": True}})
    check = backend.check_classification(run, expected(False), rule_allows=lambda _: False)
    assert check.verdict == "cancel"
    assert check.decision.cause == "classification_mismatch"
    assert check.cancel is not None and check.cancel_error is None
    assert stops(fakebin) == [["stop", run.csession_id, "--actor=auto:t", "--json"]]
    error = check.error(AGENT)
    assert (error.exit_code, error.code, error.details) == (
        12,
        "launch_failed",
        {"cause": "classification_mismatch"},
    )


def test_a_trusted_actual_run_in_a_forbidden_place_is_cancelled(
    fakebin: Fakebin, tmp_path: Path
) -> None:
    backend = fake_backend(fakebin)
    run = started(fakebin, tmp_path, backend)
    check = backend.check_classification(run, expected(True), rule_allows=lambda actual: actual)
    assert check.verdict == "cancel" and check.decision.cause == "classification_mismatch"


def test_a_taint_by_this_run_counts_as_third_party_reading(
    fakebin: Fakebin, tmp_path: Path
) -> None:
    backend = fake_backend(fakebin)
    run = started(fakebin, tmp_path, backend)
    fakebin.script("claude-session", "status", merge={"tainted": True, "tainted_by_run": run.run})
    check = backend.check_classification(run, expected(True), rule_allows=lambda _: False)
    assert check.verdict == "accept"


def test_a_taint_by_an_earlier_run_does_not(fakebin: Fakebin, tmp_path: Path) -> None:
    backend = fake_backend(fakebin)
    run = started(fakebin, tmp_path, backend)
    fakebin.script("claude-session", "status", merge={"tainted": True, "tainted_by_run": 99})
    check = backend.check_classification(run, expected(True), rule_allows=lambda _: False)
    assert (check.verdict, check.decision.actual) == ("cancel", False)


def test_the_result_file_check_decides_an_allowed_mismatch(
    fakebin: Fakebin, tmp_path: Path
) -> None:
    backend = fake_backend(fakebin)
    run = started(fakebin, tmp_path, backend)
    fakebin.script("claude-session", "status", merge={"run": {"reads_third_party": True}})
    check = backend.check_classification(
        run, expected(False), rule_allows=lambda _: True, result_file_allows=lambda: False
    )
    assert check.verdict == "cancel" and stops(fakebin)


def test_a_status_failure_cancels_as_unknown(fakebin: Fakebin, tmp_path: Path) -> None:
    backend = fake_backend(fakebin)
    run = started(fakebin, tmp_path, backend)
    fakebin.script("claude-session", "status", exit=12, error={"code": "failed", "message": "db"})
    check = backend.check_classification(run, expected(False), rule_allows=lambda _: True)
    assert check.verdict == "cancel" and check.decision.cause == "classification_unknown"
    assert check.status is None
    assert len(stops(fakebin)) == 1
    assert check.error(AGENT).details == {"cause": "classification_unknown"}


@pytest.mark.parametrize(
    "stdout",
    ["", "not json", "[]", "null", '{"csession_id": 5}', "{" * 5000, "\x00"],
)
def test_unparseable_status_cancels_as_unknown(
    fakebin: Fakebin, tmp_path: Path, stdout: str
) -> None:
    backend = fake_backend(fakebin)
    run = started(fakebin, tmp_path, backend)
    fakebin.script("claude-session", "status", stdout=stdout)
    check = backend.check_classification(run, expected(False), rule_allows=lambda _: True)
    assert (check.verdict, check.decision.cause) == ("cancel", "classification_unknown")


@pytest.mark.parametrize(
    "patch",
    [
        {"current_run": 7},
        {"run": {"run": 7}},
        {"csession_id": "cs-bbbbbbbb"},
        {"run": {"reads_third_party": "yes"}},
        {"run": {"reads_third_party": None}},
        {"tainted": "no"},
    ],
)
def test_a_status_for_another_run_or_with_odd_types_cancels_as_unknown(
    fakebin: Fakebin, tmp_path: Path, patch: dict[str, Any]
) -> None:
    backend = fake_backend(fakebin)
    run = started(fakebin, tmp_path, backend)
    fakebin.script("claude-session", "status", merge=patch)
    check = backend.check_classification(run, expected(False), rule_allows=lambda _: True)
    assert (check.verdict, check.decision.cause) == ("cancel", "classification_unknown")


def test_a_failing_cancel_is_reported_not_raised(fakebin: Fakebin, tmp_path: Path) -> None:
    backend = fake_backend(fakebin)
    run = started(fakebin, tmp_path, backend)
    fakebin.script("claude-session", "status", exit=12, error={"code": "failed", "message": "x"})
    fakebin.script("claude-session", "stop", exit=12, error={"code": "failed", "message": "x"})
    check = backend.check_classification(run, expected(False), rule_allows=lambda _: True)
    assert check.verdict == "cancel" and check.cancel is None
    assert check.cancel_error is not None and check.cancel_error.exit_code == 12


# --- the outcome file ---------------------------------------------------------------------------


def done_outcome(session: str = SESSION, run: Any = 1, **changes: Any) -> dict[str, Any]:
    return outcome_object("done", session, run, **changes)


def write(path: Path, data: object) -> None:
    path.write_text(data if isinstance(data, str) else json.dumps(data))


def test_the_outcome_file_is_used_when_it_names_the_run(fakebin: Fakebin, tmp_path: Path) -> None:
    fakebin.seed_session(SESSION, outcome="failed")
    path = tmp_path / "o.json"
    write(path, done_outcome())
    result = fake_backend(fakebin).run_outcome(str(path), SESSION, 1)
    assert (result.source, result.outcome, result.exit_code, result.phase) == (
        "file",
        "done",
        0,
        "session",
    )
    assert result.report is not None and result.report["status"] == "done"


@pytest.mark.parametrize(
    "content",
    [
        done_outcome(run=2),
        done_outcome(session="cs-bbbbbbbb"),
        done_outcome(outcome="victory"),
        done_outcome(outcome=None),
        done_outcome(exit_code="0"),
        done_outcome(exit_code=True),
        done_outcome(phase="middle"),
        done_outcome(run="1"),
        done_outcome(run=True),
        done_outcome(report="x"),
        done_outcome(end_reason=5),
        "",
        "{",
        "not json at all",
        "[1, 2]",
        "null",
        '{"csession_id": "cs-abcdefgh"}',
        "[" * 100000,
    ],
)
def test_a_bad_outcome_file_falls_back_to_status(
    fakebin: Fakebin, tmp_path: Path, content: object
) -> None:
    fakebin.seed_session(SESSION, outcome="failed")
    path = tmp_path / "o.json"
    write(path, content)
    result = fake_backend(fakebin).run_outcome(str(path), SESSION, 1)
    assert (result.source, result.outcome) == ("status", "failed")


def test_a_missing_outcome_file_falls_back_to_status(fakebin: Fakebin, tmp_path: Path) -> None:
    fakebin.seed_session(SESSION, outcome="timeout")
    result = fake_backend(fakebin).run_outcome(str(tmp_path / "nope.json"), SESSION, 1)
    assert (result.source, result.outcome, result.exit_code) == ("status", "timeout", 16)


def test_a_symlink_directory_fifo_and_huge_file_are_not_outcome_files(
    fakebin: Fakebin, tmp_path: Path
) -> None:
    fakebin.seed_session(SESSION, outcome="failed")
    real = tmp_path / "real.json"
    write(real, done_outcome())
    link = tmp_path / "link.json"
    link.symlink_to(real)
    directory = tmp_path / "dir.json"
    directory.mkdir()
    fifo = tmp_path / "fifo.json"
    os.mkfifo(fifo)
    huge = tmp_path / "huge.json"
    write(huge, json.dumps(done_outcome(message="x" * (sb.MAX_OUTCOME_FILE + 10))))
    for path in (link, directory, fifo, huge):
        assert sb.read_outcome_file(str(path), SESSION, 1) is None
    assert sb.read_outcome_file(str(real), SESSION, 1) is not None


def test_a_vanished_session_makes_the_fallback_fail(fakebin: Fakebin, tmp_path: Path) -> None:
    with pytest.raises(CliError) as caught:
        fake_backend(fakebin).run_outcome(str(tmp_path / "none.json"), "cs-zzzzzzzz", 1)
    assert caught.value.exit_code == 10


def test_first_working_at_is_filled_from_status_events(fakebin: Fakebin, tmp_path: Path) -> None:
    fakebin.seed_session(SESSION)
    path = tmp_path / "o.json"
    write(path, done_outcome())
    events = [
        {"event_id": 5, "event_ts": "2026-11-14T10:00:05Z", "run": 1, "to_activity": "working"},
        {"event_id": 3, "event_ts": "2026-11-14T10:00:03Z", "run": 1, "to_activity": "working"},
        {"event_id": 2, "event_ts": "2026-11-14T10:00:02Z", "run": 1, "to_activity": "idle"},
        {"event_id": 1, "event_ts": "2026-11-14T09:00:00Z", "run": 0, "to_activity": "working"},
    ]
    fakebin.script("claude-session", "status", merge={"recent_events": events})
    result = fake_backend(fakebin).run_outcome(str(path), SESSION, 1)
    assert (result.source, result.first_working_at) == ("file", "2026-11-14T10:00:03Z")


def test_first_working_at_stays_null_without_events_or_status(
    fakebin: Fakebin, tmp_path: Path
) -> None:
    path = tmp_path / "o.json"
    write(path, done_outcome())
    # an unknown session: the status call fails, the file still wins
    result = fake_backend(fakebin).run_outcome(str(path), SESSION, 1)
    assert (result.source, result.first_working_at) == ("file", None)


def test_new_outcome_file_paths(tmp_path: Path) -> None:
    state = tmp_path / "state"
    state.mkdir()
    first = sb.new_outcome_file(str(state), AGENT)
    second = sb.new_outcome_file(str(state), AGENT)
    assert first != second
    assert os.path.dirname(first) == str(state / "outcomes" / AGENT) == os.path.dirname(second)
    assert os.path.basename(first).endswith(".json") and len(os.path.basename(first)) == 37
    for directory in (state / "outcomes", state / "outcomes" / AGENT):
        assert oct(directory.stat().st_mode & 0o777) == "0o700"
    assert not os.path.exists(first)


def test_new_outcome_file_rejects_bad_input_and_symlinks(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        sb.new_outcome_file("relative", AGENT)
    with pytest.raises(ValueError):
        sb.new_outcome_file(str(tmp_path), "../escape")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    state = tmp_path / "state"
    state.mkdir()
    (state / "outcomes").symlink_to(elsewhere)
    with pytest.raises(CliError):
        sb.new_outcome_file(str(state), AGENT)
    assert list(elsewhere.iterdir()) == []


# --- status, stop, forget, labels, secret, list, reconcile --------------------------------------


def test_status_is_parsed_and_validated(fakebin: Fakebin) -> None:
    fakebin.seed_session(SESSION, outcome="done")
    status = fake_backend(fakebin).status(SESSION)
    sibling_schemas.validate(status.raw, "claude-session", "cli/status.json")
    assert (status.csession_id, status.current_run, status.run.outcome) == (SESSION, 1, "done")
    assert status.outcome().exit_code == 0
    fakebin.assert_argv("claude-session", ["status", SESSION, "--json"])


def test_status_of_a_run(fakebin: Fakebin) -> None:
    fakebin.seed_session(SESSION)
    fake_backend(fakebin).status(SESSION, 1)
    fakebin.assert_argv("claude-session", ["status", SESSION, "--run=1", "--json"])


def test_status_of_an_unknown_session_is_not_found(fakebin: Fakebin) -> None:
    with pytest.raises(CliError) as caught:
        fake_backend(fakebin).status("cs-zzzzzzzz")
    assert (caught.value.exit_code, caught.value.code, caught.value.tool) == (
        10,
        "not_found",
        "claude-session",
    )


def test_stop_cancels_a_pending_run_and_carries_actor_and_decisions(
    fakebin: Fakebin, tmp_path: Path
) -> None:
    backend = fake_backend(fakebin, actor="auto:q", decision_ids=["d9"])
    run = started(fakebin, tmp_path, backend)
    result = backend.stop(run.csession_id, grace="5s")
    sibling_schemas.validate(result, "claude-session", "cli/stop.json")
    assert result["ended"] is True
    fakebin.assert_argv(
        "claude-session",
        ["stop", run.csession_id, "--actor=auto:q", "--decision-id=d9", "--grace=5s", "--json"],
    )


def test_stop_reports_a_run_that_is_still_live(fakebin: Fakebin) -> None:
    fakebin.seed_session(SESSION, final=False)
    fakebin.script("claude-session", "stop", exit=12, merge={"ended": False})
    result = fake_backend(fakebin).stop(SESSION)
    assert result["ended"] is False


def test_stop_failure_is_passed_through(fakebin: Fakebin) -> None:
    with pytest.raises(CliError) as caught:
        fake_backend(fakebin).stop("cs-zzzzzzzz")
    assert caught.value.exit_code == 10


def test_forget(fakebin: Fakebin) -> None:
    fakebin.seed_session(SESSION)
    backend = fake_backend(fakebin, actor="auto:q")
    result = backend.forget(SESSION)
    assert SESSION in json.dumps(result["forgotten"])
    fakebin.assert_argv("claude-session", ["forget", SESSION, "--actor=auto:q", "--json"])
    with pytest.raises(CliError) as caught:
        backend.forget(SESSION)
    assert (caught.value.exit_code, caught.value.code) == (10, "not_found")


def test_label_add_and_remove(fakebin: Fakebin) -> None:
    fakebin.seed_session(SESSION)
    backend = fake_backend(fakebin, actor="auto:q")
    added = backend.label_add(SESSION, ["pr=12", "team=a"])
    assert set(added["added"]) == {"pr=12", "team=a"}
    removed = backend.label_remove(SESSION, ["pr=12"])
    assert removed["removed"] == ["pr=12"]
    fakebin.assert_calls(
        "claude-session",
        [
            ["label", "add", SESSION, "pr=12", "team=a", "--json"],
            ["label", "remove", SESSION, "pr=12", "--json"],
        ],
    )


@pytest.mark.parametrize("pair", ["", "novalue", "UPPER=x", "k=v w", "k=" + "v" * 300, "-x=1"])
def test_bad_label_pairs_are_usage_errors_and_call_nothing(fakebin: Fakebin, pair: str) -> None:
    backend = fake_backend(fakebin)
    with pytest.raises(CliError) as caught:
        backend.label_add(SESSION, [pair])
    assert caught.value.exit_code == 2
    assert fakebin.invocations("claude-session") == []


def test_label_needs_a_pair(fakebin: Fakebin) -> None:
    with pytest.raises(ValueError):
        fake_backend(fakebin).label_add(SESSION, [])


def test_verify_secret_sends_the_secret_on_stdin_only(fakebin: Fakebin) -> None:
    fakebin.seed_session(SESSION)
    fakebin.script("claude-session", "verify-secret", secret="hunter2")
    backend = fake_backend(fakebin, actor="auto:q", decision_ids=["d"])
    assert backend.verify_secret(SESSION, "hunter2") is True
    assert backend.verify_secret(SESSION, "wrong") is False
    calls = fakebin.invocations("claude-session")
    assert [c.argv for c in calls] == [["verify-secret", SESSION, "--json"]] * 2
    assert [c.stdin for c in calls] == ["hunter2", "wrong"]
    assert all("hunter2" not in " ".join(c.env.values()) for c in calls)


def test_list_by_label_finds_the_agents_session(fakebin: Fakebin, tmp_path: Path) -> None:
    backend = fake_backend(fakebin)
    run = started(fakebin, tmp_path, backend)
    assert backend.list_by_label(f"agent={AGENT}") == [run.csession_id]
    assert backend.list_by_label("agent=ag-bbbbbbbb") == []
    fakebin.assert_argv("claude-session", ["list", "--label=agent=ag-bbbbbbbb", "--json"])


def test_reconcile(fakebin: Fakebin) -> None:
    result = fake_backend(fakebin).reconcile()
    sibling_schemas.validate(result, "claude-session", "cli/reconcile.json")
    fake_backend(fakebin).reconcile(dry_run=True)
    fakebin.assert_calls(
        "claude-session", [["reconcile", "--json"], ["reconcile", "--dry-run", "--json"]]
    )


def test_actor_and_decisions_are_added_only_where_the_command_takes_them(
    fakebin: Fakebin, tmp_path: Path
) -> None:
    fakebin.seed_session(SESSION)
    fakebin.script("claude-session", "verify-secret", secret="s")
    backend = fake_backend(fakebin, actor="auto:q", decision_ids=["d"])
    backend.status(SESSION)
    backend.list_by_label("a=b")
    backend.label_add(SESSION, ["a=b"])
    backend.verify_secret(SESSION, "s")
    backend.reconcile()
    for argv in fakebin.argvs("claude-session"):
        assert not any(a.startswith(("--actor", "--decision-id")) for a in argv)


# --- the tool itself ----------------------------------------------------------------------------


def test_a_missing_claude_session_is_tool_missing(tmp_path: Path) -> None:
    backend = SessionBackend([str(tmp_path / "no-such-claude-session")], gateway=SystemGateway())
    with pytest.raises(CliError) as caught:
        backend.reconcile()
    assert (caught.value.exit_code, caught.value.code, caught.value.tool) == (
        12,
        "tool_missing",
        "claude-session",
    )


def test_a_start_that_cannot_run_is_a_start_error(tmp_path: Path) -> None:
    backend = SessionBackend([str(tmp_path / "no-such-claude-session")], gateway=SystemGateway())
    with pytest.raises(StartError) as caught:
        backend.start(spec(), outcome_file=str(tmp_path / "o.json"), hook=RELAY)
    assert (caught.value.code, caught.value.csession_id) == ("tool_missing", None)


def test_a_hung_claude_session_times_out_and_is_reaped(fakebin: Fakebin) -> None:
    fakebin.script("claude-session", "reconcile", hang=True)
    backend = fake_backend(fakebin, timeout=1.5)
    with pytest.raises(CliError) as caught:
        backend.reconcile()
    assert (caught.value.exit_code, caught.value.code) == (12, "failed")
