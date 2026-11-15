"""The outcome table, as a pure function and end to end through a headless launch."""

from __future__ import annotations

import pytest

from agent_spawn.agent_view import lifecycle_events, outcome_object
from agent_spawn.outcome_table import park_row
from agent_spawn.session_backend import RunOutcome
from tests import schema_registry
from tests.conftest import Isolation
from tests.fakebin.harness import Fakebin
from tests.launch_support import Env, result_json

LAUNCH = ("launch", "--mode", "headless", "--skill", "do-it", "--json", "--wait")

# (outcome, end_reason, exit code, park_reason, needs_attention, escalated_by)
TABLE: list[tuple[str, str | None, str | None, str | None, str | None]] = [
    ("done", "normal", "done", None, None),
    ("blocked", "normal", "blocked", "blocked", None),
    ("needs-input", "normal", "escalated", "escalated", None),
    ("failed", "normal", "failed", "failed", None),
    ("exited-unreported", "no_report", "unreported", "unreported", None),
    ("killed", "stopped", "killed", None, None),
    ("killed", "forwarded_term", "killed", None, None),
    ("killed", "forwarded_hup", "killed", None, None),
    ("killed", "forwarded_int", "killed", None, None),
    ("killed", "signal_kill", "killed", "failed", None),
    ("killed", "internal_error", "killed", "failed", None),
    ("timeout", "timeout", "timeout", "timeout", None),
    ("lost", "lost", "lost", "lost", None),
    ("not-started", "canceled", "killed", None, None),
    ("not-started", "launch_interrupted", "killed", None, None),
    ("not-started", "never_adopted", "launch_failed", "launch_failed", None),
    ("not-started", "sandbox_unavailable", "launch_failed", "launch_failed", None),
    ("vetoed", "hook_veto", "vetoed", None, None),
    ("escalated", "hook_escalate", "escalated", "escalated", "claude-session"),
]


@pytest.mark.parametrize(("outcome", "reason", "park", "attention", "by"), TABLE)
def test_park_row(
    outcome: str, reason: str | None, park: str, attention: str | None, by: str | None
) -> None:
    row = park_row(outcome, reason)
    assert (row.park_reason, row.needs_attention, row.escalated_by) == (park, attention, by)


def test_unknown_outcome_parks_failed_with_attention() -> None:
    row = park_row("something-new", None)
    assert (row.park_reason, row.needs_attention) == ("failed", "failed")


@pytest.fixture
def env(fakebin: Fakebin, isolation_env: Isolation) -> Env:
    return Env(fakebin, isolation_env)


@pytest.mark.parametrize(
    ("outcome", "park", "attention", "by", "code", "phase"),
    [
        ("done", "done", None, None, 0, "session"),
        ("blocked", "blocked", "blocked", None, 13, "session"),
        ("needs-input", "escalated", "escalated", None, 13, "session"),
        ("failed", "failed", "failed", None, 18, "session"),
        ("exited-unreported", "unreported", "unreported", None, 14, "session"),
        ("killed", "killed", None, None, 15, "session"),
        ("timeout", "timeout", "timeout", None, 16, "session"),
        ("lost", "lost", "lost", None, 17, "session"),
        ("not-started", "killed", None, None, 12, "launch"),  # launch_interrupted
        ("vetoed", "vetoed", None, None, 11, "launch"),
        ("escalated", "escalated", "escalated", "claude-session", 13, "launch"),
    ],
)
def test_launch_wait_applies_the_outcome_table(
    env: Env,
    fakebin: Fakebin,
    outcome: str,
    park: str,
    attention: str | None,
    by: str | None,
    code: int,
    phase: str,
) -> None:
    fakebin.script("claude-session", "exec", outcome=outcome)
    done = env.run(*LAUNCH)
    body = result_json(done)
    assert done.returncode == code, (done.stdout, done.stderr)
    schema_registry.validate(body, "agent-spawn/cli/launch.json")
    assert body["outcome"] == outcome and body["phase"] == phase and body["exit_code"] == code
    agent = body["agent"]
    assert agent["status"] == "parked" and agent["park_reason"] == park
    assert agent["needs_attention"] == attention and agent["escalated_by"] == by
    assert agent["last_outcome"] == outcome and agent["supervisor_lost"] is False
    assert body["run"]["agent_id"] == agent["agent_id"]
    stored = env.agent(agent["agent_id"])
    assert stored["last_outcome"]["outcome"] == outcome


def test_outcome_object_needs_a_final_run() -> None:
    live = RunOutcome("cs-aaaaaaaa", 1, "session", None, None, None, "working", None, "status")
    with pytest.raises(ValueError, match="final"):
        outcome_object("ag-aaaaaaaa", live)


def test_lifecycle_events_for_a_park() -> None:
    from tests.fixtures import sample_agent

    before = sample_agent(status="live", park_reason=None, needs_attention=None)
    after = sample_agent(status="parked", park_reason="failed", needs_attention="failed")
    events = lifecycle_events(before, after, reads_third_party=False)
    assert [e.event for e in events] == ["post-park", "post-escalate", "post-state-change"]
    assert events[2].payload["previous_status"] == "live"
    unchanged = lifecycle_events(after, after, reads_third_party=None)
    assert unchanged == []
