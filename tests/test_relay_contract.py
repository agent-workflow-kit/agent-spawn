"""The relay against claude-session's pinned hook schema: enums, and every outcome combination.

The sibling schema is the source of truth for what claude-session sends. The relay must accept
exactly the final/outcome/end_reason/exit_code/phase combinations that the pinned schema accepts
and whose relayed form agent-spawn's own hook schema accepts, and drop the rest.
"""

from __future__ import annotations

import io
import itertools
import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from agent_spawn import relay
from agent_spawn.store import Store
from tests import schema_registry, sibling_schemas
from tests.test_relay import AGENT, EVENT_TS, SESSION, payload

OUTCOME_SCHEMA = json.loads(
    (sibling_schemas.pin_dir("claude-session") / "cli" / "outcome.json").read_text()
)
DEFS: dict[str, Any] = OUTCOME_SCHEMA["$defs"]
STATES: list[str] = DEFS["state"]["enum"]
OUTCOMES: list[str] = [v for v in DEFS["outcome_value"]["enum"] if v is not None]
CAUSES: list[str] = DEFS["cause"]["enum"]
END_ENUM: list[str] = DEFS["end_reason"]["anyOf"][1]["enum"]
END_REASONS = [*END_ENUM, "exit_5", "signal_kill"]
FINAL_STATE = {
    "done": "exited",
    "blocked": "exited",
    "needs-input": "exited",
    "failed": "exited",
    "exited-unreported": "exited-unreported",
    "killed": "killed",
    "timeout": "timeout",
    "lost": "lost",
    "not-started": "not-started",
    "vetoed": "not-started",
    "escalated": "not-started",
}
EXIT_CODES = {
    "done": 0, "blocked": 13, "needs-input": 13, "failed": 18, "exited-unreported": 14,
    "killed": 15, "timeout": 16, "lost": 17, "not-started": 12, "vetoed": 11, "escalated": 13,
}  # fmt: skip


def _report(outcome: str) -> dict[str, Any] | None:
    if outcome not in ("done", "blocked", "needs-input", "failed"):
        return None
    return {
        "status": outcome,
        "implicit": False,
        "summary": "s",
        "reason": None,
        "artifact": None,
        "reported_at": EVENT_TS,
    }


def sibling_payload(**changes: Any) -> dict[str, Any]:
    """A payload carrying every field the sibling hook schema requires."""
    body = payload(
        repo=None,
        ts=EVENT_TS,
        pending=False,
        forked_from=None,
        launch_kind="work",
        run_kind="work",
        reads_third_party=False,
        tainted=False,
        mode="headless",
        decision_ids=[],
        previous_activity=None,
        live=True,
        cost_usd=None,
        claude_session_id=None,
        session_started_at=None,
        session_end_reason=None,
    )
    body.update(changes)
    return body


def final_payload(outcome: str, end: str, phase: str, code: int | None = None) -> dict[str, Any]:
    return sibling_payload(
        final=True,
        live=False,
        state=FINAL_STATE[outcome],
        activity=None,
        previous_state="working",
        cause="exit",
        report=_report(outcome),
        outcome=outcome,
        end_reason=end,
        exit_code=EXIT_CODES[outcome] if code is None else code,
        phase=phase,
        pending=False,
    )


def sibling_valid(body: dict[str, Any]) -> bool:
    return not sibling_schemas.errors(body, "claude-session", "hooks/post-state-change.json")


def relayed(body: dict[str, Any]) -> relay.Event | None:
    try:
        return relay.Event(body)
    except ValueError:
        return None


def outbox_form(event: relay.Event) -> dict[str, Any]:
    return {
        "tool": "agent-spawn",
        "hook": "post-state-change",
        "object_type": "agent",
        "object_id": AGENT,
        "event_id": 1,
        "ts": "2026-11-14T00:00:00.000Z",
        "repo": None,
        "labels": [f"agent={AGENT}"],
        "kind": "work",
        "run_kind": "work",
        "reads_third_party": False,
        "tainted": False,
        "mode": "headless",
        "status": "live",
        "needs_attention": None,
        "agent": {},  # the agent object is not under test here
        **event.extra(),
    }


def _outbox_errors(event: relay.Event) -> list[str]:
    """Violations of agent-spawn's hook schema, except the (separately tested) agent object."""
    found = schema_registry.errors(outbox_form(event), "agent-spawn/hooks/post-state-change.json")
    return [e for e in found if not e.startswith("agent:")]


def test_the_relay_enums_equal_the_pinned_sibling_schema() -> None:
    assert relay._STATES == STATES  # pyright: ignore[reportPrivateUsage]
    assert relay._CAUSES == CAUSES  # pyright: ignore[reportPrivateUsage]
    assert sorted(relay._OUTCOMES) == sorted(OUTCOMES)  # pyright: ignore[reportPrivateUsage]
    literals = {
        part
        for part in relay._END_REASON.pattern.split("|")  # pyright: ignore[reportPrivateUsage]
        if part.isidentifier() and not part.endswith("_")
    }
    assert literals == set(END_ENUM)
    for value in END_REASONS:
        assert relay._END_REASON.fullmatch(value)  # pyright: ignore[reportPrivateUsage]
    for value in ("bogus", "exit_", "signal_KILL", "exit_5x"):
        assert not relay._END_REASON.fullmatch(value)  # pyright: ignore[reportPrivateUsage]


def test_the_relay_accepts_exactly_the_combinations_both_schemas_accept() -> None:
    valid = 0
    for outcome, end, phase in itertools.product(OUTCOMES, END_REASONS, ("launch", "session")):
        body = final_payload(outcome, end, phase)
        event = relayed(body)
        ours = event is not None and not _outbox_errors(event)
        if sibling_valid(body):
            valid += 1
            assert event is not None, f"valid payload dropped: {outcome} {end} {phase}"
        if event is not None:
            assert ours, f"relayed a payload the outbox schema rejects: {outcome} {end} {phase}"
        # the relay may be stricter than the sibling only where the sibling schema is silent;
        # it must never accept what agent-spawn's own schema rejects
        assert event is None or ours
    assert valid > 40


def test_an_outcome_with_the_wrong_exit_code_is_dropped() -> None:
    assert relayed(final_payload("done", "normal", "session", code=0)) is not None
    assert relayed(final_payload("done", "normal", "session", code=5)) is None
    assert relayed(final_payload("failed", "normal", "session", code=0)) is None


@pytest.mark.parametrize("state", STATES)
def test_every_sibling_state_is_relayed(state: str) -> None:
    final = state in ("exited", "exited-unreported", "killed", "timeout", "lost", "not-started")
    if final:
        outcome = {"exited": "done", "not-started": "not-started"}.get(state, state)
        body = final_payload(
            outcome,
            "normal" if outcome in ("done", "exited-unreported") else _end_for(outcome),
            "launch" if state == "not-started" else "session",
        )
    else:
        body = sibling_payload(state=state, activity=None if state == "starting" else "working")
        if state == "needs-input":
            body["needs_input_reason"] = "permission"
    assert sibling_valid(body), sibling_schemas.errors(
        body, "claude-session", "hooks/post-state-change.json"
    )
    assert relayed(body) is not None


def _end_for(outcome: str) -> str:
    return {
        "killed": "stopped",
        "timeout": "timeout",
        "lost": "lost",
        "not-started": "canceled",
    }[outcome]


def test_every_sibling_end_reason_is_recorded_and_relayed(tmp_path: Path) -> None:
    state = tmp_path / "state"
    with Store.open(state) as store:
        store.insert_agent(
            AGENT,
            {
                "status": "live",
                "name": "n",
                "mode": "headless",
                "csession_id": SESSION,
                "run": 1,
                "launch_kind": "work",
            },
        )
    sent: list[dict[str, Any]] = []
    for end in END_REASONS:
        for outcome, phase in itertools.product(OUTCOMES, ("launch", "session")):
            body = final_payload(outcome, end, phase)
            if sibling_valid(body):
                sent.append({**body, "event_id": len(sent) + 1})
                break
        else:  # pragma: no cover - a sibling value no outcome can carry
            pytest.fail(f"no valid payload for end_reason {end}")
    for body in sent:
        assert (
            relay.run(
                ["--state-dir", str(state), "--config", str(tmp_path / "c.yaml"), "_session-event"],
                io.BytesIO(json.dumps(body).encode()),
            )
            == 0
        )
    db = sqlite3.connect(f"file:{state / 'state.db'}?mode=ro", uri=True)
    try:
        rows = [json.loads(r[0]) for r in db.execute("SELECT payload FROM events")]
    finally:
        db.close()
    assert [row["end_reason"] for row in rows] == [body["end_reason"] for body in sent]
    assert len(sent) == len(END_REASONS)
