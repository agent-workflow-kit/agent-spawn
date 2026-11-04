"""The ``state.db`` schema: DDL per version, additive migrations and the column codecs.

Every migration is an explicit, frozen list of SQL statements: a version never changes after it
shipped, and a later change is a new migration that only adds (tables, columns, indexes). The
column specs below drive value encoding and the write whitelist; a test pins them to the DDL, so
a column cannot be added to one and forgotten in the other.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable, Sequence

from .procs import ProcessIdentity

__all__ = [
    "AGENT_COLUMNS",
    "MIGRATIONS",
    "RUN_COLUMNS",
    "SCHEMA_VERSION",
    "Migration",
    "ValueDecodeError",
    "decode_value",
    "encode_value",
]

Migration = Callable[[sqlite3.Connection], None]

STATUSES = ("launching", "live", "parked", "tearing_down", "torn_down")
NEEDS_ATTENTION = (
    "unreported",
    "escalated",
    "blocked",
    "failed",
    "timeout",
    "lost",
    "launch_failed",
    "session_missing",
)
PARK_REASONS = (
    "done",
    "blocked",
    "escalated",
    "failed",
    "unreported",
    "killed",
    "timeout",
    "lost",
    "vetoed",
    "launch_failed",
    "session_missing",
)
ESCALATED_BY = ("wtctl", "termctl", "claude-session")
TEARDOWN_STEPS = (
    "started",
    "hooks_done",
    "worktree_partial",
    "worktree_done",
    "windows_done",
    "session_done",
)


def _one_of(values: Sequence[str]) -> str:
    return "(" + ", ".join("'" + value + "'" for value in values) + ")"


_ID_GLOB = "'ag-" + "[a-z2-7]" * 8 + "'"

# Column kinds: text, int, real, bool (0/1), json (any JSON value), proc (a ProcessIdentity).
AGENT_COLUMNS: dict[str, str] = {
    "csession_id": "text",
    "forked_from": "text",
    "launch_kind": "text",
    "run_kind": "text",
    "mode": "text",
    "placement": "text",
    "term_placement": "text",
    "name": "text",
    "repo": "text",
    "cwd": "text",
    "wt_record_id": "text",
    "wt_path": "text",
    "wt_owned": "bool",
    "host_proc": "proc",
    "tainted": "bool",
    "expect_run_after": "int",
    "agent_spawn_path": "text",
    "state_dir": "text",
    "config_path": "text",
    "claude_session_path": "text",
    "termctl_path": "text",
    "wtctl_path": "text",
    "actor": "text",
    "decision_ids": "json",
    "hooks": "json",
    "settings": "json",
    "pre_resume": "json",
    "launcher": "proc",
    "supervisor": "proc",
    "exec_proc": "proc",
    "starter": "proc",
    "teardown_proc": "proc",
    "promote": "bool",
    "ephemeral": "bool",
    "status": "text",
    "needs_attention": "text",
    "park_reason": "text",
    "escalated_by": "text",
    "promotion_pending": "bool",
    "last_outcome": "json",
    "last_end_reason": "text",
    "run": "int",
    "session_state": "text",
    "activity": "text",
    "needs_input_reason": "text",
    "supervisor_lost": "bool",
    "teardown_step": "text",
    "teardown_partial_record": "text",
    "created": "real",
    "last_live_at": "real",
    "updated": "real",
    "torn_down_at": "real",
}

RUN_COLUMNS: dict[str, str] = {
    "run_kind": "text",
    "plan_run_kind": "text",
    "mode": "text",
    "placement": "text",
    "cwd": "text",
    "reads_third_party": "bool",
    "term_id": "text",
    "socket_path": "text",
    "pane_proc": "proc",
    "outcome": "json",
    "end_reason": "text",
    "exit_code": "int",
    "report_status": "text",
    "auto_resumed": "bool",
    "promoted": "bool",
    "first_working_at": "real",
    "outcome_token": "text",
    "hooks": "json",
    "session_state": "text",
    "activity": "text",
    "needs_input_reason": "text",
    "started_at": "real",
    "ended_at": "real",
}


class ValueDecodeError(ValueError):
    """A stored value that does not decode (the caller maps it to its own error type)."""


def encode_value(kind: str, value: object) -> object:
    """The SQL parameter for ``value``; a wrong Python type is a ``TypeError``, never coerced."""
    if value is None:
        return None
    if kind == "text":
        if not isinstance(value, str):
            raise TypeError(f"expected str, got {type(value).__name__}")
        return value
    if kind == "int":
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(f"expected int, got {type(value).__name__}")
        return value
    if kind == "real":
        if not isinstance(value, int | float) or isinstance(value, bool):
            raise TypeError(f"expected a number, got {type(value).__name__}")
        return float(value)
    if kind == "bool":
        if not isinstance(value, bool):
            raise TypeError(f"expected bool, got {type(value).__name__}")
        return int(value)
    if kind == "json":
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    if kind == "proc":
        if not isinstance(value, ProcessIdentity):
            raise TypeError(f"expected ProcessIdentity, got {type(value).__name__}")
        return json.dumps(value.to_dict(), sort_keys=True, separators=(",", ":"))
    raise ValueError(f"unknown column kind {kind!r}")


def decode_value(kind: str, raw: object) -> object:
    if raw is None:
        return None
    try:
        if kind in ("text", "int", "real"):
            return raw
        if kind == "bool":
            return bool(raw)
        if kind == "json":
            return json.loads(str(raw))
        if kind == "proc":
            data = json.loads(str(raw))
            if not isinstance(data, dict):
                raise ValueError("not an object")
            return ProcessIdentity.from_dict(data)
    except ValueError as exc:
        raise ValueDecodeError(f"stored {kind} value does not decode: {exc}") from exc
    raise ValueError(f"unknown column kind {kind!r}")


_V1: tuple[str, ...] = (
    "CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)",
    # Every id ever allocated, kept after the agent row is deleted: ids are never reused.
    "CREATE TABLE allocated_ids (agent_id TEXT PRIMARY KEY, allocated REAL NOT NULL)",
    f"""CREATE TABLE agents (
    agent_id TEXT PRIMARY KEY CHECK (agent_id GLOB {_ID_GLOB}),
    csession_id TEXT,
    forked_from TEXT,
    launch_kind TEXT,
    run_kind TEXT,
    mode TEXT,
    placement TEXT,
    term_placement TEXT,
    name TEXT NOT NULL,
    repo TEXT,
    cwd TEXT,
    wt_record_id TEXT,
    wt_path TEXT,
    wt_owned INTEGER CHECK (wt_owned IN (0, 1)),
    host_proc TEXT,
    tainted INTEGER CHECK (tainted IN (0, 1)),
    expect_run_after INTEGER,
    agent_spawn_path TEXT,
    state_dir TEXT,
    config_path TEXT,
    claude_session_path TEXT,
    termctl_path TEXT,
    wtctl_path TEXT,
    actor TEXT,
    decision_ids TEXT,
    hooks TEXT,
    settings TEXT,
    pre_resume TEXT,
    launcher TEXT,
    supervisor TEXT,
    exec_proc TEXT,
    starter TEXT,
    teardown_proc TEXT,
    promote INTEGER CHECK (promote IN (0, 1)),
    ephemeral INTEGER NOT NULL DEFAULT 0 CHECK (ephemeral IN (0, 1)),
    status TEXT NOT NULL CHECK (status IN {_one_of(STATUSES)}),
    needs_attention TEXT CHECK (needs_attention IN {_one_of(NEEDS_ATTENTION)}),
    park_reason TEXT CHECK (park_reason IN {_one_of(PARK_REASONS)}),
    escalated_by TEXT CHECK (escalated_by IN {_one_of(ESCALATED_BY)}),
    promotion_pending INTEGER NOT NULL DEFAULT 0 CHECK (promotion_pending IN (0, 1)),
    last_outcome TEXT,
    last_end_reason TEXT,
    run INTEGER,
    session_state TEXT,
    activity TEXT,
    needs_input_reason TEXT,
    supervisor_lost INTEGER NOT NULL DEFAULT 0 CHECK (supervisor_lost IN (0, 1)),
    teardown_step TEXT CHECK (teardown_step IN {_one_of(TEARDOWN_STEPS)}),
    teardown_partial_record TEXT,
    created REAL NOT NULL,
    last_live_at REAL,
    updated REAL NOT NULL,
    torn_down_at REAL
)""",
    "CREATE INDEX agents_status ON agents (status)",
    "CREATE INDEX agents_csession ON agents (csession_id)",
    """CREATE TABLE agent_runs (
    agent_id TEXT NOT NULL REFERENCES agents (agent_id) ON DELETE CASCADE,
    run INTEGER NOT NULL CHECK (run >= 0),
    run_kind TEXT,
    plan_run_kind TEXT,
    mode TEXT,
    placement TEXT,
    cwd TEXT,
    reads_third_party INTEGER CHECK (reads_third_party IN (0, 1)),
    term_id TEXT,
    socket_path TEXT,
    pane_proc TEXT,
    outcome TEXT,
    end_reason TEXT,
    exit_code INTEGER,
    report_status TEXT,
    auto_resumed INTEGER NOT NULL DEFAULT 0 CHECK (auto_resumed IN (0, 1)),
    promoted INTEGER NOT NULL DEFAULT 0 CHECK (promoted IN (0, 1)),
    first_working_at REAL,
    outcome_token TEXT,
    hooks TEXT,
    session_state TEXT,
    activity TEXT,
    needs_input_reason TEXT,
    started_at REAL,
    ended_at REAL,
    PRIMARY KEY (agent_id, run)
)""",
    """CREATE TABLE agent_holds (
    agent_id TEXT NOT NULL REFERENCES agents (agent_id) ON DELETE CASCADE,
    record_id TEXT NOT NULL,
    path TEXT NOT NULL,
    owned INTEGER NOT NULL CHECK (owned IN (0, 1)),
    pos INTEGER NOT NULL,
    released INTEGER NOT NULL DEFAULT 0 CHECK (released IN (0, 1)),
    PRIMARY KEY (agent_id, record_id)
)""",
    """CREATE TABLE agent_labels (
    agent_id TEXT NOT NULL REFERENCES agents (agent_id) ON DELETE CASCADE,
    pos INTEGER NOT NULL,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    PRIMARY KEY (agent_id, pos),
    UNIQUE (agent_id, key, value)
)""",
    "CREATE INDEX agent_labels_pair ON agent_labels (key, value)",
    """CREATE TABLE label_mirror_pending (
    agent_id TEXT NOT NULL REFERENCES agents (agent_id) ON DELETE CASCADE,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    op TEXT NOT NULL CHECK (op IN ('add', 'remove')),
    PRIMARY KEY (agent_id, key, value)
)""",
    """CREATE TABLE relay_events (
    csession_id TEXT NOT NULL,
    event_id TEXT NOT NULL,
    received REAL NOT NULL,
    PRIMARY KEY (csession_id, event_id)
)""",
    "CREATE INDEX relay_events_received ON relay_events (received)",
    """CREATE TABLE events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id TEXT NOT NULL REFERENCES agents (agent_id) ON DELETE CASCADE,
    event TEXT NOT NULL,
    payload TEXT NOT NULL,
    created REAL NOT NULL,
    delivered INTEGER NOT NULL DEFAULT 0 CHECK (delivered IN (0, 1))
)""",
    "CREATE INDEX events_pending ON events (agent_id, delivered, event_id)",
    """CREATE TABLE leases (
    key TEXT PRIMARY KEY,
    holder TEXT NOT NULL,
    acquired REAL NOT NULL
)""",
)


def _v1(conn: sqlite3.Connection) -> None:
    for statement in _V1:
        conn.execute(statement)


# Index n is the migration from schema version n to n + 1. Append only; never edit a shipped one.
MIGRATIONS: tuple[Migration, ...] = (_v1,)
SCHEMA_VERSION = len(MIGRATIONS)
