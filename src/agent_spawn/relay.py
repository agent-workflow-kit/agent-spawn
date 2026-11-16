"""``_session-event``: the state relay (interface.md, State relay; design D35, D42).

claude-session runs this as its ``post-state-change`` hook, with the payload on stdin. The relay
only records: in one transaction it stores the event (deduplicated by ``(csession_id,
event_id)``), the run's own fields, the agent's session fields when the event is not from an
earlier run, and queues agent-spawn's ``post-state-change`` outbox event. It calls no lower tool,
runs no hooks, loads no config, and creates or migrates nothing.

It exits 0, always: every failure (malformed argv, hostile stdin, a missing or foreign database)
is logged and never returned to claude-session. Nothing is printed on stdout; the few stderr
lines carry fixed text only, never payload text.
"""

from __future__ import annotations

import json
import os
import re
import signal
import sys
from collections.abc import Mapping, Sequence
from typing import Any

from .agent_view import agent_object
from .hooks import agent_event
from .output import CliError, safe_text
from .records import RecordsError, append_log, parse_record_ts
from .store import SchemaNewer, SchemaOlder, Store, StoreError, StoreMissing

__all__ = ["MAX_PAYLOAD_BYTES", "run"]

MAX_PAYLOAD_BYTES = 1024 * 1024
DEADLINE_SECONDS = 25  # under claude-session's default 30 s hook timeout
MAX_RUN = 100_000
MAX_EVENT_ID = 2**53
MAX_MESSAGE = 4096
MAX_TEXT = 256 * 1024  # report summary/reason, each

_REQUIRED = (
    "previous_state final cause report event_ts phase outcome end_reason exit_code message "
    "sandbox_active"
).split()
_AGENT_ID = re.compile(r"ag-[a-z2-7]{8}")
_CSESSION_ID = re.compile(r"cs-[a-z2-7]{8}")
_STATES = (
    "starting working idle needs-input reported exited exited-unreported killed timeout lost "
    "not-started"
).split()
_ACTIVITIES = ("working", "idle")
_REASONS = ("permission", "question", "elicitation", "escalation")
_CAUSES = (
    "SessionStart UserPromptSubmit PermissionRequest Notification PostToolUse Stop SessionEnd "
    "report launch adopt exit timeout auto_exit stop forget reconcile liveness"
).split()
_OUTCOMES = (
    "done blocked needs-input failed exited-unreported killed timeout lost not-started vetoed "
    "escalated"
).split()
_REPORT_STATUSES = ("done", "blocked", "needs-input", "failed")
_END_REASON = re.compile(
    r"normal|auto_exit|user_exit|forwarded_term|forwarded_hup|forwarded_int|stopped|timeout"
    r"|max_turns|api_error|no_result|no_report|result_rejected|no_session_start|wrapper_lost"
    r"|lost|internal_error|hook_veto|hook_escalate|hook_timeout|canceled|never_adopted"
    r"|launch_interrupted|config_invalid|claude_not_found|sandbox_unavailable"
    r"|socket_path_too_long|invalid_permission_profile|unsafe_cwd|unsafe_path|unsafe_plugin_dir"
    r"|unsafe_read_dir|plugin_dir_missing|git_dir_changed|no_git_cwd|fork_cwd_unsupported"
    r"|no_tty|tmpdir_unsafe|session_dir_unsafe|unsafe_claude_args|unsafe_setting_sources"
    r"|claude_too_old|exit_[0-9]+|signal_[a-z0-9]+"
)
_SHA256 = re.compile(r"[0-9a-f]{64}")
_TS = re.compile(
    r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]+)?"
    r"(?:Z|[+-][0-9]{2}:[0-9]{2})"
)
# The consistency rules of schemas/hooks/post-state-change.json (the same rules claude-session's
# hook schema carries): outcome -> exit code, outcome -> end_reason, phase -> outcome/end_reason.
_EXIT_CODES = {
    "done": 0,
    "blocked": 13,
    "needs-input": 13,
    "failed": 18,
    "exited-unreported": 14,
    "killed": 15,
    "timeout": 16,
    "lost": 17,
    "not-started": 12,
    "vetoed": 11,
    "escalated": 13,
}
_START_FAILURES = frozenset(
    "canceled never_adopted launch_interrupted config_invalid claude_not_found "
    "sandbox_unavailable socket_path_too_long invalid_permission_profile unsafe_cwd unsafe_path "
    "unsafe_plugin_dir unsafe_read_dir plugin_dir_missing git_dir_changed no_git_cwd "
    "fork_cwd_unsupported no_tty tmpdir_unsafe session_dir_unsafe unsafe_claude_args "
    "unsafe_setting_sources claude_too_old".split()
)
_LAUNCH_END = _START_FAILURES | {"hook_veto", "hook_escalate", "hook_timeout"}
_LAUNCH_OUTCOMES = ("not-started", "vetoed", "escalated")
_END_BY_OUTCOME: dict[str, tuple[frozenset[str], re.Pattern[str] | None]] = {
    "timeout": (frozenset(("timeout", "max_turns")), None),
    "lost": (frozenset(("lost",)), None),
    "killed": (
        frozenset("stopped internal_error forwarded_term forwarded_hup forwarded_int".split()),
        re.compile(r"signal_[a-z0-9]+"),
    ),
    "exited-unreported": (
        frozenset(
            "normal user_exit api_error no_result no_report result_rejected "
            "no_session_start".split()
        ),
        re.compile(r"exit_[0-9]+"),
    ),
    "vetoed": (frozenset(("hook_veto", "hook_timeout")), None),
    "escalated": (frozenset(("hook_escalate",)), None),
    "not-started": (_START_FAILURES, None),
}


def _outcome_consistent(
    final: bool, outcome: str | None, end: str | None, code: int | None, phase: str
) -> bool:
    """Whether the final/outcome/end_reason/exit_code/phase combination breaks none of the
    outcome/exit_code/end_reason/phase consistency rules of agent-spawn's schema (``outcome``
    and ``end`` are already known to be well formed). The other rules, such as outcome to state,
    outcome to ``report.status`` and final to state, are not checked by the relay."""
    if not final:
        return outcome is None and end is None and code is None
    if outcome is None or end is None or code != _EXIT_CODES[outcome]:
        return False
    allowed = _END_BY_OUTCOME.get(outcome)
    if allowed is not None:
        names, pattern = allowed
        if end not in names and not (pattern is not None and pattern.fullmatch(end)):
            return False
    if phase == "launch":
        return outcome in _LAUNCH_OUTCOMES and end in _LAUNCH_END
    return outcome not in _LAUNCH_OUTCOMES and end not in _LAUNCH_END


class _Deadline(BaseException):
    """Raised by SIGALRM; a BaseException so no inner ``except Exception`` swallows it."""


class _Ignored(Exception):
    """The event is valid but not for us (reason in ``args[0]``)."""


# --- argv ------------------------------------------------------------------------------------


def parse_argv(argv: Sequence[str]) -> tuple[str, str] | None:
    """``(state_dir, config)`` from the tokens before ``_session-event``; ``None`` when either is
    missing, relative or contains a NUL. Flags the relay does not know are skipped."""
    values: dict[str, str] = {}
    tokens = list(argv)
    index = 0
    while index < len(tokens) and tokens[index] != "_session-event":
        token = tokens[index]
        name, eq, inline = token.partition("=")
        if name in ("--state-dir", "--config"):
            if eq:
                values[name] = inline
            elif index + 1 < len(tokens):
                index += 1
                values[name] = tokens[index]
        elif name in ("--actor", "--decision-id") and not eq:
            index += 1
        index += 1
    state_dir, config = values.get("--state-dir"), values.get("--config")
    for value in (state_dir, config):
        if value is None or not value.startswith("/") or "\0" in value:
            return None
    assert state_dir is not None and config is not None
    return state_dir, config


# --- payload ---------------------------------------------------------------------------------


def _is_int(value: object, low: int, high: int) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and low <= value <= high


def _enum(value: object, allowed: Sequence[str], *, nullable: bool = False) -> bool:
    return (value is None and nullable) or (isinstance(value, str) and value in allowed)


def _text(value: object, limit: int) -> str | None:
    return value[:limit] if isinstance(value, str) else None


def _ts(value: object) -> str | None:
    return value if isinstance(value, str) and _TS.fullmatch(value) else None


class Event:
    """A validated claude-session ``post-state-change`` payload (only the fields we use)."""

    def __init__(self, raw: Mapping[str, Any]) -> None:
        if raw.get("hook") != "post-state-change" or raw.get("object_type") != "session":
            raise ValueError("not a session post-state-change payload")
        object_id = raw.get("object_id")
        if not isinstance(object_id, str) or not _CSESSION_ID.fullmatch(object_id):
            raise ValueError("invalid object_id")
        if not _is_int(raw.get("event_id"), 1, MAX_EVENT_ID):
            raise ValueError("invalid event_id")
        if not _is_int(raw.get("run"), 1, MAX_RUN):
            raise ValueError("invalid run")
        if not _enum(raw.get("state"), _STATES):
            raise ValueError("invalid state")
        if not _enum(raw.get("activity"), _ACTIVITIES, nullable=True):
            raise ValueError("invalid activity")
        if not _enum(raw.get("needs_input_reason"), _REASONS, nullable=True):
            raise ValueError("invalid needs_input_reason")
        self._check_session_fields(raw)
        labels = raw.get("labels")
        if not isinstance(labels, list):
            raise ValueError("invalid labels")
        agents = {
            item.partition("=")[2]
            for item in labels  # pyright: ignore[reportUnknownVariableType]
            if isinstance(item, str) and item.startswith("agent=")
        }
        if len(agents) != 1 or not _AGENT_ID.fullmatch(next(iter(agents))):
            raise ValueError("no single valid agent label")
        self.agent_id: str = next(iter(agents))
        self.csession_id: str = object_id
        self.event_id: int = raw["event_id"]
        self.run: int = raw["run"]
        self.state: str = raw["state"]
        self.activity: str | None = raw.get("activity")
        self.reason: str | None = raw.get("needs_input_reason")
        self.raw = raw

    @staticmethod
    def _check_session_fields(raw: Mapping[str, Any]) -> None:
        """Every field the hook schema requires of ``change: session`` must be present and
        well formed, else the payload is rejected (nothing partial reaches the outbox)."""
        missing = [k for k in _REQUIRED if k not in raw]
        if missing:
            raise ValueError("missing fields")
        final, outcome = raw["final"], raw["outcome"]
        end, code = raw["end_reason"], raw["exit_code"]
        checks = (
            _enum(raw["previous_state"], _STATES, nullable=True),
            isinstance(final, bool),
            _enum(raw["cause"], _CAUSES),
            _ts(raw["event_ts"]) is not None,
            raw["phase"] in ("launch", "session"),
            _enum(outcome, _OUTCOMES, nullable=True),
            end is None or (isinstance(end, str) and _END_REASON.fullmatch(end) is not None),
            code is None or _is_int(code, -(2**31), 2**31),
            raw["message"] is None or isinstance(raw["message"], str),
            raw["sandbox_active"] is None or isinstance(raw["sandbox_active"], bool),
        )
        if not all(checks) or not _outcome_consistent(final, outcome, end, code, raw["phase"]):
            raise ValueError("invalid session fields")

    def first_working_at(self) -> float | None:
        stamp = _ts(self.raw.get("event_ts"))
        if self.activity != "working" or stamp is None:
            return None
        try:
            return parse_record_ts(stamp).timestamp()
        except (ValueError, OverflowError, OSError):
            return None

    def extra(self) -> dict[str, Any]:
        """The ``change: session`` fields of the outbox event, each kept only when well formed."""
        raw = self.raw
        body: dict[str, Any] = {
            "change": "session",
            "csession_id": self.csession_id,
            "run": self.run,
            "state": self.state,
            "activity": self.activity,
            "needs_input_reason": self.reason,
            "session_event_id": self.event_id,
        }
        body["event_ts"] = raw["event_ts"]
        for key in _REQUIRED:
            body[key] = raw[key]
        body["message"] = _text(raw["message"], MAX_MESSAGE)
        report = _report(raw["report"])
        body["report"] = None if report is False else report
        return body


def _report(value: object) -> dict[str, Any] | None | bool:
    """The report when well formed (``None`` for none); ``False`` when malformed (omitted)."""
    if value is None:
        return None
    if not isinstance(value, dict):
        return False
    report: dict[str, Any] = value  # pyright: ignore[reportUnknownVariableType]
    if set(report) != {"status", "implicit", "summary", "reason", "artifact", "reported_at"}:
        return False
    artifact = report["artifact"]
    if artifact is not None:
        if (
            not isinstance(artifact, dict)
            or set(artifact) != {"name", "path", "sha256", "size"}  # pyright: ignore[reportUnknownArgumentType]
            or not isinstance(artifact["name"], str)
            or not isinstance(artifact["path"], str)
            or not isinstance(artifact["sha256"], str)
            or not _SHA256.fullmatch(artifact["sha256"])
            or not _is_int(artifact["size"], 0, 2**53)
        ):
            return False
    if (
        report["status"] not in _REPORT_STATUSES
        or not isinstance(report["implicit"], bool)
        or not all(report[k] is None or isinstance(report[k], str) for k in ("summary", "reason"))
        or _ts(report["reported_at"]) is None
    ):
        return False
    return {
        **report,
        "summary": _text(report["summary"], MAX_TEXT),
        "reason": _text(report["reason"], MAX_TEXT),
    }


def read_payload(stream: Any) -> Mapping[str, Any]:
    """The JSON object on ``stream``: at most ``MAX_PAYLOAD_BYTES``, UTF-8, an object.

    Raises ``ValueError`` (``RecursionError`` is mapped to it) for anything else.
    """
    data: bytes = stream.read(MAX_PAYLOAD_BYTES + 1)
    if len(data) > MAX_PAYLOAD_BYTES:
        raise ValueError("payload too large")
    try:
        parsed = json.loads(data.decode("utf-8"))
    except (RecursionError, UnicodeDecodeError, MemoryError) as exc:
        raise ValueError(f"unreadable payload ({type(exc).__name__})") from None
    if not isinstance(parsed, dict):
        raise ValueError("payload is not an object")
    return parsed  # pyright: ignore[reportUnknownVariableType]


# --- recording -------------------------------------------------------------------------------


def record(store: Store, event: Event) -> str:
    """Record ``event`` in one transaction. Returns ``recorded`` or ``duplicate``; raises
    ``_Ignored`` when the event is not for a known agent's session."""
    with store.transaction():
        agent = store.get_agent(event.agent_id)
        if agent is None:
            raise _Ignored("unknown_agent")
        if agent.get("csession_id") is None:
            raise _Ignored("agent_without_session")
        if agent["csession_id"] != event.csession_id:
            raise _Ignored("session_mismatch")
        if agent["status"] == "torn_down":
            raise _Ignored("torn_down")
        if not store.record_relay_event(event.csession_id, str(event.event_id)):
            return "duplicate"
        store.prune_relay_events()
        store.relay_record_run(
            event.agent_id,
            event.run,
            session_state=event.state,
            activity=event.activity,
            needs_input_reason=event.reason,
            first_working_at=event.first_working_at(),
        )
        recorded_run = agent.get("run")
        if not isinstance(recorded_run, int) or event.run >= recorded_run:
            store.relay_set_agent(
                event.agent_id,
                run=event.run,
                session_state=event.state,
                activity=event.activity,
                needs_input_reason=event.reason,
            )
        after = store.get_agent(event.agent_id)
        assert after is not None
        run_row = store.get_run(event.agent_id, after["run"]) if after.get("run") else None
        reads = None if run_row is None else run_row.get("reads_third_party")
        if after.get("run") != event.run:  # a late event of an earlier run: redact if either is
            own = store.get_run(event.agent_id, event.run)
            reads = _either_reads(reads, None if own is None else own.get("reads_third_party"))
        store.add_events(
            event.agent_id,
            [
                agent_event(
                    "post-state-change",
                    agent_object(store, after),
                    reads_third_party=reads,
                    extra=event.extra(),
                )
            ],
        )
    return "recorded"


def _either_reads(current: bool | None, own: bool | None) -> bool | None:
    """``reads_third_party`` for an event of an earlier run: true when either run is true,
    unknown (``None``, which redacts) when either is unknown, else false."""
    if current is True or own is True:
        return True
    return None if current is None or own is None else False


# --- entry point -----------------------------------------------------------------------------


def _stderr(message: str) -> None:
    """One diagnostic line on fd 2, unbuffered: nothing is left in a stream buffer whose flush
    at interpreter shutdown would fail (a closed pipe) and change the exit code."""
    try:
        line = safe_text(f"agent-spawn: _session-event: {message}") + "\n"
        os.write(2, line.encode("utf-8", "replace"))
    except OSError:
        pass


def _log(state_dir: str, event: str, *, level: str = "warn", **fields: Any) -> None:
    try:
        append_log(state_dir, event, level=level, **fields)
    except (RecordsError, ValueError, OSError):
        pass


def _handle(state_dir: str, stdin: Any) -> None:
    try:
        store = Store.open(state_dir, create=False, migrate=False)
    except StoreMissing as exc:
        _stderr(f"no state database ({type(exc).__name__}); event dropped")
        return
    except SchemaOlder as exc:
        _stderr("state database has an older schema; event dropped")
        _log(
            state_dir,
            "schema_older",
            message="the state database has an older schema; relay event dropped",
            details={"found": exc.found, "expected": exc.expected},
        )
        return
    except SchemaNewer as exc:  # any schema other than ours is `schema_older` in the log
        _stderr(f"cannot open the state database ({exc.code}); event dropped")
        _log(
            state_dir,
            "schema_older",
            message="the state database has a newer schema; relay event dropped",
            details={"found": exc.found, "expected": exc.expected},
        )
        return
    except CliError as exc:
        _stderr(f"cannot open the state database ({exc.code}); event dropped")
        return
    except StoreError as exc:
        _stderr(f"cannot open the state database ({type(exc).__name__}); event dropped")
        _log(state_dir, "warning", message=f"relay: state database unusable ({type(exc).__name__})")
        return
    with store:
        try:
            event = Event(read_payload(stdin))
        except (ValueError, KeyError, TypeError) as exc:
            _stderr(f"payload rejected: {exc}")
            _log(state_dir, "relay_event_ignored", details={"reason": "malformed_payload"})
            return
        ids: dict[str, Any] = {
            "agent_id": event.agent_id,
            "csession_id": event.csession_id,
            "run": event.run,
        }
        try:
            outcome = record(store, event)
        except _Ignored as ignored:
            _log(
                state_dir,
                "relay_event_ignored",
                details={"reason": ignored.args[0], "event_id": event.event_id},
                **{k: v for k, v in ids.items() if k != "agent_id"},
                agent_id=event.agent_id,
            )
            return
        except StoreError as exc:
            _stderr(f"event not recorded ({type(exc).__name__})")
            _log(state_dir, "warning", message=f"relay: not recorded ({type(exc).__name__})", **ids)
            return
        _log(
            state_dir,
            "relay_event",
            level="debug",
            details={
                "event_id": event.event_id,
                "state": event.state,
                "duplicate": outcome == "duplicate",
            },
            **ids,
        )


def _quiet_streams() -> None:
    """Make sure nothing buffered in ``sys.stdout``/``sys.stderr`` can fail the shutdown flush
    (a closed pipe turns it into exit 120): a stream that cannot flush is replaced by a sink."""
    for name in ("stdout", "stderr"):
        try:
            getattr(sys, name).flush()
        except (OSError, ValueError, AttributeError):
            try:
                setattr(sys, name, open(os.devnull, "w", encoding="utf-8"))  # noqa: SIM115
            except OSError:
                pass


def _on_alarm(_signum: int, _frame: object) -> None:
    raise _Deadline


def run(argv: Sequence[str], stdin: Any = None) -> int:
    """The relay. Always returns 0."""
    previous: Any = None
    try:
        try:
            previous = signal.signal(signal.SIGALRM, _on_alarm)
            signal.alarm(DEADLINE_SECONDS)
        except (ValueError, OSError):
            pass  # not the main thread: no deadline
        parsed = parse_argv(argv)
        if parsed is None:
            _stderr("needs an absolute --state-dir and --config; event dropped")
            return 0
        source = stdin if stdin is not None else sys.stdin.buffer
        _handle(parsed[0], source)
    except _Deadline:
        _stderr("deadline exceeded; event dropped")
    except BaseException as exc:  # the relay never fails claude-session's hook
        _stderr(f"internal error ({type(exc).__name__}); event dropped")
    finally:
        try:
            signal.alarm(0)
            if previous is not None:
                signal.signal(signal.SIGALRM, previous)
        except (ValueError, OSError):
            pass
        _quiet_streams()
    return 0
