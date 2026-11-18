"""What the browse TUI shows and runs, without any Textual import.

The screen is a thin layer over this module: rows and filters come from ``list --json``, the detail
pane from ``show --json``, and every action is one CLI command run as a child process (the same
code paths a script would use). Every string that came from an agent (names, labels, paths, report
text) is passed through ``safe_text`` and bounded here, so no widget ever receives raw untrusted
text.
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol

from .. import gateway as gw
from ..ids import is_agent_id
from ..output import ResultFileTarget, render_json, safe_text, write_result_file

__all__ = [
    "ATTENTION_ORDER",
    "CliResult",
    "Backend",
    "FilterState",
    "STATUS_CYCLE",
    "SubprocessBackend",
    "age_text",
    "agent_cells",
    "detail_text",
    "intent_text",
    "parse_agents",
    "shown",
    "write_intent",
]

#: Statuses the status filter cycles through (``list`` hides torn-down agents).
STATUS_CYCLE: tuple[str | None, ...] = (None, "launching", "live", "parked", "tearing_down")
ATTENTION_ORDER = (
    "unreported",
    "escalated",
    "blocked",
    "failed",
    "timeout",
    "lost",
    "launch_failed",
    "session_missing",
)

MAX_CELL = 40  # characters of one table cell
MAX_VALUE = 300  # characters of one value in the detail pane
MAX_MESSAGE = 300  # characters of an error shown on the status line
MAX_LABELS_SHOWN = 50  # labels listed in the detail pane
MAX_RUNS_SHOWN = 50
MAX_OUTPUT = 8 * 1024 * 1024  # bytes read from one child command


def shown(value: object, limit: int = MAX_VALUE) -> str:
    """``value`` as one escaped, bounded line of plain text (never markup, never raw controls)."""
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=True, default=str)
    text = safe_text(text[: limit * 4])  # cut before escaping: a huge value costs little
    if len(text) > limit:
        text = text[: limit - 1] + "…"
    return text


# --- running the CLI -------------------------------------------------------------------------


@dataclass(frozen=True)
class CliResult:
    """What one CLI child printed: its exit code and its JSON (a result object or an envelope)."""

    exit_code: int
    payload: Any = None
    message: str = ""

    @property
    def ok(self) -> bool:
        return self.exit_code == 0

    @property
    def error_code(self) -> str | None:
        error = self.payload.get("error") if isinstance(self.payload, dict) else None  # pyright: ignore[reportUnknownMemberType]
        code = error.get("code") if isinstance(error, dict) else None  # pyright: ignore[reportUnknownMemberType]
        return code if isinstance(code, str) else None

    def describe(self) -> str:
        """One safe line: what failed, or an empty string on success."""
        if self.ok:
            return ""
        text = self.message
        error = self.payload.get("error") if isinstance(self.payload, dict) else None  # pyright: ignore[reportUnknownMemberType]
        if isinstance(error, dict):
            message = error.get("message")  # pyright: ignore[reportUnknownMemberType]
            if isinstance(message, str) and message:
                text = message
        return f"exit {self.exit_code}: {shown(text or 'no message', MAX_MESSAGE)}"


class Backend(Protocol):
    """How the screen runs agent-spawn commands. ``command`` is the argv after the program name
    (``["ack", "ag-abcdefgh"]``); ``acting`` marks a command that changes something, which then
    carries the human actor and the decision ids."""

    def call(
        self, command: Sequence[str], *, acting: bool = False, timeout: float | None = None
    ) -> CliResult: ...

    def run_on_terminal(self, argv: Sequence[str]) -> int: ...

    def close(self) -> None: ...


class SubprocessBackend:
    """Runs ``agent-spawn --json [--config C] [--actor A] [--decision-id D]... <command>``."""

    def __init__(
        self,
        *,
        config_path: str | None,
        actor: str,
        decision_ids: Sequence[str] = (),
        prefix: Sequence[str] | None = None,
    ) -> None:
        self._prefix = list(prefix) if prefix is not None else [sys.executable, "-m", "agent_spawn"]
        self._config_path = config_path
        self._actor = actor
        self._decision_ids = list(decision_ids)
        self._children: set[subprocess.Popen[bytes]] = set()
        self._lock = threading.Lock()
        self._closed = False

    def argv_for(self, command: Sequence[str], *, acting: bool) -> list[str]:
        argv = [*self._prefix, "--json"]
        if self._config_path is not None:
            argv += ["--config", self._config_path]
        if acting:
            argv += ["--actor", self._actor]
            for decision_id in self._decision_ids:
                argv += ["--decision-id", decision_id]
        return [*argv, *command]

    def call(
        self, command: Sequence[str], *, acting: bool = False, timeout: float | None = None
    ) -> CliResult:
        argv = self.argv_for(command, acting=acting)
        with self._lock:
            if self._closed:
                return CliResult(130, None, "the browser is closing")
            try:
                proc = gw.get_gateway().spawn(
                    argv,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
            except OSError as exc:
                return CliResult(1, None, f"could not run agent-spawn ({exc.strerror})")
            self._children.add(proc)
        try:
            out, err = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()
            return CliResult(124, None, "the command took too long")
        finally:
            with self._lock:
                self._children.discard(proc)
        return _result_of(proc.returncode, out[:MAX_OUTPUT], err[:4096])

    def run_on_terminal(self, argv: Sequence[str]) -> int:
        """Run ``argv`` (what ``attach`` printed) on this terminal and return its exit code."""
        try:
            done = gw.get_gateway().run_on_terminal(argv)
        except OSError:
            return 127
        return done.returncode

    def close(self) -> None:
        """Ask the children this backend started to stop (exact processes, by handle)."""
        with self._lock:
            self._closed = True
            children = list(self._children)
        for proc in children:
            if proc.poll() is None:
                proc.terminate()


def _result_of(code: int, out: bytes, err: bytes) -> CliResult:
    text = out.decode("utf-8", errors="replace").strip()
    payload: Any = None
    if text:
        try:
            payload = json.loads(text)
        except ValueError:
            payload = None
    stderr_line = err.decode("utf-8", errors="replace").strip().splitlines()
    return CliResult(code, payload, stderr_line[-1] if stderr_line else "")


# --- the list ---------------------------------------------------------------------------------


def parse_agents(payload: Any) -> list[dict[str, Any]]:
    """The usable agent objects of ``list --json`` (a malformed entry is skipped)."""
    agents = payload.get("agents") if isinstance(payload, dict) else None  # pyright: ignore[reportUnknownMemberType]
    if not isinstance(agents, list):
        return []
    usable: list[dict[str, Any]] = []
    for agent in agents:  # pyright: ignore[reportUnknownVariableType]
        if not isinstance(agent, dict):
            continue
        agent_id = agent.get("agent_id")  # pyright: ignore[reportUnknownMemberType]
        if (
            isinstance(agent_id, str)
            and is_agent_id(agent_id)
            and isinstance(agent.get("status"), str)
        ):  # pyright: ignore[reportUnknownMemberType]
            usable.append(agent)  # pyright: ignore[reportUnknownArgumentType]
    return usable


def _labels(agent: Mapping[str, Any]) -> list[str]:
    labels = agent.get("labels")
    return [label for label in labels if isinstance(label, str)] if isinstance(labels, list) else []  # pyright: ignore[reportUnknownVariableType]


@dataclass
class FilterState:
    """The three list filters: label text, one status, attention only."""

    label: str = ""
    status: str | None = None
    attention_only: bool = False

    def matches(self, agent: Mapping[str, Any]) -> bool:
        if self.status is not None and agent.get("status") != self.status:
            return False
        if self.attention_only and not agent.get("needs_attention"):
            return False
        terms = self.label.lower().split()
        if terms:
            labels = [label.lower() for label in _labels(agent)]
            return all(any(term in label for label in labels) for term in terms)
        return True

    def next_status(self) -> str | None:
        index = STATUS_CYCLE.index(self.status) if self.status in STATUS_CYCLE else 0
        return STATUS_CYCLE[(index + 1) % len(STATUS_CYCLE)]

    @property
    def active(self) -> bool:
        return bool(self.label.split()) or self.status is not None or self.attention_only

    def summary(self) -> str:
        parts: list[str] = []
        if self.status is not None:
            parts.append(f"status={self.status}")
        if self.attention_only:
            parts.append("attention")
        if self.label.split():
            parts.append(f"label~{shown(self.label, 40)}")
        return " ".join(parts) if parts else "none"


def age_text(stamp: Any, now: datetime) -> str:
    """``5m``, ``3h``, ``2d``: how long ago an RFC 3339 time was (``-`` when unusable)."""
    if not isinstance(stamp, str):
        return "-"
    try:
        then = datetime.fromisoformat(stamp)
    except ValueError:
        return "-"
    if then.tzinfo is None:
        return "-"
    seconds = int((now - then).total_seconds())
    if seconds < 0:
        return "0s"
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m"
    if seconds < 86400:
        return f"{seconds // 3600}h"
    return f"{seconds // 86400}d"


def agent_cells(agent: Mapping[str, Any], now: datetime, *, marked: bool) -> list[str]:
    """The plain-text cells of one row: mark, id, status, attention, kind, name, labels, age."""
    return [
        "*" if marked else " ",
        str(agent["agent_id"]),
        shown(agent.get("status"), MAX_CELL),
        shown(agent.get("needs_attention") or "-", MAX_CELL),
        shown(agent.get("kind") or "-", MAX_CELL),
        shown(agent.get("name") or "-", MAX_CELL),
        shown(" ".join(_labels(agent)) or "-", MAX_CELL),
        age_text(agent.get("created"), now),
    ]


# --- the detail pane --------------------------------------------------------------------------


def _field(lines: list[str], name: str, value: Any) -> None:
    lines.append(f"{name}: {shown('-' if value in (None, '') else value)}")


def detail_text(agent: Mapping[str, Any], show: Mapping[str, Any] | None) -> str:
    """The detail pane as plain text. ``show`` is the payload of ``show --json`` (``None`` while
    only the list's own data is known)."""
    lines: list[str] = []
    full = show.get("agent") if show is not None else None
    data: Mapping[str, Any] = full if isinstance(full, dict) else agent  # pyright: ignore[reportUnknownVariableType]
    _field(lines, "agent", data.get("agent_id"))
    _field(lines, "name", data.get("name"))
    for key in ("status", "needs_attention", "park_reason", "kind", "run_kind", "mode"):
        _field(lines, key, data.get(key))
    for key in ("placement", "repo", "cwd", "csession_id", "run", "session_state"):
        _field(lines, key, data.get(key))
    for key in ("activity", "needs_input_reason", "forked_from", "escalated_by"):
        _field(lines, key, data.get(key))
    for key in ("last_outcome", "last_end_reason", "tainted", "supervisor_lost", "teardown_step"):
        _field(lines, key, data.get(key))
    for key in ("created", "last_live_at", "updated"):
        _field(lines, key, data.get(key))
    labels = _labels(data)
    lines.append("labels:" + ("" if labels else " -"))
    lines.extend(f"  {shown(label)}" for label in labels[:MAX_LABELS_SHOWN])
    if len(labels) > MAX_LABELS_SHOWN:
        lines.append(f"  ... {len(labels) - MAX_LABELS_SHOWN} more")
    for key in ("worktree", "window"):
        value = data.get(key)
        _field(lines, key, _ref_text(value))
    holds = data.get("holds")
    if isinstance(holds, list) and holds:  # pyright: ignore[reportUnknownArgumentType]
        lines.append("holds:")
        lines.extend(f"  {shown(_ref_text(hold))}" for hold in holds[:MAX_LABELS_SHOWN])  # pyright: ignore[reportUnknownVariableType]
    if show is None:
        lines.append("")
        lines.append("(press enter for the full detail)")
        return "\n".join(lines)
    lines.extend(_show_sections(show))
    return "\n".join(lines)


def _ref_text(value: Any) -> str | None:
    if not isinstance(value, dict):
        return None
    parts = [
        str(v) for v in value.values() if isinstance(v, (str, int)) and not isinstance(v, bool)
    ]  # pyright: ignore[reportUnknownVariableType]
    return " ".join(parts) or None


def _show_sections(show: Mapping[str, Any]) -> list[str]:
    lines: list[str] = []
    session = show.get("session")
    if isinstance(session, dict):
        lines.append("session:")
        for key, value in session.items():  # pyright: ignore[reportUnknownVariableType]
            lines.append(f"  {shown(key, 40)}: {shown(value)}")
    supervisor = show.get("supervisor")
    if isinstance(supervisor, dict):
        lines.append(
            "supervisor: "
            + " ".join(f"{shown(k, 20)}={shown(v, 40)}" for k, v in supervisor.items())
        )  # pyright: ignore[reportUnknownVariableType]
    for key in ("third_party_run", "repo_config_changed"):
        if key in show:
            _field(lines, key, show[key])
    runs = show.get("runs")
    if isinstance(runs, list) and runs:  # pyright: ignore[reportUnknownArgumentType]
        lines.append("runs:")
        for run in runs[:MAX_RUNS_SHOWN]:  # pyright: ignore[reportUnknownVariableType]
            if not isinstance(run, dict):
                continue
            outcome = run.get("outcome")  # pyright: ignore[reportUnknownMemberType]
            outcome_name = outcome.get("outcome") if isinstance(outcome, dict) else None  # pyright: ignore[reportUnknownMemberType]
            lines.append(
                f"  run {shown(run.get('run'), 10)}: {shown(run.get('mode'), 20)} "  # pyright: ignore[reportUnknownMemberType]
                f"{shown(run.get('run_kind'), 20)} outcome={shown(outcome_name or '-', 40)} "  # pyright: ignore[reportUnknownMemberType]
                f"ended={shown(run.get('ended_at') or '-', 40)}"  # pyright: ignore[reportUnknownMemberType]
            )
    warnings = show.get("warnings")
    if isinstance(warnings, list) and warnings:  # pyright: ignore[reportUnknownArgumentType]
        lines.append("warnings:")
        lines.extend(f"  {shown(w)}" for w in warnings[:MAX_LABELS_SHOWN])  # pyright: ignore[reportUnknownVariableType]
    return lines


# --- the intent file --------------------------------------------------------------------------


def intent_text(intent: str, selected: Sequence[str]) -> str:
    """The content of ``--result-file`` (schemas/cli/browse-result.json)."""
    if intent not in ("attach", "resume", "back"):
        raise ValueError(f"unknown intent {intent!r}")
    unique = list(dict.fromkeys(selected))
    if intent == "attach" and len(unique) != 1:
        raise ValueError("attach carries exactly one agent id")
    if intent == "resume" and not unique:
        raise ValueError("resume carries at least one agent id")
    return render_json({"intent": intent, "selected": unique}) + "\n"


def write_intent(target: ResultFileTarget, intent: str, selected: Sequence[str]) -> str | None:
    """Write the intent file (atomic, mode 0600, no link followed); a warning on failure."""
    return write_result_file(target, intent_text(intent, selected))
