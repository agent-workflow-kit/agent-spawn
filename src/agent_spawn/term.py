"""termctl as agent-spawn uses it: the window preflight, ``open``, markers, labels and the
cleanup of dead panes (interface.md, Launch order step 1 and 7, Supervision step 2, Labels on
lower tools, Teardown order step 5).

Every call goes through the gateway as an argv list (no shell). Values travel as
``--flag=value`` so a value that starts with ``-`` stays a value. Labels, names and paths are
validated before they reach argv, and nothing an agent wrote (a prompt, a summary, a report) is
ever put in argv or the environment: the only free text is the agent's own labels, which pass the
label grammar and are capped. Everything termctl prints is untrusted: it is parsed defensively,
never quoted in a message and carried to the caller only as ``lower_error``.

This module knows no agent state. ``launch.py`` and ``resume.py`` own the order of the calls and
the parking; ``supervise.py`` owns the markers; ``teardown.py`` owns the window step.
"""

from __future__ import annotations

import json
import math
import re
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from . import gateway as gw
from .labels import parse_label
from .output import CliError, from_lower_tool
from .validate import GrammarError

__all__ = [
    "MAX_AGENT_LABELS",
    "ATTENTION_MARKER",
    "Pane",
    "TermBackend",
    "window_labels",
]

TOOL = "termctl"
TERM_ID = re.compile(r"tc-[a-z2-7]{8}")
#: termctl allows 32 labels per pane: `kind=` and `agent=` and then this many of the agent's.
MAX_AGENT_LABELS = 30
#: termctl's built-in marker name (interface.md, Supervision step 2).
ATTENTION_MARKER = "attention"
#: The hint of a failed window preflight.
PREFLIGHT_HINT = "use `--placement here` or `--mode headless`"
CALL_TIMEOUT = 120.0
#: Marker, label and list calls take no hooks and run in watch loops: they never block long.
QUICK_TIMEOUT = 5.0
#: `open` runs termctl's `pre-open` and `post-open` hooks.
OPEN_TIMEOUT = 300.0
_MAX_OUTPUT_ERROR = 8 * 1024 * 1024
_VALUE = re.compile(r"[A-Za-z0-9._:/#@+=,-]{0,256}")


def _flag(name: str, value: str) -> str:
    if "\0" in value:
        raise CliError.usage(f"{name} cannot be passed to termctl")
    return f"{name}={value}"


def _reject_constant(name: str) -> Any:
    raise ValueError(f"non-finite JSON constant {name}")


def _parse_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value):
        raise ValueError(f"non-finite JSON number {text}")
    return value


def _loads(text: str) -> dict[str, Any] | None:
    if len(text) > _MAX_OUTPUT_ERROR:
        return None
    try:
        data = json.loads(text, parse_constant=_reject_constant, parse_float=_parse_float)
    except (ValueError, RecursionError):
        return None
    return data if isinstance(data, dict) else None  # pyright: ignore[reportUnknownVariableType]


def window_labels(
    kind: str | None, agent_id: str, labels: Sequence[str]
) -> tuple[list[str], list[str]]:
    """The labels of a pane: ``kind=<run kind>``, ``agent=<agent_id>``, then the first
    ``MAX_AGENT_LABELS`` agent labels in stored order. Returns the labels and the warnings (the
    rest are dropped with a warning; a kind that does not fit the label grammar is left out)."""
    result: list[str] = []
    warnings: list[str] = []
    if kind is not None:
        if _VALUE.fullmatch(kind) is not None:
            result.append(f"kind={kind}")
        else:
            warnings.append("the window has no kind= label: the run kind is not a label value")
    result.append(f"agent={agent_id}")
    kept = list(labels[:MAX_AGENT_LABELS])
    for text in kept:
        try:
            label = parse_label(text)
        except GrammarError:
            warnings.append("a label was left off the window: it does not fit termctl's grammar")
            continue
        result.append(str(label))
    if len(labels) > MAX_AGENT_LABELS:
        warnings.append(
            f"{len(labels) - MAX_AGENT_LABELS} label(s) were left off the window: termctl allows "
            f"at most 32 per pane (kind=, agent= and {MAX_AGENT_LABELS} more)"
        )
    return result, warnings


@dataclass(frozen=True)
class Pane:
    """The parts of ``termctl open --json`` agent-spawn uses (validated)."""

    term_id: str
    socket_path: str
    pid: int | None
    state: str  # running, dead or gone


def _parse_pane(data: dict[str, Any] | None) -> Pane | None:
    if data is None:
        return None
    term_id, socket_path = data.get("id"), data.get("socket_path")
    pid, state = data.get("pid"), data.get("state")
    if not isinstance(term_id, str) or TERM_ID.fullmatch(term_id) is None:
        return None
    if not isinstance(socket_path, str) or not socket_path.startswith("/") or "\0" in socket_path:
        return None
    if state not in ("running", "dead", "gone"):
        return None
    if pid is not None and (not isinstance(pid, int) or isinstance(pid, bool) or pid < 1):
        return None
    return Pane(term_id, socket_path, pid, state)


@dataclass
class _Reply:
    code: int
    data: dict[str, Any] | None


class TermBackend:
    """termctl as agent-spawn uses it.

    ``command`` is the resolved argv prefix. ``actor`` and ``decision_ids`` go only to ``open``
    and ``close``, the commands that run hooks and are audited.
    """

    def __init__(
        self,
        command: Sequence[str],
        *,
        actor: str | None = None,
        decision_ids: Sequence[str] = (),
        gateway: gw.Gateway | None = None,
    ) -> None:
        if not command:
            raise ValueError("an empty termctl command")
        self.command = list(command)
        self.actor = actor
        self.decision_ids = list(decision_ids)
        self._gateway = gateway

    @property
    def gateway(self) -> gw.Gateway:
        return self._gateway if self._gateway is not None else gw.get_gateway()

    def audit_flags(self) -> list[str]:
        flags: list[str] = []
        if self.actor is not None:
            flags.append(_flag("--actor", self.actor))
        for value in self.decision_ids:
            flags.append(_flag("--decision-id", value))
        return flags

    def call(self, args: Sequence[str], *, timeout: float = CALL_TIMEOUT) -> _Reply:
        argv = [*self.command, *args]
        try:
            done = self.gateway.run(argv, timeout=timeout)
        except (FileNotFoundError, PermissionError):
            raise CliError(
                "tool_missing",
                "termctl is not installed or not executable",
                tool=TOOL,
                details={"hint": "install termctl"},
            ) from None
        except subprocess.TimeoutExpired:
            raise CliError("failed", "termctl did not answer in time", tool=TOOL) from None
        except gw.OutputTooLarge:
            raise CliError("failed", "termctl printed too much output", tool=TOOL) from None
        except OSError as error:
            raise CliError(
                "failed", f"termctl could not run ({error.strerror})", tool=TOOL
            ) from None
        return _Reply(done.returncode, _loads(done.stdout))

    @staticmethod
    def error_of(reply: _Reply) -> object:
        return reply.data.get("error") if reply.data is not None else None

    def fail(self, reply: _Reply, *, window_preflight: bool = False) -> CliError:
        return from_lower_tool(
            TOOL, reply.code, self.error_of(reply), window_preflight=window_preflight
        )

    # -- the calls ------------------------------------------------------------------------------

    @staticmethod
    def open_args(
        *,
        name: str,
        labels: Sequence[str],
        placement: str | None,
        cwd: str | None,
        argv: Sequence[str],
        keep_on_exit: bool,
    ) -> list[str]:
        args = [_flag("--name", name)]
        if cwd is not None:
            args.append(_flag("--cwd", cwd))
        args.extend(_flag("--label", label) for label in labels)
        if placement is not None:
            args.append(_flag("--placement", placement))
        if keep_on_exit:
            args.append("--keep-on-exit=failed")
        return [*args, "--json"]

    def preflight(
        self,
        *,
        name: str,
        labels: Sequence[str],
        placement: str | None,
        cwd: str | None,
        argv: Sequence[str],
    ) -> dict[str, Any]:
        """``termctl open --dry-run --json`` (launch step 1): termctl's tmux checks. Exit 2
        (``not_supported`` and the like) is a usage error; any other failure exits 12."""
        args = self.open_args(
            name=name, labels=labels, placement=placement, cwd=cwd, argv=argv, keep_on_exit=True
        )
        reply = self.call(["open", "--dry-run", *args, *self.audit_flags(), "--", *argv])
        if reply.code == 0 and reply.data is not None:
            return reply.data
        if reply.code == 0:
            raise CliError("failed", "termctl printed unusable output", tool=TOOL)
        code = reply.code if reply.code == 2 else 12
        error = from_lower_tool(TOOL, code, self.error_of(reply), window_preflight=True)
        if code == 12:
            error.details = {**(error.details or {}), "hint": PREFLIGHT_HINT}
        raise error

    def placements(self, labels: Sequence[str]) -> dict[str, Any]:
        """``termctl placements --label … --json`` (the read-only plan of a dry run)."""
        args = [_flag("--label", label) for label in labels]
        reply = self.call(["placements", *args, "--json"])
        if reply.code != 0:
            raise self.fail(reply, window_preflight=True)
        if reply.data is None:
            raise CliError("failed", "termctl printed unusable output", tool=TOOL)
        return reply.data

    def open(
        self,
        *,
        name: str,
        labels: Sequence[str],
        placement: str | None,
        cwd: str | None,
        argv: Sequence[str],
        override_escalation: bool = False,
    ) -> Pane:
        """The real ``open`` (launch step 7). A failure raises ``CliError`` (``from_lower_tool``:
        termctl's exit 11 and 13 map to ``vetoed`` and ``escalated``)."""
        args = self.open_args(
            name=name, labels=labels, placement=placement, cwd=cwd, argv=argv, keep_on_exit=True
        )
        if override_escalation:
            args.append("--override-escalation")
        reply = self.call(["open", *args, *self.audit_flags(), "--", *argv], timeout=OPEN_TIMEOUT)
        if reply.code != 0:
            # termctl's exit 2 here is not the caller's usage error: the preflight accepted the
            # same request, so it is a placement failure like any other (exit 12).
            raise from_lower_tool(TOOL, 12 if reply.code == 2 else reply.code, self.error_of(reply))
        pane = _parse_pane(reply.data)
        if pane is None:
            raise CliError("failed", "termctl printed unusable output", tool=TOOL)
        return pane

    def mark(self, term_id: str, socket_path: str, marker: str = ATTENTION_MARKER) -> None:
        self._change(["mark", term_id, marker], term_id, socket_path)

    def unmark(self, term_id: str, socket_path: str) -> None:
        self._change(["unmark", term_id], term_id, socket_path)

    def label_add(self, term_id: str, socket_path: str, pairs: Sequence[str]) -> None:
        self._change(["label", "add", term_id, *pairs], term_id, socket_path)

    def label_remove(self, term_id: str, socket_path: str, pairs: Sequence[str]) -> None:
        self._change(["label", "remove", term_id, *pairs], term_id, socket_path)

    def _change(self, args: Sequence[str], term_id: str, socket_path: str) -> None:
        self._check_id(term_id)
        reply = self.call([*args, _flag("--socket", socket_path), "--json"], timeout=QUICK_TIMEOUT)
        if reply.code != 0:
            raise self.fail(reply)

    @staticmethod
    def _check_id(term_id: str) -> None:
        """A pane id from the store is never trusted to be one: a value that is not termctl's id
        grammar (a flag, say) never reaches argv."""
        if TERM_ID.fullmatch(term_id) is None:
            raise CliError("internal", "a termctl call with something that is not a pane id")

    def list_dead(self, agent_id: str, socket_path: str) -> list[str]:
        """The ids of the dead panes labeled ``agent=<agent_id>`` on one server."""
        reply = self.call(
            [
                "list",
                _flag("--label", f"agent={agent_id}"),
                "--dead",
                _flag("--socket", socket_path),
                "--json",
            ],
            timeout=QUICK_TIMEOUT,
        )
        if reply.code != 0:
            raise self.fail(reply)
        panes = reply.data.get("panes") if reply.data is not None else None
        if not isinstance(panes, list):
            raise CliError("failed", "termctl printed unusable output", tool=TOOL)
        found: list[str] = []
        for pane in panes:  # pyright: ignore[reportUnknownVariableType]
            term_id = pane.get("id") if isinstance(pane, dict) else None  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
            if not isinstance(term_id, str) or TERM_ID.fullmatch(term_id) is None:
                raise CliError("failed", "termctl printed unusable output", tool=TOOL)
            found.append(term_id)
        return found

    def find_panes(self, agent_id: str, socket_path: str | None) -> list[tuple[str, str]]:
        """``(id, socket path)`` of every pane labeled ``agent=<agent_id>`` (dead or running) on
        one server (the default one when ``socket_path`` is None): how a pane whose ``open`` did
        not report back is found again."""
        args = ["list", _flag("--label", f"agent={agent_id}")]
        if socket_path is not None:
            args.append(_flag("--socket", socket_path))
        reply = self.call([*args, "--json"], timeout=QUICK_TIMEOUT)
        if reply.code != 0:
            raise self.fail(reply)
        panes = reply.data.get("panes") if reply.data is not None else None
        if not isinstance(panes, list):
            raise CliError("failed", "termctl printed unusable output", tool=TOOL)
        found: list[tuple[str, str]] = []
        for pane in panes:  # pyright: ignore[reportUnknownVariableType]
            term_id = pane.get("id") if isinstance(pane, dict) else None  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
            where = pane.get("socket_path", socket_path)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
            if (
                isinstance(term_id, str)
                and TERM_ID.fullmatch(term_id) is not None
                and isinstance(where, str)
                and where.startswith("/")
                and "\0" not in where
            ):
                found.append((term_id, where))
        return found

    def close(self, term_id: str, socket_path: str) -> None:
        """``termctl close`` of a dead pane (no ``--force``: a running pane exits 20)."""
        self._check_id(term_id)
        reply = self.call(
            ["close", term_id, _flag("--socket", socket_path), *self.audit_flags(), "--json"]
        )
        if reply.code != 0:
            raise self.fail(reply)
