"""wtctl as agent-spawn uses it: worktree records for launch step 4, resume step 5, the records of
a cwd and the holds (interface.md, Launch order, Resume order, Records of a cwd).

Every call goes through the gateway as an argv list (no shell). Values travel as ``--flag=value``
so a value that starts with ``-`` stays a value; ids and paths are validated before they reach
argv, and nothing an agent wrote (a prompt, a label, a record's free text) is ever put in argv or
the environment. Everything wtctl prints is untrusted: it is parsed defensively (a record with a
field of the wrong type makes the whole answer unusable, because a skipped record could hide a
throwaway), never quoted in a message, and carried to the caller only as ``lower_error``.

This module knows no agent state. ``launch.py`` and ``resume.py`` own the order of the calls, the
parking and the stored ids; they use:

* ``WtctlBackend.create`` / ``adopt`` / ``show`` / ``list_records`` / ``forget``: the calls;
* ``failure_kind`` and ``park_fields``: the exit mapping of a failed ``create`` or ``adopt``;
* the pure rules of the records of a cwd: ``records_containing``, ``is_throwaway_root``,
  ``inside_throwaway`` and ``worktrees_containing`` (the third-party rule over the records is
  ``third_party.cwd_rule_allows``).

Behavior that rests on the wtctl interface document alone (the real wtctl was a scaffold when
this was written) is listed in docs/design.md, "Live verification".
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

from . import gateway as gw
from .output import CliError, from_lower_tool
from .validate import REPO_ID

TOOL = "wtctl"
RECORD_ID = re.compile(r"wt-[a-z2-7]{8}")
#: wtctl's name grammar (a worktree's display name and last path component).
WT_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
SHA = re.compile(r"[0-9a-f]{40}")
#: Record states that are not tombstones (wtctl interface, Records).
LIVE_STATES = frozenset({"creating", "active", "unlinked", "missing", "clone_missing", "partial"})
TOMBSTONES = frozenset({"removed", "forgotten", "gone"})
#: ``wtctl create`` may clone, fetch and bootstrap; the other calls only read or write the state.
CREATE_TIMEOUT = 3600.0
CALL_TIMEOUT = 120.0
_PATH_MAX = 4096
_BRANCH_FORBIDDEN = re.compile(r"[\x00-\x20\x7f~^:?*\[\\]")

FailureKind = Literal["vetoed", "escalated", "busy", "bootstrap_failed", "other"]


# --- grammars of the values agent-spawn passes ---------------------------------------------------


def check_repo(value: str) -> str:
    if REPO_ID.fullmatch(value) is None:
        raise CliError.usage("--repo is not a repo id (host[:port]/path)")
    return value


def check_branch(value: str, flag: str) -> str:
    """A conservative subset of ``git check-ref-format --branch``: wtctl is the authority, but a
    value that cannot be a branch name never reaches it (nor starts with ``-``)."""
    bad = (
        not value
        or len(value) > 255
        or value.startswith(("-", "/"))
        or value.endswith(("/", ".", ".lock"))
        or ".." in value
        or "@{" in value
        or "//" in value
        or value == "@"
        or value == "HEAD"  # wtctl exits 2 for a branch or base named HEAD
        or value.startswith("refs/")
        or _BRANCH_FORBIDDEN.search(value) is not None
        or any(part.startswith(".") or part.endswith(".lock") for part in value.split("/"))
    )
    if bad:
        raise CliError.usage(f"{flag} is not a valid branch name")
    return value


def check_sha(value: str) -> str:
    if SHA.fullmatch(value) is None:
        raise CliError.usage("--ref must be a full 40-character lowercase hex sha")
    return value


def check_worktree_name(value: str) -> str:
    if WT_NAME.fullmatch(value) is None:
        raise CliError.usage(
            "--worktree-name must be 1 to 64 characters of letters, digits, '.', '_' or '-' "
            "and start with a letter or digit"
        )
    return value


def resolve_in(root: str, cwd: str | None) -> str:
    """The realpath of ``cwd`` taken relative to ``root`` (absolute ``cwd`` as given); ``root``
    itself without one. The caller checks that the result is still inside ``root``."""
    real_root = os.path.realpath(root)
    return real_root if cwd is None else os.path.realpath(os.path.join(real_root, cwd))


def check_abs_path(value: str, what: str) -> str:
    if not value or "\0" in value or not os.path.isabs(value) or len(value) > _PATH_MAX:
        raise CliError.usage(f"{what} is not an absolute path")
    return value


# --- records ------------------------------------------------------------------------------------


@dataclass(frozen=True)
class WtRecord:
    """The fields of a wtctl record agent-spawn reads (``id``, ``path``, ``object_type``,
    ``state``, ``name``, ``owned`` and ``holders``)."""

    id: str
    path: str
    object_type: str
    state: str
    name: str | None
    owned: bool | None
    holders: tuple[str, ...]

    @property
    def live(self) -> bool:
        return self.state in LIVE_STATES

    @property
    def tombstone(self) -> bool:
        return self.state in TOMBSTONES


def parse_record(data: object) -> WtRecord | None:
    """A record out of wtctl's output, or ``None`` when any field we rely on is unusable."""
    if not isinstance(data, dict):
        return None
    record_id = data.get("id")
    path = data.get("path")
    object_type = data.get("object_type")
    state = data.get("state")
    name = data.get("name")
    owned = data.get("owned")
    holders = data.get("holders")
    if not isinstance(record_id, str) or RECORD_ID.fullmatch(record_id) is None:
        return None
    if not isinstance(path, str) or "\0" in path or not os.path.isabs(path):
        return None
    if len(path) > _PATH_MAX or object_type not in ("worktree", "throwaway"):
        return None
    if not isinstance(state, str) or not (state in LIVE_STATES or state in TOMBSTONES):
        return None
    if name is not None and not isinstance(name, str):
        return None
    if owned is not None and not isinstance(owned, bool):
        return None
    if not isinstance(holders, list) or not all(isinstance(h, str) for h in holders):  # pyright: ignore[reportUnknownVariableType]
        return None
    return WtRecord(
        record_id,
        path,
        object_type,
        state,
        name,
        owned,
        tuple(holders),  # pyright: ignore[reportUnknownArgumentType]
    )


# --- the exit mapping ----------------------------------------------------------------------------


class WtctlError(CliError):
    """A failed wtctl call (``tool: wtctl``, wtctl's error verbatim in ``lower_error``).

    ``record`` / ``record_id`` are the record a ``create`` that exited 12 ``bootstrap_failed``
    names in ``details.record`` (wtctl's schema: the whole record object; an id string is also
    accepted), which the caller verifies and stores before anything else can fail. They are
    ``None`` for every other error: another failure's ``details.record`` (``path_exists`` names a
    foreign holder) is never taken. ``operation`` is the wtctl command (only ``create`` runs
    hooks, so only it can be vetoed or escalated).
    """

    def __init__(
        self,
        base: CliError,
        record_id: str | None,
        operation: str = "",
        record: WtRecord | None = None,
    ) -> None:
        super().__init__(
            base.code,
            base.message,
            exit_code=base.exit_code,
            reason=base.reason,
            tool=base.tool,
            lower_code=base.lower_code,
            lower_error=base.lower_error,
            details=base.details,
            decision_ids=base.decision_ids,
        )
        self.record_id = record_id
        self.record = record
        self.operation = operation  # the wtctl command that failed


def failure_kind(error: CliError) -> FailureKind:
    """How a failed ``create``/``adopt`` parks the agent (interface.md, Launch order step 4)."""
    if error.tool != TOOL:
        return "other"
    # `adopt` fires no hooks and has no override, so it never vetoes or escalates: whatever it
    # exits with is a failure (busy keeps its own rule).
    hooks = getattr(error, "operation", "create") == "create"
    if hooks and error.exit_code == 11:
        return "vetoed"
    if hooks and error.exit_code == 13 and error.code == "escalated":
        return "escalated"
    if error.exit_code == 12 and error.lower_code == "busy":
        return "busy"
    if hooks and error.exit_code == 12 and error.lower_code == "bootstrap_failed":
        return "bootstrap_failed"
    return "other"


def park_fields(kind: FailureKind) -> dict[str, str | None]:
    """The ``park_reason``, ``needs_attention`` and ``escalated_by`` of a failed worktree step."""
    if kind == "vetoed":
        return {"park_reason": "vetoed", "needs_attention": None, "escalated_by": None}
    if kind == "escalated":
        return {"park_reason": "escalated", "needs_attention": "escalated", "escalated_by": TOOL}
    if kind == "busy":  # nothing is wrong with the machinery: no attention, retryable
        return {"park_reason": "launch_failed", "needs_attention": None, "escalated_by": None}
    return {
        "park_reason": "launch_failed",
        "needs_attention": "launch_failed",
        "escalated_by": None,
    }


# --- the calls -----------------------------------------------------------------------------------


@dataclass(frozen=True)
class CreatePlan:
    """Everything one ``wtctl create`` is built from (all values already validated)."""

    agent_id: str
    repo: str
    name: str
    branch: str | None = None
    base: str | None = None
    ref: str | None = None
    detach: bool = False
    no_bootstrap: bool = False
    copy_env: bool = False


@dataclass
class _Reply:
    code: int
    data: dict[str, Any] | None


def _flag(name: str, value: str) -> str:
    if "\0" in value:
        raise CliError.usage(f"{name} cannot be passed to wtctl")
    return f"{name}={value}"


def planned_name(agent_id: str, worktree_name: str | None) -> str:
    """The worktree name: ``--worktree-name``, else the ``agent_id`` (deterministic, so a resume
    finds the record)."""
    return worktree_name if worktree_name else agent_id


class WtctlBackend:
    """wtctl as agent-spawn uses it.

    ``command`` is the resolved argv prefix. ``actor`` and ``decision_ids`` go only to the
    commands that change state and are audited (``create``, ``adopt``, ``forget``); the reads
    (``list``, ``show``) carry neither.
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
            raise ValueError("an empty wtctl command")
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
                "wtctl is not installed or not executable",
                tool=TOOL,
                details={"hint": "install wtctl"},
            ) from None
        except subprocess.TimeoutExpired:
            if args and args[0] == "create":  # the outcome of a create is unknown
                raise CliError("failed", "wtctl did not answer in time", tool=TOOL) from None
            # adopt, forget, show and list change nothing that a retry cannot repeat: retryable
            raise self._fail(
                _Reply(12, {"error": {"code": "busy", "message": "wtctl did not answer in time"}}),
                args[0] if args else "",
            ) from None
        except gw.OutputTooLarge:
            raise CliError("failed", "wtctl printed too much output", tool=TOOL) from None
        except OSError as error:
            raise CliError("failed", f"wtctl could not run ({error.strerror})", tool=TOOL) from None
        return _Reply(done.returncode, _loads(done.stdout))

    def _fail(self, reply: _Reply, operation: str) -> WtctlError:
        error = reply.data.get("error") if reply.data is not None else None
        base = from_lower_tool(TOOL, reply.code, error)
        record_id: str | None = None
        record: WtRecord | None = None
        if (
            operation == "create"
            and reply.code == 12
            and isinstance(error, dict)
            and error.get("code") == "bootstrap_failed"  # pyright: ignore[reportUnknownMemberType]
        ):
            details = error.get("details")  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
            named = details.get("record") if isinstance(details, dict) else None  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
            if isinstance(named, str) and RECORD_ID.fullmatch(named):
                record_id = named
            else:
                record = parse_record(named)
                record_id = record.id if record is not None else None
        wrapped = WtctlError(base, record_id, operation, record)
        if failure_kind(wrapped) == "busy":
            wrapped.details = {"retryable": True}
        return wrapped

    def _unusable(self, what: str) -> CliError:
        return CliError("internal", f"wtctl printed unusable {what}", tool=TOOL)

    # -- the calls ------------------------------------------------------------------------------

    def create_argv(self, plan: CreatePlan, *, override_escalation: bool = False) -> list[str]:
        argv = [
            "create",
            _flag("--repo", plan.repo),
            _flag("--holder", plan.agent_id),
            _flag("--name", plan.name),
        ]
        if plan.branch is not None:
            argv.append(_flag("--branch", plan.branch))
        if plan.base is not None:
            argv.append(_flag("--base", plan.base))
        if plan.ref is not None:
            argv.append(_flag("--ref", plan.ref))
        if plan.detach:
            argv.append("--detach")
        if plan.no_bootstrap:
            argv.append("--no-bootstrap")
        if plan.copy_env:
            argv.append("--copy-env")
        if override_escalation:
            argv.append("--override-escalation")
        return [*argv, *self.audit_flags()]

    def create(
        self, plan: CreatePlan, *, override_escalation: bool = False
    ) -> tuple[WtRecord, bool]:
        """``create``; returns the record and ``reused``. A failure raises ``WtctlError``."""
        reply = self.call(
            [*self.create_argv(plan, override_escalation=override_escalation), "--json"],
            timeout=CREATE_TIMEOUT,
        )
        if reply.code != 0:
            raise self._fail(reply, "create")
        record = parse_record(reply.data.get("record") if reply.data is not None else None)
        if record is None or reply.data is None:
            raise self._unusable("a create result")
        return record, reply.data.get("reused") is True

    def create_dry_run(self, plan: CreatePlan) -> dict[str, Any]:
        """``create --dry-run --json`` verbatim (the dry run's ``worktree_plan``)."""
        reply = self.call([*self.create_argv(plan), "--dry-run", "--json"], timeout=CREATE_TIMEOUT)
        if reply.code != 0:
            raise self._fail(reply, "create")
        if reply.data is None:
            raise self._unusable("a create plan")
        return reply.data

    def adopt(self, path: str, holder: str, *, repo: str | None = None) -> tuple[WtRecord, bool]:
        """``adopt <path> --holder <agent_id> [--repo <id>]``; the record and ``existing``."""
        check_abs_path(path, "the worktree path")
        argv = ["adopt", path, _flag("--holder", holder)]
        if repo is not None:
            argv.append(_flag("--repo", repo))
        reply = self.call([*argv, *self.audit_flags(), "--json"])
        if reply.code != 0:
            raise self._fail(reply, "adopt")
        record = parse_record(reply.data.get("record") if reply.data is not None else None)
        if record is None or reply.data is None:
            raise self._unusable("an adopt result")
        return record, reply.data.get("existing") is True

    def show(self, record_id: str) -> WtRecord | None:
        """``show <id>``: the record (a tombstone too); ``None`` when wtctl does not know it."""
        if RECORD_ID.fullmatch(record_id) is None:
            raise CliError.usage("not a wtctl record id")
        reply = self.call(["show", record_id, "--json"])
        if reply.code == 10:
            return None
        if reply.code != 0:
            raise self._fail(reply, "show")
        record = parse_record(reply.data)
        if record is None:
            raise self._unusable("a record")
        return record

    def list_records(self, *, repo: str | None = None) -> list[WtRecord]:
        """``list [--repo <id>] --json``: the live records (tombstones, if listed, are dropped)."""
        argv = ["list"]
        if repo is not None:
            argv.append(_flag("--repo", repo))
        reply = self.call([*argv, "--json"])
        if reply.code != 0:
            raise self._fail(reply, "list")
        raw = reply.data.get("records") if reply.data is not None else None
        if not isinstance(raw, list):
            raise self._unusable("a record list")
        found: list[WtRecord] = []
        for item in raw:  # pyright: ignore[reportUnknownVariableType]
            record = parse_record(item)
            if record is None:
                raise self._unusable("a record list")
            if record.live:
                found.append(record)
        return found

    def forget(self, record_id: str, holder: str) -> list[str]:
        """``forget <id> --holder <h>``: the holder is released. Returns wtctl's warnings; exit 10
        for a record that is already a tombstone counts as done. Any other failure raises."""
        if RECORD_ID.fullmatch(record_id) is None:
            raise CliError.usage("not a wtctl record id")
        reply = self.call(
            ["forget", record_id, _flag("--holder", holder), *self.audit_flags(), "--json"]
        )
        if reply.code == 10:
            details = _details(reply)
            if details.get("state") in TOMBSTONES:
                return []
            raise self._fail(reply, "forget")
        if reply.code != 0:
            raise self._fail(reply, "forget")
        warnings = reply.data.get("warnings") if reply.data is not None else None
        return [w for w in warnings if isinstance(w, str)] if isinstance(warnings, list) else []  # pyright: ignore[reportUnknownVariableType]


def _details(reply: _Reply) -> dict[str, Any]:
    error = reply.data.get("error") if reply.data is not None else None
    details = error.get("details") if isinstance(error, dict) else None  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    return details if isinstance(details, dict) else {}  # pyright: ignore[reportUnknownVariableType]


def _loads(text: str) -> dict[str, Any] | None:
    if len(text) > 8 * 1024 * 1024:
        return None
    try:
        data = json.loads(text)
    except ValueError:
        return None
    return data if isinstance(data, dict) else None  # pyright: ignore[reportUnknownVariableType]


# --- the records of a cwd (pure) --------------------------------------------------------------


def _real(path: str) -> str:
    return os.path.realpath(path)


def contains(outer: str, inner: str) -> bool:
    """Whether ``outer`` equals ``inner`` or is a path-component ancestor of it (realpaths)."""
    return inner == outer or inner.startswith(outer.rstrip("/") + "/")


def records_containing(records: Sequence[WtRecord], cwd: str) -> list[WtRecord]:
    """Every live record whose recorded path contains ``cwd`` (both resolved with realpath)."""
    real = _real(cwd)
    return [r for r in records if r.live and contains(_real(r.path), real)]


def inside_throwaway(records: Sequence[WtRecord], cwd: str) -> bool:
    return any(r.object_type == "throwaway" for r in records_containing(records, cwd))


def is_throwaway_root(records: Sequence[WtRecord], cwd: str) -> bool:
    """Whether ``cwd`` equals the root path of a live throwaway record and lies inside no
    worktree record."""
    real = _real(cwd)
    root = any(r.live and r.object_type == "throwaway" and _real(r.path) == real for r in records)
    return root and not any(r.object_type == "worktree" for r in records_containing(records, cwd))


def worktrees_containing(records: Sequence[WtRecord], cwd: str) -> list[WtRecord]:
    return [r for r in records_containing(records, cwd) if r.object_type == "worktree"]
