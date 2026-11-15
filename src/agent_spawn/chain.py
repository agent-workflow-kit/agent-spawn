"""Supervisor-initiated runs of a run chain (interface.md, Supervision and Third-party runs).

Today that is the automatic report resume: after a headless run ended ``exited-unreported`` the
supervisor starts one more run of the same session with a fixed prompt. The run goes through the
pre-start checks and the classification check like any other ``start``. A refusal or failure at
or before ``start`` raises ``ChainFailure``; the supervisor then parks the agent per the run that
triggered the chain (the chain rule), with the failure in ``post-park``.

TODO(#21): the pre-start checks 2 and 4 (the records of the cwd: throwaway and worktree
containment, ``third_party_run``) join here once wtctl records exist.
TODO(#29): promotion is the other supervisor-initiated run; it will reuse this module.
"""

from __future__ import annotations

import os
import re
import secrets
import stat
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from . import lifecycle, third_party
from .agent_view import outcome_object
from .outcome_table import DELIBERATE_END_REASONS
from .output import CliError
from .session_backend import (
    RunOutcome,
    RunSpec,
    SessionBackend,
    StartError,
    StartResult,
    new_outcome_file,
    relay_hook,
)
from .store import StoreError

if TYPE_CHECKING:
    from .supervise import Supervisor

#: The fixed prompt of the automatic report resume (interface.md, Supervision step 3).
REPORT_PROMPT = (
    "Your last run ended without a status report. "
    "File your status report now with claude-session report.\n"
)
#: End reasons after which no automatic resume or promotion follows.
NO_RESUME_END_REASONS = DELIBERATE_END_REASONS
_WORD = re.compile(r"[a-z][a-z0-9_]*")


@dataclass
class RunCtx:
    """One run of the chain the supervisor is watching."""

    run: int
    outcome_file: str
    auto_resumed: bool = False
    started: bool = False


@dataclass
class Final:
    """A run that ended: claude-session's outcome and agent-spawn's outcome object."""

    ctx: RunCtx
    result: RunOutcome
    outcome: dict[str, Any]


def reason_word(text: object, fallback: str = "failed") -> str:
    """``text`` as a ``chain_failure.reason`` (lower case, ``[a-z][a-z0-9_]*``)."""
    if isinstance(text, str):
        word = text.lower().replace("-", "_")
        if _WORD.fullmatch(word):
            return word
    return fallback


class ChainFailure(Exception):
    """A supervisor-initiated run was refused or failed before or at its ``start``, or ended in
    its launch phase: ``{tool, reason, run, end_reason}`` of ``post-park``'s ``chain_failure``."""

    def __init__(
        self, tool: str, reason: str, *, run: int | None = None, end_reason: str | None = None
    ) -> None:
        super().__init__(f"{tool}: {reason}")
        self.tool = tool
        self.reason = reason
        self.run = run
        self.end_reason = end_reason

    def as_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "reason": self.reason,
            "run": self.run,
            "end_reason": self.end_reason,
        }


class ChainCanceled(Exception):
    """A signal arrived while no `exec` child existed: the pending chain run (if any) was
    canceled, and the supervisor parks per the previous run."""


def failure_of(error: CliError, *, run: int | None = None) -> ChainFailure:
    """The chain failure of a refusal or failure of a lower tool or of agent-spawn itself."""
    reason = error.reason or error.lower_code or error.code
    lower = error.lower_error or {}
    end_reason = lower.get("end_reason") if isinstance(lower, dict) else None  # pyright: ignore[reportUnknownMemberType]
    return ChainFailure(
        error.tool,
        reason_word(reason),
        run=run,
        end_reason=end_reason if isinstance(end_reason, str) else None,
    )


# --- the fixed prompt ---------------------------------------------------------------------------


def write_prompt(state_dir: str, agent_id: str) -> str:
    """The fixed prompt in a private file (mode 0600, never followed through a link)."""
    base = os.path.join(state_dir, "prompts")
    directory = os.path.join(base, agent_id)
    for path in (base, directory):
        try:
            os.mkdir(path, 0o700)
        except FileExistsError:
            pass
        info = os.lstat(path)
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid():
            raise OSError(f"{path} is not a directory of the user")
    path = os.path.join(directory, f"auto-report-{secrets.token_hex(8)}.txt")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(REPORT_PROMPT)
    return path


def remove_prompt(path: str) -> None:
    """Delete the prompt file, and its directory when nothing else is in it."""
    try:
        os.unlink(path)
    except OSError:
        pass
    try:
        os.rmdir(os.path.dirname(path))
    except OSError:
        pass


# --- the automatic report resume ----------------------------------------------------------------


def record_unstarted(
    sup: Supervisor, session: str, run: int, end_reason: str, *, hooks: object, token: str | None
) -> None:
    """Record a chain run that was canceled before it started, with its ``not-started`` outcome."""
    assert sup.store is not None
    result = RunOutcome(
        session, run, "launch", "not-started", end_reason, 12, "not-started", None, "status"
    )
    outcome = outcome_object(sup.agent_id, result, auto_resumed=True)
    fields: dict[str, object] = {
        "mode": "headless",
        "auto_resumed": True,
        "hooks": hooks,
        "outcome": outcome,
        "end_reason": end_reason,
        "exit_code": 12,
        "ended_at": sup.gateway.wall_time(),
    }
    if token is not None:
        fields["outcome_token"] = token
    with sup.store.lease(lifecycle.lease_key(sup.agent_id), wait=lifecycle.LEASE_WAIT) as lease:
        with sup.store.transaction():
            sup.store.update_agent(lease, sup.agent_id, {"run": run})
            sup.store.put_run(lease, sup.agent_id, run, fields)


def cancel_pending(
    sup: Supervisor,
    backend: SessionBackend,
    session: str,
    run: int,
    *,
    hooks: object,
    token: str | None,
) -> None:
    """Cancel the chain run (no process can adopt it yet) and record it."""
    try:
        backend.stop(session)
    except CliError as error:
        sup.log("warning", level="warn", message=f"cancel failed: {error.code}")
    record_unstarted(sup, session, run, "canceled", hooks=hooks, token=token)


def start_report_resume(
    sup: Supervisor,
    backend: SessionBackend,
    row: dict[str, Any],
    session: str,
    trigger: Final,
) -> RunCtx:
    """Start the automatic report resume run; the pending run is adopted by the next `exec`.

    Raises ``ChainFailure`` (parks per ``trigger``) or ``ChainCanceled`` (a signal arrived).
    """
    assert sup.store is not None
    store, agent_id = sup.store, sup.agent_id
    if sup.terminated or sup.collect_signals():
        raise ChainCanceled
    trigger_row = store.get_run(agent_id, trigger.ctx.run) or {}
    cwd = trigger_row.get("cwd")
    hooks = trigger_row.get("hooks")
    try:
        prompt = write_prompt(sup.state_dir, agent_id)
    except OSError as error:
        sup.log("warning", level="warn", message=f"could not write the prompt: {error.strerror}")
        raise ChainFailure("agent-spawn", "failed") from None
    try:
        return _start(sup, backend, row, session, trigger, prompt, cwd, hooks)
    finally:
        remove_prompt(prompt)


def _pre_start_checks(
    sup: Supervisor, backend: SessionBackend, spec: RunSpec, cwd: object
) -> third_party.Classification:
    """Pre-start checks 1, 3 and 4 (the records of the cwd are TODO(#21))."""
    try:
        classification = backend.classify(spec)
    except CliError as error:
        raise failure_of(error) from None
    if classification.reads_third_party:
        try:
            third_party.require_cwd_rule(True, cwd if isinstance(cwd, str) else None)
        except CliError as error:
            sup.log("third_party_refused", level="warn", details={"reason": error.reason})
            raise failure_of(error) from None
    return classification


def _start(
    sup: Supervisor,
    backend: SessionBackend,
    row: dict[str, Any],
    session: str,
    trigger: Final,
    prompt: str,
    cwd: object,
    hooks: object,
) -> RunCtx:
    assert sup.store is not None
    store, agent_id = sup.store, sup.agent_id
    spec = RunSpec(agent_id=agent_id, mode="headless", resume=session, prompt_file=prompt)
    classification = _pre_start_checks(sup, backend, spec, cwd)
    outcome_file = new_outcome_file(sup.state_dir, agent_id)
    token = os.path.basename(outcome_file).removesuffix(".json")
    relay = relay_hook(row["agent_spawn_path"], sup.state_dir, row["config_path"])
    try:
        started = backend.start(spec, outcome_file=outcome_file, hook=relay)
    except StartError as error:
        run = _recover_run(sup, backend, session, trigger, error, hooks, token)
        raise failure_of(error, run=run) from None
    work = os.path.join(started.csession_dir, "work")
    with store.lease(lifecycle.lease_key(agent_id), wait=lifecycle.LEASE_WAIT) as lease:
        with store.transaction():
            store.update_agent(
                lease, agent_id, {"run": started.run, "run_kind": classification.run_kind}
            )
            store.put_run(
                lease,
                agent_id,
                started.run,
                {
                    "plan_run_kind": classification.run_kind,
                    "mode": "headless",
                    "placement": None,
                    "cwd": cwd if isinstance(cwd, str) else None,
                    "reads_third_party": classification.reads_third_party,
                    "outcome_token": token,
                    "hooks": hooks,
                    "auto_resumed": True,
                },
            )
    _classification_check(sup, backend, started, classification, work, cwd, hooks, token)
    if sup.collect_signals():
        # No `exec` child: cancel our own pending run, skip everything else, park.
        cancel_pending(sup, backend, session, started.run, hooks=hooks, token=token)
        raise ChainCanceled
    return RunCtx(started.run, outcome_file, auto_resumed=True)


def _recover_run(
    sup: Supervisor,
    backend: SessionBackend,
    session: str,
    trigger: Final,
    error: StartError,
    hooks: object,
    token: str,
) -> int | None:
    """The run a failed ``start`` allocated, if any (it is canceled and recorded)."""
    run = error.run if error.csession_id == session else None
    if run is None and error.maybe_allocated:
        try:
            status = backend.status(session)
        except CliError:
            return None
        if status.current_run > trigger.ctx.run:  # the pending run `start` may have left behind
            run = status.current_run
    if run is None:
        return None
    try:
        backend.stop(session)
    except CliError:
        pass
    try:
        record_unstarted(
            sup, session, run, "canceled" if error.run is None else "never_adopted",
            hooks=hooks, token=token,
        )  # fmt: skip
    except StoreError as problem:
        sup.log("warning", level="warn", message=f"could not record the run: {problem}")
    return run


def _classification_check(
    sup: Supervisor,
    backend: SessionBackend,
    started: StartResult,
    classification: third_party.Classification,
    work: str,
    cwd: object,
    hooks: object,
    token: str,
) -> None:
    assert sup.store is not None
    check = backend.check_classification(
        started,
        classification,
        rule_allows=lambda actual, run_cwd: third_party.cwd_rule_allows(
            actual, run_cwd, session_work=work
        ),
        # The supervisor does not know the launcher's `--result-file`; a mismatch toward
        # third-party-reading therefore cancels rather than continues (fail closed).
        result_file_allows=lambda: False,
    )
    status = check.status
    if status is not None:
        reads = bool(status.run.reads_third_party or status.tainted_by_run == started.run)
        with sup.store.lease(lifecycle.lease_key(sup.agent_id), wait=lifecycle.LEASE_WAIT) as lease:
            with sup.store.transaction():
                sup.store.update_agent(
                    lease,
                    sup.agent_id,
                    {
                        "run_kind": status.run.run_kind or classification.run_kind,
                        "tainted": bool(status.tainted),
                    },
                )
                sup.store.put_run(
                    lease,
                    sup.agent_id,
                    started.run,
                    {"run_kind": status.run.run_kind, "reads_third_party": reads},
                )
    if check.verdict == "warn":
        sup.log(
            "classification_mismatch",
            level="warn",
            csession_id=started.csession_id,
            details={"expected": classification.classification},
        )
        return
    if check.verdict == "accept":
        return
    cause = check.decision.cause or "classification_unknown"
    sup.log(cause, level="error", csession_id=started.csession_id)
    if check.cancel_error is not None:
        sup.log(
            "warning",
            level="warn",
            csession_id=started.csession_id,
            message=f"the pending run could not be canceled ({check.cancel_error.code})",
        )
    else:
        try:
            record_unstarted(
                sup, started.csession_id, started.run, "canceled", hooks=hooks, token=token
            )
        except StoreError as problem:
            sup.log("warning", level="warn", message=f"could not record the run: {problem}")
    raise ChainFailure("agent-spawn", cause, run=started.run)
