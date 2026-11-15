"""Supervisor-initiated runs of a run chain (interface.md, Supervision and Third-party runs).

Today that is the automatic report resume: after a headless run ended ``exited-unreported`` the
supervisor starts one more run of the same session with a fixed prompt. The run goes through the
pre-start checks and the classification check like any other ``start``. A refusal or failure at
or before ``start`` raises ``ChainFailure``; the supervisor then parks the agent per the run that
triggered the chain (the chain rule), with the failure in ``post-park``.

Of the five pre-start checks only 1 (the classification dry run) and 3 (the third-party cwd
rule) run here.
TODO(#21): checks 2, 4 and 5 (the records of the cwd, the throwaway check ``third_party_run`` and
the holding of worktree records) join here once wtctl records exist.
TODO(#29): promotion is the other supervisor-initiated run; it will reuse this module.
"""

from __future__ import annotations

import os
import re
import secrets
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


def _private_dir(name: str, *, dir_fd: int | None = None) -> int:
    """An open descriptor of the user's own directory ``name`` (created 0700, never followed
    through a link); a pre-existing one that others can reach is tightened to 0700."""
    try:
        os.mkdir(name, 0o700, dir_fd=dir_fd)
    except FileExistsError:
        pass
    fd = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=dir_fd)
    try:
        info = os.fstat(fd)
        if info.st_uid != os.getuid():
            raise OSError(f"{name} is not a directory of the user")
        if info.st_mode & 0o077:
            os.fchmod(fd, 0o700)
    except BaseException:
        os.close(fd)
        raise
    return fd


def write_prompt(state_dir: str, agent_id: str) -> str:
    """The fixed prompt in a private file (mode 0600, never followed through a link)."""
    base_fd = _private_dir(os.path.join(state_dir, "prompts"))
    try:
        dir_fd = _private_dir(agent_id, dir_fd=base_fd)
    finally:
        os.close(base_fd)
    name = f"auto-report-{secrets.token_hex(8)}.txt"
    try:
        fd = os.open(
            name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
            0o600,
            dir_fd=dir_fd,
        )
        try:
            handle = os.fdopen(fd, "w", encoding="utf-8")
        except BaseException:
            os.close(fd)
            _unlink_quietly(name, dir_fd)
            raise
        try:
            with handle:
                handle.write(REPORT_PROMPT)
        except BaseException:
            _unlink_quietly(name, dir_fd)
            raise
    finally:
        os.close(dir_fd)
    return os.path.join(state_dir, "prompts", agent_id, name)


def _unlink_quietly(name: str, dir_fd: int) -> None:
    try:
        os.unlink(name, dir_fd=dir_fd)
    except OSError:
        pass


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
    sup: Supervisor,
    session: str,
    run: int,
    end_reason: str | None,
    *,
    hooks: object,
    token: str | None,
) -> None:
    """Record a chain run that was canceled before it started, with its ``not-started`` outcome.

    ``end_reason`` None records a run that could not be confirmed ended (``stop`` failed or left
    it live): the row exists with ``ended_at`` but with a null outcome, never a fabricated one.
    """
    assert sup.store is not None
    fields: dict[str, object] = {
        "mode": "headless",
        "auto_resumed": True,
        "hooks": hooks,
        "outcome": None,
        "end_reason": None,
        "exit_code": None,
        "ended_at": sup.gateway.wall_time(),
    }
    if end_reason is not None:
        result = RunOutcome(
            session, run, "launch", "not-started", end_reason, 12, "not-started", None, "status"
        )
        fields.update(
            outcome=outcome_object(sup.agent_id, result, auto_resumed=True),
            end_reason=end_reason,
            exit_code=12,
        )
    else:
        sup.log(
            "warning",
            level="warn",
            csession_id=session,
            message=f"run {run} could not be confirmed ended; recorded without an outcome",
        )
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
    """Cancel the chain run (no process can adopt it yet) and record it; a run that ``stop``
    did not end is recorded without an outcome."""
    ended = stop_pending(sup, backend, session)
    record_unstarted(sup, session, run, "canceled" if ended else None, hooks=hooks, token=token)


def stop_pending(sup: Supervisor, backend: SessionBackend, session: str) -> bool:
    """``claude-session stop``; True only when it answered ``ended: true``."""
    try:
        result = backend.stop(session)
    except CliError as error:
        sup.log("warning", level="warn", message=f"cancel failed: {error.code}")
        return False
    if result.get("ended") is not True:
        sup.log("warning", level="warn", message="cancel failed: the run is still live")
        return False
    return True


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
    """Pre-start checks 1 (the classification dry run) and 3 (the third-party cwd rule); checks
    2, 4 and 5 are TODO(#21)."""
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
    agent_id = sup.agent_id
    spec = RunSpec(agent_id=agent_id, mode="headless", resume=session, prompt_file=prompt)
    classification = _pre_start_checks(sup, backend, spec, cwd)
    if sup.collect_signals():
        raise ChainCanceled  # no run exists yet
    outcome_file = new_outcome_file(sup.state_dir, agent_id)
    token = os.path.basename(outcome_file).removesuffix(".json")
    relay = relay_hook(row["agent_spawn_path"], sup.state_dir, row["config_path"])
    try:
        started = backend.start(spec, outcome_file=outcome_file, hook=relay)
    except StartError as error:
        run, end_reason = _recover_run(sup, backend, session, trigger, error, hooks, token)
        failure = failure_of(error, run=run)
        if run is not None:
            failure.end_reason = end_reason  # the run row and the failure agree
        raise failure from None
    try:
        return _after_start(
            sup, backend, session, started, classification, cwd, hooks, token, outcome_file
        )
    except (ChainFailure, ChainCanceled):
        raise
    except Exception as error:  # noqa: BLE001 - a started run must never be left unrecorded
        sup.log("warning", level="error", message=f"chain run {started.run} failed: {error}")
        try:
            cancel_pending(sup, backend, session, started.run, hooks=hooks, token=token)
        except Exception as problem:  # noqa: BLE001 - best effort
            sup.log("warning", level="warn", message=f"could not cancel the run: {problem}")
        raise ChainFailure("agent-spawn", "failed", run=started.run) from None


def _after_start(
    sup: Supervisor,
    backend: SessionBackend,
    session: str,
    started: StartResult,
    classification: third_party.Classification,
    cwd: object,
    hooks: object,
    token: str,
    outcome_file: str,
) -> RunCtx:
    assert sup.store is not None
    store, agent_id = sup.store, sup.agent_id
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
) -> tuple[int | None, str | None]:
    """The run a failed ``start`` allocated, if any, and its recorded end reason.

    A run named by the error already ended in the launch phase with a real end reason (from the
    error, else ``status --run N``): it is recorded as it ended and not stopped. A run found
    through ``status`` is stopped only when it is the session's pending current run, newer than
    the triggering run (another actor's live run is never touched); it is recorded as canceled
    only when ``stop`` ended it.
    """
    run = error.run if error.csession_id == session else None
    ended_reason: str | None = None
    if run is not None:
        ended_reason = _real_end_reason(backend, session, run, error)
    elif error.maybe_allocated:
        found = _pending_run_after(sup, backend, session, trigger)
        if found is None:
            return None, None
        run = found
    if run is None:
        return None, None
    if ended_reason is None:
        ended_reason = "canceled" if stop_pending(sup, backend, session) else None
    try:
        record_unstarted(sup, session, run, ended_reason, hooks=hooks, token=token)
    except StoreError as problem:
        sup.log("warning", level="warn", message=f"could not record the run: {problem}")
    return run, ended_reason


def _real_end_reason(
    backend: SessionBackend, session: str, run: int, error: StartError
) -> str | None:
    lower = error.lower_error
    reason = lower.get("end_reason") if isinstance(lower, dict) else None  # pyright: ignore[reportUnknownMemberType]
    if isinstance(reason, str):
        return reason
    try:
        status = backend.status(session, run)
    except CliError:
        return None
    if status.run.run == run and status.run.outcome is not None:
        return status.run.end_reason
    return None


def _pending_run_after(
    sup: Supervisor, backend: SessionBackend, session: str, trigger: Final
) -> int | None:
    """The pending run a ``start`` of unknown fate may have left behind, else None."""
    status = None
    for _ in range(2):  # one retry
        try:
            status = backend.status(session)
            break
        except CliError:
            continue
    if status is None:
        sup.log(
            "warning",
            level="warn",
            csession_id=session,
            message="start failed and status is unavailable: a pending run may be an orphan",
        )
        return None
    if (
        status.pending
        and status.run.run == status.current_run
        and status.current_run > trigger.ctx.run
    ):
        return status.current_run
    return None


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
    try:
        # A run the cancel did not end is recorded without an outcome (never as canceled).
        record_unstarted(
            sup,
            started.csession_id,
            started.run,
            "canceled" if check.cancel_error is None else None,
            hooks=hooks,
            token=token,
        )
    except StoreError as problem:
        sup.log("warning", level="warn", message=f"could not record the run: {problem}")
    raise ChainFailure("agent-spawn", cause, run=started.run)
