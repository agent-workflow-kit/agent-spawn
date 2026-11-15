"""Supervisor-initiated runs of a run chain (interface.md, Supervision and Third-party runs).

Today that is the automatic report resume: after a headless run ended ``exited-unreported`` the
supervisor starts one more run of the same session with a fixed prompt. The run goes through the
pre-start checks and the classification check like any other ``start``. A refusal or failure at
or before ``start`` raises ``ChainFailure``; the supervisor then parks the agent per the run that
triggered the chain (the chain rule), with the failure in ``post-park``.

All five pre-start checks run here, before ``start``: the classification dry run, the records of
the cwd (one ``wtctl list``), the cwd rule of a third-party-reading run or the throwaway check of
a trusted one, and the holding of the worktree records of the cwd. A refusal or a failure of
wtctl is a chain failure (``tool`` and ``reason`` of the refusal).
TODO(#29): promotion is the other supervisor-initiated run; it will reuse this module.
"""

from __future__ import annotations

import os
import re
import secrets
import shlex
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
from .worktrees import WtctlBackend, WtRecord, worktrees_containing

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
    actual: RunOutcome | None = None,
) -> None:
    """Record a chain run that was canceled before it started, with its ``not-started`` outcome.

    ``end_reason`` None records a run that could not be confirmed ended (``stop`` failed or left
    it live): the row exists with ``ended_at`` but with a null outcome, never a fabricated one.
    ``actual`` is the real final outcome of a run that ended itself (read from ``status``): it is
    recorded as it is, whatever it was.
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
    if actual is not None and actual.final:
        fields.update(
            outcome=outcome_object(sup.agent_id, actual, auto_resumed=True),
            end_reason=actual.end_reason,
            exit_code=actual.exit_code,
        )
    elif end_reason is not None:
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


class _Wtctl:
    """wtctl as the launcher resolved it, resolved on first use: config ``wtctl_cmd``, else the
    absolute path stored with the agent. ``get()`` is ``None`` under ``wtctl_cmd: null`` (no
    wtctl on this machine: no records) and raises exit 12 ``tool_missing`` (a chain failure)
    when neither resolves, only once a lookup of the records of a cwd needs it."""

    def __init__(self, sup: Supervisor, backend: SessionBackend, row: dict[str, Any]) -> None:
        self._sup = sup
        self._actor = backend.actor
        self._row = row
        self._backend: WtctlBackend | None = None

    def get(self) -> WtctlBackend | None:
        from .launch import resolve_tool  # late: launch is a big module, this is all we need

        command = self._sup.config.wtctl_cmd
        if command is None:
            return None
        if self._backend is None:
            stored = self._row.get("wtctl_path")
            try:
                argv = resolve_tool(command, "wtctl")
            except CliError as missing:
                if not isinstance(stored, str) or not stored:
                    raise
                try:
                    argv = resolve_tool(shlex.split(stored), "wtctl")
                except (CliError, ValueError):
                    raise missing from None
            self._backend = WtctlBackend(
                argv,
                actor=self._actor,
                decision_ids=self._row.get("decision_ids") or [],
                gateway=self._sup.gateway,
            )
        return self._backend

    def list_records(self) -> list[WtRecord]:
        backend = self.get()
        return backend.list_records() if backend is not None else []


def _same_path(a: str, b: str) -> bool:
    try:
        return os.path.realpath(a) == os.path.realpath(b)
    except (ValueError, OSError):
        return False


def _real_cwd(cwd: object) -> str | None:
    """The cwd of the triggering run as the rule judges it: ``session-dir`` as it is, a path by
    its realpath, anything else (null, not a string, a path the OS refuses) as unknown."""
    if not isinstance(cwd, str) or not cwd:
        return None
    if cwd == third_party.SESSION_DIR:
        return cwd
    try:
        return os.path.realpath(cwd)
    except (ValueError, OSError):
        return None


def _pre_start_checks(
    sup: Supervisor,
    backend: SessionBackend,
    spec: RunSpec,
    row: dict[str, Any],
    session: str,
    cwd: object,
) -> tuple[third_party.Classification, str | None, _Wtctl]:
    """Pre-start checks 1 to 4 (the holds, check 5, follow in ``_take_holds``): the
    classification dry run, the records of the cwd, the cwd rule of a third-party-reading run or
    the throwaway check of a trusted one. Returns the classification, the realpath cwd the run
    uses and wtctl (``None`` without one)."""
    try:
        classification = backend.classify(spec)
    except CliError as error:
        raise failure_of(error) from None
    reads = classification.reads_third_party
    real = _real_cwd(cwd)
    wt = _Wtctl(sup, backend, row)
    session_work: str | None = None
    if reads and real is not None and real != third_party.SESSION_DIR:
        try:  # the one real path besides a throwaway root a third-party run may use
            session_work = os.path.join(backend.status(session).csession_dir, "work")
        except CliError:
            session_work = None  # unknown: the rule fails closed
    try:
        in_work = session_work is not None and real == os.path.realpath(session_work)
        needs = real is not None and real != third_party.SESSION_DIR and not in_work
        records = wt.list_records() if needs else []
        third_party.require_cwd_rule(reads, real, session_work=session_work, records=records)
    except CliError as error:
        if error.reason in ("third_party_cwd", "third_party_run"):
            sup.log("third_party_refused", level="warn", details={"reason": error.reason})
        raise failure_of(error) from None
    return classification, real, wt


def _take_holds(sup: Supervisor, wt: _Wtctl, row: dict[str, Any], cwd: str | None) -> None:
    """Pre-start check 5: hold every worktree record of the cwd the agent does not hold yet
    (``wtctl adopt --holder``). Throwaways are never held. The records are looked up again, so a
    record created since the checks is not missed."""
    assert sup.store is not None
    if cwd is None or cwd == third_party.SESSION_DIR:
        return
    store, agent_id = sup.store, sup.agent_id
    try:
        backend = wt.get()
        if backend is None:
            return
        around = worktrees_containing(backend.list_records(), cwd)
        held = {h["record_id"] for h in store.get_holds(agent_id)}
        for record in around:
            if record.id in held and agent_id in record.holders:
                continue  # both sides agree that the agent holds it
            adopted, _ = backend.adopt(record.path, agent_id)  # idempotent: re-adds a lost holder
            with store.lease(lifecycle.lease_key(agent_id), wait=lifecycle.LEASE_WAIT) as lease:
                store.reacquire_hold(
                    lease, agent_id, adopted.id, adopted.path, owned=adopted.owned is True
                )
        _store_nearest(sup, around, row)
    except CliError as error:
        raise failure_of(error) from None


def _store_nearest(sup: Supervisor, around: list[WtRecord], row: dict[str, Any]) -> None:
    """No worktree was created or adopted at launch: the agent's ``worktree`` is the nearest
    record it holds for its cwd (as the launcher does)."""
    assert sup.store is not None
    current = sup.store.get_agent(sup.agent_id)
    if not around or current is None or current.get("wt_record_id") is not None:
        return
    nearest = max(around, key=lambda r: len(os.path.realpath(r.path)))
    with sup.store.lease(lifecycle.lease_key(sup.agent_id), wait=lifecycle.LEASE_WAIT) as lease:
        sup.store.update_agent(
            lease,
            sup.agent_id,
            {"wt_record_id": nearest.id, "wt_path": nearest.path, "wt_owned": False},
            touch=False,
        )


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
    classification, real_cwd, wt = _pre_start_checks(sup, backend, spec, row, session, cwd)
    if sup.collect_signals():
        raise ChainCanceled  # no run exists yet
    _take_holds(sup, wt, row, real_cwd)
    if sup.collect_signals():
        raise ChainCanceled  # no run exists yet (a hold taken stays: teardown releases it)
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
            sup, backend, session, started, classification, cwd, hooks, token, outcome_file, wt
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
    wt: _Wtctl,
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
    _classification_check(sup, backend, started, classification, work, hooks, token, wt)
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
    actual: RunOutcome | None = None
    if run is not None:
        # The run ended itself in the launch phase: never stop it, never invent its outcome.
        actual = _real_outcome(backend, session, run)
        ended_reason = actual.end_reason if actual is not None else None
    elif error.maybe_allocated:
        found = _pending_run_after(sup, backend, session, trigger)
        if found is None:
            return None, None
        run = found
    if run is None:
        return None, None
    if error.run is None or error.csession_id != session:
        ended_reason = "canceled" if stop_pending(sup, backend, session) else None
    try:
        record_unstarted(sup, session, run, ended_reason, hooks=hooks, token=token, actual=actual)
    except StoreError as problem:
        sup.log("warning", level="warn", message=f"could not record the run: {problem}")
    return run, ended_reason


def _real_outcome(backend: SessionBackend, session: str, run: int) -> RunOutcome | None:
    """The final outcome of ``run`` from ``status --run N``, else None (unreadable or live)."""
    try:
        status = backend.status(session, run)
    except CliError:
        return None
    outcome = status.outcome()
    return outcome if status.run.run == run and outcome.final else None


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
    hooks: object,
    token: str,
    wt: _Wtctl,
) -> None:
    assert sup.store is not None
    looked_up: list[list[WtRecord]] = []

    def rule_allows(actual: bool, run_cwd: str | None) -> bool:
        # The run's actual cwd is judged by the records as they are now (one `wtctl list` for
        # the whole check); a lookup error raises and cancels the run as unknown. The session's
        # own `work/` lies in no record: no lookup for it.
        if not looked_up:
            in_work = run_cwd is not None and _same_path(run_cwd, work)
            needs = run_cwd is not None and run_cwd != third_party.SESSION_DIR and not in_work
            looked_up.append(wt.list_records() if needs else [])
        return third_party.cwd_rule_allows(actual, run_cwd, session_work=work, records=looked_up[0])

    check = backend.check_classification(
        started,
        classification,
        rule_allows=rule_allows,
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
