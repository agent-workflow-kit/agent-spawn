"""``launch`` for headless agents without a worktree (interface.md, Launch order, steps 1 to 9).

The order: usage checks and preflight (tool resolution, prompt copies, classification, pre-start
checks), ``pre-launch`` hooks, allocation, the session (``claude-session start`` and the
post-start classification check), the detached supervisor, and the return.

Signals (interface.md, Signals). Until a supervisor is recorded the handlers raise
``LaunchInterrupted`` in the main thread, which stops the in-flight lower-tool call (the gateway
kills exactly its own child), cancels a pending run, parks the agent ``killed`` without attention
and exits 130 or 143. Once the supervisor is recorded (just before it is acknowledged) the
handlers only set a flag and the wait loop forwards SIGTERM to the exact verified supervisor.
"""

from __future__ import annotations

import os
import shlex
import shutil
import signal
import stat
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import FrameType
from typing import Any

from . import gateway as gw
from . import hooks as hooklib
from . import lifecycle, procs, third_party
from .agent_view import agent_object, ts
from .config import Config
from .ids import new_agent_id
from .labels import LabelSet, job_label, parse_run_labels
from .output import (
    EXIT_SIGINT,
    EXIT_SIGTERM,
    EXIT_TIMED_OUT,
    CliError,
    Outcome,
    PayloadKind,
    error_outcome,
    result_object,
)
from .records import RecordsError, append_audit, append_log
from .session_backend import (
    RunSpec,
    SessionBackend,
    StartError,
    StartResult,
    new_outcome_file,
    relay_hook,
)
from .store import Store, StoreError, default_state_dir
from .validate import GrammarError, default_name, parse_duration

POLL = 0.1  # seconds between looks at the state while waiting
PRE_LAUNCH_BODY_LABEL_CAP = 64
ACK_TIMEOUT = 60.0
DEAD_SUPERVISOR_POLLS = 5  # looks at an agent whose supervisor is dead before giving up
WORKTREE_FLAGS = ("branch", "base", "ref", "worktree_name")
WORKTREE_SWITCHES = ("detach", "no_bootstrap", "copy_env")
PASS_THROUGH_FLAGS = (
    "model",
    "timeout",
    "max_turns",
    "permission_profile",
    "sandbox",
    "setting_sources",
    "reads_third_party",
    "env_allow",
    "env_allowlist_only",
    "auto_exit",
)
SIGNAL_NAMES: dict[int, str] = {
    int(signal.SIGTERM): "SIGTERM",
    int(signal.SIGHUP): "SIGHUP",
    int(signal.SIGINT): "SIGINT",
}


class LaunchInterrupted(BaseException):
    """A termination signal arrived before a supervisor was recorded (not an ``Exception``, so
    no ``except Exception`` of ours or a library's swallows it)."""

    def __init__(self, signum: int) -> None:
        super().__init__(signum)
        self.signum = signum


class SignalState:
    """What the launcher does with SIGTERM, SIGHUP and SIGINT (interface.md, Signals)."""

    def __init__(self) -> None:
        self.raising = True
        self.received: list[int] = []

    def handle(self, signum: int, frame: FrameType | None) -> None:
        if self.raising:
            raise LaunchInterrupted(signum)
        self.received.append(signum)

    def take(self) -> list[int]:
        taken, self.received = self.received, []
        return taken


# --- small helpers --------------------------------------------------------------------------


def resolve_tool(argv: Sequence[str], name: str) -> list[str]:
    """``argv`` with its program resolved to an absolute path, or exit 12 ``tool_missing``."""
    first = argv[0]
    path = os.path.abspath(first) if "/" in first else shutil.which(first)
    if path is None or not os.path.isfile(path) or not os.access(path, os.X_OK):
        raise CliError(
            "tool_missing",
            f"{name} is not installed or not executable",
            tool=name if name != "agent-spawn" else "agent-spawn",
            details={"hint": f"install {name}"},
        )
    return [path, *argv[1:]]


def resolve_self() -> str:
    """The absolute path of this agent-spawn, which must exec it in place (a console script)."""
    candidates: list[str] = []
    argv0 = sys.argv[0] if sys.argv else ""
    if argv0 and os.path.basename(argv0) == "agent-spawn":
        candidates.append(os.path.abspath(argv0))
    candidates.append(os.path.join(os.path.dirname(os.path.abspath(sys.executable)), "agent-spawn"))
    found = shutil.which("agent-spawn")
    if found is not None:
        candidates.append(os.path.abspath(found))
    for path in candidates:
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    raise CliError("tool_missing", "agent-spawn cannot find its own executable path")


def _inside(path: str, directory: str) -> bool:
    """Whether ``path`` equals ``directory`` or lies below it (both realpaths)."""
    return path == directory or path.startswith(directory.rstrip("/") + "/")


def _same_file(info: os.stat_result, fd_info: os.stat_result) -> bool:
    return (info.st_dev, info.st_ino) == (fd_info.st_dev, fd_info.st_ino)


def _copy_private(source: str | None, destination: str) -> None:
    """Copy a regular file (or stdin for ``None``) to ``destination`` (mode 0600, exclusive)."""
    out = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        if source is None:
            reader = sys.stdin.buffer
            while chunk := reader.read(1 << 16):
                os.write(out, chunk)
            return
        fd = os.open(source, os.O_RDONLY | os.O_NONBLOCK | os.O_CLOEXEC)
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                raise CliError.usage(f"{os.path.basename(source)!r} is not a regular file")
            os.set_blocking(fd, True)
            while chunk := os.read(fd, 1 << 16):
                os.write(out, chunk)
        finally:
            os.close(fd)
    except OSError as error:
        raise CliError.usage(
            f"cannot read {os.path.basename(source or '-')!r} ({error.strerror})"
        ) from None
    finally:
        os.close(out)


@dataclass
class _Copies:
    directory: str
    prompt: str | None = None
    inputs: list[str] = field(default_factory=lambda: [])
    input_names: list[str] = field(default_factory=lambda: [])


def make_copies(state_dir: str, agent_id: str, args: Any) -> _Copies:
    """Copy the prompt and the inputs into ``<state>/prompts/<agent_id>/`` (launch step 1).

    The directory is made exclusively (mode 0700): two launchers that drew the same id never
    share it. Later steps pass these copies, never the caller's paths.
    """
    base = os.path.join(state_dir, "prompts")
    os.makedirs(base, mode=0o700, exist_ok=True)
    directory = os.path.join(base, agent_id)
    try:
        os.mkdir(directory, 0o700)
    except FileExistsError:
        raise CliError("failed", "a prompt directory for this agent id already exists") from None
    copies = _Copies(directory)
    try:
        if args.prompt is not None:
            path = os.path.join(directory, "prompt")
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
            try:
                os.write(fd, args.prompt.encode("utf-8", "surrogateescape"))
            finally:
                os.close(fd)
            copies.prompt = path
        elif args.prompt_file is not None:
            path = os.path.join(directory, "prompt")
            _copy_private(None if args.prompt_file == "-" else args.prompt_file, path)
            copies.prompt = path
        for index, source in enumerate(args.input_file or []):
            name = os.path.basename(source)
            if not name or name in (".", ".."):
                raise CliError.usage("--input-file must name a file")
            folder = os.path.join(directory, "inputs", str(index))
            os.makedirs(folder, mode=0o700)
            path = os.path.join(folder, name)
            _copy_private(source, path)
            copies.inputs.append(path)
            copies.input_names.append(name)
    except BaseException:
        shutil.rmtree(directory, ignore_errors=True)
        raise
    return copies


def remove_copies(directory: str | None) -> None:
    if directory is not None:
        shutil.rmtree(directory, ignore_errors=True)


# --- the launch -----------------------------------------------------------------------------


class Launch:
    """One ``launch`` invocation (headless, no worktree)."""

    def __init__(self, inv: Any, sig: SignalState) -> None:
        self.inv = inv
        self.args = inv.args
        self.sig = sig
        self.config: Config = inv.config
        self.warnings: list[str] = list(inv.warnings)
        self.gateway = gw.get_gateway()
        self.state_dir = inv.state_dir or str(default_state_dir())
        self.store: Store | None = None
        self.backend: SessionBackend | None = None
        self.agent_id: str | None = None
        self.agent_inserted = False
        self.copies: _Copies | None = None
        self.csession_id: str | None = None
        self.detached: procs.DetachedStart | None = None
        self.forwarder: procs.SignalForwarder | None = None
        self.decision_ids: list[str] = list(inv.decision_ids)
        self.run_hooks: list[dict[str, str]] = []
        self.pre_hooks: list[dict[str, str]] = []
        self.started = False

    # -- logging ------------------------------------------------------------------------------

    def log(self, event: str, *, level: str = "info", **fields: Any) -> None:
        try:
            append_log(self.state_dir, event, level=level, agent_id=self.agent_id, **fields)
        except (RecordsError, ValueError, OSError) as error:
            self.warnings.append(f"could not write the log ({type(error).__name__})")

    def audit(self, action: str, result: str, inputs: Mapping[str, Any]) -> None:
        if self.agent_id is None:
            return
        try:
            append_audit(
                self.state_dir,
                actor=self.inv.actor,
                decision_ids=self.decision_ids,
                target=self.agent_id,
                action=action,
                inputs=inputs,
                result=result,
            )
        except (RecordsError, ValueError, OSError) as error:
            self.warnings.append(f"could not write the audit log ({type(error).__name__})")

    def deliver(self) -> None:
        """Deliver the queued hook events of a failed or interrupted launch (best effort)."""
        if self.store is None or not self.agent_inserted or self.agent_id is None:
            return
        try:
            hooklib.deliver_outbox(
                self.store,
                self.agent_id,
                self.config.hooks,
                state_dir=self.state_dir,
                gateway=self.gateway,
            )
        except (StoreError, OSError):
            pass  # hook failures never change the command's result

    # -- step 1: usage checks and preflight -------------------------------------------------

    def usage_checks(self) -> None:
        args = self.args
        worktree = args.worktree or "none"
        if worktree not in ("new", "none") and not os.path.isabs(worktree):
            raise CliError.usage("--worktree needs new, none or an absolute path")
        has_flags = any(getattr(args, name) for name in (*WORKTREE_FLAGS, *WORKTREE_SWITCHES))
        if worktree == "new":
            if not args.repo:
                raise CliError.usage("--worktree new needs --repo")
            if args.cwd and (args.cwd == "session-dir" or os.path.isabs(args.cwd)):
                raise CliError.usage("--worktree new takes a relative --cwd inside the worktree")
        elif has_flags:
            raise CliError.usage(
                "--branch, --base, --ref, --detach, --worktree-name, --no-bootstrap and "
                "--copy-env need --worktree new"
            )
        if args.cwd == "session-dir" and worktree != "none":
            raise CliError.usage("--cwd session-dir cannot be combined with a worktree")
        if (worktree != "none" or has_flags) and self.config.wtctl_cmd is None:
            raise CliError.usage("worktree flags need wtctl, but wtctl_cmd is null in the config")
        mode = args.mode or "interactive"
        if args.placement == "here" and args.wait_timeout is not None:
            raise CliError.usage("--wait-timeout cannot be combined with --placement here")
        if args.placement == "here" and mode == "interactive" and not _tty_pair():
            raise CliError.usage("--placement here needs a terminal on stdin and stdout")
        if mode == "headless" and not (args.skill or args.prompt is not None or args.prompt_file):
            raise CliError.usage("a headless launch needs --skill or a prompt")
        hook_values = hooklib.parse_invocation_hooks(args.hook or [], ("pre-launch", "post-launch"))
        self.pre_hooks = [h for h in hook_values if h["hook"] == "pre-launch"]
        self.run_hooks = [h for h in hook_values if h["hook"] == "post-launch"]
        if args.max_turns is not None:
            self.max_turns()
        # Seams owned by later issues; the usage checks above already ran.
        if worktree != "none":
            raise CliError.of(
                "internal", "`launch --worktree` is not implemented yet (tracked in issue #20)"
            )  # TODO(#20): worktrees, holds and the records of a cwd.
        if mode != "headless":
            raise CliError.of(
                "internal",
                "interactive launches are not implemented yet (windows: #27, here: #30)",
            )  # TODO(#27): windows; TODO(#30): --placement here.
        if args.placement is not None:
            self.warnings.append("--placement is ignored for a headless launch")

    def max_turns(self) -> int | None:
        value = self.args.max_turns
        if value is None:
            return None
        try:
            return int(value)
        except ValueError:
            raise CliError.usage("--max-turns must be an integer") from None

    def tool_argv(self, name: str, command: Sequence[str]) -> list[str]:
        return resolve_tool(command, name)

    def labels(self) -> LabelSet:
        given = parse_run_labels(self.args.label or [])
        extra = list(given)
        job = os.environ.get(self.config.job_id_env)
        if job is not None and not os.environ.get("CSESSION_ID"):
            label = job_label(job)
            if label is None:
                self.warnings.append(
                    f"{self.config.job_id_env} does not fit a label value: ignored"
                )
            else:
                extra.append(label)
        change = LabelSet().add_run(extra)
        if change.skipped:
            self.warnings.append("job= was not added: the agent already has 64 labels")
        return change.labels

    def build_spec(
        self, agent_id: str, labels: LabelSet, cwd: str | None, copies: _Copies | None
    ) -> RunSpec:
        a = self.args
        return RunSpec(
            agent_id=agent_id,
            mode="headless",
            kind=a.kind,
            skill=a.skill,
            plugin_dirs=tuple(os.path.abspath(p) for p in (a.plugin_dir or [])),
            model=a.model,
            timeout=a.timeout,
            max_turns=self.max_turns(),
            permission_profile=a.permission_profile,
            sandbox=a.sandbox,
            setting_sources=a.setting_sources,
            reads_third_party=bool(a.reads_third_party),
            env_allow=tuple(a.env_allow or ()),
            env_allowlist_only=None
            if a.env_allowlist_only is None
            else tuple(v for v in a.env_allowlist_only if v),
            auto_exit=a.auto_exit,
            repo=a.repo,
            cwd=cwd,
            labels=tuple(labels.texts()),
            prompt_file=None if copies is None else copies.prompt,
            input_files=() if copies is None else tuple(copies.inputs),
        )

    def settings(
        self, labels: LabelSet, input_names: Sequence[str], repo: str | None
    ) -> dict[str, Any]:
        a = self.args
        return {
            "kind": a.kind,
            "mode": "headless",
            "placement": None,
            "term_placement": a.term_placement,
            "name": a.name or a.worktree_name,
            "repo": repo,
            "worktree": a.worktree or "none",
            "branch": a.branch,
            "base": a.base,
            "ref": a.ref,
            "detach": bool(a.detach),
            "worktree_name": a.worktree_name,
            "no_bootstrap": bool(a.no_bootstrap),
            "copy_env": bool(a.copy_env),
            "cwd": a.cwd,
            "labels": labels.texts(),
            "skill": a.skill,
            "has_prompt": a.prompt is not None or a.prompt_file is not None,
            "input_names": list(input_names),
            "plugin_dirs": [os.path.abspath(p) for p in (a.plugin_dir or [])],
            "promote": a.promote,
            "ephemeral": bool(a.ephemeral),
            "wait": bool(a.wait),
            "wait_timeout": a.wait_timeout,
        }

    def pass_throughs(self) -> dict[str, Any]:
        found: dict[str, Any] = {}
        for name in PASS_THROUGH_FLAGS:
            value = getattr(self.args, name)
            if value is not None and value != []:
                found[name] = value
        return found

    def effective_cwd(self, classification: third_party.Classification) -> str | None:
        """The run's cwd: ``--cwd``, else the plan's (the kind's default, possibly session-dir)."""
        given = self.args.cwd
        if given == third_party.SESSION_DIR:
            return given
        if given is not None:
            return os.path.realpath(given)
        plan = classification.cwd
        if plan is None or plan == third_party.SESSION_DIR:
            return plan
        return os.path.realpath(plan)

    def result_file_inside(self, directories: Sequence[str]) -> bool:
        """Whether ``--result-file`` lies inside one of ``directories`` (the third-party rule)."""
        target = self.inv.result_file
        if target is None:
            return False
        return any(_inside(target.parent, os.path.realpath(d)) for d in directories)

    def pre_start_checks(self, classification: third_party.Classification) -> str | None:
        """Pre-start checks 3 and 4 (the records of the cwd belong to #20); returns the cwd."""
        cwd = self.effective_cwd(classification)
        if classification.reads_third_party:
            try:
                third_party.require_cwd_rule(True, cwd)
            except CliError as error:
                self.log("third_party_refused", level="warn", details={"reason": error.reason})
                raise
            if (
                cwd is not None
                and cwd != third_party.SESSION_DIR
                and self.result_file_inside([cwd])
            ):
                raise CliError.usage("--result-file lies inside a third-party-reading run's cwd")
        # TODO(#20): the records of the cwd (throwaway and worktree containment, third_party_run).
        return cwd

    # -- hooks --------------------------------------------------------------------------------

    def hook_entries(self, name: str) -> list[hooklib.HookEntry]:
        stored = self.pre_hooks if name == "pre-launch" else self.run_hooks
        return [
            *hooklib.config_entries(self.config.hooks, name),
            *hooklib.invocation_entries(stored, name),
        ]

    def run_pre_launch(
        self,
        labels: LabelSet,
        classification: third_party.Classification,
        cwd: str | None,
        copies: _Copies,
    ) -> None:
        a = self.args
        settings = self.settings(labels, copies.input_names, classification.repo or a.repo)
        body: dict[str, Any] = {
            "repo": settings["repo"],
            "labels": labels.texts(),
            "kind": a.kind,
            "run_kind": classification.run_kind,
            "reads_third_party": classification.reads_third_party,
            "tainted": None,
            "mode": "headless",
            "status": None,
            "needs_attention": None,
            "placement": None,
            "term_placement": a.term_placement,
            "worktree": a.worktree or "none",
            "branch": a.branch,
            "cwd": cwd,
            "name": a.name or a.worktree_name,
            "skill": a.skill,
            "has_prompt": settings["has_prompt"],
            "input_names": list(copies.input_names),
            "plugin_dirs": settings["plugin_dirs"],
            "promote": a.promote,
            "ephemeral": bool(a.ephemeral),
            "wait": bool(a.wait),
            "settings": settings,
            "pass_throughs": self.pass_throughs(),
            "actor": self.inv.actor,
            "decision_ids": list(self.decision_ids),
            "override_escalation": bool(a.override_escalation),
        }
        verdict = hooklib.run_pre_hooks(
            "pre-launch",
            body,
            self.hook_entries("pre-launch"),
            state_dir=self.state_dir,
            gateway=self.gateway,
        )
        for decision in verdict.decision_ids:
            if decision not in self.decision_ids and len(self.decision_ids) < 32:
                self.decision_ids.append(decision)
        if verdict.verdict == "ok":
            return
        if verdict.verdict == "escalate" and a.override_escalation:
            self.override_hook = verdict.hook
            return
        error = hooklib.verdict_error(verdict)
        assert error is not None
        error.decision_ids = list(self.decision_ids)
        raise error

    override_hook: str | None = None

    # -- the whole launch ----------------------------------------------------------------------

    def run(self) -> Outcome:
        self.usage_checks()
        config = self.config
        cs_argv = self.tool_argv("claude-session", config.claude_session_cmd)
        self_path = resolve_self()
        state = Path(self.state_dir)
        state.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.store = Store.open(self.state_dir, gateway=self.gateway)
        store = self.store
        self.backend = SessionBackend(
            cs_argv,
            actor=self.inv.actor,
            decision_ids=self.inv.decision_ids,
            gateway=self.gateway,
        )
        labels = self.labels()
        proposed = new_agent_id(store)
        self.agent_id = None  # no record yet: log lines and errors carry no agent id
        self.copies = make_copies(self.state_dir, proposed, self.args)
        try:
            spec = self.build_spec(proposed, labels, self._classification_cwd(), self.copies)
            classification = self.backend.classify(spec, refuse=not self.args.dry_run)
            cwd = self.pre_start_checks(classification) if not self.args.dry_run else None
            if self.args.dry_run:
                return self.dry_run(spec, labels, classification, cs_argv, proposed)
            self.run_pre_launch(labels, classification, cwd, self.copies)
        except BaseException:
            remove_copies(self.copies.directory)
            raise
        return self.after_hooks(proposed, labels, classification, cwd, cs_argv, self_path)

    def _classification_cwd(self) -> str | None:
        given = self.args.cwd
        if given is None:
            return None
        if given == third_party.SESSION_DIR:
            return given
        real = os.path.realpath(given)
        return real if os.path.isdir(real) else None

    # -- dry run ------------------------------------------------------------------------------

    def dry_run(
        self,
        spec: RunSpec,
        labels: LabelSet,
        classification: third_party.Classification,
        cs_argv: Sequence[str],
        agent_id: str,
    ) -> Outcome:
        assert self.backend is not None and self.copies is not None
        try:
            cwd = self.effective_cwd(classification)
            would_refuse = classification.would_refuse
            if would_refuse is not None and would_refuse not in third_party.WOULD_REFUSE_REASONS:
                would_refuse = "unknown"
            if would_refuse is None and classification.reads_third_party:
                if not third_party.cwd_rule_allows(True, cwd):
                    would_refuse = "third_party_cwd"
            outcome_path = os.path.join(self.state_dir, "outcomes", agent_id, "TOKEN.json")
            relay = relay_hook(resolve_self(), self.state_dir, self.inv.config_file or "")
            argv = [
                *cs_argv,
                "start",
                *self.start_flags(spec),
                f"--hook={relay}",
                f"--outcome-file={outcome_path}",
                *self.audit_flags(),
                "--json",
            ]
            plan = {
                "dry_run": True,
                "resume": False,
                "agent_settings": self.settings(
                    labels, self.copies.input_names, classification.repo or self.args.repo
                ),
                "reads_third_party": classification.reads_third_party,
                "session_plan": self.plan_of(spec),
                "would_refuse": would_refuse,
                "tools": {"claude-session": list(cs_argv), "termctl": None, "wtctl": None},
                "worktree_plan": None,
                "term_plan": None,
                "session_start_argv": argv,
                "hooks": self.hook_listing(),
                "warnings": self.warnings,
            }
        finally:
            remove_copies(self.copies.directory)
        return Outcome(20 if would_refuse else 0, plan, PayloadKind.DRY_RUN)

    def start_flags(self, spec: RunSpec) -> list[str]:
        from .session_backend import _launch_flags  # pyright: ignore[reportPrivateUsage]

        return _launch_flags(spec)

    def audit_flags(self) -> list[str]:
        assert self.backend is not None
        return self.backend._audit_flags()  # pyright: ignore[reportPrivateUsage]

    def plan_of(self, spec: RunSpec) -> dict[str, Any]:
        """The classification dry run's plan, verbatim."""
        assert self.backend is not None
        reply = self.backend._call(self.backend.dry_run_argv(spec))  # pyright: ignore[reportPrivateUsage]
        return reply.data or {}

    def hook_listing(self) -> list[dict[str, str]]:
        listing: list[dict[str, str]] = []
        for name in ("pre-launch", "post-launch"):
            for entry in self.hook_entries(name):
                listing.append(
                    {
                        "hook": name,
                        "cmd": entry.cmd,
                        "source": "config" if entry.source == "config" else "flag",
                    }
                )
        return listing

    # -- steps 3 to 6 -------------------------------------------------------------------------

    def after_hooks(
        self,
        proposed: str,
        labels: LabelSet,
        classification: third_party.Classification,
        cwd: str | None,
        cs_argv: Sequence[str],
        self_path: str,
    ) -> Outcome:
        assert self.store is not None and self.backend is not None and self.copies is not None
        store, args = self.store, self.args
        identity = store.holder_identity()
        settings_row = {
            "settings": self.settings(
                labels, self.copies.input_names, classification.repo or args.repo
            ),
            "pass_throughs": self.pass_throughs(),
        }
        name = _name_of(proposed, args)
        row = store.insert_new_agent(
            {
                "status": "launching",
                "launch_kind": args.kind,
                "mode": "headless",
                "placement": None,
                "term_placement": args.term_placement,
                "name": name,
                "repo": args.repo or classification.repo,
                "cwd": cwd,
                "agent_spawn_path": self_path,
                "state_dir": self.state_dir,
                "config_path": self.inv.config_file,
                "claude_session_path": shlex.join(cs_argv),
                "actor": self.inv.actor,
                "decision_ids": list(self.decision_ids),
                "settings": settings_row,
                "launcher": identity,
                "starter": identity,
                "promote": args.promote,
                "ephemeral": bool(args.ephemeral),
            },
            proposed=proposed,
            labels=labels,
        )
        self.agent_id = row["agent_id"]
        self.agent_inserted = True
        assert self.agent_id is not None
        if self.agent_id != proposed:
            self.copies = _rename_copies(self.state_dir, self.copies, self.agent_id, args)
        self.log("launch", details={"mode": "headless", "kind": args.kind})
        if self.override_hook is not None:
            self.audit(
                "launch_escalation_override",
                "ok",
                {
                    "labels": labels.texts(),
                    "repo": row["repo"],
                    "worktree": None,
                    "csession_id": None,
                    "merged_head": None,
                    "force": False,
                },
            )
        # The session (step 6) and the placement (step 7).
        started = self.start_session(row, labels, classification, cwd)
        return self.place_and_wait(started)

    def start_session(
        self,
        row: dict[str, Any],
        labels: LabelSet,
        classification: third_party.Classification,
        cwd: str | None,
    ) -> StartResult:
        assert self.store is not None and self.backend is not None and self.copies is not None
        backend = self.backend
        agent_id = row["agent_id"]
        spec = self.build_spec(agent_id, labels, cwd, self.copies)
        outcome_file = new_outcome_file(self.state_dir, agent_id)
        relay = relay_hook(row["agent_spawn_path"], self.state_dir, row["config_path"])
        try:
            result = backend.start(spec, outcome_file=outcome_file, hook=relay)
        except StartError as error:
            self.fail_start(error, row)
            raise  # fail_start always raises; this keeps the type checker honest
        self.csession_id = result.csession_id
        token = os.path.basename(outcome_file).removesuffix(".json")
        self.record_session(row, result, classification, token, cwd)
        remove_copies(self.copies.directory)
        self.check_classification(row, result, classification, cwd)
        return result

    def record_session(
        self,
        row: dict[str, Any],
        result: StartResult,
        classification: third_party.Classification,
        token: str,
        cwd: str | None,
    ) -> None:
        assert self.store is not None
        store = self.store
        agent_id = row["agent_id"]
        with store.lease(lifecycle.lease_key(agent_id), wait=lifecycle.LEASE_WAIT) as lease:
            with store.transaction():
                store.update_agent(
                    lease,
                    agent_id,
                    {
                        "csession_id": result.csession_id,
                        "run": result.run,
                        "run_kind": classification.run_kind,
                    },
                )
                store.put_run(
                    lease,
                    agent_id,
                    result.run,
                    {
                        "plan_run_kind": classification.run_kind,
                        "mode": "headless",
                        "placement": None,
                        "cwd": cwd,
                        "reads_third_party": classification.reads_third_party,
                        "outcome_token": token,
                        "hooks": self.run_hooks,
                    },
                )

    def fail_start(self, error: StartError, row: dict[str, Any]) -> None:
        """Park after a failed ``start`` (launch step 6) and raise the pass-through error."""
        assert self.backend is not None and self.store is not None
        agent_id = row["agent_id"]
        session, run = error.csession_id, None
        if session is not None:
            run = error.run
        elif error.maybe_allocated:
            try:
                found = self.backend.list_by_label(f"agent={agent_id}")
            except CliError:
                found = []
            if len(found) == 1:
                session = found[0]
                try:
                    self.backend.stop(session)
                except CliError:
                    pass
        if session is not None:
            self.csession_id = session
            with self.store.lease(
                lifecycle.lease_key(agent_id), wait=lifecycle.LEASE_WAIT
            ) as lease:
                with self.store.transaction():
                    self.store.update_agent(
                        lease, agent_id, {"csession_id": session, **({"run": run} if run else {})}
                    )
                    if run:
                        self.store.put_run(
                            lease, agent_id, run, {"mode": "headless", "hooks": self.run_hooks}
                        )
            remove_copies(None if self.copies is None else self.copies.directory)
        code = error.exit_code
        if code == 11:
            fields: dict[str, Any] = {"park_reason": "vetoed", "needs_attention": None}
        elif code == 13:
            fields = {
                "park_reason": "escalated",
                "needs_attention": "escalated",
                "escalated_by": "claude-session",
            }
        elif code == 20:
            fields = {"park_reason": "launch_failed", "needs_attention": None}
        else:
            fields = {"park_reason": "launch_failed", "needs_attention": "launch_failed"}
        lifecycle.park(self.store, agent_id, from_status="launching", **fields)
        self.log("status_change", level="warn", details={"to": "parked", **fields})
        error.agent_id = agent_id
        error.decision_ids = list(self.decision_ids)
        raise error

    def check_classification(
        self,
        row: dict[str, Any],
        result: StartResult,
        classification: third_party.Classification,
        cwd: str | None,
    ) -> None:
        assert self.backend is not None and self.store is not None
        agent_id = row["agent_id"]
        work = os.path.join(result.csession_dir, "work")

        def rule_allows(actual: bool, run_cwd: str | None) -> bool:
            return third_party.cwd_rule_allows(actual, run_cwd, session_work=work)

        def result_file_allows() -> bool:
            return not self.result_file_inside([result.csession_dir, *([cwd] if cwd else [])])

        check = self.backend.check_classification(
            result,
            classification,
            rule_allows=rule_allows,
            result_file_allows=result_file_allows,
        )
        status = check.status
        if status is not None:
            self.record_status(agent_id, result.run, status, classification)
        if check.verdict == "warn":
            self.log(
                "classification_mismatch",
                level="warn",
                csession_id=result.csession_id,
                details={"expected": classification.classification},
            )
            return
        if check.verdict == "accept":
            return
        cause = check.decision.cause or "classification_unknown"
        self.log(
            "classification_unknown"
            if cause == "classification_unknown"
            else "classification_mismatch",
            level="error",
            csession_id=result.csession_id,
        )
        lifecycle.park(
            self.store,
            agent_id,
            park_reason="launch_failed",
            needs_attention="launch_failed",
            from_status="launching",
        )
        error = check.error(agent_id)
        error.decision_ids = list(self.decision_ids)
        raise error

    def record_status(
        self, agent_id: str, run: int, status: Any, classification: third_party.Classification
    ) -> None:
        assert self.store is not None
        reads = bool(status.run.reads_third_party or status.tainted_by_run == run)
        with self.store.lease(lifecycle.lease_key(agent_id), wait=lifecycle.LEASE_WAIT) as lease:
            with self.store.transaction():
                self.store.update_agent(
                    lease,
                    agent_id,
                    {
                        "run_kind": status.run.run_kind or classification.run_kind,
                        "tainted": bool(status.tainted),
                        "repo": status.repo or self.args.repo or classification.repo,
                    },
                )
                self.store.put_run(
                    lease,
                    agent_id,
                    run,
                    {"run_kind": status.run.run_kind, "reads_third_party": reads},
                )

    # -- step 7: the detached supervisor ---------------------------------------------------------

    def place_and_wait(self, result: StartResult) -> Outcome:
        assert self.store is not None and self.backend is not None
        store, agent_id = self.store, self.agent_id
        assert agent_id is not None
        row = store.get_agent(agent_id)
        assert row is not None
        supervise_dir = os.path.join(self.state_dir, "supervise")
        os.makedirs(supervise_dir, mode=0o700, exist_ok=True)
        argv = [
            row["agent_spawn_path"],
            "supervise",
            agent_id,
            "--state-dir",
            self.state_dir,
            "--config",
            row["config_path"],
        ]
        try:
            self.detached = procs.spawn_detached(
                argv,
                log_path=os.path.join(supervise_dir, f"{agent_id}.log"),
                ack_timeout=ACK_TIMEOUT,
                gateway=self.gateway,
            )
            identity = self.detached.identity
            with store.lease(lifecycle.lease_key(agent_id), wait=lifecycle.LEASE_WAIT) as lease:
                store.update_agent(lease, agent_id, {"supervisor": identity})
            self.forwarder = procs.SignalForwarder(identity, gateway=self.gateway)
            self.sig.raising = False  # from here a signal is forwarded, not a reason to cancel
            self.detached.acknowledge()
        except (procs.DetachedForkError, OSError) as error:
            self.sig.raising = True
            self.placement_failed(result, error)
            raise  # placement_failed raises; unreachable
        self.log("launch", message="supervisor started", pid=identity.pid)
        return self.wait()

    def placement_failed(self, result: StartResult, error: BaseException) -> None:
        """No supervisor exists: cancel the pending run and park ``launch_failed``."""
        assert self.store is not None and self.backend is not None and self.agent_id is not None
        agent_id = self.agent_id
        if self.detached is not None:
            self.detached.abort()
        with self.store.lease(lifecycle.lease_key(agent_id), wait=lifecycle.LEASE_WAIT) as lease:
            self.store.update_agent(lease, agent_id, {"supervisor": None})
        try:
            self.backend.stop(result.csession_id)
        except CliError:
            pass
        lifecycle.park(
            self.store,
            agent_id,
            park_reason="launch_failed",
            needs_attention="launch_failed",
            from_status="launching",
        )
        raise CliError(
            "launch_failed",
            f"could not start the detached supervisor ({type(error).__name__})",
            agent_id=agent_id,
            decision_ids=list(self.decision_ids),
        )

    # -- step 9: return --------------------------------------------------------------------------

    def wait(self) -> Outcome:
        assert self.store is not None and self.agent_id is not None
        store, agent_id, args = self.store, self.agent_id, self.args
        wait = bool(args.wait)
        deadline = None
        if wait and args.wait_timeout is not None:
            deadline = time.monotonic() + _seconds(args.wait_timeout)
        dead_looks = 0
        while True:
            row = store.get_agent(agent_id)
            if row is None:
                raise CliError("failed", "the agent record disappeared", agent_id=agent_id)
            status = row["status"]
            self.started = self.started or status != "launching"
            for received in self.sig.take():
                forward = self.forward_signal(received)
                if forward is not None:
                    return forward
            supervisor = row.get("supervisor")
            alive = supervisor is not None and procs.is_alive(supervisor, gateway=self.gateway)
            ended = status not in ("launching", "live")
            if not wait:
                if status == "live":
                    return self.result(row, final=False)
                if ended:  # the run ended before this launcher saw it started
                    return self.result(row, final=True)
            elif ended and not row["promotion_pending"] and not alive:
                return self.result(row, final=True)
            if not alive and status in ("launching", "live"):
                dead_looks += 1
                if dead_looks >= DEAD_SUPERVISOR_POLLS:
                    # TODO(#32): the liveness check parks an agent whose supervisor died.
                    raise CliError(
                        "failed",
                        "the supervisor exited before the agent parked (see reconcile)",
                        agent_id=agent_id,
                    )
            else:
                dead_looks = 0
            if deadline is not None and time.monotonic() >= deadline:
                payload = result_object(
                    agent=agent_object(store, row, warnings=self.warnings),
                    timed_out=True,
                    warnings=self.warnings,
                )
                return Outcome(EXIT_TIMED_OUT, payload, PayloadKind.RESULT)
            self.gateway.sleep(POLL)

    def forward_signal(self, received: int) -> Outcome | None:
        """Handle a flagged signal: forward SIGTERM to the supervisor and keep waiting, or (a
        waiter whose run started) stop waiting on SIGHUP and SIGINT."""
        if received != signal.SIGTERM and self.started and self.args.wait:
            return Outcome(EXIT_SIGINT, None, PayloadKind.NONE)
        if self.forwarder is not None:
            outcome = self.forwarder.forward(signal.SIGTERM)
            if outcome is not None:
                self.log(
                    "signal_sent", pid=outcome.pid, signal="SIGTERM", details={"to": "supervisor"}
                )
        return None

    def result(self, row: dict[str, Any], *, final: bool) -> Outcome:
        assert self.store is not None
        store = self.store
        agent_id = row["agent_id"]
        agent = agent_object(store, row, warnings=self.warnings)
        run_row = store.get_run(agent_id, row["run"]) if row.get("run") else None
        first = ts(run_row.get("first_working_at")) if run_row else None
        if not final:
            payload = result_object(agent=agent, first_working_at=first, warnings=self.warnings)
            return Outcome(0, payload, PayloadKind.RESULT)
        last = row.get("last_outcome")
        if not isinstance(last, dict):
            raise CliError(
                "launch_failed",
                "the run ended without an outcome",
                agent_id=agent_id,
                decision_ids=list(self.decision_ids),
            )
        if last.get("outcome") == "not-started" and last.get("end_reason") in (  # pyright: ignore[reportUnknownMemberType]
            "canceled",
            "never_adopted",
        ):
            raise CliError(
                "launch_failed",
                "the run never started",
                agent_id=agent_id,
                tool="claude-session",
                lower_code="launch_failed",
                lower_error={"code": "launch_failed", "end_reason": last.get("end_reason")},  # pyright: ignore[reportUnknownMemberType]
                decision_ids=list(self.decision_ids),
            )
        payload = result_object(
            agent=agent, run=last, first_working_at=first, warnings=self.warnings
        )
        return Outcome(int(last["exit_code"]), payload, PayloadKind.RESULT)  # pyright: ignore[reportUnknownArgumentType]

    # -- interruption (before any supervisor exists) ---------------------------------------------

    def interrupted(self, signum: int) -> Outcome:
        """Cancel, park ``killed`` without attention and exit 130 or 143."""
        self.sig.raising = False
        code = EXIT_SIGTERM if signum == signal.SIGTERM else EXIT_SIGINT
        if self.detached is not None:
            self.detached.abort()
        agent_id = self.agent_id
        if self.agent_inserted and agent_id is not None and self.store is not None:
            try:
                self.cancel_pending(agent_id)
            except (CliError, StoreError):
                pass
        elif self.copies is not None:
            remove_copies(self.copies.directory)
        self.log(
            "interrupted",
            level="warn",
            signal=SIGNAL_NAMES.get(signum),
            details={"cause": "interrupted"},
        )
        error = CliError(
            "launch_failed",
            f"launch interrupted by {SIGNAL_NAMES.get(signum, 'a signal')}",
            exit_code=code,
            agent_id=agent_id if self.agent_inserted else None,
            details={"cause": "interrupted"},
            decision_ids=list(self.decision_ids),
        )
        return error_outcome(error, self.inv.decision_ids)

    def cancel_pending(self, agent_id: str) -> None:
        assert self.store is not None and self.backend is not None
        session = self.csession_id
        if session is None:
            try:
                found = self.backend.list_by_label(f"agent={agent_id}")
            except CliError:
                found = []
            session = found[0] if len(found) == 1 else None
            if session is not None:
                self.csession_id = session
                with self.store.lease(lifecycle.lease_key(agent_id), wait=5.0) as lease:
                    self.store.update_agent(lease, agent_id, {"csession_id": session})
        if session is not None:
            try:
                self.backend.stop(session)
            except CliError:
                pass
        row = self.store.get_agent(agent_id)
        if row is not None and row.get("supervisor") is not None:
            with self.store.lease(lifecycle.lease_key(agent_id), wait=5.0) as lease:
                self.store.update_agent(lease, agent_id, {"supervisor": None})
        lifecycle.park(
            self.store,
            agent_id,
            park_reason="killed",
            needs_attention=None,
            from_status="launching",
        )


def _name_of(agent_id: str, args: Any) -> str:
    try:
        return default_name(
            agent_id, name=args.name, worktree_name=args.worktree_name, kind=args.kind
        )
    except GrammarError as error:
        raise CliError.usage(str(error)) from None


def _rename_copies(state_dir: str, copies: _Copies, new_id: str, args: Any) -> _Copies:
    """The id changed at insertion: move ``prompts/<old>/`` to ``prompts/<new>/``."""
    target = os.path.join(state_dir, "prompts", new_id)
    os.rename(copies.directory, target)
    moved = _Copies(target)
    moved.prompt = None if copies.prompt is None else os.path.join(target, "prompt")
    moved.input_names = list(copies.input_names)
    old = copies.directory
    moved.inputs = [target + path[len(old) :] for path in copies.inputs]
    return moved


def _seconds(text: object) -> float:
    return parse_duration(str(text))


def _tty_pair() -> bool:
    try:
        return bool(sys.stdin.isatty() and sys.stdout.isatty())
    except (ValueError, OSError, AttributeError):
        return False


def launch(inv: Any) -> Outcome:
    """Run ``launch``; the handler of ``commands.launching``."""
    sig = SignalState()
    launch_run = Launch(inv, sig)
    restore = procs.install_signal_handlers(
        {signal.SIGTERM: sig.handle, signal.SIGHUP: sig.handle, signal.SIGINT: sig.handle}
    )
    try:
        try:
            return launch_run.run()
        except LaunchInterrupted as exc:
            outcome = launch_run.interrupted(exc.signum)
            launch_run.deliver()
            return outcome
        except CliError as error:
            if launch_run.copies is not None and not launch_run.agent_inserted:
                remove_copies(launch_run.copies.directory)
            if error.agent_id is None and launch_run.agent_inserted:
                error.agent_id = launch_run.agent_id
            launch_run.deliver()
            raise
    finally:
        restore()
        if launch_run.store is not None:
            launch_run.store.close()
