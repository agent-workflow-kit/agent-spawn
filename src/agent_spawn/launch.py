"""``launch`` for headless agents (interface.md, Launch order, steps 1 to 9).

The order: usage checks and preflight (tool resolution, prompt copies, classification, pre-start
checks), ``pre-launch`` hooks, allocation, the worktree (``wtctl create`` or ``adopt``, step 4)
and the holds (step 5), the session (``claude-session start`` and the post-start classification
check), the detached supervisor, and the return.

Window placement (interactive, ``--placement window``, step 7): the launcher listens on the
environment handoff socket, opens the window with ``termctl open`` (after the same call with
``--dry-run`` as the preflight of step 1, before any hook or allocation), records the pane's
process as the supervisor and serves the handoff. The window preflight, labels and cwd rules
live here; ``term.py`` makes the calls.

Signals (interface.md, Signals). Until a supervisor is recorded the handlers raise
``LaunchInterrupted`` in the main thread, which stops the in-flight lower-tool call (the gateway
kills exactly its own child), cancels a pending run, parks the agent ``killed`` without attention
and exits 130 or 143. Once the supervisor is recorded (just before it is acknowledged) the
handlers only set a flag and the wait loop forwards SIGTERM to the exact verified supervisor.
"""

from __future__ import annotations

import os
import re
import shlex
import shutil
import signal
import stat
import sys
import tempfile
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import FrameType
from typing import Any

from . import gateway as gw
from . import hooks as hooklib
from . import lifecycle, procs, third_party
from . import worktrees as wtlib
from .agent_view import agent_object, chain_first_working_at, outcome_object
from .config import Config
from .handoff import HandoffFailed, HandoffListener
from .ids import IdEntry, new_agent_id
from .labels import LabelSet, job_label, parse_run_labels
from .liveness import Liveness, row_moved, unreadable
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
    RunOutcome,
    RunSpec,
    SessionBackend,
    StartError,
    StartResult,
    launch_flags,
    new_outcome_file,
    relay_hook,
)
from .store import Store, StoreError, default_state_dir
from .term import Pane, TermBackend, sync_window_labels, window_labels
from .validate import GrammarError, default_name, parse_duration
from .worktrees import CreatePlan, WtctlBackend, WtctlError, WtRecord

POLL = 0.1  # seconds between looks at the state while waiting
PRE_LAUNCH_BODY_LABEL_CAP = 64
ACK_TIMEOUT = 60.0
LIVENESS_EVERY = 1.0  # seconds between liveness checks of an agent nothing drives
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
HANDLED_SIGNALS = (signal.SIGTERM, signal.SIGHUP, signal.SIGINT)
INPUT_NAME = re.compile(r"[A-Za-z0-9._-]{1,128}")
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
            # Only the first signal interrupts: a second one (even while the interrupt is being
            # handled) is recorded, never raised again.
            self.raising = False
            raise LaunchInterrupted(signum)
        self.received.append(signum)

    def take(self) -> list[int]:
        taken, self.received = self.received, []
        return taken


# --- small helpers --------------------------------------------------------------------------


def resolve_tool(argv: Sequence[str], name: str) -> list[str]:
    """``argv`` with its program resolved to an absolute path, or exit 12 ``tool_missing``."""
    first = argv[0]
    found = shutil.which(first)
    path = (
        os.path.abspath(first)
        if "/" in first
        else None
        if found is None
        else os.path.abspath(found)
    )
    if path is None or not os.path.isfile(path) or not os.access(path, os.X_OK):
        raise CliError(
            "tool_missing",
            f"{name} is not installed or not executable",
            tool=name if name != "agent-spawn" else "agent-spawn",
            details={"hint": f"install {name}"},
        )
    return [path, *argv[1:]]


def args_mode(args: Any) -> str:
    """The run mode a launch asks for (claude-session's default is interactive)."""
    return getattr(args, "mode", None) or "interactive"


def resolve_wtctl(command: Sequence[str], stored: str | None) -> list[str]:
    return resolve_stored_tool(command, "wtctl", stored)


def resolve_stored_tool(command: Sequence[str], name: str, stored: str | None) -> list[str]:
    """``name`` as configured, else as stored with the agent (an absolute path, possibly with
    arguments); exit 12 ``tool_missing`` when neither resolves. A blank stored value counts as
    none."""
    try:
        return resolve_tool(command, name)
    except CliError as missing:
        if not stored:
            raise
        try:
            argv = shlex.split(stored)
            if not argv:
                raise missing from None
            return resolve_tool(argv, name)
        except (CliError, ValueError):
            raise missing from None


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


def _runs(path: str) -> bool:
    return os.path.isfile(path) and os.access(path, os.X_OK)


def stored_self(row: Mapping[str, Any]) -> str:
    """The agent-spawn path internal argvs of a long-lived process use: the one stored with the
    agent, or, when that no longer runs (an upgrade or a cleaned `uvx` cache removed it), the
    path this process resolves for itself (exit 12 ``tool_missing`` when neither runs)."""
    stored = row.get("agent_spawn_path")
    if isinstance(stored, str) and os.path.isabs(stored) and _runs(stored):
        return stored
    return resolve_self()


def stored_tool(stored: str | None, command: Sequence[str], name: str) -> list[str]:
    """A lower tool as stored with the agent (an absolute path, possibly with arguments); when
    that no longer runs, the configured command resolved now; exit 12 ``tool_missing`` when
    neither does. The reverse preference of ``resolve_stored_tool``: a running agent keeps the
    tool it was launched with."""
    try:
        argv = shlex.split(stored) if stored else []
    except ValueError:
        argv = []
    if argv and os.path.isabs(argv[0]) and _runs(argv[0]):
        return argv
    return resolve_tool(command, name)


_inside = third_party.inside


def _same_real_path(a: str, b: str) -> bool:
    try:
        return os.path.realpath(a) == os.path.realpath(b)
    except (ValueError, OSError):
        return False


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


def input_names(args: Any) -> list[str]:
    """The basenames of the ``--input-file`` values: each fits ``[A-Za-z0-9._-]{1,128}`` and
    none repeats (exit 2)."""
    names: list[str] = []
    for source in args.input_file or []:
        name = os.path.basename(source)
        if name in ("", ".", "..") or INPUT_NAME.fullmatch(name) is None:
            raise CliError.usage(
                "--input-file names must be 1 to 128 characters of letters, digits, '.', '_' or '-'"
            )
        if name in names:
            raise CliError.usage(f"two --input-file values have the base name {name!r}")
        names.append(name)
    return names


@dataclass
class _Copies:
    directory: str
    prompt: str | None = None
    inputs: list[str] = field(default_factory=lambda: [])
    input_names: list[str] = field(default_factory=lambda: [])


def make_copies(state_dir: str, agent_id: str, args: Any, *, subdir: str | None = None) -> _Copies:
    """Copy the prompt and the inputs into ``<state>/prompts/<agent_id>/`` (launch step 1).

    The directory is made exclusively (mode 0700): two launchers that drew the same id never
    share it. Later steps pass these copies, never the caller's paths. A resume of an agent
    that already has a prompt directory copies into ``subdir`` below it instead.
    """
    names = input_names(args)  # exit 2 before anything is created
    base = os.path.join(state_dir, "prompts")
    os.makedirs(base, mode=0o700, exist_ok=True)
    directory = os.path.join(base, agent_id)
    parent: str | None = None  # the per-agent directory this call created (subdir mode)
    if subdir is not None:
        if not os.path.isdir(directory):
            parent = directory
        os.makedirs(directory, mode=0o700, exist_ok=True)
        directory = os.path.join(directory, subdir)
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
        for index, (source, name) in enumerate(zip(args.input_file or [], names, strict=True)):
            folder = os.path.join(directory, "inputs", str(index))
            os.makedirs(folder, mode=0o700)
            path = os.path.join(folder, name)
            _copy_private(source, path)
            copies.inputs.append(path)
            copies.input_names.append(name)
    except BaseException:
        shutil.rmtree(directory, ignore_errors=True)
        if parent is not None:
            try:
                os.rmdir(parent)  # only when empty
            except OSError:
                pass
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
        self.supervised = False  # a supervisor is recorded and acknowledged
        self.label_set: LabelSet | None = None
        self.wt: WtctlBackend | None = None  # resolved on first use; None under wtctl_cmd: null
        self.wt_argv: list[str] | None = None
        self.wt_records: list[WtRecord] | None = None  # the last `wtctl list`
        self.holds_started = False  # `take_holds` may have adopted a record
        self.stored_wtctl: str | None = None  # the absolute wtctl path stored with the agent
        self.mode = args_mode(self.args)  # interactive or headless
        self.term: TermBackend | None = None  # resolved when a window will be opened
        self.window_socket: str | None = None  # the tmux server the preflight named
        self.term_argv: list[str] | None = None
        self.stored_termctl: str | None = None  # the absolute termctl path stored with the agent
        self.term_override = False  # --override-escalation goes to `termctl open` (resume)

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
        if getattr(args, "override_escalation", False) and not self.inv.actor.startswith("human:"):
            # The CLI already gates this flag; this keeps the handler safe when called directly.
            raise CliError.usage("--override-escalation needs an actor human:<user>")
        worktree = args.worktree or "none"
        if worktree not in ("new", "none") and not os.path.isabs(worktree):
            raise CliError.usage("--worktree needs new, none or an absolute path")
        # a value flag counts when given at all (an empty value is still a worktree flag)
        has_flags = any(getattr(args, name) is not None for name in WORKTREE_FLAGS) or any(
            getattr(args, name) for name in WORKTREE_SWITCHES
        )
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
        if worktree != "none":
            self.worktree_usage(worktree)
        mode = self.mode
        if self.here:  # interactive only: headless ignores --placement (with a warning)
            if args.wait_timeout is not None:
                raise CliError.usage("--wait-timeout cannot be combined with --placement here")
            if not args.dry_run and not _tty_pair():
                raise CliError.usage("--placement here needs a terminal on stdin and stdout")
        if mode == "headless" and not (args.skill or args.prompt is not None or args.prompt_file):
            raise CliError.usage("a headless launch needs --skill or a prompt")
        hook_values = hooklib.parse_invocation_hooks(args.hook or [], ("pre-launch", "post-launch"))
        self.pre_hooks = [h for h in hook_values if h["hook"] == "pre-launch"]
        self.run_hooks = [h for h in hook_values if h["hook"] == "post-launch"]
        if args.max_turns is not None:
            self.max_turns()
        input_names(args)
        self.label_set = self.labels()  # a reserved label key is exit 2, before any tool lookup
        if mode == "headless":
            if args.placement is not None:
                self.warnings.append("--placement is ignored for a headless launch")

    def worktree_usage(self, worktree: str) -> None:
        """Grammars and containment of the worktree flags (exit 2, nothing created)."""
        args = self.args
        if args.repo:
            wtlib.check_repo(args.repo)
        if args.branch is not None:
            wtlib.check_branch(args.branch, "--branch")
        if args.base is not None:
            wtlib.check_branch(args.base, "--base")
        if args.ref is not None:
            wtlib.check_sha(args.ref)
        if args.worktree_name is not None:
            wtlib.check_worktree_name(args.worktree_name)
        if args.branch is not None and args.detach:
            raise CliError.usage("--branch cannot be combined with --detach")
        if worktree == "new":
            if args.cwd is not None:
                # Lexical only: a symlink inside the worktree is checked once it exists.
                normal = os.path.normpath(args.cwd)
                if normal == ".." or normal.startswith("../"):
                    raise CliError.usage("--cwd must stay inside the new worktree")
            return
        wtlib.check_abs_path(worktree, "--worktree")
        if args.cwd is not None:
            root = os.path.realpath(worktree)
            if not _inside(wtlib.resolve_in(root, args.cwd), root):
                raise CliError.usage("--cwd lies outside the worktree")

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

    @property
    def here(self) -> bool:
        """The supervisor runs in this process, in the foreground (interactive, ``--placement
        here``)."""
        return self.mode == "interactive" and self.args.placement == "here"

    @property
    def waits(self) -> bool:
        """``--wait``; ``--placement here`` implies it."""
        return bool(self.args.wait) or self.here

    @property
    def windowed(self) -> bool:
        """A window will host the supervisor: interactive mode with placement ``window``."""
        return self.mode == "interactive" and (self.args.placement or "window") == "window"

    def termctl_backend(self) -> TermBackend:
        """termctl, resolved once (exit 12 ``tool_missing`` before anything changes): config's
        command, else the absolute path stored with the agent."""
        if self.term is not None:
            return self.term
        argv = self.resolve_termctl()
        self.term = TermBackend(
            argv, actor=self.inv.actor, decision_ids=self.decision_ids, gateway=self.gateway
        )
        return self.term

    def resolve_termctl(self) -> list[str]:
        if self.term_argv is None:
            self.term_argv = resolve_stored_tool(
                self.config.termctl_cmd, "termctl", self.stored_termctl
            )
        return self.term_argv

    def window_name(self, agent_id: str) -> str:
        return _name_of(agent_id, self.args)

    def window_preflight(
        self,
        agent_id: str,
        run_kind: str | None,
        label_texts: Sequence[str],
        cwd: str | None,
        self_path: str,
        name: str | None = None,
        term_placement: str | None = None,
    ) -> list[str]:
        """Launch step 1, window preflight: ``termctl open --dry-run`` with the real open's
        name, labels and placement (``term_placement`` is the value the real open will use: a
        resume passes the stored one when ``--term-placement`` is not given), ``--cwd`` only for
        a directory that exists (never ``session-dir``). Returns the window labels (with their
        warnings noted)."""
        labels, notes = window_labels(run_kind, agent_id, label_texts)
        self.warnings.extend(w for w in notes if w not in self.warnings)
        directory = None
        if cwd is not None and cwd != third_party.SESSION_DIR and os.path.isdir(cwd):
            directory = os.path.realpath(cwd)
        placement = self.args.term_placement or term_placement
        reply = self.termctl_backend().preflight(
            name=name or self.window_name(agent_id),
            labels=labels,
            placement=placement,
            cwd=directory,
            argv=[self_path, "supervise"],
        )
        socket = reply.get("socket_path")
        if isinstance(socket, str) and socket.startswith("/") and "\0" not in socket:
            self.window_socket = socket  # where a pane of this launch would be created
        return labels

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
            mode=self.mode,
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
        worktree = a.worktree or "none"
        return {
            "kind": a.kind,
            "mode": self.mode,
            "placement": self.placement_value(),
            "term_placement": a.term_placement,
            "name": a.name or a.worktree_name,
            # `--worktree <path>` stores the literal --repo (or null): a retry of its `adopt`
            # at resume passes exactly what the user gave, not the repo the plan derived.
            "repo": a.repo if worktree not in ("new", "none") else repo,
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
            "wait": self.waits,
            "wait_timeout": a.wait_timeout,
        }

    def placement_value(self) -> str | None:
        """The stored placement: ``window`` or ``here`` for an interactive run, else null."""
        return (self.args.placement or "window") if self.mode == "interactive" else None

    def pass_throughs(self) -> dict[str, Any]:
        found: dict[str, Any] = {}
        for name in PASS_THROUGH_FLAGS:
            value = getattr(self.args, name)
            if value is not None and value != []:
                found[name] = value
        return found

    def worktree_mode(self) -> str:
        return self.args.worktree or "none"

    def effective_cwd(self, classification: third_party.Classification) -> str | None:
        """The run's cwd: inside an adopted worktree, else ``--cwd``, else the plan's (the
        kind's default, possibly session-dir). ``None`` for ``--worktree new`` (step 4)."""
        mode = self.worktree_mode()
        if mode == "new":
            return None
        if mode != "none":
            return wtlib.resolve_in(mode, self.args.cwd)
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

    def refuse_result_file(self, message: str) -> CliError:
        """The usage error of a refused ``--result-file``; the target is dropped on the way, so
        nothing is written to it, not even the exit-2 envelope."""
        self.inv.result_file = None
        return CliError.usage(message)

    # -- wtctl: resolution, the records of a cwd, the rules, the holds -----------------------------

    def wtctl_backend(self, stored: str | None = None) -> WtctlBackend | None:
        """wtctl, resolved once: ``None`` under config ``wtctl_cmd: null`` (no wtctl on this
        machine); else config's command, else the absolute path stored with the agent; exit 12
        ``tool_missing`` when neither resolves."""
        if self.wt is not None:
            return self.wt
        command = self.config.wtctl_cmd
        if command is None:
            return None
        stored = stored or self.stored_wtctl
        argv = resolve_wtctl(command, stored)
        self.wt_argv = argv
        self.wt = WtctlBackend(
            argv, actor=self.inv.actor, decision_ids=self.decision_ids, gateway=self.gateway
        )
        return self.wt

    def live_records(self, *, fresh: bool = False) -> list[WtRecord]:
        """The live wtctl records (one ``wtctl list``, cached until ``fresh``); ``[]`` when there
        is no wtctl on this machine. A lookup error fails the command (wtctl's code)."""
        backend = self.wtctl_backend()
        if backend is None:
            return []
        if fresh or self.wt_records is None:
            self.wt_records = backend.list_records()
        return self.wt_records

    @staticmethod
    def needs_records(cwd: str | None) -> bool:
        """The records of a cwd are looked up for a real path (``session-dir`` lies in no
        record: it is under claude-session's own directory)."""
        return cwd is not None and cwd != third_party.SESSION_DIR

    def records_for(self, cwd: str | None) -> list[WtRecord]:
        return self.live_records(fresh=True) if self.needs_records(cwd) else []

    def enforce_rules(
        self,
        reads_third_party: bool,
        cwd: str | None,
        records: Sequence[WtRecord],
        *,
        session_work: str | None = None,
    ) -> None:
        """Pre-start checks 3 and 4: the cwd rule of a third-party-reading run and the throwaway
        check of a trusted one (exit 20, logged ``third_party_refused``)."""
        try:
            third_party.require_cwd_rule(
                reads_third_party, cwd, session_work=session_work, records=records
            )
        except CliError as error:
            self.log("third_party_refused", level="warn", details={"reason": error.reason})
            raise

    def pre_start_checks(self, classification: third_party.Classification) -> str | None:
        """Pre-start checks 2 to 4; returns the cwd (``None`` for ``--worktree new``: step 4)."""
        reads = classification.reads_third_party
        mode = self.worktree_mode()
        if mode == "new":
            if reads:  # the cwd will be a worktree: a third-party-reading run is always refused
                self.enforce_rules(True, None, [])
            return None  # a trusted run's checks 2 and 4 run at step 4, once the cwd exists
        cwd = self.effective_cwd(classification)
        records = self.records_for(cwd)
        if mode != "none" and cwd is not None and wtlib.inside_throwaway(records, cwd):
            raise CliError.usage("--worktree lies inside a wtctl throwaway")
        self.enforce_rules(reads, cwd, records)
        self.check_result_file(reads, cwd)
        return cwd

    def check_result_file(self, reads_third_party: bool, cwd: str | None) -> None:
        """The third-party path rule of ``--result-file``: not inside the run's cwd (a throwaway's
        root). The session directory does not exist before ``start``; the classification check
        covers it."""
        if (
            reads_third_party
            and cwd is not None
            and cwd != third_party.SESSION_DIR
            and self.result_file_inside([cwd])
        ):
            raise self.refuse_result_file(
                "--result-file lies inside a third-party-reading run's cwd"
            )

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
            "mode": self.mode,
            "status": None,
            "needs_attention": None,
            "placement": self.placement_value(),
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
            "wait": self.waits,
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
        if self.windowed:
            self.termctl_backend()  # tool resolution: exit 12 tool_missing before anything changes
        if self.worktree_mode() != "none":
            self.wtctl_backend()  # tool resolution: exit 12 tool_missing before anything changes
        self_path = resolve_self()
        if self.args.dry_run:
            return self.dry_run_launch(cs_argv)
        state = Path(self.state_dir)
        state.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.store = Store.open(self.state_dir, gateway=self.gateway)
        store = self.store
        # No `kill_grace` config key: the launcher never passes `--grace` to `stop`
        # (claude-session's own grace applies), so the backend's `kill_grace` only widens a call
        # timeout that is never lengthened here (decided in #13; a key is needed only when a
        # command takes a grace).
        self.backend = SessionBackend(
            cs_argv,
            actor=self.inv.actor,
            decision_ids=self.inv.decision_ids,
            gateway=self.gateway,
        )
        labels = self.label_set if self.label_set is not None else self.labels()
        proposed = new_agent_id(store)
        self.agent_id = None  # no record yet: log lines and errors carry no agent id
        self.copies = make_copies(self.state_dir, proposed, self.args)
        try:
            spec = self.build_spec(proposed, labels, self._classification_cwd(), self.copies)
            classification = self.backend.classify(spec)
            cwd = self.pre_start_checks(classification)
            if self.windowed:
                # Before any hook or allocation: a machine without a usable tmux fails here.
                self.window_preflight(
                    proposed, classification.run_kind, labels.texts(), cwd, self_path
                )
            self.run_pre_launch(labels, classification, cwd, self.copies)
            # The hooks' decision ids join every later call of the launcher.
            self.backend.decision_ids = list(self.decision_ids)
            if self.wt is not None:
                self.wt.decision_ids = list(self.decision_ids)
            if self.term is not None:
                self.term.decision_ids = list(self.decision_ids)
        except BaseException:
            remove_copies(self.copies.directory)
            raise
        return self.after_hooks(proposed, labels, classification, cwd, cs_argv, self_path)

    def _classification_cwd(self) -> str | None:
        mode = self.worktree_mode()
        if mode == "new":
            return None  # the worktree does not exist yet: `--repo` is passed instead
        if mode != "none":
            real = wtlib.resolve_in(mode, self.args.cwd)
            return real if os.path.isdir(real) else None
        given = self.args.cwd
        if given is None:
            return None
        if given == third_party.SESSION_DIR:
            return given
        real = os.path.realpath(given)
        return real if os.path.isdir(real) else None

    # -- dry run ------------------------------------------------------------------------------

    def dry_run_launch(self, cs_argv: Sequence[str]) -> Outcome:
        """``--dry-run`` creates nothing: no state directory, no database, no kept prompt copy.

        The copies the classification needs live in a private temporary directory that is
        removed before returning; the plan is the classification's own dry run, reused.
        """
        labels = self.label_set if self.label_set is not None else self.labels()
        agent_id = new_agent_id(_NoIndex())
        scratch = tempfile.mkdtemp(prefix="agent-spawn-dry-run-")
        try:
            copies = make_copies(scratch, agent_id, self.args)
            self.backend = SessionBackend(
                cs_argv,
                actor=self.inv.actor,
                decision_ids=self.inv.decision_ids,
                gateway=self.gateway,
            )
            spec = self.build_spec(agent_id, labels, self._classification_cwd(), copies)
            classification, session_plan = self.backend.classify_with_plan(spec, refuse=False)
            return self.dry_run(
                spec, labels, classification, session_plan, cs_argv, agent_id, copies
            )
        finally:
            shutil.rmtree(scratch, ignore_errors=True)

    def dry_run(
        self,
        spec: RunSpec,
        labels: LabelSet,
        classification: third_party.Classification,
        session_plan: dict[str, Any],
        cs_argv: Sequence[str],
        agent_id: str,
        copies: _Copies,
    ) -> Outcome:
        assert self.backend is not None
        cwd = self.effective_cwd(classification)
        would_refuse = classification.would_refuse
        if would_refuse is not None and would_refuse not in third_party.WOULD_REFUSE_REASONS:
            would_refuse = "unknown"
        worktree_plan: dict[str, Any] | None = None
        if would_refuse is None:
            would_refuse, worktree_plan = self.dry_run_worktree(classification, cwd, agent_id)
        term_plan: dict[str, Any] | None = None
        if self.windowed and would_refuse is None:
            # The preflight of step 1 and termctl's read-only plan; a failure is the real run's.
            window = self.window_preflight(
                agent_id, classification.run_kind, labels.texts(), cwd, resolve_self()
            )
            term_plan = self.termctl_backend().placements(window)
        outcome_path = os.path.join(self.state_dir, "outcomes", agent_id, "TOKEN.json")
        relay = relay_hook(resolve_self(), self.state_dir, self.inv.config_file or "")
        argv = [
            *cs_argv,
            "start",
            *launch_flags(spec),
            f"--hook={relay}",
            f"--outcome-file={outcome_path}",
            *self.backend.audit_flags(),
            "--json",
        ]
        plan = {
            "dry_run": True,
            "resume": False,
            "agent_settings": self.settings(
                labels, copies.input_names, classification.repo or self.args.repo
            ),
            "reads_third_party": classification.reads_third_party,
            "session_plan": session_plan,
            "would_refuse": would_refuse,
            "tools": {
                "claude-session": list(cs_argv),
                "termctl": self.term_argv if self.windowed else None,
                "wtctl": self.wt_argv,
            },
            "worktree_plan": worktree_plan,
            "term_plan": term_plan,
            "session_start_argv": argv,
            "hooks": self.hook_listing(),
            "warnings": self.warnings,
        }
        return Outcome(20 if would_refuse else 0, plan, PayloadKind.DRY_RUN)

    def dry_run_worktree(
        self, classification: third_party.Classification, cwd: str | None, agent_id: str
    ) -> tuple[str | None, dict[str, Any] | None]:
        """Pre-start checks 2 to 4, read-only, and ``wtctl create --dry-run`` for a new
        worktree: the refusal a real run would give and the worktree plan."""
        reads = classification.reads_third_party
        mode = self.worktree_mode()
        if mode == "new":
            if reads:
                return "third_party_cwd", None
            backend = self.wtctl_backend()
            assert backend is not None
            return None, backend.create_dry_run(self.create_plan(agent_id))
        records = self.records_for(cwd)
        if mode != "none" and cwd is not None and wtlib.inside_throwaway(records, cwd):
            raise CliError.usage("--worktree lies inside a wtctl throwaway")
        if not third_party.cwd_rule_allows(reads, cwd, records=records):
            return ("third_party_cwd" if reads else "third_party_run"), None
        self.check_result_file(reads, cwd)
        return None, None

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
        fields: dict[str, Any] = {
            "status": "launching",
            "launch_kind": args.kind,
            "mode": self.mode,
            "placement": self.placement_value(),
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
        }
        if self.wt_argv is not None:
            fields["wtctl_path"] = shlex.join(self.wt_argv)
        if self.windowed and self.term_argv is not None:
            fields["termctl_path"] = shlex.join(self.term_argv)
        # A signal is held back until the record and the flag agree: an interrupt must never
        # find a record the launcher does not know about.
        with _signals_held():
            row = store.insert_new_agent(fields, proposed=proposed, labels=labels)
            self.agent_id = row["agent_id"]
            self.agent_inserted = True
        assert self.agent_id is not None
        if self.agent_id != proposed:
            self.copies = _rename_copies(self.state_dir, self.copies, self.agent_id, args)
        self.log("launch", details={"mode": self.mode, "kind": args.kind})
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
        # The worktree (step 4), the holds (step 5), the session (step 6), the placement (step 7).
        cwd = self.prepare_worktree(row, classification, cwd)
        started = self.start_session(row, labels, classification, cwd)
        return self.place_and_wait(started)

    # -- steps 4 and 5: the worktree and the holds ------------------------------------------------

    def create_plan(self, agent_id: str) -> CreatePlan:
        a = self.args
        return CreatePlan(
            agent_id=agent_id,
            repo=a.repo,
            name=wtlib.planned_name(agent_id, a.worktree_name),
            branch=a.branch,
            base=a.base,
            ref=a.ref,
            detach=bool(a.detach),
            no_bootstrap=bool(a.no_bootstrap),
            copy_env=bool(a.copy_env),
        )

    def prepare_worktree(
        self, row: dict[str, Any], classification: third_party.Classification, cwd: str | None
    ) -> str | None:
        """Launch steps 4 and 5: create or adopt the worktree, resolve the cwd inside it, run a
        trusted run's checks 2 and 4 now, and hold every worktree record of the cwd. A failure
        parks the agent (``park_worktree_failure``) and raises."""
        agent_id = row["agent_id"]
        mode = self.worktree_mode()
        try:
            if mode == "new":
                record = self.run_create(agent_id, self.create_plan(agent_id))
                cwd = self.cwd_in_worktree(record.path)
            elif mode != "none":
                backend = self.wtctl_backend()
                assert backend is not None
                record, _ = backend.adopt(mode, agent_id, repo=self.args.repo)
                # `adopt` never creates a record: the agent releases it at teardown (forget),
                # whatever wtctl's own `owned` says of a record it adopted.
                self.store_worktree(agent_id, record.id, path=record.path, owned=False)
                cwd = self.cwd_in_worktree(mode)
            if mode != "none":
                self.update_cwd(agent_id, cwd)
            if mode == "new":
                self.enforce_rules(False, cwd, self.records_for(cwd))
            self.take_holds(agent_id, cwd)
        except CliError as error:
            self.park_worktree_failure(agent_id, error)
        return cwd

    def cwd_in_worktree(self, root: str) -> str:
        """The run's cwd: the worktree root, or ``--cwd`` resolved inside it."""
        real_root = os.path.realpath(root)
        cwd = wtlib.resolve_in(real_root, self.worktree_cwd_arg())
        if not _inside(cwd, real_root):
            raise CliError(
                "launch_failed", "--cwd resolves outside the worktree", details={"retryable": False}
            )
        return cwd

    def worktree_cwd_arg(self) -> str | None:
        return self.args.cwd

    def run_create(
        self, agent_id: str, plan: CreatePlan, *, override_escalation: bool = False
    ) -> WtRecord:
        """``wtctl create``; the record is stored from every result that names it: success (a
        reused record included; ``owned`` is wtctl's) and ``bootstrap_failed`` (when it is this
        agent's own record), before anything else can fail."""
        backend = self.wtctl_backend()
        assert backend is not None
        try:
            record, _ = backend.create(plan, override_escalation=override_escalation)
        except WtctlError as error:
            self.remember_named_record(agent_id, error, plan.name, backend)
            raise
        self.store_worktree(agent_id, record.id, path=record.path, owned=record.owned is True)
        return record

    def remember_named_record(
        self, agent_id: str, error: WtctlError, name: str, backend: WtctlBackend
    ) -> None:
        """A ``create`` that exited 12 ``bootstrap_failed`` named the record it kept
        (``details.record``): store it at once (a retry finds it by id). Anything wtctl names is
        untrusted: the record must be a live worktree of the planned name that this agent holds,
        else the failure is plain and nothing is stored."""
        record = error.record
        if record is None and error.record_id is not None:  # only an id: verify it with `show`
            try:
                record = backend.show(error.record_id)
            except CliError:
                return
        if (
            record is None
            or not record.live
            or record.object_type != "worktree"
            or record.name != name
            or agent_id not in record.holders
        ):
            return
        self.store_worktree(agent_id, record.id, path=record.path, owned=record.owned is True)

    def store_worktree(
        self, agent_id: str, record_id: str, *, path: str | None, owned: bool
    ) -> None:
        """Store the record the agent created or adopted (``wt_record_id`` and friends) and its
        hold row, in one transaction with signals held back."""
        assert self.store is not None
        changes: dict[str, object] = {"wt_record_id": record_id, "wt_owned": owned}
        if path is not None:
            changes["wt_path"] = path
        with _signals_held():
            with self.store.lease(
                lifecycle.lease_key(agent_id), wait=lifecycle.LEASE_WAIT
            ) as lease:
                with self.store.transaction():
                    self.store.update_agent(lease, agent_id, changes)
                    if path is not None:
                        self.store.put_hold(lease, agent_id, record_id, path, owned=owned)

    def update_cwd(self, agent_id: str, cwd: str | None) -> None:
        assert self.store is not None
        with self.store.lease(lifecycle.lease_key(agent_id), wait=lifecycle.LEASE_WAIT) as lease:
            self.store.update_agent(lease, agent_id, {"cwd": cwd}, touch=False)

    def take_holds(self, agent_id: str, cwd: str | None) -> None:
        """Pre-start check 5: hold every worktree record of the cwd the agent does not hold
        yet (``wtctl adopt --holder``); throwaways are never held. The records are looked up
        again, so a record created since the checks is not missed."""
        assert self.store is not None
        if not self.needs_records(cwd):
            return
        assert cwd is not None
        backend = self.wtctl_backend()
        if backend is None:
            return
        around = wtlib.worktrees_containing(self.live_records(fresh=True), cwd)
        if not around:
            return
        self.holds_started = True
        row = self.store.get_agent(agent_id)
        assert row is not None
        held = {hold["record_id"] for hold in self.store.get_holds(agent_id)}
        for record in around:
            if record.id in held and agent_id in record.holders:
                continue  # both sides agree that the agent holds it
            # A missing local row or a hold wtctl lost (holders emptied externally): `adopt` is
            # idempotent, so it re-adds the holder; the local row is reopened or put.
            with _signals_held():  # the hold wtctl took must reach the local row
                adopted, _ = backend.adopt(record.path, agent_id)
                self.wt_records = None
                # a hold taken by `adopt` is released at teardown, never removed: the agent did not
                # create the record (wtctl's `owned` is about whoever created it)
                self.store_hold(agent_id, adopted.id, adopted.path, owned=False)
        row = self.store.get_agent(agent_id)
        assert row is not None
        if row.get("wt_record_id") is None:
            nearest = max(around, key=lambda r: len(os.path.realpath(r.path)))
            self.store_nearest(agent_id, nearest)

    def store_hold(self, agent_id: str, record_id: str, path: str, *, owned: bool) -> None:
        assert self.store is not None
        with _signals_held():
            with self.store.lease(
                lifecycle.lease_key(agent_id), wait=lifecycle.LEASE_WAIT
            ) as lease:
                self.store.reacquire_hold(lease, agent_id, record_id, path, owned=owned)

    def store_nearest(self, agent_id: str, record: WtRecord) -> None:
        """No worktree was created or adopted at launch: the agent's ``worktree`` is the nearest
        record it holds for its cwd."""
        assert self.store is not None
        with self.store.lease(lifecycle.lease_key(agent_id), wait=lifecycle.LEASE_WAIT) as lease:
            self.store.update_agent(
                lease,
                agent_id,
                {"wt_record_id": record.id, "wt_path": record.path, "wt_owned": False},
                touch=False,
            )

    def park_worktree_failure(self, agent_id: str, error: CliError) -> None:
        """Park after a failed worktree step or hold and raise the pass-through error
        (interface.md, Launch order step 4)."""
        assert self.store is not None
        if isinstance(error, WtctlError):
            fields = wtlib.park_fields(wtlib.failure_kind(error))
        elif error.code == "refused":
            fields = wtlib.park_fields("busy")  # a refusal: launch_failed without attention
        else:
            fields = wtlib.park_fields("other")
        lifecycle.park(
            self.store,
            agent_id,
            from_status="launching",
            park_reason=str(fields["park_reason"]),
            needs_attention=fields["needs_attention"],
            escalated_by=fields["escalated_by"],
        )
        self.log("status_change", level="warn", details={"to": "parked", **fields})
        error.agent_id = agent_id
        error.decision_ids = list(self.decision_ids)
        raise error

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
                        "mode": self.mode,
                        "placement": self.run_placement(),
                        "cwd": cwd,
                        "reads_third_party": classification.reads_third_party,
                        "outcome_token": token,
                        "hooks": self.run_hooks,
                    },
                )

    def run_placement(self) -> str | None:
        """The run's placement: ``window``, ``here`` or null (headless)."""
        if self.windowed:
            return "window"
        return "here" if self.here else None

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
        recorded: dict[str, Any] | None = None
        if session is not None:
            self.csession_id = session
            agent_changes: dict[str, object] = {"csession_id": session}
            run_fields: dict[str, object] = {"mode": self.mode, "hooks": self.run_hooks}
            if run:
                # The run was allocated and never started: it is recorded with its not-started
                # outcome, updating `run`, `last_outcome` and `last_end_reason` together.
                lower = error.lower_error or {}
                reason = lower.get("end_reason")
                end_reason = reason if isinstance(reason, str) else None
                recorded = outcome_object(
                    agent_id,
                    RunOutcome(
                        session,
                        run,
                        "launch",
                        "not-started",
                        end_reason,
                        12,
                        "not-started",
                        None,
                        "status",
                    ),  # fmt: skip
                )
                agent_changes.update(run=run, last_outcome=recorded, last_end_reason=end_reason)
                run_fields.update(
                    outcome=recorded,
                    end_reason=end_reason,
                    exit_code=12,
                    ended_at=self.gateway.wall_time(),
                )
            with self.store.lease(
                lifecycle.lease_key(agent_id), wait=lifecycle.LEASE_WAIT
            ) as lease:
                with self.store.transaction():
                    self.store.update_agent(lease, agent_id, agent_changes)
                    if run:
                        self.store.put_run(lease, agent_id, run, run_fields)
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
        lifecycle.park(self.store, agent_id, from_status="launching", outcome=recorded, **fields)
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

        looked_up: list[list[WtRecord]] = []

        def rule_allows(actual: bool, run_cwd: str | None) -> bool:
            # The run's actual cwd is judged by the records as they are now (one `wtctl list`
            # for the whole check); a lookup error raises and cancels the run as unknown.
            # The session dir and its `work/` lie in no record: no lookup (as in step 1).
            if not looked_up:
                in_work = run_cwd is not None and _same_real_path(run_cwd, work)
                needs = self.needs_records(run_cwd) and not in_work
                looked_up.append(self.live_records(fresh=True) if needs else [])
            return third_party.cwd_rule_allows(
                actual, run_cwd, session_work=work, records=looked_up[0]
            )

        def result_file_allows() -> bool:
            if not self.result_file_inside([result.csession_dir, *([cwd] if cwd else [])]):
                return True
            self.inv.result_file = None  # refused: never written, not even the error envelope
            return False

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
        # The run is third-party-reading or unknown: fail closed, so the error envelope is never
        # written into its session directory or cwd (only the stdout copy remains).
        if self.result_file_inside([result.csession_dir, *([cwd] if cwd else [])]):
            third_party.drop_result_file(self.inv)
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
        if check.cancel_error is not None:
            # The run could not be canceled and may still be live: say so, keep the session.
            code = check.cancel_error.code
            self.log(
                "warning",
                level="warn",
                csession_id=result.csession_id,
                message=f"the pending run could not be canceled ({code})",
            )
            error.details = {**(error.details or {}), "cancel_failed": True, "cancel_code": code}
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
        if self.windowed:
            return self.place_window(result, row)
        if self.here:
            return self.place_here(result, row)
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
                # The supervisor starts with these blocked and unblocks them once its handlers
                # are installed, so an early signal is delivered instead of killing it.
                keep_blocked=HANDLED_SIGNALS,
                gateway=self.gateway,
            )
            identity = self.detached.identity
            with store.lease(lifecycle.lease_key(agent_id), wait=lifecycle.LEASE_WAIT) as lease:
                store.update_agent(lease, agent_id, {"supervisor": identity})
            self.forwarder = procs.SignalForwarder(identity, gateway=self.gateway)
            self.sig.raising = False  # from here a signal is forwarded, not a reason to cancel
            self.detached.acknowledge()
            self.supervised = True
        except (procs.DetachedForkError, OSError, StoreError) as error:
            self.sig.raising = True
            self.placement_failed(result, error)
            raise  # placement_failed raises; unreachable
        self.log("launch", message="supervisor started", pid=identity.pid)
        return self.wait()

    def place_here(self, result: StartResult, row: dict[str, Any]) -> Outcome:
        """Launch step 7, interactive ``--placement here``: this process is the supervisor, in
        the foreground, with the terminal as ``exec``'s stdio (implied ``--wait``).

        The launcher's signal rows apply until ``exec`` is spawned (an interrupt cancels the
        pending run and parks it: this process is then the only one that could, so exactly one
        process parks); ``Supervisor.spawn_exec`` swaps in the supervisor's handlers when it
        spawns ``exec``. Nothing is written to the terminal while ``exec`` owns it: the result
        goes to the output after ``exec`` ended and to ``--result-file``."""
        from .supervise import Signals, Supervisor  # late: supervise imports this module

        assert self.store is not None and self.agent_id is not None
        store, agent_id = self.store, self.agent_id
        # SIGQUIT is ignored from the moment this process becomes the supervisor, before the
        # (possibly long) wait for the lease that records it.
        restore_quit = procs.install_signal_handlers(
            {signal.SIGQUIT: signal.SIG_IGN}, gateway=self.gateway
        )
        supervisor = Supervisor(
            self.inv,
            Signals(interactive=True),
            agent_id=agent_id,
            state_dir=self.state_dir,
            in_process=True,
        )
        try:
            with store.lease(lifecycle.lease_key(agent_id), wait=lifecycle.LEASE_WAIT) as lease:
                # Recorded before `exec` can adopt the run (the supervisor is this process).
                store.update_agent(lease, agent_id, {"supervisor": store.holder_identity()})
            self.log("launch", message="in-process supervisor started", pid=os.getpid())
            supervisor.run_in_process(store, row)
        except (StoreError, CliError, OSError) as error:
            if supervisor.child is not None:
                # `exec` was started (and stopped again by the supervisor): this process was its
                # supervisor, so only the agent record is left for the liveness check.
                # The record stays for the liveness check: once this process (the recorded
                # supervisor) is gone, the next check applies the outcome table or flags it.
                raise CliError(
                    "failed",
                    "the supervisor failed before the agent parked (see reconcile)",
                    agent_id=agent_id,
                ) from None
            if isinstance(error, StoreError):
                raise  # no `exec` yet: `store_failed` cancels the pending run and parks the agent
            # No `exec` yet, and no other process can adopt the run: cancel it and park.
            self.sig.raising = False
            try:
                self.cancel_pending(agent_id)
            except (CliError, StoreError):
                pass  # the agent stays `launching`: the liveness check settles it (dead launcher)
            if isinstance(error, CliError):
                if error.agent_id is None:
                    error.agent_id = agent_id
                raise
            raise CliError(
                "launch_failed",
                f"the supervisor could not be started ({type(error).__name__})",
                agent_id=agent_id,
                decision_ids=list(self.decision_ids),
            ) from None
        finally:
            self.sig.raising = False  # a late signal is recorded, never raised
            if supervisor.restore_handlers is not None:
                supervisor.restore_handlers()
            restore_quit()
        self.supervised = True
        final = store.get_agent(agent_id)
        if final is None:
            raise CliError("failed", "the agent record disappeared", agent_id=agent_id)
        if final["status"] in ("launching", "live"):
            raise CliError(
                "failed",
                "the supervisor ended before the agent parked (see reconcile)",
                agent_id=agent_id,
            )
        return self.result(final, final=True)

    def window_cwd(self, row: dict[str, Any], result: StartResult) -> str | None:
        """The window's cwd (launch step 7): the run's cwd; for ``session-dir`` the realpath of
        ``<csession_dir>/work``. ``None`` leaves it to termctl (the launcher's directory)."""
        cwd = row.get("cwd")
        if cwd == third_party.SESSION_DIR:
            return os.path.realpath(os.path.join(result.csession_dir, "work"))
        return os.path.realpath(cwd) if isinstance(cwd, str) and cwd else None

    def supervise_argv(self, row: dict[str, Any], listener: HandoffListener) -> list[str]:
        """The window's command: absolute paths only, no secret and no agent text."""
        return [
            row["agent_spawn_path"],
            "supervise",
            row["agent_id"],
            "--handoff",
            str(listener.socket_path),
            "--handoff-timeout",
            str(self.config.handoff_timeout),
            "--state-dir",
            self.state_dir,
            "--config",
            row["config_path"],
        ]

    def place_window(self, result: StartResult, row: dict[str, Any]) -> Outcome:
        """Launch step 7, interactive window: listen on the handoff socket, ``termctl open``,
        record the pane's process as the supervisor and serve the handoff. A failure before the
        handoff completed means no supervisor exists: the pending run is canceled and the agent
        parked (exactly one process parks)."""
        assert self.store is not None and self.agent_id is not None
        store, agent_id = self.store, self.agent_id
        term_backend = self.termctl_backend()
        try:
            listener = HandoffListener.create(
                self.state_dir, agent_id=agent_id, gateway=self.gateway
            )
        except HandoffFailed as failure:
            self.placement_failed(result, self.handoff_error(failure))
            raise  # unreachable
        try:
            run_kind = row.get("run_kind")
            labels, notes = window_labels(
                run_kind if isinstance(run_kind, str) else None,
                agent_id,
                store.get_labels(agent_id).texts(),
            )
            self.warnings.extend(w for w in notes if w not in self.warnings)
            before = self.panes_before_open(term_backend)
            try:
                pane = term_backend.open(
                    name=row["name"],
                    labels=labels,
                    placement=row.get("term_placement"),
                    cwd=self.window_cwd(row, result),
                    argv=self.supervise_argv(row, listener),
                    override_escalation=self.term_override,
                )
            except LaunchInterrupted:
                self.recover_window(result, term_backend, before)  # the pane may exist unrecorded
                raise
            except CliError as error:
                if error.exit_code not in (11, 13):  # a veto or escalation created no pane
                    self.recover_window(result, term_backend, before)
                if self.term_override:
                    self.term_override_audit(
                        "refused" if error.exit_code == 20 else "error", agent_id
                    )
                self.placement_failed(result, error)
                raise  # unreachable
            if self.term_override:
                self.term_override_audit("ok", agent_id)
            try:
                identity = self.record_window(result, pane)
            except BaseException:
                self.recover_window(result, term_backend, before)
                raise
            try:
                listener.serve(
                    pane_pid=pane.pid,
                    pane_state=pane.state,
                    timeout=self.config.handoff_timeout,
                    env=dict(os.environ),
                )
            except HandoffFailed as failure:
                self.placement_failed(result, self.handoff_error(failure))
                raise  # unreachable
            # The handoff is done: a signal from here is forwarded (below), never raised.
            self.sig.raising = False
        finally:
            listener.close()
        if identity is None:  # the pane process could not be read when it was recorded
            self.placement_failed(result, self.handoff_error(HandoffFailed("pane_unreadable")))
            raise  # unreachable
        self.sync_window_labels(term_backend, pane, run_kind, labels)
        self.forwarder = procs.SignalForwarder(identity, gateway=self.gateway)
        self.sig.raising = False  # from here a signal is forwarded, not a reason to cancel
        self.supervised = True
        self.log("launch", message="window supervisor started", pid=identity.pid)
        return self.wait()

    def sync_window_labels(
        self, term_backend: TermBackend, pane: Pane, run_kind: object, opened: Sequence[str]
    ) -> None:
        """Labels were read before ``open`` (which can take long) and a ``label add`` or
        ``remove`` in that gap found no window to mirror to: push the difference between the
        labels now stored and the ones the window was opened with. Best effort."""
        assert self.store is not None and self.agent_id is not None
        if not sync_window_labels(term_backend, self.store, self.agent_id, pane, run_kind, opened):
            self.log("warning", level="warn", message="the window's labels could not be synced")

    def panes_before_open(self, term_backend: TermBackend) -> set[tuple[str, str]] | None:
        """The agent's labeled panes (an older run's dead pane among them) before the real
        ``open``, so that recovery tells the new pane from them; None when unknown. Best effort."""
        assert self.agent_id is not None
        try:
            return set(term_backend.find_panes(self.agent_id, self.window_socket))
        except CliError:
            return None

    def recover_window(
        self,
        result: StartResult,
        term_backend: TermBackend,
        before: set[tuple[str, str]] | None,
    ) -> None:
        """A ``termctl open`` that failed or was interrupted may have created its pane without
        reporting it: find it by its ``agent=`` label (on the server the preflight named), among
        the panes that were not there before the open, and record it on the run, so that teardown
        closes it. When ``before`` is unknown and several panes match, only their socket path is
        recorded (no pane id is guessed). Best effort, never raises."""
        assert self.store is not None and self.agent_id is not None
        self.sig.raising = False  # a signal during this cleanup is recorded, never aborts it
        agent_id = self.agent_id
        try:
            found = term_backend.find_panes(agent_id, self.window_socket)
            if before is not None:
                found = [pane for pane in found if pane not in before]
            if not found:
                return
            fields: dict[str, Any]
            if before is None and len(found) > 1:
                sockets = {socket_path for _, socket_path in found}
                if len(sockets) != 1:
                    return
                fields = {"socket_path": sockets.pop()}
            else:
                fields = {"term_id": found[0][0], "socket_path": found[0][1]}
            with self.store.lease(
                lifecycle.lease_key(agent_id), wait=lifecycle.LEASE_WAIT
            ) as lease:
                self.store.put_run(lease, agent_id, result.run, fields)
        except (CliError, StoreError, OSError):
            self.log("warning", level="warn", message="a window that failed to open was not found")

    def term_override_audit(self, result: str, agent_id: str) -> None:
        """``--override-escalation`` went to ``termctl open`` (resume only)."""

    @staticmethod
    def handoff_error(failure: HandoffFailed) -> CliError:
        return CliError(
            "handoff_failed",
            "the environment handoff to the window failed",
            details={"cause": failure.cause},
        )

    def record_window(self, result: StartResult, pane: Pane) -> procs.ProcessIdentity | None:
        """Store the window on the run (``term_id``, ``socket_path``, the pane process) and the
        pane process as the supervisor, before the handoff completes (the supervisor checks its
        record once it has its environment)."""
        assert self.store is not None and self.agent_id is not None
        identity: procs.ProcessIdentity | None = None
        if pane.pid is not None and pane.state != "gone":
            try:
                identity = procs.read_identity(pane.pid, gateway=self.gateway)
            except (procs.ProcessIdentityError, OSError):
                identity = None
        agent_id = self.agent_id
        with self.store.lease(lifecycle.lease_key(agent_id), wait=lifecycle.LEASE_WAIT) as lease:
            with self.store.transaction():
                self.store.put_run(
                    lease,
                    agent_id,
                    result.run,
                    {
                        "term_id": pane.term_id,
                        "socket_path": pane.socket_path,
                        "pane_proc": identity,
                    },
                )
                if identity is not None:
                    self.store.update_agent(lease, agent_id, {"supervisor": identity})
        return identity

    def placement_failed(self, result: StartResult, error: BaseException) -> None:
        """No supervisor exists: cancel the pending run and park (``launch_failed``; a termctl
        veto or escalation parks as such)."""
        assert self.store is not None and self.backend is not None and self.agent_id is not None
        agent_id = self.agent_id
        self.sig.raising = False  # a signal during this cleanup is recorded, never aborts it
        if self.detached is not None:
            self.detached.abort()
        try:
            with self.store.lease(
                lifecycle.lease_key(agent_id), wait=lifecycle.LEASE_WAIT
            ) as lease:
                self.store.update_agent(lease, agent_id, {"supervisor": None})
        except StoreError:
            self.log("warning", level="warn", message="could not clear the supervisor record")
        try:
            self.backend.stop(result.csession_id)
        except CliError:
            pass
        fields: dict[str, Any] = {
            "park_reason": "launch_failed",
            "needs_attention": "launch_failed",
        }
        if isinstance(error, CliError) and error.tool == "termctl":
            if error.exit_code == 11:
                fields = {"park_reason": "vetoed", "needs_attention": None}
            elif error.exit_code == 13:
                fields = {
                    "park_reason": "escalated",
                    "needs_attention": "escalated",
                    "escalated_by": "termctl",
                }
        try:
            lifecycle.park(self.store, agent_id, from_status="launching", **fields)
        except StoreError:
            # the agent stays `launching`: the liveness check settles it (dead launcher)
            self.log("warning", level="warn", message="could not park the agent")
        self.log("status_change", level="warn", details={"to": "parked", **fields})
        if isinstance(error, CliError):
            error.agent_id = agent_id
            error.decision_ids = list(self.decision_ids)
            raise error
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
        wait = self.waits
        deadline = None
        if wait and args.wait_timeout is not None:
            deadline = self.gateway.monotonic() + _seconds(args.wait_timeout)
        next_liveness = 0.0
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
            try:
                alive = supervisor is not None and procs.is_alive(supervisor, gateway=self.gateway)
            except procs.ProcessIdentityError:
                # unreadable is an error, never "dead" (interface.md, Liveness)
                raise unreadable(agent_id, "supervisor") from None
            ended = status not in ("launching", "live")
            if not wait:
                if status == "live":
                    return self.result(row, final=False)
                if ended and not alive:  # the run ended before this launcher saw it started
                    return self.result(row, final=True)
            elif lifecycle.run_settled(row, supervisor_alive=alive):
                return self.result(row, final=True)
            if not alive and status in ("launching", "live"):
                # Nothing drives the agent: the liveness check decides on each poll (it never
                # signals, never stops the run and delivers no hook; it is rate limited because
                # it calls claude-session).
                now = self.gateway.monotonic()
                if now >= next_liveness:
                    next_liveness = now + LIVENESS_EVERY
                    checked = Liveness(
                        store,
                        state_dir=self.state_dir,
                        fallback_command=self.backend.command if self.backend else None,
                        gateway=self.gateway,
                    ).check(agent_id)
                    if row_moved(row, checked.row):
                        continue  # a real transition: look at the new row once, at once
            if deadline is not None and self.gateway.monotonic() >= deadline:
                payload = result_object(
                    agent=agent_object(store, row, warnings=self.warnings),
                    timed_out=True,
                    first_working_at=self.first_working_at(row),
                    warnings=self.warnings,
                )
                return Outcome(EXIT_TIMED_OUT, payload, PayloadKind.RESULT)
            self.gateway.sleep(POLL)

    def first_working_at(self, row: dict[str, Any], run: int | None = None) -> str | None:
        assert self.store is not None
        return chain_first_working_at(self.store, row, run)

    def forward_signal(self, received: int) -> Outcome | None:
        """Handle a flagged signal: forward SIGTERM to the supervisor and keep waiting, or (a
        waiter whose run started) stop waiting on SIGHUP and SIGINT."""
        if received != signal.SIGTERM and self.started and self.waits:
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
        first = self.first_working_at(row)  # the chain's first run, whichever run ended last
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
        if self.agent_inserted and self.csession_id is not None and self.copies is not None:
            remove_copies(self.copies.directory)  # a session was allocated: the copies are spent
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

    def cleanup_failed(self, failure: CliError) -> CliError:
        """Undo what a failed command left that its error path did not (called once, with
        signals recorded rather than raised); returns the failure to report. A launch before its
        record exists drops its prompt copies; once the record exists they stay for the next
        resume."""
        if self.copies is not None and not self.agent_inserted:
            remove_copies(self.copies.directory)
        return failure

    def store_failed(self, error: StoreError) -> CliError:
        """The state database failed mid-launch: end what is pending and report, never escape.

        Before a supervisor is recorded and acknowledged the pending run is canceled and the
        agent parked; afterwards the supervisor owns the run and is left alone.
        """
        agent_id = self.agent_id if self.agent_inserted else None
        if agent_id is not None and not self.supervised and self.store is not None:
            try:
                self.cancel_pending(agent_id)
            except (CliError, StoreError):
                pass  # the agent stays `launching`: the liveness check settles it (dead launcher)
            if self.csession_id is not None and self.copies is not None:
                remove_copies(self.copies.directory)
        self.log("warning", level="warn", message=f"state error ({type(error).__name__})")
        return CliError(
            "launch_failed" if agent_id is not None else "failed",
            f"the state database could not be used ({type(error).__name__})",
            agent_id=agent_id,
            decision_ids=list(self.decision_ids),
        )

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


class _NoIndex:
    """An empty agent index: a dry run allocates an id without opening the database."""

    def entries_with_prefix(self, prefix: str) -> list[IdEntry]:
        return []

    def is_reserved(self, agent_id: str) -> bool:
        return False


@contextmanager
def _signals_held() -> Iterator[None]:
    """Block SIGTERM, SIGHUP and SIGINT; a signal that arrived meanwhile is delivered on exit."""
    old = signal.pthread_sigmask(signal.SIG_BLOCK, HANDLED_SIGNALS)
    try:
        yield
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, old)


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


_pending_restore: Callable[[], None] | None = None


def defer_restore(restore: Callable[[], None]) -> None:
    """Keep handlers installed until ``restore_signal_handlers`` (see ``launch``)."""
    global _pending_restore
    _pending_restore = restore


def restore_signal_handlers() -> None:
    """Put back the handlers ``launch(..., restore_handlers=False)`` replaced (idempotent)."""
    global _pending_restore
    restore, _pending_restore = _pending_restore, None
    if restore is not None:
        restore()


def launch(inv: Any, *, restore_handlers: bool = True) -> Outcome:
    """Run ``launch``; the handler of ``commands.launching``.

    With ``restore_handlers=False`` (the command line) the recording handlers stay installed
    after this returns, so a signal that arrives while the result is printed cannot lose it;
    the caller then calls ``restore_signal_handlers`` once the output is flushed.
    """
    return run_command(inv, Launch, restore_handlers=restore_handlers)


def run_command(
    inv: Any, make: Callable[[Any, SignalState], Launch], *, restore_handlers: bool
) -> Outcome:
    """The signal handling and failure cleanup shared by ``launch`` and ``resume``."""
    global _pending_restore
    sig = SignalState()
    launch_run = make(inv, sig)
    restore = procs.install_signal_handlers(
        {signal.SIGTERM: sig.handle, signal.SIGHUP: sig.handle, signal.SIGINT: sig.handle}
    )
    if not restore_handlers:
        _pending_restore = restore
    try:
        failure: CliError
        try:
            return launch_run.run()
        except LaunchInterrupted as exc:
            outcome = launch_run.interrupted(exc.signum)
            launch_run.deliver()
            return outcome
        except StoreError as error:
            sig.raising = False  # a second signal must not interrupt the cleanup below
            failure = launch_run.store_failed(error)
        except CliError as error:
            sig.raising = False
            failure = error
        failure = launch_run.cleanup_failed(failure)
        if failure.agent_id is None and launch_run.agent_inserted:
            failure.agent_id = launch_run.agent_id
        launch_run.deliver()
        raise failure
    finally:
        sig.raising = False
        if restore_handlers:
            restore()
        if launch_run.store is not None:
            launch_run.store.close()
