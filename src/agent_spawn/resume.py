"""``resume`` and ``resume --fork`` for headless agents without a worktree (interface.md, Resume
order, Resume without a session, Resume failures).

A resume reuses the launcher of ``launch.py`` (``Resume`` is a ``Launch``): the signal handling,
the post-start classification check, the detached supervisor and the wait are the same code. What
differs is the lifecycle entry and its undoing:

* **plain resume** moves ``parked`` to ``launching`` first (``Store.enter_resume``, a
  compare-and-set that saves the park fields). Every failure before a run of this resume exists
  puts those fields back (``Store.restore_pre_resume``, also in the signal path); a failure after
  the run exists parks the agent like a launch does.
* **fork** inserts the fork's record; a failure before the fork's session exists deletes it again
  (its id stays reserved) and drops its prompt copies.
* an agent **without a session** re-runs its launch from the stored settings
  (``agents.settings``: ``{"settings": ..., "pass_throughs": ...}``). Its stored prompt and input
  copies are never deleted by a failed resume; only an allocated session removes them. The same
  holds for the new ``resume-*`` copies of a plain resume of an agent with a session (D26: "a
  failed resume never deletes them"; ``reconcile`` removes orphans older than an hour).

Prompt and input text travels only in files (the private copies), never in argv or environment.

Worktrees: an agent without a session re-runs launch step 4 as resume step 5 (``resume_worktree``:
``wtctl show`` by the stored id, else the ``wtctl list --repo`` lookup of the planned name, else
``create``/``adopt``); every run takes its holds (``take_holds``) before ``start``. A busy wtctl,
or any failure that left the agent's state unchanged, restores the pre-resume row; a wtctl veto
or escalation of ``create``, a ``creating`` record and a record in another state park it.

TODO(#27): interactive resumes with a window; TODO(#30): ``--placement here``. TODO(#31): the
liveness check that precedes the lifecycle entry.
"""

from __future__ import annotations

import os
import secrets
import shlex
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from typing import Any, NoReturn

from . import hooks as hooklib
from . import lifecycle, third_party
from . import worktrees as wtlib
from .agent_view import outcome_object
from .ids import resolve_agent_id
from .labels import Label, LabelSet, job_label, parse_run_labels, parse_selector
from .launch import (
    Launch,
    SignalState,
    _Copies,
    _signals_held,
    _tty_pair,
    input_names,
    make_copies,
    new_agent_id,
    remove_copies,
    resolve_self,
    run_command,
)
from .output import (
    EXIT_SIGINT,
    EXIT_SIGTERM,
    CliError,
    Outcome,
    PayloadKind,
    error_outcome,
)
from .records import RecordsError, append_audit
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
from .store import AgentMissing, CasLost, Store, StoreError, StoreMissing
from .validate import GrammarError, validate_name
from .worktrees import CreatePlan, WtctlBackend, WtctlError, WtRecord

SIGNAL_NAME = {2: "SIGINT", 1: "SIGHUP", 15: "SIGTERM"}
DEFAULT_SETTINGS: dict[str, Any] = {
    "kind": None,
    "mode": "headless",
    "placement": None,
    "term_placement": None,
    "name": None,
    "repo": None,
    "worktree": "none",
    "branch": None,
    "base": None,
    "ref": None,
    "detach": False,
    "worktree_name": None,
    "no_bootstrap": False,
    "copy_env": False,
    "cwd": None,
    "labels": [],
    "skill": None,
    "has_prompt": False,
    "input_names": [],
    "plugin_dirs": [],
    "promote": None,
    "ephemeral": False,
    "wait": False,
    "wait_timeout": None,
}
#: ``claude-session start`` flags a stored pass-through name maps to (RunSpec field names).
PASS_THROUGH_NAMES = (
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


def _refusal(reason: str, message: str, agent_id: str | None = None) -> CliError:
    return CliError("refused", message, reason=reason, agent_id=agent_id)


class Resume(Launch):
    """One ``resume`` invocation."""

    def __init__(self, inv: Any, sig: SignalState) -> None:
        super().__init__(inv, sig)
        args = self.args
        self.fork: bool = bool(args.fork)
        self.args.repo = None  # set from the agent once it is resolved (record_status reads it)
        self.source: dict[str, Any] = {}
        self.source_id = ""
        self.source_session: str | None = None  # the session resumed or forked
        self.stored: dict[str, Any] = {}
        self.stored_pass: dict[str, Any] = {}
        self.mode = "headless"
        self.given_labels: list[Label] = []
        self.start_labels: list[str] = []
        self.new_labels: LabelSet = LabelSet()  # the labels to store at record_session
        self.label_applied: tuple[Label, ...] = ()
        self.skill: str | None = None
        self.kind_flag: str | None = None
        self.plugin_dirs: tuple[str, ...] = ()
        self.run_pass: dict[str, Any] = {}
        self.pass_override = False
        self.resume_cwd: str | None = None  # the cwd the run's pre-start checks use
        self.expect_run_after: int | None = None
        self.entered = False  # a plain resume left `parked` (the pre-resume row is saved)
        self.fork_inserted = False
        self.kept_fork_failure: CliError | None = None  # a fork kept because a hold stayed
        self.start_attempted = False
        self.settled = False  # a run of this resume is recorded (or the agent parked): no restore
        self.run_recorded = False
        self.override_audit: list[str] = []
        self.new_copies: _Copies | None = None
        self.unseen_session: str | None = None  # a plain resume's session whose status failed
        self.wt_override = False  # --override-escalation goes to `wtctl create` (step 5)

    # -- step 1: usage checks ---------------------------------------------------------------

    def usage_checks(self) -> None:
        a = self.args
        if a.cwd is not None:
            if not self.fork:
                raise CliError.usage("--cwd needs --fork")
            if a.cwd != third_party.SESSION_DIR and not os.path.isabs(a.cwd):
                a.cwd = os.path.realpath(a.cwd)  # launch resolves a relative --cwd the same way
        if a.prompt is not None and a.prompt_file is not None:
            raise CliError.usage("--prompt cannot be combined with --prompt-file")
        if a.placement == "here" and not _tty_pair():
            raise CliError.usage("--placement here needs a terminal on stdin and stdout")
        if a.placement == "here" and a.wait_timeout is not None:
            raise CliError.usage("--wait-timeout cannot be combined with --placement here")
        for value in a.hook or []:
            if value.partition("=")[0] == "pre-launch":
                raise CliError.usage(
                    "--hook pre-launch does not apply to resume (agent-spawn's pre-launch hooks "
                    "never gate a resume)"
                )
        self.run_hooks = hooklib.parse_invocation_hooks(a.hook or [], ("post-launch",))
        if a.max_turns is not None:
            self.max_turns()
        input_names(a)
        if a.name is not None:
            try:
                validate_name(a.name)
            except GrammarError as error:
                raise CliError.usage(str(error)) from None
        for setting in a.reset or []:
            if not setting or "\0" in setting:
                raise CliError.usage("--reset needs a setting name")
        if (a.allow_tainted or a.override_escalation) and not self.inv.actor.startswith("human:"):
            # The CLI already gates these flags; this keeps the handler safe when called directly.
            raise CliError.usage(
                "--allow-tainted and --override-escalation need an actor human:<user>"
            )
        try:
            self.given_labels = parse_run_labels(a.label or [])
        except GrammarError as error:
            raise CliError.usage(str(error)) from None

    # -- the agent ------------------------------------------------------------------------

    def open_store(self) -> Store:
        try:
            return Store.open(self.state_dir, create=False, gateway=self.gateway)
        except StoreMissing:
            raise CliError.of("not_found", "no agents exist yet") from None

    def resolve_source(self, store: Store) -> dict[str, Any]:
        a = self.args
        if a.agent_id is not None:
            resolved = resolve_agent_id(a.agent_id, store)
            if resolved.torn_down:
                raise CliError.of(
                    "not_found",
                    f"agent {resolved.agent_id} is torn down",
                    agent_id=resolved.agent_id,
                    details={"status": "torn_down"},
                )
            agent_id = resolved.agent_id
        else:
            found = store.find_agents([parse_selector(text) for text in a.select])
            if not found:
                raise CliError.of("not_found", "no agent matches the --select pairs")
            if len(found) > 1:
                raise CliError.of(
                    "ambiguous", f"--select matches {len(found)} agents", candidates=sorted(found)
                )
            agent_id = found[0]
        row = store.get_agent(agent_id)
        if row is None or row["status"] == "torn_down":
            raise CliError.of(
                "not_found",
                f"agent {agent_id} is torn down",
                agent_id=agent_id,
                details={"status": "torn_down"},
            )
        return row

    # -- step 4: resolve the run ----------------------------------------------------------

    def last_mode(self, store: Store, row: Mapping[str, Any]) -> str | None:
        for run in reversed(store.get_runs(row["agent_id"])):
            if not run.get("promoted") and isinstance(run.get("mode"), str):
                return run["mode"]
        mode = row.get("mode")
        return mode if isinstance(mode, str) else None

    def never_started(self, store: Store, agent_id: str) -> bool:
        """Every run of the session ended in its launch phase (so the skill was never used)."""
        runs = store.get_runs(agent_id)
        if not runs:
            return False
        for run in runs:
            outcome = run.get("outcome")
            if not isinstance(outcome, dict) or outcome.get("phase") != "launch":  # pyright: ignore[reportUnknownMemberType]
                return False
        return True

    def plan_run(self, store: Store, row: dict[str, Any]) -> None:
        """Everything that is decided from the stored agent and the flags, before any state
        changes (the usage errors that need the agent included)."""
        a = self.args
        agent_id = row["agent_id"]
        self.source, self.source_id = row, agent_id
        settings = row.get("settings")
        session = row.get("csession_id")
        if isinstance(settings, dict):
            raw_settings = settings.get("settings")  # pyright: ignore[reportUnknownMemberType]
            raw_pass = settings.get("pass_throughs")  # pyright: ignore[reportUnknownMemberType]
            self.stored = dict(raw_settings) if isinstance(raw_settings, dict) else {}  # pyright: ignore[reportUnknownArgumentType]
            self.stored_pass = dict(raw_pass) if isinstance(raw_pass, dict) else {}  # pyright: ignore[reportUnknownArgumentType]
            damaged = not (isinstance(raw_settings, dict) and isinstance(raw_pass, dict))
        else:
            damaged = True
        if damaged and session is None and not self.fork:
            # A launch re-run needs its stored flags: dropping them silently would run
            # something else than was asked for.
            raise CliError(
                "failed",
                "the stored launch settings of the agent are unusable",
                agent_id=agent_id,
                details={"hint": "launch the agent again"},
            )
        self.source_session = session if isinstance(session, str) else None
        self.stored_wtctl = (
            row.get("wtctl_path") if isinstance(row.get("wtctl_path"), str) else None
        )
        self.args.repo = row.get("repo") or self.stored.get("repo")
        if self.source_session is None and self.fork:
            if row["status"] == "tearing_down":
                return  # step 3 lists tearing_down first: run() refuses it
            raise CliError("no_run", f"agent {agent_id} has no session to fork", agent_id=agent_id)
        if self.source_session is None:
            if a.reset or a.allow_tainted:
                raise CliError.usage("--reset and --allow-tainted need an agent with a session")
        # Mode, then the checks that depend on it.
        stored_mode = self.stored.get("mode") if self.source_session is None else None
        mode = a.mode or stored_mode or self.last_mode(store, row) or self.stored.get("mode")
        self.mode = mode if mode in ("interactive", "headless") else "headless"
        if self.mode != "headless":
            raise CliError.of(
                "internal",
                "interactive resumes are not implemented yet (windows: #27, here: #30)",
            )  # TODO(#27): windows; TODO(#30): --placement here.
        if a.placement is not None:
            self.warnings.append("--placement is ignored for a headless resume")
        self.plan_override(row)
        self.plan_skill(store, row)
        self.plan_flags()
        self.plan_labels(store, row)

    def plan_override(self, row: Mapping[str, Any]) -> None:
        """``--override-escalation`` is passed to the tool that escalated, else exit 2."""
        a = self.args
        if not a.override_escalation:
            return
        if self.fork:
            # Only a window's `termctl open` could take it for a fork; never claude-session.
            self.warnings.append("--override-escalation has no effect on a headless fork")
            return
        by = row.get("escalated_by")
        last = row.get("last_outcome")
        ended_escalated = isinstance(last, dict) and last.get("outcome") == "escalated"  # pyright: ignore[reportUnknownMemberType]
        if by == "claude-session" and ended_escalated and self.source_session is not None:
            self.pass_override = True
            return
        if by == "wtctl" and self.source_session is None and self.worktree_mode() == "new":
            self.wt_override = True  # goes to `wtctl create` at step 5, and to nothing else
            return
        raise CliError.usage(
            "--override-escalation has no target: the agent was not escalated by a tool this "
            "resume calls"
        )  # TODO(#27): the termctl override

    def plan_skill(self, store: Store, row: Mapping[str, Any]) -> None:
        a = self.args
        stored_skill = self.stored.get("skill")
        stored_skill = stored_skill if isinstance(stored_skill, str) else None
        if self.fork:
            self.skill = a.skill
        elif self.source_session is None:
            self.skill = a.skill or stored_skill
        elif a.skill is None and self.never_started(store, row["agent_id"]):
            self.skill = stored_skill
        else:
            self.skill = a.skill

    def plan_flags(self) -> None:
        a = self.args
        given = self.pass_throughs()
        if self.source_session is None:
            # A launch re-run: the stored flags with the ones given now laid over them.
            self.run_pass = {
                **{k: v for k, v in self.stored_pass.items() if k in PASS_THROUGH_NAMES},
                **given,
            }
            stored_kind = self.stored.get("kind")
            self.kind_flag = a.kind or (stored_kind if isinstance(stored_kind, str) else None)
            stored_dirs = self.stored.get("plugin_dirs")
            dirs = [os.path.abspath(p) for p in (a.plugin_dir or [])]
            self.plugin_dirs = tuple(
                dirs
                or [
                    d
                    for d in (stored_dirs if isinstance(stored_dirs, list) else [])
                    if isinstance(d, str)
                ]  # pyright: ignore[reportUnknownVariableType]
            )
        else:
            self.run_pass = given
            self.kind_flag = a.kind  # never a stored kind: claude-session carries it
            self.plugin_dirs = tuple(os.path.abspath(p) for p in (a.plugin_dir or []))

    def plan_labels(self, store: Store, row: Mapping[str, Any]) -> None:
        extra = list(self.given_labels)
        if not self.fork:  # a fork gets job= only from an explicit --label job=
            job = os.environ.get(self.config.job_id_env)
            if job is not None and not os.environ.get("CSESSION_ID"):
                label = job_label(job)
                if label is None:
                    self.warnings.append(
                        f"{self.config.job_id_env} does not fit a label value: ignored"
                    )
                else:
                    extra.append(label)
        current = LabelSet() if self.fork else store.get_labels(row["agent_id"])
        try:
            change = current.add_run(extra)
        except GrammarError as error:
            raise CliError.usage(str(error)) from None
        if change.skipped:
            self.warnings.append("job= was not added: the agent already has 64 labels")
        self.new_labels = change.labels
        self.label_applied = change.applied
        if self.fork or self.source_session is None:
            self.start_labels = change.labels.texts()  # a new session gets every label
        else:
            self.start_labels = [str(pair) for pair in change.applied]

    # -- the spec -------------------------------------------------------------------------

    def worktree_mode(self) -> str:
        """The stored worktree mode of an agent without a session (``new``, a path or ``none``);
        ``none`` for every other run: those run in the cwd they recorded."""
        if self.fork or self.source_session is not None:
            return "none"
        mode = self.stored.get("worktree")
        return mode if isinstance(mode, str) and mode else "none"

    def run_cwd(self, row: Mapping[str, Any]) -> str | None:
        """The cwd of the run, for the pre-start checks (``session-dir``, a realpath or None)."""
        if self.fork:
            return self.fork_cwd(row)
        mode = self.worktree_mode()
        if mode == "new":
            recorded = row.get("cwd")  # known only once an earlier attempt got past step 5
            return os.path.realpath(recorded) if isinstance(recorded, str) else None
        if mode != "none":
            given = self.stored.get("cwd")
            return wtlib.resolve_in(mode, given if isinstance(given, str) else None)
        recorded = row.get("cwd")
        if isinstance(recorded, str):
            return recorded if recorded == third_party.SESSION_DIR else os.path.realpath(recorded)
        given = self.stored.get("cwd") if self.source_session is None else None
        if isinstance(given, str):
            return given if given == third_party.SESSION_DIR else os.path.realpath(given)
        return None

    def fork_cwd(self, row: Mapping[str, Any]) -> str | None:
        a = self.args
        if a.cwd is not None:
            return a.cwd if a.cwd == third_party.SESSION_DIR else os.path.realpath(a.cwd)
        recorded = row.get("cwd")
        if not isinstance(recorded, str):
            return None
        if recorded == third_party.SESSION_DIR:
            return third_party.SESSION_DIR
        # The source's own work directory is recorded by path when it was resolved: then the
        # fork gets its own (claude-session refuses a fork without --cwd, fork_cwd_unsupported).
        assert self.backend is not None and self.source_session is not None
        try:
            status = self.backend.status(self.source_session)
        except CliError:
            return os.path.realpath(recorded)
        work = os.path.join(status.csession_dir, "work")
        if os.path.realpath(recorded) == os.path.realpath(work):
            return third_party.SESSION_DIR
        return os.path.realpath(recorded)

    def build_spec(  # type: ignore[override]
        self, agent_id: str, labels: Any, cwd: str | None, copies: _Copies | None
    ) -> RunSpec:
        pt = self.run_pass
        resume = self.source_session
        turns = pt.get("max_turns")
        env_only = pt.get("env_allowlist_only")
        return RunSpec(
            agent_id=agent_id,
            mode=self.mode,
            resume=resume,
            fork=self.fork,
            kind=self.kind_flag,
            skill=self.skill,
            plugin_dirs=self.plugin_dirs,
            model=pt.get("model"),
            timeout=pt.get("timeout"),
            max_turns=int(turns) if turns is not None else None,
            permission_profile=pt.get("permission_profile"),
            sandbox=pt.get("sandbox"),
            setting_sources=pt.get("setting_sources"),
            reads_third_party=bool(pt.get("reads_third_party")),
            env_allow=tuple(pt.get("env_allow") or ()),
            env_allowlist_only=None if env_only is None else tuple(v for v in env_only if v),
            auto_exit=pt.get("auto_exit"),
            reset=tuple(self.args.reset or ()) if resume else (),
            allow_tainted=bool(self.args.allow_tainted) if resume else False,
            override_escalation=self.pass_override,
            repo=None if resume else self.args.repo,
            cwd=self.spec_cwd(cwd),
            labels=tuple(self.start_labels),
            prompt_file=None if copies is None else copies.prompt,
            input_files=() if copies is None else tuple(copies.inputs),
        )

    def spec_cwd(self, cwd: str | None) -> str | None:
        if self.source_session is not None and not self.fork:
            return None  # a plain resume runs in the stored cwd
        if cwd is None or cwd == third_party.SESSION_DIR or self.fork:
            return cwd  # a fork always passes its cwd: claude-session refuses a bad one
        return cwd if os.path.isdir(cwd) else None  # a sessionless relaunch falls back to --repo

    # -- prompt and input copies ------------------------------------------------------------

    def stored_copies(self, row: Mapping[str, Any]) -> _Copies:
        """The copies a launch kept for an agent without a session (never deleted by a failed
        resume). A missing one fails closed: the run would not be the one that was asked for."""
        agent_id = row["agent_id"]
        directory = os.path.join(self.state_dir, "prompts", agent_id)
        copies = _Copies(directory)
        if self.stored.get("has_prompt"):
            path = os.path.join(directory, "prompt")
            if not os.path.isfile(path):
                raise self.copies_missing(agent_id)
            copies.prompt = path
        names = self.stored.get("input_names")
        for index, name in enumerate(names if isinstance(names, list) else []):  # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]
            path = os.path.join(directory, "inputs", str(index), str(name))
            if not os.path.isfile(path):
                raise self.copies_missing(agent_id)
            copies.inputs.append(path)
            copies.input_names.append(str(name))
        return copies

    @staticmethod
    def copies_missing(agent_id: str) -> CliError:
        return CliError(
            "failed",
            "a stored prompt or input copy of the agent is missing",
            agent_id=agent_id,
            details={"hint": "pass --prompt-file and --input-file again"},
        )

    def make_run_copies(self, row: Mapping[str, Any], proposed: str) -> _Copies | None:
        a = self.args
        gave_prompt = a.prompt is not None or a.prompt_file is not None
        gave_inputs = bool(a.input_file)
        if self.fork:
            return make_copies(self.state_dir, proposed, a)
        agent_id = row["agent_id"]
        if self.source_session is not None:
            if not gave_prompt and not gave_inputs:
                return None
            self.new_copies = make_copies(
                self.state_dir, agent_id, a, subdir=f"resume-{secrets.token_hex(4)}"
            )
            return self.new_copies
        stored = self.stored_copies(row)
        if not gave_prompt and not gave_inputs:
            return stored
        fresh = make_copies(self.state_dir, agent_id, a, subdir=f"resume-{secrets.token_hex(4)}")
        self.new_copies = fresh
        return self.install_copies(stored, fresh, gave_prompt, gave_inputs)

    def install_copies(
        self, stored: _Copies, fresh: _Copies, gave_prompt: bool, gave_inputs: bool
    ) -> _Copies:
        """Newly given text replaces the stored copy (claude-session's carry rule). The new
        tree was built in ``fresh`` (same layout as the stored one); it is swapped in by renames
        under held signals and the stored settings (``has_prompt``, ``input_names``) follow at
        once, so the next resume sees what is on disk."""
        base = stored.directory
        result = _Copies(base, stored.prompt, list(stored.inputs), list(stored.input_names))
        inputs = os.path.join(base, "inputs")
        old_inputs = inputs + ".old"
        prompt = os.path.join(base, "prompt")
        old_prompt = prompt + ".old"
        swapped_inputs = swapped_prompt = False
        had_inputs = had_prompt = False
        previous = dict(self.stored)
        with _signals_held():
            try:
                # Inputs first (the step that can fail on its own), then the prompt; every
                # replaced copy is kept as ``*.old`` until the settings are stored, so one
                # failure anywhere puts the previous copies back and disk and settings agree.
                if gave_inputs:
                    shutil.rmtree(old_inputs, ignore_errors=True)
                    had_inputs = os.path.isdir(inputs)
                    if had_inputs:
                        os.rename(inputs, old_inputs)
                    swapped_inputs = True
                    os.rename(os.path.join(fresh.directory, "inputs"), inputs)
                    result.inputs = [
                        os.path.join(inputs, str(index), name)
                        for index, name in enumerate(fresh.input_names)
                    ]
                    result.input_names = list(fresh.input_names)
                if gave_prompt and fresh.prompt is not None:
                    if os.path.lexists(old_prompt):
                        os.remove(old_prompt)
                    had_prompt = os.path.lexists(prompt)
                    if had_prompt:
                        os.rename(prompt, old_prompt)
                    swapped_prompt = True
                    os.replace(fresh.prompt, prompt)
                    result.prompt = prompt
                self.store_copy_settings(result, gave_prompt, gave_inputs)
            except BaseException:
                self.stored = previous
                if swapped_prompt:
                    if os.path.lexists(prompt):
                        os.remove(prompt)
                    if had_prompt:
                        os.rename(old_prompt, prompt)
                if swapped_inputs:
                    shutil.rmtree(inputs, ignore_errors=True)
                    if had_inputs:
                        os.rename(old_inputs, inputs)
                raise
            if had_prompt:
                os.remove(old_prompt)
            shutil.rmtree(old_inputs, ignore_errors=True)
        shutil.rmtree(fresh.directory, ignore_errors=True)
        self.new_copies = None
        return result

    def store_copy_settings(self, copies: _Copies, gave_prompt: bool, gave_inputs: bool) -> None:
        assert self.store is not None
        agent_id = self.source_id
        settings = {**self.stored}
        if gave_prompt:
            settings["has_prompt"] = True
        if gave_inputs:
            settings["input_names"] = list(copies.input_names)
        self.stored = settings
        with self.store.lease(lifecycle.lease_key(agent_id), wait=lifecycle.LEASE_WAIT) as lease:
            self.store.update_agent(
                lease,
                agent_id,
                {"settings": {"settings": settings, "pass_throughs": self.stored_pass}},
                touch=False,
            )

    # -- checks -----------------------------------------------------------------------------

    def check_cwd_rule(self, classification: third_party.Classification, cwd: str | None) -> None:
        """Pre-start checks 2 to 4: the records of the cwd (one ``wtctl list``), the cwd rule
        of a third-party-reading run, the throwaway check of a trusted one."""
        reads = classification.reads_third_party
        self.enforce_rules(reads, cwd, self.records_for(cwd))
        if reads:
            self.check_session_result_file(cwd)

    def check_session_result_file(self, cwd: str | None) -> None:
        """The third-party path rule of ``--result-file`` (interface.md, Result file): not inside
        the run's cwd (a throwaway's root) nor inside the session's directory, ``outbox/``
        included. The session directory is known from ``claude-session status``; a session that
        cannot be read fails the command (the rule cannot be checked) rather than skipping it.
        A fork's own directory does not exist yet: the source's is checked, and the classification
        check covers the fork's."""
        if self.inv.result_file is None:
            return
        directories: list[str] = []
        if cwd is not None and cwd != third_party.SESSION_DIR:
            directories.append(cwd)
        if self.source_session is not None:
            assert self.backend is not None
            directories.append(self.backend.status(self.source_session).csession_dir)
        if self.result_file_inside(directories):
            raise CliError.usage(
                "--result-file lies inside a third-party-reading run's cwd or session directory"
            )

    def check_plan_cwd(self, classification: third_party.Classification, cwd: str | None) -> None:
        """A plain resume passes no ``--cwd``: the run starts where claude-session's plan says.
        When that differs from the recorded cwd the cwd rule applies to the plan's cwd too
        (the session's own ``work/`` directory is the one real path a third-party run may use)."""
        plan = classification.cwd
        if (
            self.fork
            or self.source_session is None
            or not classification.reads_third_party
            or plan is None
            or plan == third_party.SESSION_DIR
            or cwd is None
        ):
            return
        real = os.path.realpath(plan)
        if cwd != third_party.SESSION_DIR and real == os.path.realpath(cwd):
            return
        assert self.backend is not None
        work: str | None = None
        try:
            work = os.path.realpath(
                os.path.join(self.backend.status(self.source_session).csession_dir, "work")
            )
        except CliError:
            pass  # unknown: the rule below fails closed
        # The session's own `work/` lies in no record: no lookup for it.
        records = [] if work is not None and real == work else self.live_records()
        self.enforce_rules(True, real, records, session_work=work)

    # -- the whole resume ---------------------------------------------------------------------

    def run(self) -> Outcome:
        self.usage_checks()
        cs_argv = self.tool_argv("claude-session", self.config.claude_session_cmd)
        self_path = resolve_self()
        self.store = self.open_store()
        store = self.store
        row = self.resolve_source(store)
        self.agent_id = None
        self.backend = SessionBackend(
            cs_argv,
            actor=self.inv.actor,
            decision_ids=self.inv.decision_ids,
            gateway=self.gateway,
        )
        if row["status"] == "tearing_down" and not self.args.dry_run:
            raise _refusal("tearing_down", "the agent is being torn down", row["agent_id"])
        self.plan_run(store, row)
        if row["status"] == "tearing_down":
            return self.dry_run_refused(row, "tearing_down", cs_argv)
        if self.args.dry_run:
            return self.dry_run_resume(row, cs_argv)
        proposed = new_agent_id(store) if self.fork else row["agent_id"]
        return self.resume_run(row, proposed, cs_argv, self_path)

    def resume_run(
        self, row: dict[str, Any], proposed: str, cs_argv: Sequence[str], self_path: str
    ) -> Outcome:
        assert self.store is not None and self.backend is not None
        identity = self.store.holder_identity()
        # The lifecycle entry (step 3), then the copies (step 4): a failure of either restores
        # the pre-resume row or deletes the fork.
        if self.fork:
            self.resume_cwd = self.fork_cwd(row)
            run_id = self.insert_fork(row, proposed, cs_argv, self_path, identity)["agent_id"]
        else:
            self.enter(row, identity)
            run_id = row["agent_id"]
        self.copies = copies = self.make_run_copies(row, run_id)
        self.log("fork" if self.fork else "resume", details={"mode": self.mode})
        self.run_checks_and_start(row, run_id, copies, cs_argv, self_path)
        assert self.csession_id is not None
        return self.place_and_wait(self.start_result)  # type: ignore[attr-defined]

    # -- lifecycle entry -----------------------------------------------------------------------

    def enter(self, row: dict[str, Any], identity: Any) -> None:
        """Plain resume: the compare-and-set ``parked`` → ``launching`` under the agent's lease."""
        assert self.store is not None
        a = self.args
        agent_id = row["agent_id"]
        changes: dict[str, object] = {}
        if a.name is not None:
            changes["name"] = a.name
        if a.term_placement is not None:
            changes["term_placement"] = a.term_placement
        if a.promote is not None:
            changes["promote"] = a.promote
        run = row.get("run") if self.source_session is not None else None
        self.expect_run_after = run if isinstance(run, int) else None
        with _signals_held():
            with self.store.lease(
                lifecycle.lease_key(agent_id), wait=lifecycle.LEASE_WAIT
            ) as lease:
                try:
                    self.store.enter_resume(
                        lease,
                        agent_id,
                        launcher=identity,
                        expect_run_after=self.expect_run_after,
                        changes=changes,
                    )
                except CasLost as lost:
                    status = lost.status
                    if status == "tearing_down":
                        raise _refusal(
                            "tearing_down", "the agent is being torn down", agent_id
                        ) from None
                    if status == "torn_down":
                        raise CliError.of(
                            "not_found",
                            f"agent {agent_id} is torn down",
                            agent_id=agent_id,
                            details={"status": "torn_down"},
                        ) from None
                    raise _refusal("live", f"the agent is {status}", agent_id) from None
                except AgentMissing:
                    raise CliError.of("not_found", f"no agent {agent_id}") from None
            self.entered = True
            self.agent_id = agent_id
            self.agent_inserted = True

    def fork_settings(self, row: Mapping[str, Any], labels: LabelSet) -> dict[str, Any]:
        a = self.args
        names = input_names(a)
        settings = {
            **DEFAULT_SETTINGS,
            **{k: v for k, v in self.stored.items() if k in DEFAULT_SETTINGS},
        }
        settings.update(
            {
                "kind": a.kind if a.kind is not None else settings["kind"],
                "mode": self.mode,
                "placement": None,
                "term_placement": a.term_placement or row.get("term_placement"),
                "name": a.name or row["name"],
                "repo": row.get("repo"),
                "worktree": "none",
                "cwd": a.cwd,
                "labels": labels.texts(),
                "skill": a.skill,
                "has_prompt": a.prompt is not None or a.prompt_file is not None,
                "input_names": list(names),
                "plugin_dirs": list(self.plugin_dirs),
                "promote": a.promote if a.promote is not None else row.get("promote"),
                "ephemeral": False,
                "wait": bool(a.wait),
                "wait_timeout": a.wait_timeout,
            }
        )
        return settings

    def insert_fork(
        self,
        row: dict[str, Any],
        proposed: str,
        cs_argv: Sequence[str],
        self_path: str,
        identity: Any,
    ) -> dict[str, Any]:
        """Step 3 of a fork: the fork's record (status ``launching``, no session)."""
        assert self.store is not None
        a = self.args
        fields: dict[str, Any] = {
            "status": "launching",
            "forked_from": row["agent_id"],
            "launch_kind": a.kind if a.kind is not None else row.get("launch_kind"),
            "mode": self.mode,
            "placement": None,
            "term_placement": a.term_placement or row.get("term_placement"),
            "name": a.name or row["name"],
            "repo": row.get("repo"),
            "cwd": self.resume_cwd,
            "agent_spawn_path": self_path,
            "state_dir": self.state_dir,
            "config_path": self.inv.config_file,
            "claude_session_path": shlex.join(cs_argv),
            "actor": self.inv.actor,
            "decision_ids": list(self.decision_ids),
            "settings": {
                "settings": self.fork_settings(row, self.new_labels),
                "pass_throughs": self.pass_throughs(),
            },
            "launcher": identity,
            "starter": identity,
            "promote": a.promote if a.promote is not None else row.get("promote"),
            "ephemeral": False,
        }
        with _signals_held():
            created = self.store.insert_new_agent(fields, proposed=proposed, labels=self.new_labels)
            self.agent_id = created["agent_id"]
            self.agent_inserted = True
            self.fork_inserted = True
        return created

    # -- steps 4 and 7: classification, checks, start -------------------------------------------

    def run_checks_and_start(
        self,
        source_row: dict[str, Any],
        run_id: str,
        copies: _Copies | None,
        cs_argv: Sequence[str],
        self_path: str,
    ) -> None:
        assert self.store is not None and self.backend is not None
        backend = self.backend
        try:
            cwd = self.run_cwd(source_row)
            spec = self.build_spec(run_id, None, cwd, copies)
            classification = backend.classify(spec)
        except CliError as error:
            if (
                not self.fork
                and self.source_session is not None
                and error.tool == "claude-session"
                and error.exit_code == 10
            ):
                self.session_missing(run_id)
            raise
        if self.worktree_mode() != "none":
            if classification.reads_third_party:
                # A stored worktree: the cwd will be one, so the run is refused before any wtctl
                # call (`create` or `adopt`).
                self.enforce_rules(True, None, [])
            cwd = self.resume_worktree(source_row, run_id)  # step 5
        elif cwd is None and self.source_session is None:
            cwd = self.effective_cwd(classification)  # the plan's cwd, as launch step 4
        elif cwd is None:
            cwd = classification.cwd
        self.check_cwd_rule(classification, cwd)
        self.check_plan_cwd(classification, cwd)
        self.resume_cwd = cwd
        self.take_holds(run_id, cwd)  # step 6
        # The session (step 7).
        spec = self.build_spec(run_id, None, cwd, copies)
        outcome_file = new_outcome_file(self.state_dir, run_id)
        relay = relay_hook(self_path, self.state_dir, self.inv.config_file or "")
        # The human-only flags are audited before the call that takes them (as launch does), so
        # an interrupt during `start` still leaves a record; a failed start adds its own.
        self.audit_overrides("ok", run_id)
        self.start_attempted = True
        try:
            result = backend.start(spec, outcome_file=outcome_file, hook=relay)
        except StartError as error:
            self.audit_overrides("error" if error.exit_code != 20 else "refused", run_id)
            self.fail_start(error, {"agent_id": run_id})
            raise
        self.csession_id = result.csession_id
        self.start_result = result
        token = os.path.basename(outcome_file).removesuffix(".json")
        self.record_resumed(run_id, result, classification, token, cwd, cs_argv, self_path)
        remove_copies(os.path.join(self.state_dir, "prompts", run_id))
        row = self.store.get_agent(run_id)
        assert row is not None
        self.check_classification(row, result, classification, cwd)

    start_result: StartResult

    # -- step 5: the worktree of an agent without a session --------------------------------------

    def create_plan(self, agent_id: str) -> CreatePlan:
        """The stored launch settings as a ``wtctl create`` (re-validated: the store is not
        trusted to hold only what the launch checked)."""
        stored = self.stored

        def text(key: str) -> str | None:
            value = stored.get(key)
            return value if isinstance(value, str) else None

        branch, base, ref = text("branch"), text("base"), text("ref")
        name = text("worktree_name")
        return CreatePlan(
            agent_id=agent_id,
            repo=wtlib.check_repo(str(self.args.repo or "")),
            name=wtlib.planned_name(
                agent_id, wtlib.check_worktree_name(name) if name is not None else None
            ),
            branch=wtlib.check_branch(branch, "--branch") if branch is not None else None,
            base=wtlib.check_branch(base, "--base") if base is not None else None,
            ref=wtlib.check_sha(ref) if ref is not None else None,
            detach=bool(stored.get("detach")),
            no_bootstrap=bool(stored.get("no_bootstrap")),
            copy_env=bool(stored.get("copy_env")),
        )

    def worktree_cwd_arg(self) -> str | None:
        given = self.stored.get("cwd")
        return given if isinstance(given, str) else None

    def resume_worktree(self, row: dict[str, Any], agent_id: str) -> str:
        """Resume step 5: find or create the agent's worktree and return the run's cwd.

        ``wt_record_id`` set: ``wtctl show``; unset (``--worktree new``): the ``wtctl list --repo``
        lookup of the live record of the planned name that this agent holds. ``active`` skips
        ``create``, a tombstone is ``worktree_gone``, ``creating`` parks with the reconcile hint,
        any other live state parks ``worktree_state``; no record runs launch step 4 again with
        the stored settings.
        """
        backend = self.wtctl_backend(row.get("wtctl_path"))
        if backend is None:  # config says there is no wtctl on this machine now
            raise CliError.usage("this agent's worktree needs wtctl, but wtctl_cmd is null")
        mode = self.worktree_mode()
        record = self.known_record(row, backend, agent_id)
        if record is None:
            record = self.make_record(row, backend, agent_id)
        else:
            if record.state == "creating":
                self.park_launch_failed(
                    agent_id,
                    CliError(
                        "launch_failed",
                        "the worktree record is still being created",
                        details={"hint": "run `wtctl reconcile`"},
                    ),
                )
            if record.state != "active":
                self.park_launch_failed(
                    agent_id,
                    CliError(
                        "failed",
                        "the worktree record is not active",
                        details={
                            "cause": "worktree_state",
                            "retryable": False,
                            "hint": f"`wtctl show {record.id}`; a human repairs or forgets the "
                            "record",
                        },
                    ),
                )
            self.store_worktree(agent_id, record.id, path=record.path, owned=record.owned is True)
        root = record.path if mode == "new" else mode
        cwd = self.cwd_in_worktree(root)
        self.update_cwd(agent_id, cwd)
        return cwd

    def known_record(
        self, row: dict[str, Any], backend: WtctlBackend, agent_id: str
    ) -> WtRecord | None:
        stored_id = row.get("wt_record_id")
        if isinstance(stored_id, str):
            record = backend.show(stored_id)
            if record is None or record.tombstone:
                raise CliError(
                    "worktree_gone",
                    "the agent's worktree record no longer exists",
                    details={"hint": "tear the agent down, or launch again"},
                )
            return record
        if self.worktree_mode() != "new":
            return None  # `--worktree <path>`: adopt is idempotent
        plan = self.create_plan(agent_id)
        found = [
            r
            for r in backend.list_records(repo=plan.repo)
            if r.object_type == "worktree" and r.name == plan.name and agent_id in r.holders
        ]
        if len(found) > 1:  # names are unique per repo among live records: never expected
            raise CliError("failed", "several worktree records match the planned name")
        return found[0] if found else None

    def make_record(self, row: dict[str, Any], backend: WtctlBackend, agent_id: str) -> WtRecord:
        """Launch step 4 again: ``create`` (with the wtctl override when the human gave it) or
        ``adopt``. A veto or an escalation parks the agent; any other failure raises and the
        pre-resume row is restored (a busy wtctl keeps the agent's attention)."""
        mode = self.worktree_mode()
        try:
            if mode == "new":
                try:
                    record = self.run_create(
                        agent_id, self.create_plan(agent_id), override_escalation=self.wt_override
                    )
                except BaseException as error:
                    # audited once, with the real result (a refusal is exit 20)
                    refused = isinstance(error, CliError) and error.exit_code == 20
                    if self.wt_override:
                        self.audit_wt_override("refused" if refused else "error", agent_id)
                    raise
                if self.wt_override:
                    self.audit_wt_override("ok", agent_id)
                return record
            # the literal --repo given at launch (stored), never the one derived for the row
            given = self.stored.get("repo")
            record, _ = backend.adopt(
                mode, agent_id, repo=given if isinstance(given, str) else None
            )
            self.store_worktree(agent_id, record.id, path=record.path, owned=record.owned is True)
            return record
        except WtctlError as error:
            kind = wtlib.failure_kind(error)
            if kind in ("vetoed", "escalated"):
                fields = wtlib.park_fields(kind)
                self.park_with(agent_id, fields, error)
            raise

    def park_with(self, agent_id: str, fields: Mapping[str, str | None], error: CliError) -> None:
        """Park the agent (the second row of Resume failures) and raise ``error``."""
        assert self.store is not None
        lifecycle.park(
            self.store,
            agent_id,
            from_status="launching",
            park_reason=str(fields["park_reason"]),
            needs_attention=fields["needs_attention"],
            escalated_by=fields["escalated_by"],
        )
        self.settled = True
        self.log("status_change", level="warn", details={"to": "parked", **fields})
        error.agent_id = agent_id
        error.decision_ids = list(self.decision_ids)
        raise error

    def park_launch_failed(self, agent_id: str, error: CliError) -> None:
        self.park_with(agent_id, wtlib.park_fields("other"), error)

    def audit_wt_override(self, result: str, agent_id: str) -> None:
        self.audit_one("resume_escalation_override", result, agent_id)

    def audit_overrides(self, result: str, agent_id: str) -> None:
        """``resume_tainted`` and ``resume_escalation_override`` (audit.jsonl), once the
        human-only flag was actually passed to ``claude-session start``."""
        a = self.args
        if a.allow_tainted and self.source_session is not None:
            self.audit_one("resume_tainted", result, agent_id)
        if self.pass_override:
            self.audit_one("resume_escalation_override", result, agent_id)

    def audit_one(self, action: str, result: str, agent_id: str) -> None:
        try:
            append_audit(
                self.state_dir,
                actor=self.inv.actor,
                decision_ids=self.decision_ids,
                target=self.source_id if self.fork else agent_id,
                action=action,
                inputs={
                    "labels": self.new_labels.texts(),
                    "repo": self.args.repo,
                    "worktree": None,
                    "csession_id": self.source_session,
                    "merged_head": None,
                    "force": False,
                },
                result=result,
            )
        except (RecordsError, ValueError, OSError) as error:
            self.warnings.append(f"could not write the audit log ({type(error).__name__})")

    def record_resumed(
        self,
        agent_id: str,
        result: StartResult,
        classification: third_party.Classification,
        token: str,
        cwd: str | None,
        cs_argv: Sequence[str],
        self_path: str,
    ) -> None:
        """The run exists: record the session and run, the labels and the stored paths. From here
        on the agent is parked (never restored) by any failure."""
        assert self.store is not None
        store = self.store
        changes: dict[str, object] = {
            "csession_id": result.csession_id,
            "run": result.run,
            "run_kind": classification.run_kind,
            "mode": self.mode,
            "placement": None,
            "agent_spawn_path": self_path,
            "state_dir": self.state_dir,
            "config_path": self.inv.config_file,
            "claude_session_path": shlex.join(cs_argv),
            "actor": self.inv.actor,
            "decision_ids": list(self.decision_ids),
            "starter": store.holder_identity(),
            "supervisor": None,
            "exec_proc": None,
            "supervisor_lost": False,
        }
        if cwd is not None:
            changes["cwd"] = cwd
        if self.wt_argv is not None:
            changes["wtctl_path"] = shlex.join(self.wt_argv)
        if self.source_session is None and not self.fork:
            changes["settings"] = {
                "settings": self.resumed_settings(),
                "pass_throughs": self.run_pass,
            }
        with store.lease(lifecycle.lease_key(agent_id), wait=lifecycle.LEASE_WAIT) as lease:
            with store.transaction():
                store.update_agent(lease, agent_id, changes)
                store.put_run(
                    lease,
                    agent_id,
                    result.run,
                    {
                        "plan_run_kind": classification.run_kind,
                        "mode": self.mode,
                        "placement": None,
                        "cwd": cwd,
                        "reads_third_party": classification.reads_third_party,
                        "outcome_token": token,
                        "hooks": self.run_hooks,
                    },
                )
                self.store_run_labels(lease, agent_id)
        self.settled = True
        self.run_recorded = True

    def store_run_labels(self, lease: Any, agent_id: str) -> None:
        """Store the labels given to this resume (they were passed to `start`: nothing is left
        to mirror). A fork's labels were stored with its record."""
        assert self.store is not None
        if self.fork or not self.label_applied:
            return
        try:
            self.store.edit_labels(lease, agent_id, "add_run", list(self.label_applied))
        except GrammarError:
            self.log("warning", level="warn", message="labels not stored: over cap")
        else:
            self.store.mirror_done(lease, agent_id, [(pair, "add") for pair in self.label_applied])

    def resumed_settings(self) -> dict[str, Any]:
        a = self.args
        settings = {
            **DEFAULT_SETTINGS,
            **{k: v for k, v in self.stored.items() if k in DEFAULT_SETTINGS},
        }
        names = self.copies.input_names if self.copies is not None else settings["input_names"]
        settings.update(
            {
                "kind": self.kind_flag,
                "mode": self.mode,
                "skill": self.skill,
                "labels": self.new_labels.texts(),
                "has_prompt": bool(settings["has_prompt"])
                or a.prompt is not None
                or a.prompt_file is not None,
                "input_names": list(names),
                "plugin_dirs": list(self.plugin_dirs),
            }
        )
        return settings

    # -- failures ------------------------------------------------------------------------------

    def session_missing(self, agent_id: str) -> NoReturn:
        """claude-session reports the session missing: park ``session_missing`` and exit 10."""
        assert self.store is not None
        lifecycle.park(
            self.store,
            agent_id,
            park_reason="session_missing",
            needs_attention="session_missing",
            from_status="launching",
        )
        self.settled = True
        self.log("status_change", level="warn", details={"to": "parked", "cause": "session"})
        raise CliError.of(
            "not_found",
            "the agent's session is missing",
            agent_id=agent_id,
            details={"status": "session_missing"},
            decision_ids=list(self.decision_ids),
        )

    def record_unstarted(
        self, agent_id: str, session: str, run: int, end_reason: str | None
    ) -> Any:
        """Record a run that was allocated and never started (``not-started``)."""
        assert self.store is not None
        recorded = outcome_object(
            agent_id,
            RunOutcome(
                session, run, "launch", "not-started", end_reason, 12, "not-started", None, "status"
            ),  # fmt: skip
        )
        with self.store.lease(lifecycle.lease_key(agent_id), wait=lifecycle.LEASE_WAIT) as lease:
            with self.store.transaction():
                self.store.update_agent(
                    lease,
                    agent_id,
                    {
                        "csession_id": session,
                        "run": run,
                        "last_outcome": recorded,
                        "last_end_reason": end_reason,
                    },
                )
                self.store.put_run(
                    lease,
                    agent_id,
                    run,
                    {
                        "mode": self.mode,
                        "hooks": self.run_hooks,
                        "outcome": recorded,
                        "end_reason": end_reason,
                        "exit_code": 12,
                        "ended_at": self.gateway.wall_time(),
                    },
                )
                self.store_run_labels(lease, agent_id)
        self.settled = True
        self.run_recorded = True
        return recorded

    def fail_start(self, error: StartError, row: dict[str, Any]) -> None:  # type: ignore[override]
        """A failed ``start``: a run that was allocated is recorded and the agent parked
        ``launch_failed``; a missing session parks ``session_missing``; anything else leaves
        the undoing to ``cleanup_failed`` (restore the pre-resume row, or drop the fork)."""
        assert self.backend is not None and self.store is not None
        agent_id = row["agent_id"]
        if (
            error.csession_id is not None
            and error.run
            and (
                self.fork or self.source_session is None or error.csession_id == self.source_session
            )
        ):
            self.csession_id = error.csession_id
            reason = (error.lower_error or {}).get("end_reason")
            recorded = self.record_unstarted(
                agent_id, error.csession_id, error.run, reason if isinstance(reason, str) else None
            )
            remove_copies(os.path.join(self.state_dir, "prompts", agent_id))
            fields: dict[str, Any] = {
                "park_reason": "launch_failed",
                "needs_attention": "launch_failed",
            }
            lifecycle.park(
                self.store, agent_id, from_status="launching", outcome=recorded, **fields
            )
            self.log("status_change", level="warn", details={"to": "parked", **fields})
        elif error.exit_code == 10 and not self.fork and self.source_session is not None:
            self.session_missing(agent_id)
        # A start exit 13 (an escalation) is treated like exit 11: pre-launch hooks run in exec,
        # so claude-session's start never escalates and there is no override path to offer.
        error.agent_id = agent_id
        error.decision_ids = list(self.decision_ids)
        raise error

    def find_orphan(self, agent_id: str) -> tuple[str, int | None] | None:
        """The session (and its pending run) a ``start`` of unknown fate left behind."""
        assert self.backend is not None
        backend = self.backend
        session = self.source_session if not self.fork and self.source_session else None
        if session is None:
            try:
                found = backend.list_by_label(f"agent={agent_id}")
            except CliError:
                return None
            if len(found) != 1:
                return None
            session = found[0]
        try:
            status = backend.status(session)
        except CliError:
            if self.source_session == session and not self.fork:
                self.unseen_session = session
                return None
            return (session, None)
        if (
            status.pending
            and status.run.run == status.current_run
            and status.current_run > (self.expect_run_after or 0 if not self.fork else 0)
        ):
            return session, status.current_run
        return None if self.source_session == session and not self.fork else (session, None)

    def settle_orphan(self, agent_id: str, *, interrupted: bool) -> bool:
        """Cancel an orphaned pending run and park the agent; False when there is none."""
        assert self.backend is not None and self.store is not None
        orphan = self.find_orphan(agent_id)
        if orphan is None:
            return False
        session, run = orphan
        ended = False
        try:
            ended = self.backend.stop(session).get("ended") is True
        except CliError:
            pass
        self.csession_id = session
        recorded = None
        remove_copies(os.path.join(self.state_dir, "prompts", agent_id))  # a session exists now
        if run is not None:
            recorded = self.record_unstarted(agent_id, session, run, "canceled" if ended else None)
        else:
            with self.store.lease(lifecycle.lease_key(agent_id), wait=5.0) as lease:
                self.store.update_agent(lease, agent_id, {"csession_id": session})
            self.settled = True
        if interrupted:
            lifecycle.park(
                self.store,
                agent_id,
                park_reason="killed",
                needs_attention=None,
                from_status="launching",
                outcome=recorded,
            )
        else:
            lifecycle.park(
                self.store,
                agent_id,
                park_reason="launch_failed",
                needs_attention="launch_failed",
                from_status="launching",
                outcome=recorded,
            )
        return True

    def restore(self, agent_id: str) -> None:
        """Put the pre-resume row back (a plain resume that failed before its run existed)."""
        assert self.store is not None
        try:
            with self.store.lease(
                lifecycle.lease_key(agent_id), wait=lifecycle.LEASE_WAIT
            ) as lease:
                self.store.restore_pre_resume(lease, agent_id)
        except CasLost:
            pass  # another process already settled the agent
        self.settled = True

    def release_fork_holds(self, agent_id: str) -> CliError | None:
        """``wtctl forget --holder <fork id>`` for every record the failed fork holds: the stored
        rows and, since a crash between ``adopt`` and the row would hide one, the worktree
        records of the fork's cwd that list the fork as a holder. ``not_a_holder`` (exit 0) and
        a tombstone or unknown record (exit 10) count as released. Returns the first other
        failure (every record is still tried); the caller then keeps the fork."""
        assert self.store is not None
        record_ids = [hold["record_id"] for hold in self.store.get_holds(agent_id)]
        if not record_ids and not self.holds_started:
            return None
        failure: CliError | None = None
        try:
            backend = self.wtctl_backend(self.stored_wtctl)
        except CliError as error:
            backend, failure = None, error
        cwd = self.resume_cwd
        if backend is not None and self.holds_started and self.needs_records(cwd):
            assert cwd is not None
            try:
                for record in wtlib.worktrees_containing(backend.list_records(), cwd):
                    if agent_id in record.holders and record.id not in record_ids:
                        record_ids.append(record.id)
            except CliError as error:  # the holds are unknown: the fork must stay
                failure = failure or error
        for record_id in record_ids:
            try:
                if backend is None:
                    raise CliError("tool_missing", "wtctl is not available", tool="wtctl")
                backend.forget(record_id, agent_id)
            except CliError as error:
                if error.exit_code == 10:
                    continue
                self.log(
                    "warning",
                    level="warn",
                    message=f"could not release the fork's hold on {record_id}",
                )
                failure = failure or error
        return failure

    def drop_fork(self, agent_id: str) -> None:
        """A fork that failed before its session existed: the holds it took are released, its
        record and prompt copies go, its id stays reserved. When a hold could not be released
        the fork row (the only handle on that hold) stays, parked ``launch_failed``, and the
        release failure is what the command reports."""
        assert self.store is not None
        failure = self.release_fork_holds(agent_id)
        if failure is not None:
            fields = wtlib.park_fields("other")
            lifecycle.park(
                self.store,
                agent_id,
                from_status="launching",
                park_reason=str(fields["park_reason"]),
                needs_attention=fields["needs_attention"],
            )
            self.log("status_change", level="warn", details={"to": "parked", **fields})
            self.kept_fork_failure = failure
            self.settled = True
            return
        with self.store.lease(lifecycle.lease_key(agent_id), wait=lifecycle.LEASE_WAIT) as lease:
            self.store.delete_agent(lease, agent_id)
        remove_copies(os.path.join(self.state_dir, "prompts", agent_id))
        self.agent_inserted = False
        self.fork_inserted = False
        self.settled = True

    def undo_entry(self, *, interrupted: bool, search_orphan: bool = True) -> None:
        """Whatever the failure, an entry whose run does not exist is undone: the orphaned
        pending run of a ``start`` of unknown fate is canceled and recorded, else the
        pre-resume row is restored (plain) or the fork is deleted."""
        if self.settled or self.store is None or self.agent_id is None:
            return
        agent_id = self.agent_id
        if not (self.entered or self.fork_inserted):
            return
        try:
            # A definite refusal of `start` (search_orphan false) allocated nothing: looking for
            # a pending run could stop another caller's.
            if self.start_attempted and search_orphan:
                if self.settle_orphan(agent_id, interrupted=interrupted):
                    return
                if self.unseen_session is not None:
                    try:  # status failed: cancel whatever the start may have left pending
                        self.backend.stop(self.unseen_session)  # type: ignore[union-attr]
                    except CliError:
                        pass
            if self.fork_inserted:
                self.drop_fork(agent_id)
            else:
                self.restore(agent_id)
        except (StoreError, CliError):
            # TODO(#31): the liveness check restores a launching agent whose launcher died.
            self.log("warning", level="warn", message="could not undo the resume's entry")

    def cleanup_failed(self, failure: CliError) -> CliError:
        definite = isinstance(failure, StartError) and not failure.maybe_allocated
        self.undo_entry(interrupted=False, search_orphan=not definite)
        if self.kept_fork_failure is not None:  # the fork stays: report why its hold stayed
            failure = self.kept_fork_failure
            failure.agent_id = self.agent_id
            failure.decision_ids = list(self.decision_ids)
        elif self.fork and not self.agent_inserted:
            failure.agent_id = None  # no fork remains
        # Prompt and input copies of a plain or sessionless resume stay (see the module doc).
        return failure

    def interrupted(self, signum: int) -> Outcome:
        self.sig.raising = False
        code = EXIT_SIGTERM if signum == 15 else EXIT_SIGINT
        if self.detached is not None:
            self.detached.abort()
        if self.run_recorded and self.agent_id is not None and self.store is not None:
            try:
                self.cancel_pending(self.agent_id)
            except (CliError, StoreError):
                pass
            remove_copies(os.path.join(self.state_dir, "prompts", self.agent_id))
        else:
            self.undo_entry(interrupted=True)
        self.log(
            "interrupted",
            level="warn",
            signal=SIGNAL_NAME.get(signum),
            details={"cause": "interrupted"},
        )
        error = CliError(
            "launch_failed",
            f"resume interrupted by {SIGNAL_NAME.get(signum, 'a signal')}",
            exit_code=code,
            agent_id=self.agent_id if self.agent_inserted else None,
            details={"cause": "interrupted"},
            decision_ids=list(self.decision_ids),
        )
        return error_outcome(error, self.inv.decision_ids)

    def store_failed(self, error: StoreError) -> CliError:
        agent_id = self.agent_id if self.agent_inserted else None
        if agent_id is not None and not self.supervised and self.store is not None:
            if self.run_recorded:
                try:
                    self.cancel_pending(agent_id)
                except (CliError, StoreError):
                    pass  # TODO(#31): the liveness check parks an agent that stayed launching.
            else:
                self.undo_entry(interrupted=False)
        self.log("warning", level="warn", message=f"state error ({type(error).__name__})")
        return CliError(
            "launch_failed" if agent_id is not None and self.run_recorded else "failed",
            f"the state database could not be used ({type(error).__name__})",
            agent_id=agent_id if self.agent_inserted else None,
            decision_ids=list(self.decision_ids),
        )

    # -- dry run --------------------------------------------------------------------------------

    def dry_run_refused(self, row: dict[str, Any], reason: str, cs_argv: Sequence[str]) -> Outcome:
        return self.dry_plan(row, cs_argv, {}, False, reason, None)

    def dry_run_resume(self, row: dict[str, Any], cs_argv: Sequence[str]) -> Outcome:
        """Steps 1 to 4, read-only: no lease, no kept copy, no record."""
        assert self.backend is not None
        scratch = tempfile.mkdtemp(prefix="agent-spawn-dry-run-")
        try:
            run_id = "ag-aaaaaaaa" if self.fork else row["agent_id"]
            copies = None
            if (
                self.args.prompt is not None
                or self.args.prompt_file is not None
                or self.args.input_file
            ):
                copies = make_copies(scratch, run_id, self.args)
            elif self.source_session is None:
                copies = self.stored_copies(row)
            cwd = self.run_cwd(row)
            spec = self.build_spec(run_id, None, cwd, copies)
            try:
                classification, plan = self.backend.classify_with_plan(spec, refuse=False)
            except CliError as error:
                if error.tool == "claude-session" and error.exit_code == 10:
                    raise CliError.of(
                        "not_found",
                        "the agent's session is missing",
                        agent_id=row["agent_id"],
                        details={"status": "session_missing"},
                        decision_ids=list(self.decision_ids),
                    ) from None
                raise
            if cwd is None:
                cwd = (
                    self.effective_cwd(classification)
                    if self.source_session is None
                    else classification.cwd
                )
            refuse = classification.would_refuse
            if refuse is not None and refuse not in third_party.WOULD_REFUSE_REASONS:
                refuse = "unknown"
            if refuse is None and not self.fork and row["status"] in ("launching", "live"):
                refuse = "live"
            if refuse is None:
                refuse = self.dry_run_rules(classification, cwd)
            names = [] if copies is None else copies.input_names
            argv = [
                *cs_argv,
                "start",
                *launch_flags(spec),
                f"--hook={relay_hook(resolve_self(), self.state_dir, self.inv.config_file or '')}",
                f"--outcome-file={os.path.join(self.state_dir, 'outcomes', run_id, 'TOKEN.json')}",
                *self.backend.audit_flags(),
                "--json",
            ]
            return self.dry_plan(
                row, cs_argv, plan, classification.reads_third_party, refuse, argv, names
            )
        finally:
            shutil.rmtree(scratch, ignore_errors=True)

    def dry_run_rules(
        self, classification: third_party.Classification, cwd: str | None
    ) -> str | None:
        """Pre-start checks 2 to 4, read-only. A stored worktree is always refused for a
        third-party-reading run; its other checks need the cwd of step 5 (not planned here)."""
        reads = classification.reads_third_party
        if self.worktree_mode() != "none":
            return "third_party_cwd" if reads else None
        if not third_party.cwd_rule_allows(reads, cwd, records=self.records_for(cwd)):
            return "third_party_cwd" if reads else "third_party_run"
        if reads:
            self.check_session_result_file(cwd)
        return None

    def dry_plan(
        self,
        row: Mapping[str, Any],
        cs_argv: Sequence[str],
        session_plan: dict[str, Any],
        reads: bool,
        would_refuse: str | None,
        argv: list[str] | None,
        names: Sequence[str] = (),
    ) -> Outcome:
        a = self.args
        settings = {
            **DEFAULT_SETTINGS,
            **{k: v for k, v in self.stored.items() if k in DEFAULT_SETTINGS},
        }
        settings.update(
            {
                "kind": self.kind_flag,
                "mode": self.mode,
                "placement": None,
                "term_placement": a.term_placement or row.get("term_placement"),
                "name": a.name or row.get("name"),
                "repo": row.get("repo"),
                "cwd": a.cwd,
                "labels": self.new_labels.texts(),
                "skill": self.skill,
                "has_prompt": bool(settings["has_prompt"] and not self.fork)
                or a.prompt is not None
                or a.prompt_file is not None,
                "input_names": list(names)
                or (list(settings["input_names"]) if not self.fork else []),
                "plugin_dirs": list(self.plugin_dirs),
                "promote": a.promote if a.promote is not None else row.get("promote"),
                "ephemeral": False if self.fork else bool(row.get("ephemeral")),
                "wait": bool(a.wait),
                "wait_timeout": a.wait_timeout,
            }
        )
        plan = {
            "dry_run": True,
            "resume": True,
            "agent_settings": settings,
            "reads_third_party": reads,
            "session_plan": session_plan,
            "would_refuse": would_refuse,
            "tools": {"claude-session": list(cs_argv), "termctl": None, "wtctl": self.wt_argv},
            "worktree_plan": None,
            "term_plan": None,
            "session_start_argv": argv or [],
            "hooks": [
                {"hook": h["hook"], "cmd": h["cmd"], "source": "flag"} for h in self.run_hooks
            ],
            "warnings": self.warnings,
        }
        return Outcome(20 if would_refuse else 0, plan, PayloadKind.DRY_RUN)


def resume(inv: Any, *, restore_handlers: bool = True) -> Outcome:
    """Run ``resume``; the handler of ``commands.launching``."""
    return run_command(inv, Resume, restore_handlers=restore_handlers)
