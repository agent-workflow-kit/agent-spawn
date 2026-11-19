"""The registry of live scenarios: what each one needs, costs and checks.

Pure data, no I/O: ``make live-list`` prints it (``python -m tests.live``) without starting
anything, and ``tests/test_live_harness.py`` checks that it agrees with the tests under
``tests/live/`` (every test names a registered scenario, every scenario has a test).

``sessions`` is the number of real Claude Code processes the scenario is expected to start
(each one is counted by the ``claude`` wrapper the harness puts first on ``PATH``, and the
suite fails before starting more than ``MAX_LIVE_SESSIONS``); ``turns`` is the number of model
turns those sessions are expected to take (one tiny prompt each). Unbilled scenarios start no
Claude Code at all: they use real tmux, termctl, wtctl and claude-session, and a stub
``claude`` where a program named ``claude`` must exist.
"""

from __future__ import annotations

from dataclasses import dataclass, field

#: Waves, in the order they are meant to be run (``LIVE_SCENARIOS=wave:<name>``).
WAVES = {
    "tools": "unbilled: real tools, no Claude Code",
    "billed-1": "billed: basic runs (headless, window, wait, flags)",
    "billed-2": "billed: continuation (resume, fork, teardown, promotion)",
    "billed-3": "billed: kills, signals, minimal environment, jobs",
}


@dataclass(frozen=True)
class Scenario:
    name: str
    wave: str
    summary: str
    tools: tuple[str, ...]
    sessions: int = 0
    turns: int = 0
    #: References the scenario confirms: issue comments (``#36 <PR>``), design.md spikes, FIDELITY.md.
    covers: tuple[str, ...] = ()
    #: Set when the scenario cannot run on Linux in this suite: the reason, recorded in the report.
    skip: str | None = None
    #: Tools that, when missing, skip the scenario instead of failing it.
    optional: tuple[str, ...] = ()
    #: Wall-clock bound of one test of the scenario, in seconds (default: the suite's).
    timeout: int | None = None
    #: Minimum claude-session version of the scenario (compared when the version is known).
    min_claude_session: str | None = None
    extra: dict[str, str] = field(default_factory=dict)

    @property
    def billed(self) -> bool:
        return self.sessions > 0


def _s(name: str, wave: str, summary: str, tools: str, **kw: object) -> Scenario:
    return Scenario(name, wave, summary, tuple(tools.split()), **kw)  # type: ignore[arg-type]


_ALL = [
    # ---- unbilled: tools -----------------------------------------------------------------
    _s(
        "tools-versions",
        "tools",
        "record every tool's version, check the minimums the design was verified with",
        "claude claude-session termctl wtctl tmux git jobsctl",
        optional=("jobsctl",),
        covers=("#36 plan: version/minimum checks",),
    ),
    _s(
        "fidelity-argv",
        "tools",
        "the fakes' argv grammar rows and tmux texts against the real programs (existing suite)",
        "claude-session termctl tmux",
        covers=("FIDELITY.md verified rows",),
        optional=("claude-session", "termctl"),
    ),
    _s(
        "tmux-real-server",
        "tools",
        "stale and missing socket texts, list-commands, -L with -S, unknown command text",
        "tmux",
        covers=("#36 #8 tmux", "FIDELITY.md tmux"),
    ),
    _s(
        "termctl-open-pid",
        "tools",
        "open reports the pane pid (A8) incl. a wrapper shell that execs; dry run vs open; "
        "bare open; label cap; argv cap",
        "termctl tmux",
        covers=("design.md A8", "#36 #26", "#36 #27", "#36 #8 termctl", "promotion check E"),
    ),
    _s(
        "termctl-sighup-leader",
        "tools",
        "closing a pane delivers SIGHUP to the pane's process only (spike B1, B2)",
        "termctl tmux",
        covers=("design.md B1-B3", "#36 #8 termctl", "#36 #27"),
    ),
    _s(
        "termctl-keep-on-exit",
        "tools",
        "keep-on-exit failed/always/never, exit_status, exit_signal, list --dead, close",
        "termctl tmux",
        covers=("design.md B6", "#36 #27 keep-on-exit", "promotion check F, G", "#36 #8 termctl"),
    ),
    _s(
        "termctl-default-server",
        "tools",
        "default server selection order, empty list, socket_path, exact label match, close",
        "termctl tmux",
        covers=("#36 #79", "FIDELITY.md termctl list --dead"),
    ),
    _s(
        "termctl-focus",
        "tools",
        "focus without a client, attach answer, unknown id, locked error",
        "termctl tmux",
        covers=("#36 #28", "#36 #8 termctl focus"),
    ),
    _s(
        "wtctl-create-list-show",
        "tools",
        "create, list, show: names, branches, paths, shapes, --flag=value, errors",
        "wtctl git",
        covers=("#36 #20", "#36 #8 wtctl", "design.md wtctl behavior"),
    ),
    _s(
        "wtctl-adopt-forget-holders",
        "tools",
        "adopt and forget by holder: existing, sorted holders, tombstones, actors",
        "wtctl git",
        covers=("#36 #20", "#36 #22", "#36 #21"),
    ),
    _s(
        "wtctl-collisions-reuse",
        "tools",
        "explicit --name collisions per case, reuse by the same holder",
        "wtctl git",
        covers=("#36 #20", "design.md wtctl behavior"),
    ),
    _s(
        "wtctl-remove-status",
        "tools",
        "remove and status: held, tombstone, --flag=value, human-only order, force_overrides",
        "wtctl git",
        covers=("#36 #23", "#36 #22"),
    ),
    _s(
        "wtctl-baseline-check",
        "tools",
        "baseline check by record id, partial/missing records, list realpaths",
        "wtctl git",
        covers=("#36 #25", "#36 #21"),
    ),
    _s(
        "cs-pending-run",
        "tools",
        "claude-session start without adoption: pending status, list --label, stop, forget, "
        "expiry never_adopted, verify-secret, label exit 10, reconcile --dry-run",
        "claude-session",
        covers=("#36 #9", "#36 #14", "#36 #15", "#36 #75 1", "#36 #81", "promotion check B"),
    ),
    _s(
        "cs-start-killed",
        "tools",
        "SIGKILL of start at many offsets never leaves an unlabeled session",
        "claude-session",
        covers=("#36 #75 2",),
    ),
    _s(
        "cs-dry-run-shapes",
        "tools",
        "dry-run plans: implicit-done flags and origins, would_refuse, override-escalation "
        "exit 2, abbreviations, values starting with '-'",
        "claude-session",
        min_claude_session="0.43.0",
        covers=("#36 #9", "#36 #88", "#36 #78", "#36 #8 claude-session"),
    ),
    _s(
        "cs-stop-grace",
        "tools",
        "stop default grace against claude-session's kill_grace with a SIGTERM-ignoring stub",
        "claude-session",
        covers=("#36 #19 DEFAULT_GRACE", "#36 #9 stop", "FIDELITY.md exec signals"),
    ),
    _s(
        "promotion-subchecks",
        "tools",
        "the seven promotion sub-checks of the issue body, against real claude-session, "
        "termctl and tmux with a stub claude",
        "claude-session termctl tmux",
        covers=("#36 #29", "design.md D30 D47", "#36 #73"),
    ),
    _s(
        "as-stub-lifecycle",
        "tools",
        "agent-spawn against the real claude-session, wtctl, termctl and tmux with a stub "
        "claude: launch, wait, show, list, ack, resume, fork, teardown with a real worktree",
        "claude-session wtctl termctl tmux git",
        covers=("#36 #20-#23", "#36 #25", "#36 #26", "#36 #15", "#36 #12"),
    ),
    _s(
        "as-cron-fake-crontab",
        "tools",
        "cron install and uninstall against a fake crontab that is first on PATH",
        "tmux",
        covers=("#36 #34", "FIDELITY.md crontab"),
    ),
    _s(
        "as-reconcile-reset-lease",
        "tools",
        "reconcile --reset-lease on a real corrupted state.db row",
        "claude-session",
        covers=("#36 #80", "design.md D52"),
    ),
    _s(
        "as-browse-pilot",
        "tools",
        "browse (Textual pilot) over a real agent-spawn list payload",
        "claude-session",
        covers=("#36 #38 browse",),
    ),
    _s(
        "as-dead-pane-reconcile",
        "tools",
        "teardown and reconcile close a dead pane on a real tmux server",
        "claude-session termctl tmux",
        covers=("#36 #79", "design.md D51"),
    ),
    # ---- billed --------------------------------------------------------------------------
    _s(
        "headless-launch-wait",
        "billed-1",
        "headless launch --wait: result object, relay states, status fields, outcome file",
        "claude claude-session",
        sessions=1,
        turns=1,
        covers=("#36 #9", "#36 #14", "#36 #15", "#36 #8 claude-session", "design.md D1 D43"),
    ),
    _s(
        "interactive-window-close",
        "billed-1",
        "interactive launch in a real window; closing the window parks killed",
        "claude claude-session termctl tmux",
        sessions=1,
        turns=1,
        covers=("#36 #27", "#36 #26", "design.md D17 B9", "#36 issue: interactive launch"),
    ),
    _s(
        "report-resume-chain",
        "billed-1",
        "--no-implicit-done run ends unreported; the automatic report resume carries the flag",
        "claude claude-session",
        sessions=2,
        turns=2,
        min_claude_session="0.43.0",
        covers=("#36 #88", "#36 #9", "interface.md report resume"),
    ),
    _s(
        "resume-continue",
        "billed-2",
        "headless resume continues the Claude session; list --label shows one session",
        "claude claude-session",
        sessions=2,
        turns=2,
        covers=("#36 #75 5", "#36 #9 resume", "issue: resume"),
    ),
    _s(
        "fork-inplace-holder",
        "billed-2",
        "fork in place holds the worktree record; the original's teardown is held",
        "claude claude-session wtctl git",
        sessions=2,
        turns=2,
        covers=("#36 #21", "#36 #22", "#36 #23", "#36 #75 5", "issue: in-place fork holder"),
    ),
    _s(
        "fork-throwaway",
        "billed-2",
        "fork into a throwaway with a third-party-reading run",
        "claude claude-session wtctl git",
        sessions=2,
        turns=2,
        covers=("#36 #21", "issue: fork into a throwaway", "design.md D49"),
    ),
    _s(
        "teardown-wtctl-real",
        "billed-2",
        "teardown of a launched --worktree new agent with the real wtctl",
        "claude claude-session wtctl git",
        sessions=1,
        turns=1,
        covers=("#36 #23", "#36 #20", "issue: teardown with real wtctl"),
    ),
    _s(
        "promotion-escalation",
        "billed-2",
        "headless run reports needs-input, is promoted into a window and continues (spike D5)",
        "claude claude-session termctl tmux",
        sessions=2,
        turns=2,
        covers=("design.md D5", "#36 #29", "#36 #73", "issue: headless escalation"),
    ),
    _s(
        "resumable-after-kill-pane",
        "billed-3",
        "claude exits 0 on kill-pane and the session resumes (spike D1)",
        "claude claude-session termctl tmux",
        sessions=2,
        turns=2,
        covers=("design.md D1", "#36 issue: spike D"),
    ),
    _s(
        "resumable-after-sigterm",
        "billed-3",
        "SIGTERM of an idle and of a mid-turn claude: exit 143, both resume (spikes D2, D3)",
        "claude claude-session termctl tmux",
        sessions=4,
        turns=4,
        covers=("design.md D2 D3", "#36 issue: spike D"),
    ),
    _s(
        "minimal-environment",
        "billed-3",
        "claude authenticates with only PATH and the private HOME (spike D4)",
        "claude",
        sessions=1,
        turns=1,
        covers=("design.md D4",),
    ),
    _s(
        "jobsctl-cancel-force",
        "billed-3",
        "jobsctl cancel --force of a job running launch --wait",
        "claude claude-session jobsctl",
        sessions=1,
        turns=2,
        optional=("jobsctl",),
        covers=("issue: jobsctl cancel --force", "design.md D20 D44"),
    ),
    _s(
        "liveness-kills",
        "billed-3",
        "SIGKILL of the supervisor only and of exec only against a real run in a pane",
        "claude claude-session termctl tmux",
        sessions=2,
        turns=2,
        covers=("#36 #75 3", "#36 #75 4", "design.md B9", "FIDELITY.md wrapper_alive"),
    ),
    # ---- recorded but not automated --------------------------------------------------------
    _s(
        "placement-here-terminal",
        "billed-3",
        "--placement here: termios, SIGWINCH, job control, hangup on a real terminal",
        "claude claude-session",
        skip="needs a human at a real terminal (job control, resize, hangup); not automatable",
        covers=("#36 #74",),
    ),
    _s(
        "upgrade-live-supervisor",
        "billed-3",
        "a newer agent-spawn migrating state.db under a live supervisor; older relay events",
        "claude claude-session",
        skip="needs two agent-spawn builds installed side by side; run by hand, see "
        "docs/live-testing.md",
        covers=("#36 #35",),
    ),
    _s(
        "real-crontab",
        "tools",
        "cron install against the user's real crontab",
        "tmux",
        skip="never touches the real crontab: only the fake-crontab scenario runs",
        covers=("FIDELITY.md crontab",),
    ),
    _s(
        "macos",
        "tools",
        "macOS handoff peer check, kqueue waits",
        "tmux",
        skip="macOS is out of scope for the Linux live suite (issue #37)",
        covers=("#37",),
    ),
]

SCENARIOS: dict[str, Scenario] = {s.name: s for s in _ALL}
if len(SCENARIOS) != len(_ALL):  # pragma: no cover - a registry typo, caught on import
    raise RuntimeError("duplicate live scenario name")


def expected_sessions(names: list[str] | None = None) -> int:
    """Total billed sessions of the named scenarios (default: every runnable one)."""
    chosen = [SCENARIOS[n] for n in names] if names else list(SCENARIOS.values())
    return sum(s.sessions for s in chosen if s.skip is None)


def expected_turns(names: list[str] | None = None) -> int:
    chosen = [SCENARIOS[n] for n in names] if names else list(SCENARIOS.values())
    return sum(s.turns for s in chosen if s.skip is None)


def select(tokens: list[str]) -> list[str]:
    """Scenario names matching ``LIVE_SCENARIOS`` tokens: a name, ``wave:<name>``, ``billed``,
    ``unbilled`` or ``all``. An unknown token is an error (a typo must not select nothing)."""
    chosen: list[str] = []
    for token in tokens:
        if token == "all":
            found = list(SCENARIOS)
        elif token == "billed":
            found = [n for n, s in SCENARIOS.items() if s.billed]
        elif token == "unbilled":
            found = [n for n, s in SCENARIOS.items() if not s.billed]
        elif token.startswith("wave:"):
            wave = token[5:]
            if wave not in WAVES:
                raise ValueError(f"unknown wave {wave!r}; waves: {', '.join(WAVES)}")
            found = [n for n, s in SCENARIOS.items() if s.wave == wave]
        elif token in SCENARIOS:
            found = [token]
        else:
            raise ValueError(f"unknown scenario {token!r} (see `make live-list`)")
        chosen.extend(n for n in found if n not in chosen)
    return chosen


def parse_selection(text: str | None) -> list[str] | None:
    """``LIVE_SCENARIOS`` (comma or space separated) as scenario names; ``None`` when unset."""
    if text is None or not text.strip():
        return None
    return select([t for t in text.replace(",", " ").split() if t])


def format_listing(max_sessions: int, names: list[str] | None = None) -> str:
    """The text of ``make live-list``: every scenario with its expected billed sessions/turns."""
    chosen = [SCENARIOS[n] for n in names] if names else list(SCENARIOS.values())
    lines = [f"{'scenario':<28} {'wave':<9} {'sessions':>8} {'turns':>5}  tools / notes"]
    for s in chosen:
        note = f"SKIP: {s.skip}" if s.skip else " ".join(s.tools)
        if s.optional and s.skip is None:
            note += f" (optional: {' '.join(s.optional)})"
        lines.append(f"{s.name:<28} {s.wave:<9} {s.sessions:>8} {s.turns:>5}  {note}")
    runnable = [s.name for s in chosen if s.skip is None]
    sessions = expected_sessions(runnable)
    lines.append("")
    lines.append(
        f"{len(runnable)} runnable scenarios; expected billed sessions: {sessions} "
        f"(MAX_LIVE_SESSIONS={max_sessions}), turns: {expected_turns(runnable)}"
    )
    if sessions > max_sessions:
        lines.append("WARNING: the selection expects more sessions than MAX_LIVE_SESSIONS allows")
    return "\n".join(lines)
