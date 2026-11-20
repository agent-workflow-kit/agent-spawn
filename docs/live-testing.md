# Live testing

The `live` suite (`tests/live/`) runs agent-spawn against the real tools it depends on: Claude
Code (`claude --model haiku`), tmux, claude-session, termctl, wtctl, git and, when installed,
jobsctl. It re-checks every relied-on third-party behavior that the fakes in `tests/fakebin/`
only assume ([FIDELITY.md](../tests/fakebin/FIDELITY.md)) and that the design records as
"assumed, check live" ([design.md](design.md), Live verification). It is **opt-in**, never part
of `make verify` and never run in CI. The suite has passed on Linux against the real tools (see [Results](#results)); what it does not cover is listed under [Not covered](#not-covered) and tracked in the issues named there.

Some scenarios cost API usage (about seventy haiku turns in total, see [Cost](#cost)); the rest
start no Claude Code at all.

## Running

```sh
make live-list                       # what would run, with expected billed sessions and turns
make live LIVE_SCENARIOS=wave:tools  # the unbilled wave: no credentials, no cost
export CLAUDE_CODE_OAUTH_TOKEN=...   # from `claude setup-token` (or ANTHROPIC_API_KEY)
make live LIVE_SCENARIOS=wave:billed-1
make live PYTEST_ARGS='-k window'    # any pytest selection works too
```

`make live` sets `AS_LIVE=1` and runs `uv run pytest -m live tests/live -rs`. Without `AS_LIVE=1`
every live test skips, even under `pytest -m live`; `addopts` deselects the marker in a normal
run. Run it serially (no `pytest-xdist`): the session counter and the wall-clock bounds are per
process.

| Variable | Meaning |
|---|---|
| `AS_LIVE=1` | Opt-in. `make live` sets it. |
| `LIVE_SCENARIOS` | Comma or space separated: scenario names, `wave:<name>`, `billed`, `unbilled`, `all`. An unknown token is an error. Unset: everything. |
| `MAX_LIVE_SESSIONS` | The most real `claude` processes the run may start (default 25). A billed test that could not finish inside the cap fails before it starts; the wrapper refuses a launch over it (exit 97) and the test fails. |
| `LIVE_TEST_TIMEOUT` | Wall-clock bound of one test, seconds (default 180 unbilled, 420 billed). |
| `LIVE_SUITE_TIMEOUT` | Wall-clock bound of the whole run, seconds (default 5400); when it runs out the run stops. |
| `AS_LIVE_STRICT` | `1` (default): a missing required tool or missing credentials **fails** the scenario with a message naming what to set; `0`: it skips. Optional tools (`jobsctl`) always skip. |
| `ANTHROPIC_API_KEY`, `CLAUDE_CODE_OAUTH_TOKEN` | Credentials for billed scenarios (first one found). |
| `AS_LIVE_COPY_LOGIN=1` | Use a copy of `~/.claude/.credentials.json` instead (see [Credentials](#credentials)). |
| `LIVE_CLAUDE`, `LIVE_CLAUDE_SESSION`, `LIVE_TERMCTL`, `LIVE_WTCTL`, `LIVE_TMUX`, `LIVE_GIT`, `LIVE_JOBSCTL` | A tool other than the one on `PATH`. |
| `LIVE_RUN_USD` | `--max-budget-usd` the wrapper adds to `-p` runs (default 0.25). |
| `LIVE_OBSERVATIONS` | Append the recorded observations (below) to this JSON-lines file. |

claude-session must be installed with `--link-mode=copy` (`uv tool install --link-mode=copy`):
its isolated (third-party-reading) runs refuse a hard-linked install, which `fork-throwaway`
needs.

Tools are found on `PATH` (or the `LIVE_*` variable); the suite never installs anything. Their
versions are recorded by `tools-versions` and compared with the minimums in
`tests/live/harness.py` (`MINIMUMS`). A version that cannot be read (an editable install of a
sibling reports `0.0.0`; wtctl 0.32.x did, a released wtctl 1.0.0 prints `wtctl 1.0.0`, checked by `wtctl-version`) is recorded as unknown, never failed. claude-session older than 0.43.0
skips the scenarios that need `--implicit-done`.

### Waves

| Wave | Scenarios | Cost |
|---|---|---|
| `tools` | tmux, termctl, wtctl, claude-session (with a stub `claude`), agent-spawn against all of them (including `as-wrapper-dead-child-alive` and its window variant, the hook, `wait` exit-code, `adopted_at` and wtctl-failure scenarios of wave 2), wtctl hooks, bootstrap and `remove --merged-head`, cron, `reconcile --reset-lease`, browse, and the wave-4 scenarios (termctl hook override, slow `pre-teardown`, `locked`, baseline digest/anchor changes, concurrent `create`, partial teardown, input files, a pty for the human gates, window-close `post-park`) | none |
| `billed-1` | headless launch, interactive window and window close, `--no-implicit-done` report-resume chain | 4 sessions |
| `billed-2` | resume, in-place fork, fork into a throwaway, teardown with real wtctl, promotion | 9 sessions |
| `billed-3` | kill-pane, SIGTERM idle and mid-turn, minimal environment, `jobsctl cancel --force`, liveness kills, `interactive-window-close-unreported`, `window-wrapper-dead-child-alive-real` | 12 sessions |

Run the waves in order; a failure in `tools` usually explains the later ones.

## Cost

Only the billed scenarios cost anything. The bounds:

- **Model and prompts.** Every launch is `--model haiku`; prompts are one sentence ("Reply with
  the single word ok.") or, where a run must still be working when the test acts, a long
  counting task that the test ends itself.
- **The `claude` wrapper.** The private `PATH` starts with a `claude` wrapper (written by
  `harness.write_claude_wrapper`) around the real binary. It counts every launch in a file
  shared by the whole run (a version probe is not a launch), refuses a launch over
  `MAX_LIVE_SESSIONS` (exit 97), refuses a launch whose last `--model <value>` is not `haiku` (or
  a `claude-haiku-*` id) or that uses the `--model=<value>` form (exit 98) and adds
  `--max-budget-usd` to `-p` runs. A test fails on the refusals recorded since it started, not on
  earlier ones. This counts runs started by agent-spawn on its own
  as well (the automatic report resume, a promotion).
- **Turns.** A tiny claude-session run takes about three turns (read the prompt file, run the
  report command, the final answer) and claude gets `--max-turns` + 1, so headless launches pass
  `--max-turns 6` (the promotion scenario 8) and `--timeout 4m`. `--max-turns` only warns in
  interactive mode, so an interactive session has no turn cap: it is bounded by the tiny prompt,
  `--no-auto-exit` plus the test closing its window or `stop`, the test's wall-clock bound and the
  end of the private tmux server.
- **Wall clock.** Every test has a bound (`LIVE_TEST_TIMEOUT`, `SIGALRM`), the run has one
  (`LIVE_SUITE_TIMEOUT`), and every child process call is bounded and ends exactly the process
  group it created.
- **Session counter.** `make live-list` prints the expected sessions and turns per scenario and
  warns when the selection expects more than `MAX_LIVE_SESSIONS`. After a billed test the
  harness fails it when it started more sessions than its registry entry says.

The full billed set expects 25 real sessions (about 71 turns); `make live-list` prints the
current numbers. These are the expected numbers for one full pass; re-running a scenario after a fix starts its sessions again, so the sessions used over a debugging session can exceed them.

## Isolation

Every test gets a private world (`harness.World`): a directory directly under `/tmp` (short,
mode 0700) with its own `HOME`, `XDG_*`, `TMPDIR`, `TMUX_TMPDIR` (an existing directory, so no
tool can fall back to the real default socket), the agent-spawn, claude-session, termctl and
wtctl state and config, a `bin` directory that is the whole `PATH` of every child (symlinks to
the discovered tools, the counting `claude`, a tripwire `crontab` that exits 70), and per-test
git repositories (a bare origin and a clone, wired to wtctl with `repos.<id>.clone` and
`clone_url`: no network).

- Child environments are built from an allowlist (locale, proxy and certificate settings),
  never from the parent's; `TMUX`, `SSH_AUTH_SOCK`, `XDG_RUNTIME_DIR`, `CLAUDE_CONFIG_DIR` are
  never set; a test cannot override the private variables; every call re-checks the
  environment for paths outside the private root.
- The real tmux server, `~/.claude`, crontab, termctl and wtctl state are never read or written.
  The real home is read only to find the login file `AS_LIVE_COPY_LOGIN=1` asks for.
- Only private tmux servers are stopped (`kill-server` on every unix socket under the private
  root, with a fixed 10 s bound that does not depend on the test's deadline). Processes a test
  must signal are signaled by exact PID after an identity check (start-time token and command
  line). At teardown the harness also `claude-session stop`s the sessions the test noted and
  kills, each by exact PID plus start-time token, the tracked processes and any process whose
  `HOME` is the world's home or whose working directory lies under the world's root (a detached
  supervisor or claude a failed test left behind). Nothing is signaled by pattern. Each teardown
  step is guarded so the credential copy and the private home are always deleted, and pytest
  turns SIGTERM and SIGHUP into `KeyboardInterrupt` so an interrupted run still tears down (the
  login copy is also removed at exit).
- Commands run in the world's private `repos` directory unless a test names another; they never
  inherit pytest's working directory.
- `as-teardown-partial` makes a directory under the world unwritable (mode 0555) and restores it
  in a `finally`; the test skips as root. A pytest killed by SIGKILL in that window leaves a
  directory `rm` cannot delete: recover with `chmod -R u+rwX /tmp/as-live-*` and remove it.
- The cron scenario runs against a fake `crontab` first on `PATH`; the real crontab is never
  touched. Real-crontab fidelity (the system's `no crontab for <user>` wording and exit code,
  localized messages) is therefore **not checked** and stays in FIDELITY.md's Unverified list.

## Credentials

Billed scenarios need Claude Code credentials, supplied as in claude-session's live suite, in
this order:

1. `ANTHROPIC_API_KEY`. Claude Code needs the key approved, so its last 20 characters are
   written to the private `~/.claude.json` (mode 0600 in the 0700 private root, deleted at
   teardown) and interactive sessions do not stop at the custom-key dialog. The full key is
   never written to a file by the harness.
2. `CLAUDE_CODE_OAUTH_TOKEN` (from `claude setup-token`). The variable name is written to the
   private claude-session config (`claude_auth_env`), never its value.
3. `AS_LIVE_COPY_LOGIN=1`: a copy of `~/.claude/.credentials.json`. It is copied once per pytest
   process into a private mode-0700 directory, installed (mode 0600) into each billed test's
   private home, and a token Claude refreshed there is carried to the next test. The real file
   is only read, never written; the copy is deleted when the run ends. An OAuth refresh can
   rotate the refresh token and sign your real installation out, so prefer a token.

A credential reaches only the children of billed tests (their environment, or the private
home's login file) and is never printed (`Auth` hides it in every repr). Files that hold
credential material, all mode 0600 inside the 0700 private root and deleted at teardown: the
login copy (`AS_LIVE_COPY_LOGIN=1`), the last 20 characters of an API key in `.claude.json`, and
for `jobsctl-cancel-force` a small 0600 env file that a 0700 wrapper script sources before it
execs `agent-spawn launch`. The credentials go this way rather than through jobsctl's own
`env_file` because jobsctl rejects an `env_file` whose ancestor directory is writable by group
or others, and `/tmp` (mode 1777) is.
Unbilled scenarios get none. Do not run pytest
with `-l`/`--showlocals`: a failure would print locals.

## Reading the results

- A failing check raises `LiveError: assumption broken: <assumption> [<document line>]
  <what was seen>`. The bracket names the doc that states the assumption (an interface.md,
  design.md, FIDELITY.md line or a #36 comment item); fix the doc or the code, or both.
- Each test's docstring lists the FIDELITY.md items and #36 comment items it confirms. After a
  run, update those docs: move a confirmed behavior from "Unverified" to "Verified" and correct
  a fake that differs.
- Values that are open questions are recorded, not asserted, with `world.observe(key, value)`.
  They print in the terminal summary (`live observation: ...`) and, with `LIVE_OBSERVATIONS`,
  go to a JSON-lines file. Run with `-rs` (the make target does) to see skip reasons.
- Tests that need a tool the host lacks fail in strict mode (the default) so a partial host
  cannot report green.

## Scenarios

`make live-list` is the authoritative list; this is the map from the issue to the scenarios.

| Issue scope | Scenario |
|---|---|
| Full interactive launch in a window and window close (parked without attention; `done` when the session had already reported, `killed` otherwise) | `interactive-window-close` (billed); unbilled twin with a stub: `as-stub-lifecycle` |
| An unreported interactive session whose window is closed mid-turn (parked `killed`, no report) | `interactive-window-close-unreported` (billed) |
| `exec` SIGKILLed under a real interactive claude in a window | `window-wrapper-dead-child-alive-real` (billed) |
| Headless escalation with promotion and continuation (spike D5) | `promotion-escalation`; unbilled parts in `promotion-subchecks` |
| Resume; fork into a throwaway; in-place fork holder | `resume-continue`, `fork-throwaway`, `fork-inplace-holder` |
| Teardown with real wtctl | `teardown-wtctl-real`, `as-stub-lifecycle` |
| `jobsctl cancel --force` through `launch --wait` | `jobsctl-cancel-force` (skips without jobsctl) |
| `claude` resumable after kill-pane, SIGTERM idle and mid-turn (D1 to D3); minimal environment (D4) | `resumable-after-kill-pane`, `resumable-after-sigterm`, `minimal-environment` |
| `termctl open --override-escalation` after a changed `pre-open` hook (#112) | `as-termctl-override-escalation` |
| A `pre-teardown` hook slower than its timeout (#113) | `as-slow-pre-teardown` |
| The termctl `locked` error (#114) | `termctl-locked`, `as-label-locked` |
| A baseline change with no `changed_classes`: digest only, anchors only (#115) | `wtctl-baseline-no-classes`, `as-wtctl-baseline-no-classes` |
| wtctl `create` `busy` and the `creating` state; `remove` exit 1 `partial`; `--input-file`; `wait --interval`; a human `--yes` at a TTY; `post-park` once on the window-close parks; `wtctl --version` (#116) | `wtctl-create-busy`; `as-teardown-partial`; `as-input-files`; `cs-wait-interval`; `tty-human-gates`; `as-window-close-post-park`; `wtctl-version` |
| Spike regressions B1 and A8 | `termctl-sighup-leader`, `termctl-open-pid`, `as-stub-lifecycle` |
| The promotion sub-checks | `promotion-subchecks`, `termctl-open-pid`, `termctl-keep-on-exit`, `promotion-escalation` |

### The #36 comments

| Comment (PR) | Where it is checked |
|---|---|
| #8 fakes: claude-session | argv forms: existing `fidelity-argv`; shapes of `start`, `status`, `list`, `stop`, `forget`, `reconcile`, exit codes, outcome file mode and single write: `cs-pending-run`, `cs-dry-run-shapes`, `cs-stop-grace`; exec signals and exit 15: `cs-stop-grace`; hooks, finalizers under a real run: `headless-launch-wait`. Not covered: option abbreviations (see `fidelity-argv`), the order of refusals when several apply, `--actor` caps, hook splitting and parallelism, the trust dialog's `starting` state, the exit code of an exec killed by SIGKILL (observed: a headless claude dies with the wrapper and the run ends `lost`, see "First billed run"); `--input-file` copies (`as-input-files`: the staged copy is readable, a resume's own input replaces the set, the refusals); `wait --interval` (`cs-wait-interval`: claude-session has no such flag, see Results; `cron install --interval` is 1 to 59); claude-session's pre-launch veto, timeout and escalation are covered through agent-spawn (`as-launch-phase-hooks`) and claude-session's outcome mapping behind the exit codes 0, 13, 14, 15, 16, 17 and 18 of agent-spawn's `wait` by `as-wait-exit-codes` (the stub files real `claude-session report`s, which `exec` ingests when it exits; the scenario's 22 and 10 are agent-spawn's own codes, not claude-session's); an ambiguous id prefix (21, needs two ids sharing four characters), `start` without `--prompt-file` (exit 2), an unknown id for every command (exit 10, only `status` and `label` are checked), `stop` of an already final run (`already_final`) and the `recent_events` cap of 20 in `status`. |
| #8 termctl / tmux | `termctl-open-pid`, `termctl-keep-on-exit`, `termctl-focus`, `tmux-real-server`. The `locked` error: `termctl-locked` (the test holds the server's `flock`: exit 12 after about 10 s, `cause: timeout`; a lock file that cannot be used is `cause: unusable`) and `as-label-locked` (agent-spawn's `label add` warns and queues, `reconcile` pushes it). Not covered: `--keep-on-exit` on tmux 3.2 (host has 3.5a); the stale-socket and missing-socket texts are checked only by `fidelity-argv` (`test_fidelity.py`), not by `tmux-real-server`. |
| #8 wtctl | `wtctl-create-list-show`, `wtctl-collisions-reuse`, `wtctl-remove-status`, `wtctl-adopt-forget-holders`. A `create` hook veto (11), timeout and escalation (13) with `--override-escalation` and a human actor: `wtctl-create-hooks` (and through agent-spawn, `as-wtctl-create-failures`); `bootstrap_failed` with `details.record` and the human-only re-run: `wtctl-bootstrap-failed`; `remove --merged-head`: `wtctl-remove-merged-head`. A human gate at a real terminal (no `--actor`, a pty as stdin, `--yes`, the `[y/N]` prompt): `tty-human-gates`; a concurrent `create` (`busy`) and the `creating` state: `wtctl-create-busy`. Not covered: `remove --flag=value` forms beyond `--holder`/`--actor`. |
| #9 session backend | `cs-pending-run`, `cs-dry-run-shapes`. Not covered: the outcome-file refusal under the caller cwd for isolated runs, a launch-failed start's not-started outcome. |
| #14 relay | `headless-launch-wait` (real hook payload, `first_working_at`), `liveness-kills` (late delivery). Not covered: a hook whose stderr closed early, an event for a session whose label names another agent, redelivery of a dropped `malformed_payload`. |
| #15 self-label | status fields: `cs-pending-run`, `headless-launch-wait`; `verify-secret`: `cs-pending-run` (11 for pending, wrong and unknown; 0 for an adopted run's own secret read from its session directory with a stub claude; the self-label flow itself needs an agent inside a session: not automated). |
| #20, #21, #22, #23, #25 wtctl | the `wtctl-*` scenarios (`wtctl-reuse-mismatch`: agent-spawn against the real wtctl, wrapped only to give the launch's holder an earlier record on the branch under another name, refuses the reused record as `worktree_mismatch` and leaves it alone, #99); `as-stub-lifecycle`; `fork-inplace-holder`, `teardown-wtctl-real`. `remove` exit 1 `partial` and agent-spawn's sticky `worktree_partial` teardown: `as-teardown-partial` (the failure needs no git or root: the worktree's parent loses its write bit). A `remove` killed mid-delete: not covered (wtctl deletes in process with no hook or gate between the first entry and the last, so no deterministic kill point exists; the record such a kill leaves is the `partial` one that `as-teardown-partial` reaches). `baseline check`: the git class change and a change of only the `claude` class (`wtctl-baseline-check`, and `repo_config_changed` in `show` by `as-wtctl-create-failures`) are covered; a digest-only change (`repo_config_changed` with only the two digests) and a moved anchor (`baseline check` says `unlinked`; only `baseline accept --expect-digest` shows `details.anchors`) by `wtctl-baseline-no-classes` and `as-wtctl-baseline-no-classes`. |
| #26, #27 window | `termctl-open-pid`, `as-stub-lifecycle`, `interactive-window-close`. Not covered: post-open hooks slower than 60 s, a user's real termctl config (rules, hooks), WSL entries like `=C:`, the marker loop against an attached client. |
| #28 attach | `termctl-focus` (no client, unknown client, unknown id). The `switched` answer needs an attached client: not automated. |
| #74 here | `placement-here-terminal`: recorded, not automated (needs a human at a terminal). |
| #75 liveness | 1: `cs-pending-run`; 2: `cs-start-killed`; 3: `liveness-kills` (supervisor only: claude lives on and `supervisor_lost` is set; exec only: a headless claude dies with the wrapper by `PR_SET_PDEATHSIG`, the run and the park are `lost`), `as-wrapper-dead-child-alive` (the wrapper-dead-child-alive row, with a SIGTERM-ignoring stub: exec SIGKILLed, claude-session reports wrapper dead, child alive, no outcome; the supervisor ends, the agent stays `live` with `supervisor_lost` and `needs_attention: lost`, set once; `stop` ends the orphan and parks it `killed`; the interactive-window variant, `as-wrapper-dead-child-alive-window`: the stub ignores SIGTERM and SIGHUP, SIGKILL of exec leaves the child running and the window dead, the agent stays `live` with `supervisor_lost`, `stop` parks it `killed`), `as-dead-pane-reconcile`, and with a real claude in a window `window-wrapper-dead-child-alive-real` (the claude dies with its terminal: see Results); 4 (no second `post-park` from a late reconcile): `liveness-kills` (after `stop` of the supervisor-only run, and after the `lost` park of the exec-only run), `as-wrapper-dead-child-alive` (after the `stop` park); 4 for the window park paths with a stub: `as-window-close-post-park` (a killed window of an unreported session, parked `killed`, and SIGKILL of exec under a stub that dies with its terminal, parked `lost`: `post-park` once, late reconciles no second one; the two billed window scenarios themselves still configure no hook, see Not covered); 5: `as-stub-lifecycle`, `fork-inplace-holder`, `resume-continue`; 6 (foreign sessions) and 7 (`forget --holder` of `creating`/`partial`): the `missing`/`unlinked` half in `wtctl-adopt-forget-holders`; the rest not covered. |
| #78 overrides | `cs-dry-run-shapes` (claude-session half), `wtctl-remove-status` (wtctl half). The human `--override-escalation` is covered for `launch` (agent-spawn's own `pre-launch`, `as-prelaunch-hooks`), `resume` towards claude-session (`as-launch-phase-hooks`) and towards wtctl (`as-wtctl-create-failures`), and for `wtctl create` (`wtctl-create-hooks`), always with an explicit `--actor human:tester`. `termctl open --override-escalation` after a changed `pre-open` hook (`as-termctl-override-escalation`; see Results for the caller-cwd divergence) and the slow `pre-teardown` hook (`as-slow-pre-teardown`; `--override-escalation` never lifts a timeout veto). |
| #79 dead panes | `termctl-default-server`, `termctl-keep-on-exit`, `as-dead-pane-reconcile` |
| #80 reset-lease | `as-reconcile-reset-lease` (the TTY path needs a real terminal; a racing second reconcile is not provoked) |
| #81 reconcile | `cs-pending-run` (`reconcile --dry-run`), `as-dead-pane-reconcile`. Teardown killed during a real hook: not covered. |
| #83 upgrades | `upgrade-live-supervisor`: recorded, not automated (needs two agent-spawn builds). |
| #84 browse | `as-browse-pilot` (real `list` and `show`). `a`, `r`, `s`, `t`, `w` against real tools, terminals and Textual versions: not automated. |
| #88 implicit-done | `cs-dry-run-shapes` (flags, plan value, exit codes), `report-resume-chain` (carry through the automatic resume). A `--kind` change on resume dropping the carry is not covered. |
| #34 cron | `as-cron-fake-crontab` (fake crontab only; the real crontab is deliberately skipped) |
| #19 `DEFAULT_GRACE` | `cs-stop-grace` |

Recorded, not automated: `placement-here-terminal`, `upgrade-live-supervisor`, `real-crontab`
(never), `macos` (issue #37). They appear in `make live-list` with their reason and skip
with it.

## Not covered

What the unbilled live suite cannot check, with the reason. The billed candidates are for a later
wave (each needs a real Claude Code session or a host capability this host lacks); each item below is tracked by the issue named in its row, except the first row (untracked, no scenario planned), the wtctl audit gap (wtctl#5, closed, no open issue) (#100 is closed by the wave-3 run; the wave-4 run covered #112 to #115 and most of #116, whose remaining items are the rows above), and macOS is tracked by #37.

| Item | Why it is not an unbilled scenario | Candidate session cost |
|---|---|---|
| The null `adopted_at` of a run that ended in its launch phase, against real claude | No issue tracks it and no billed scenario is planned: it stays in the unbilled `as-launch-phase-hooks`. The other `adopted_at` cases (launched, plain resumed, automatically resumed, promoted) ran live, see Results. | not estimated (not designed) |
| A late `reconcile` after the window-close parks of wave 3 delivers no second `post-park` (#75 item 4), **with a real claude** | The unbilled twin `as-window-close-post-park` checks the product's behavior on both paths with a stub (a killed window, a SIGKILLed exec under a child that dies with its terminal). The two billed scenarios `interactive-window-close-unreported` and `window-wrapper-dead-child-alive-real` still configure no `post-park` hook or count its runs: adding it needs a hook config in them and a billed re-run (#116, which stays open for this item). | none (a hook config on the existing sessions) but a re-run of both: 3 sessions |
| Real sandbox behavior of isolated (third-party-reading) runs | The host has no `bwrap` or `socat`; `fork-throwaway` passes `--sandbox preferred`. Needs a host with both installed (#111). | 1 to 2 sessions on such a host |
| macOS | Out of scope of the Linux suite: tracked by issue #37, not #100. | not applicable |
| `wtctl remove` **killed** mid-delete (the unknown-outcome half of the sticky `worktree_partial` teardown) | wtctl deletes the tree in its own process, through directory fds, with no hook, prompt or gate between its first and last deletion (`post-remove` runs after the record is `removed`), so no deterministic kill point exists; a kill by timing would be flaky. The state such a kill leaves (a `partial` record) and agent-spawn's handling of it are covered by `as-teardown-partial`. Kept in #116. | none |
| The `baseline_accept` audit record of wtctl | `test_baseline_accept_is_audited` (scenario `wtctl-baseline-no-classes`) is an expected failure: wtctl 1.0.0 writes none. wtctl#5 (closed; the `TODO(#5)` in `baseline_accept.py` remains; no open issue) is the reference. | none |

The wave-4 scenarios use the stub `claude`: they check what agent-spawn and the lower tools do, not what Claude does.

## Stub claude

Scenarios in the `tools` wave need a program called `claude`. They use a shell stub
(`harness.write_stub_claude`) that is not Claude Code: it reports a version, records its PID and
runs a body (`exit 0`, a SIGTERM-ignoring loop). claude-session runs it like the real thing
(it reaches `phase: session` while the child lives), so lifecycle, handoff, holds, teardown,
liveness and dead panes are checked against the real sibling tools for free. What a real
Claude adds (hook events, reports, resumable conversations) is checked by the billed scenarios.

## First unbilled run

The `tools` wave was run on Linux 5.15 with tmux 3.5a, git 2.34.1, termctl 0.25.0, wtctl 0.32.0
and claude-session 0.45.2 (the siblings built from their `origin/main`; jobsctl 0.34.0 was not
needed). Nothing billed was run. Observed differences from what the docs assumed:

- wtctl `create --dry-run` over a record the holder already has returns `reused: true`,
  `dry_run: false`, `branch_action` null. The product is right (wtctl interface.md, Create order,
  "Retry after a crash": the answer is the same under `--dry-run`); design.md and the fake were
  wrong and are corrected. Another holder over the same name is refused `path_exists`.
- wtctl, explicit `--branch` held by another holder under the same `--name`: refusal reason
  `path_exists` (the path check runs first), not `branch_in_use`; under another `--name` the
  same holder gets its existing record back (`reused: true`, the record keeps the old name).
  agent-spawn refuses that record (#99, scenario `wtctl-reuse-mismatch`).
- wtctl `baseline check` change: `error.code` is `refused` and `error.reason` is
  `repo_config_changed` (with `details.changed_classes`); the fake does not model a changed
  baseline.
- claude-session `verify-secret` of an unknown session answers like a wrong secret
  (`11`, `{"match": false}`, claude-session interface.md; the fake already does the same); an id
  prefix needs 4 characters after `cs-`.
- claude-session takes `--json` after the command (a leading `--json` is exit 2).
- `stop` of a SIGTERM-ignoring child took 11.6 s with the default `kill_grace` of 10 s, inside
  agent-spawn's `DEFAULT_GRACE` plus margin. Its `signals` list sometimes holds only the SIGTERM
  (exec's own SIGKILL won the race), so the scenario accepts both.

Everything else the `tools` wave checks passed. The per-scenario details are in the test
docstrings; run with `LIVE_OBSERVATIONS=live.jsonl` to get the observed values.

## First billed run

The billed waves ran with the real Claude Code on 2026-11-19 (haiku only). The #36 comment records
that the first run (42 haiku sessions) passed 9 of the 10 scenarios it ran; it does not name them
or the failing one, and the count is not reconciled with the 13 billed scenarios listed under
[Results](#results). PR #98 names the scenarios it fixed after that run: `report-resume-chain`,
`interactive-window-close`, `resumable-after-kill-pane`, `fork-throwaway`, `promotion-escalation`
and the exec-only case of `liveness-kills`. A rerun (13 haiku sessions) after the fixes passed;
the #36 final comment says every rerun scenario passed but does not list which ones were rerun.
The failures were harness and doc assumptions (below), not product bugs. Combined result:
[Results](#results).

Contradicted by the first run (tests and docs corrected):

- the real model filed its report in one turn even for "Reply with the single word ok.", so no
  interactive run stayed unreported or reached `needs-input`; the window-close park was
  `done`/`forwarded_hup`;
- the promotion test compared the recorded supervisor with the pane process: the recorded
  supervisor stays the original headless supervisor, and the pane runs `supervise --host`;
- SIGKILL of exec only does not leave a headless claude alive (the row is reached with a SIGTERM-ignoring child, `as-wrapper-dead-child-alive`): a headless claude dies with the wrapper
  (`PR_SET_PDEATHSIG`), the run ends `lost`, the agent parks `lost` with `needs_attention: lost`
  and `supervisor_lost` false;
- two fidelity `ACCEPTED` rows (an empty `--model`, `--permission-profile=-x`) are refused by the
  real 0.45.2 with `usage`; the fake is corrected.

## Results

Run date 2026-11-19, Linux 5.15, claude 2.1.328 (`--model haiku`), claude-session 0.45.2,
termctl 0.25.0, wtctl 0.32.0, tmux 3.5a, git 2.34.1, jobsctl (installed, used by
`jobsctl-cancel-force`). The suite has passed on Linux:

- **Unbilled wave** (`LIVE_SCENARIOS=unbilled`): 154 passed (before `as-wrapper-dead-child-alive`), 4 skipped (the recorded-only
  scenarios `placement-here-terminal`, `upgrade-live-supervisor` and `real-crontab`, and `macos`,
  issue #37).
- **`as-wrapper-dead-child-alive`** (added after the first runs, 2026-11-20, claude-session
  0.45.2, termctl 0.25.0, wtctl 0.32.2 (an `origin/main` build from `git archive`; its
  `--version` printed the `0.0.0` placeholder in 0.32.x, the package metadata said 0.32.2; the `0.0.0` remark applies to the 0.32.x runs only. `wtctl --version` of the 1.0.0 release prints `wtctl 1.0.0`, checked in wave 4 by `wtctl-version`), tmux 3.5a):
  passed. The stub child ignores SIGTERM but records each one, and the test waits for its ready
  file before killing anything. SIGKILL of exec leaves the child alive (the one SIGTERM it
  received came from `PR_SET_PDEATHSIG` at that moment, and it ignored it); claude-session
  reports `wrapper_alive: false`, `child_alive: true` and no outcome; the agent-spawn supervisor
  reads no outcome and ends; the agent stays `live` with `supervisor_lost: true` and
  `needs_attention: lost`, set once; liveness checks (`show`, `list`, `reconcile`) send the stub
  no further SIGTERM. `agent-spawn stop --grace 2s` exits 0; claude-session's `stop` result
  (`session_stop`) has `ended: true`, `run_exit_code: 15` (the `killed` outcome code, not a
  `stop` exit code), `outcome: killed`, `end_reason: stopped` and signals only the child, SIGTERM
  then SIGKILL; the stub logged that one more SIGTERM. The agent parks `killed` with
  `supervisor_lost` cleared and `post-park` runs once. The fake-level twin is
  `tests/test_liveness_processes.py` (`test_a_sigkilled_exec_under_a_surviving_child_...`).
- **Wave 2, unbilled** (2026-11-20, claude-session 0.45.2, termctl 0.25.0, wtctl 0.32.2 from `git archive`, tmux 3.5a, git 2.34.1, no `claude` and no credentials; the whole unbilled wave: 165 passed, 5 skipped (the four recorded-only scenarios and jobsctl, not installed), Python 3.12). New scenarios, all against the real tools with a stub `claude`:
  - `as-prelaunch-hooks`: agent-spawn's own `pre-launch` hook vetoes with exit 11 (envelope `hook` and `hook_stderr`), escalates with 13, a timeout vetoes and ends the hook's process, a later veto wins over an earlier escalation (both ran), `--override-escalation` is exit 2 without a human actor (also with `--actor auto:*`), counts a 13 as allow with `--actor human:tester` (payload `override_escalation: true`) and never lifts a veto; no agent, session or prompt copy exists after any refusal.
  - `as-launch-phase-hooks`: claude-session's `pre-launch` veto ends the run `vetoed`/`hook_veto` (11), a timeout `hook_timeout` (11), an escalation `escalated`/`hook_escalate` (13, parked `escalated` by `claude-session`); `wait` repeats the codes; `show` has `adopted_at` null for each of those runs and a date-time for the resumed run after `resume --override-escalation --actor human:tester`; the flag is exit 2 without a human actor and for an agent whose last run did not end escalated.
  - `as-wait-exit-codes`: `launch --wait` and `wait` exit 14 (stub exits 0), 18 (`failed` report), 13 (`blocked` and `needs-input` reports, parked `blocked` and `escalated`), 0, 16 (`--timeout 3s`, end reason `timeout`), 15 (`stop`), 17 (SIGKILL of exec while the child dies with it: parked `lost`, attention `lost`), 22 (`wait --timeout` on a live agent) and 10 (unknown id). The stub files its reports with the real `claude-session report ... --reason-file -`; `exec` ingests them when the stub exits.
  - `as-adopted-at`: `show` `runs[].adopted_at` is a date-time for a launched run (already while live), a resumed run, the automatic report resume (a stub that prints a stream-json `result` line under `--no-implicit-done`) and a promoted run (a stub that files `needs-input` once; `--promote`); each pair of runs is ordered, and the first run's value is unchanged by the resume. The real-timing check stays billed (see [Not covered](#not-covered)).
  - `as-wrapper-dead-child-alive-window`: the interactive-window variant of the wrapper-dead-child-alive row: with a stub ignoring SIGTERM and SIGHUP, SIGKILL of exec leaves the child running (it received at most one SIGHUP, never a SIGTERM, when the window's pane process, `supervise`, ended and the pane died); claude-session reports wrapper dead, child alive, no outcome; the agent stays `live` with `supervisor_lost` and attention `lost`; `reconcile` parks nothing; `stop` ends the child (SIGTERM, then SIGKILL) and parks `killed`; `teardown` closes the dead pane.
  - `wtctl-create-hooks`, `wtctl-bootstrap-failed`, `wtctl-remove-merged-head` and the claude-class test of `wtctl-baseline-check`: the real wtctl's `pre-create` veto (11), timeout (11, the hook's process killed), escalation (13, changed nothing) and human override (exit 2 for a non-human actor), a failing bootstrap step (12 `bootstrap_failed` with `details.record`, the record kept, later steps skipped, `bootstrap` re-run 13 without a human actor and 0 with one), `remove --merged-head` of a branch deleted on the remote (refused `unpushed` without it or with a sha that is not HEAD; removed and the branch deleted with it), and a change of only the `claude` class (`baseline check` exit 20 with `changed_classes: ["claude"]`, `status` reports a baseline warning).
  - `as-wtctl-create-failures`: launch over the same wtctl: 11 parks `vetoed` without attention; 13 parks `escalated` with `escalated_by: wtctl` and `resume --override-escalation --actor human:tester` reaches `wtctl create` and the launch runs; 12 `bootstrap_failed` passes through as exit 12 with `lower_code: bootstrap_failed`, stores the worktree record, and the resume continues with it without a second worktree or a second bootstrap; `show` `repo_config_changed` reports `changed_classes: ["claude"]` for an added `CLAUDE.md`; `teardown` of that worktree with only the claude class changed is decided by `wtctl remove` (exit 0, or exit 20 with a reason that is never `repo_config_changed`; the test accepts either).
  - `wtctl-reuse-mismatch` (from #99, not re-run by its reviewers) passed against the real wtctl: its priming `create` without `--actor` is accepted.
  No divergence from the docs or the product was found; the corrections made while writing the scenarios were to the harness's own stub (`claude-session report` takes `--reason-file`, not `--reason`).
- **Wave 4, unbilled** (2026-11-20, claude-session 0.45.2, termctl 0.25.0, wtctl 1.0.0 (a `git archive` of the `v1.0.0` tag: `wtctl --version` prints `wtctl 1.0.0`, so the 0.32.x `0.0.0` remark is dropped and `wtctl-remove-merged-head` was re-run against it), tmux 3.5a, git 2.34.1, stub `claude`, no credentials, 0 sessions; the whole unbilled wave: 178 passed, 5 skipped (the four recorded-only scenarios and jobsctl, not installed), 2 expected failures, Python 3.12; the new tests alone on Python 3.13: 13 passed, 2 expected failures). New scenarios:
  - `as-slow-pre-teardown`: a `pre-teardown` hook with `timeout: 3s` that sleeps is killed (its shell and the `sleep` it started are gone), `teardown` exits 11 `vetoed` naming the hook, the agent is `parked` again with no marker, the session is kept and the agent's attention is unchanged (the launch uses `--worktree none`, so no worktree is checked); `--override-escalation --actor human:tester` does not lift it (the hook runs and vetoes again); a hook that exits 0 in time lets the retry finish.
  - `as-termctl-override-escalation`: a termctl `pre-open` hook exits 13: the launch exits 13 and parks `escalated` with `escalated_by: termctl`, no pane; `resume --override-escalation` exits 2 without a human actor and with `--actor auto:bot` (the hook is not run again); after the hook script is replaced by another one that still escalates, `resume --override-escalation --actor human:tester` reaches `termctl open --override-escalation`, the changed hook runs with `override_escalation: true`, its 13 counts as allow, the window opens and the agent is `live` without attention. `test_the_override_resume_from_another_cwd` runs the same resume from the harness's default cwd, outside the repository: the pending run termctl's escalation left canceled recorded no cwd, so agent-spawn passes the agent's recorded cwd as `--cwd` (#119) and the resumed run records it (checked with `claude-session status`). Before that fix it exited 12 `launch_failed` with claude-session's `git_dir_changed`; the main test no longer needs the caller's cwd to be the repository.
  - After the fix of #119 (2026-11-20, claude-session 0.45.2, termctl 0.25.0 and tmux 3.5a, no `claude`, no credentials): `as-termctl-override-escalation` passes both tests (2 passed), the second from the harness's default cwd outside the repository; `claude-session status` shows the resumed run's cwd is the agent's repository.
  - `termctl-locked`, `as-label-locked`: with the test holding the server's `flock` (`locks/<16 hex of sha256(socket_path)>.lock`), `termctl label add` exits 12 `locked`, `details.cause: timeout` and the socket path after 10.2 s and changes nothing, a dry run exits 0, `list` is unaffected, and after release the same command works; a lock file with mode 0 is `cause: unusable` with `path` and `problem` at once. Through agent-spawn, `label add` on a live window agent succeeds on the agent and the session, warns `termctl label add failed (exit 12); queued for reconcile` (the warning does not name `locked`) and a later `reconcile` reports `labels_resynced` and the pane carries the label.
  - `wtctl-baseline-no-classes`, `as-wtctl-baseline-no-classes`: a digest-only change (the record's `baseline_digest` edited in the world's own `state.db`, the only way to present it) is exit 20 `repo_config_changed` with only `digest` and `recorded_digest`; agent-spawn `show` maps it to `changed: true`, no classes, no items, and `teardown` passes wtctl's refusal through (exit 20); a human `baseline accept` clears it. An anchors-only change is **not reachable through `baseline check`**: with `dir_id` or `common_id` edited, `baseline check` refuses `unlinked` (integrity runs first), `show` has no `repo_config_changed` and warns that the baseline was not checked, `teardown` passes `unlinked` through; only `baseline accept --expect-digest <wrong>` gives `repo_config_changed` with `details.anchors` and no `changed_classes`. The issue's premise (a `baseline check` change with no classes for anchors) was a doc assumption; corrected here. wtctl 1.0.0 writes no audit record for `baseline accept` (wtctl#5 is closed but the `TODO(#5)` in `baseline_accept.py` remains and there is no open issue): `test_baseline_accept_is_audited` is an expected failure (strict, `raises=KnownDivergence`, raised only when the accept succeeded and the audit records lack `baseline_accept`).
  - `wtctl-create-busy`: a real `wtctl create` held in a `pre-create` hook holds the clone lease: a concurrent `create` exits 12 `busy` after `lease_wait` (2.3 s with `lease_wait: 2s`), runs no hook and leaves no record; released into a blocking bootstrap step, the record is `creating` (holder set) in `list` and `show`, a `create --no-bootstrap` of another name succeeds (the clone lease is free), `remove` and `forget` of the `creating` record are `busy` after `lease_wait`, and the held create finishes `active`.
  - `as-teardown-partial`: the worktree's parent directory made unwritable makes `wtctl remove` exit 1 `partial` (`details.remaining_path`, record `partial`); agent-spawn's `teardown` exits 12 `partial`, the agent stays `tearing_down` at `worktree_partial`, a retry stays there, `reconcile` reports `teardown_incomplete` with the partial record, and after a human `wtctl remove` the retried teardown completes. Killed mid-delete: not covered (see Not covered).
  - `as-input-files`: the stub reads `inputs/1/note.txt` (mode 0600; the doc's "write-denied" is the isolated run's sandbox rule, not a read-only mode, and is not checked: no `bwrap`), the caller's later edit does not change it, `status` lists `input_files`, agent-spawn's private copy is deleted, a resume with another `--input-file` puts it in `inputs/2/` and keeps `inputs/1/`, and an oversized, duplicate-basename, badly named or `-` input exits 2 with nothing allocated and a message naming the input problem (a positive control with a small input starts).
  - `cs-wait-interval`: `claude-session wait --interval 1` and `agent-spawn wait --interval 1` are exit 2 (`unrecognized arguments`); `cron install --interval 0`, `60` and `abc` exit 2 with a message naming `--interval`, and `1` and `59` are accepted (`*/1`, `*/59`, against a fake crontab: a valid interval reads the table); `wait` after `stop` exits 12 with outcome `not-started`. **Doc bug fixed:** "`wait --interval`" in this file named a flag that does not exist in either tool.
  - `tty-human-gates`: with a pseudo-terminal (Python's `pty`) as stdin and no `--actor`, `wtctl baseline accept` prints `[y/N]`, "n" exits 13 and changes nothing, `--yes` accepts as the default human actor without a prompt (without a TTY and no actor it is exit 13); `agent-spawn launch --override-escalation` is accepted, the escalating `pre-launch` hook's payload has `override_escalation: true` and an actor starting `human:`; an explicit `--actor auto:bot` is exit 2 even at a TTY, for `launch --override-escalation` and for `teardown --force` (the cli gate itself).
  - `as-window-close-post-park` (the unbilled twin of #75 item 4 on the window paths): a killed window of an unreported stub session parks `killed` and SIGKILL of the recorded exec under a stub that dies with its terminal parks `lost` with attention `lost`; `post-park` ran once for each and two late `reconcile`s delivered no second one.
  - `wtctl-version`: `wtctl --version` is `wtctl 1.0.0`, and the harness reads the same version (the test skips on a build that prints `0.0.0` or is below the minimum, and fails only when a version of 1.0.0 or later prints another line).
- **Wave 3, billed** (2026-11-20, claude 2.1.328 `--model haiku`, claude-session 0.45.2, termctl 0.25.0, wtctl 1.0.0 (an `origin/main` build from `git archive`, the unbilled wave was re-run with it: 165 passed, 5 skipped, Python 3.12), tmux 3.5a; the login copy of `AS_LIVE_COPY_LOGIN=1`; 8 haiku sessions in all, the session budget of that run, so no re-run of anything else):
  - `interactive-window-close-unreported` passed (1 session): a long no-tools turn is still `working` when the window is closed and no report exists; the agent parks `killed`, `needs_attention` null, last end reason `forwarded_hup`; claude-session records outcome `killed`, end reason `forwarded_hup`, exit code 15, no report. This is what interface.md's outcome table says; no divergence.
  - `window-wrapper-dead-child-alive-real` (2 sessions: a first run failed on the test's own expectation, then passed after the test was fixed): SIGKILL of the recorded `exec` pid, in a window with a real interactive claude that had finished its first turn. **Observed: the real claude does not survive.** It exits within the test's 20 s look (the pane's process `supervise` ends when `exec` dies and the pane hangs up, and claude gets the hangup), so the wrapper-dead-child-alive state is not reached with a real claude; claude-session records `wrapper_alive: false`, `child_alive: false`, outcome `done`, end reason `wrapper_lost` (the model had already filed its report in its first turn, "a reported run keeps its outcome"), and the agent parks `done` without attention and without `supervisor_lost`. When the session has not reported, the outcome table's `lost` row would apply (parked `lost`, attention `lost`); that case was not observed because the real model reports in its first turn. The test accepts either row of the table for a dead claude (and the `live` with `supervisor_lost` then `stop` row for a surviving one) and fails only if the agent contradicts the outcome claude-session recorded. The first run's expectation (`lost`) was a test bug, not a product divergence.
  - `adopted_at` against real timing passed in `headless-launch-wait`, `report-resume-chain` and `promotion-escalation` in this wave, and in `resume-continue` in a separate later re-run (see the `resume-continue` live re-run entry below): a date-time, non-null for every run, not before the agent's creation, non-decreasing, not after the run's own end and not earlier than the previous run's end (supervised runs only, with 1 s of slack), inside the test's wall-clock window, and unchanged after a later action (`reconcile`, the park). Observed: the adopted time follows the run's start by a few seconds, for example 16:18:18.660 to 16:18:23.166 for a one-turn headless run.
  - Sessions in the run: 1 (unreported) + 2 (wrapper-dead, two attempts) + 1 + 2 + 2 = 8.
- **Billed scenarios**: the first run passed 9 of the 10 scenarios it ran and, after the harness
  fixes, every rerun scenario passed (see [First billed run](#first-billed-run) for what is and
  is not recorded about which ran when). The 13 billed scenarios passed live:
  `headless-launch-wait`; `interactive-window-close` (parks
  `done` with `forwarded_hup`, because the real model files a report in its first turn);
  `report-resume-chain` (run 1 `exited-unreported` with `no_report`, automatic resume run 2
  `done`, the agent parks `done`); `resume-continue`; `fork-inplace-holder` (after the fork,
  `list --label agent=<source id>` returned only the source session); `fork-throwaway`
  (needs `--reads-third-party --sandbox preferred` on a host without bwrap and socat);
  `teardown-wtctl-real`; `promotion-escalation` (D5; final park `killed`; the original headless
  supervisor stays the recorded one and the pane process runs `supervise --host`);
  `resumable-after-kill-pane` (D1; run 1 `done` with `forwarded_hup`); `resumable-after-sigterm`
  (D2/D3: SIGTERM at idle gives `done`/`signal_term`, mid-turn `killed`/`signal_term`);
  `minimal-environment` (D4); `jobsctl-cancel-force`; `liveness-kills` (supervisor-only SIGKILL:
  claude lives, the run is not finalized, the agent parks `killed` after `stop`; exec-only
  SIGKILL: claude dies with the wrapper through claude-session's parent-death SIGTERM, the run
  ends `lost`, the agent parks `lost` with `needs_attention: lost`, `supervisor_lost` false).
- **`resume-continue` live re-run** (2026-11-20, claude 2.1.328 `--model haiku`, claude-session 0.45.2, termctl 0.25.0, wtctl 1.1.0 (a `git archive` of `origin/main`), tmux 3.5a, Python 3.12; the login copy of `AS_LIVE_COPY_LOGIN=1`; 2 haiku sessions with `MAX_LIVE_SESSIONS=3`; the unbilled wave was not re-run): passed in about 25 s. The `adopted_at` checks of `billed.check_adopted_at` ran live twice: for the first run alone (a date-time, not before the agent's creation, inside the test's wall-clock window, not after its own end) and after the resume (two runs: both date-times, non-decreasing, each not after its own end, run 2 not earlier than run 1's end, run 1's value unchanged). Observed: run 1 adopted 7.2 s before it ended; run 2 adopted 4.0 s after run 1 ended and 4.6 s before its own end (UTC timestamps with millisecond precision).
- **Usage** (2026-11-19): 42 haiku sessions in the first run (budget 45) and 13 in the rerun (budget 14), 55
  in all, model haiku only; wave 3 (2026-11-20) used 8 more, 63 in all, and the `resume-continue` re-run 2 more, 65. No API key was used: the login was copied to a private directory
  (`AS_LIVE_COPY_LOGIN=1`) and deleted at each teardown.

Real-tool facts confirmed live without billing: A8 (the termctl pid is tmux's `pane_pid` and the
recorded supervisor chain), B1 (SIGHUP reaches only the leader), `--keep-on-exit failed`, the
default-server dead-pane list and `close` without `--force` (exit 20 for a running pane), the default `stop` grace of about 10.5 s (SIGTERM then SIGKILL) against
`DEFAULT_GRACE`, reconcile closing a dead pane on a real tmux server, `--reset-lease` repairs,
cron against a fake crontab, the browse pilot over real `list`/`show` output, `exited-unreported`
(exit 14) for a stub exiting 0, the wtctl facts of "First unbilled run", and claude-session
`verify-secret` of an unknown session (exit 11), the 4-character id prefix, and `start --model ""`
and `--permission-profile=-x` (exit 2, invalid flag value; the fake agrees).

Still open or not covered, each tracked:

- the items of [Not covered](#not-covered) (real-sandbox behavior #111 and the rest of #116: the billed window-close scenarios' own `post-park` hook, `remove` killed mid-delete); #112 to #115 are covered by the wave-4 run;
- two divergences found by wave 4, reported: the resume of a termctl-escalated launch from another cwd (`git_dir_changed`) and the missing wtctl `baseline_accept` audit record;
- the wtctl reuse mismatch: fixed by #99 (refused as `worktree_mismatch`; checked by `wtctl-reuse-mismatch`);
- macOS: #37;
- recorded-only: the `placement here` terminal check, the live upgrade of a running supervisor,
  the real crontab.

## Extending

- A new test names its scenario with `@pytest.mark.live_scenario("<name>")`; add the scenario
  to `tests/live/scenarios.py` first (collection fails otherwise, and
  `tests/test_live_harness.py` checks the two agree).
- Unbilled tests take the `world` fixture, billed ones `billed_world` (credentials, counting
  wrapper). Put helpers in `tests/live/harness.py` and test them in
  `tests/test_live_harness.py`; the helpers must be testable without a real tool.
- Check with `harness.check(condition, assumption, document-line, detail)` so a failure names
  what broke; record open questions with `world.observe`.
- A billed test must not start more sessions than its registry entry says.
