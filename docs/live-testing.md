# Live testing

The `live` suite (`tests/live/`) runs agent-spawn against the real tools it depends on: Claude
Code (`claude --model haiku`), tmux, claude-session, termctl, wtctl, git and, when installed,
jobsctl. It re-checks every relied-on third-party behavior that the fakes in `tests/fakebin/`
only assume ([FIDELITY.md](../tests/fakebin/FIDELITY.md)) and that the design records as
"assumed, check live" ([design.md](design.md), Live verification). It is **opt-in**, never part
of `make verify` and never run in CI. Issue #36 closes only after a full run passes.

Some scenarios cost API usage (a few dozen haiku turns in total, see [Cost](#cost)); the rest
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

Tools are found on `PATH` (or the `LIVE_*` variable); the suite never installs anything. Their
versions are recorded by `tools-versions` and compared with the minimums in
`tests/live/harness.py` (`MINIMUMS`). A version that cannot be read (an editable install of a
sibling reports `0.0.0`) is recorded as unknown, never failed. claude-session older than 0.43.0
skips the scenarios that need `--implicit-done`.

### Waves

| Wave | Scenarios | Cost |
|---|---|---|
| `tools` | tmux, termctl, wtctl, claude-session (with a stub `claude`), agent-spawn against all of them, cron, `reconcile --reset-lease`, browse | none |
| `billed-1` | headless launch, interactive window and window close, `--no-implicit-done` report-resume chain | 4 sessions |
| `billed-2` | resume, in-place fork, fork into a throwaway, teardown with real wtctl, promotion | 9 sessions |
| `billed-3` | kill-pane, SIGTERM idle and mid-turn, minimal environment, `jobsctl cancel --force`, liveness kills | 10 sessions |

Run the waves in order; a failure in `tools` usually explains the later ones.

## Cost

Only the billed scenarios cost anything. The bounds:

- **Model and prompts.** Every launch is `--model haiku`; prompts are one sentence ("Reply with
  the single word ok.") or, where a run must still be working when the test acts, a long
  counting task that the test ends itself.
- **The `claude` wrapper.** The private `PATH` starts with a `claude` wrapper (written by
  `harness.write_claude_wrapper`) around the real binary. It counts every launch in a file
  shared by the whole run (a version probe is not a launch), refuses a launch over
  `MAX_LIVE_SESSIONS` (exit 97), refuses a launch whose `--model` is not a haiku model (exit 98)
  and adds `--max-budget-usd` to `-p` runs. This counts runs started by agent-spawn on its own
  as well (the automatic report resume, a promotion).
- **Turns.** Headless launches pass `--max-turns 3` and `--timeout 4m`. `--max-turns` only warns
  in interactive mode, so an interactive session has no turn cap: it is bounded by the one-turn
  prompt, the test closing its window, `stop`, the test's wall-clock bound and the end of the
  private tmux server.
- **Wall clock.** Every test has a bound (`LIVE_TEST_TIMEOUT`, `SIGALRM`), the run has one
  (`LIVE_SUITE_TIMEOUT`), and every child process call is bounded and ends exactly the process
  group it created.
- **Session counter.** `make live-list` prints the expected sessions and turns per scenario and
  warns when the selection expects more than `MAX_LIVE_SESSIONS`. After a billed test the
  harness fails it when it started more sessions than its registry entry says.

The full billed set expects 23 real sessions (about 24 turns); `make live-list` prints the
current numbers.

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
- Only the private tmux server is stopped (`kill-server` on its socket). Processes a test must
  signal are signaled by exact PID after an identity check (start-time token and command
  line); leftovers are killed at teardown by PID plus token. Nothing is signaled by pattern.
- The cron scenario runs against a fake `crontab` first on `PATH`; the real crontab is never
  touched. Real-crontab fidelity (the system's `no crontab for <user>` wording and exit code,
  localized messages) is therefore **not checked** and stays in FIDELITY.md's Unverified list.

## Credentials

Billed scenarios need Claude Code credentials, supplied as in claude-session's live suite, in
this order:

1. `ANTHROPIC_API_KEY`. The key is also approved (its last 20 characters) in the private
   `~/.claude.json`, so interactive sessions do not stop at Claude Code's custom-key dialog.
2. `CLAUDE_CODE_OAUTH_TOKEN` (from `claude setup-token`). The variable name is written to the
   private claude-session config (`claude_auth_env`), never its value.
3. `AS_LIVE_COPY_LOGIN=1`: a copy of `~/.claude/.credentials.json`. It is copied once per pytest
   process into a private mode-0700 directory, installed (mode 0600) into each billed test's
   private home, and a token Claude refreshed there is carried to the next test. The real file
   is only read, never written; the copy is deleted when the run ends. An OAuth refresh can
   rotate the refresh token and sign your real installation out, so prefer a token.

A credential reaches only the children of billed tests (their environment, or the private
home's login file), is never written to a config file, never printed (`Auth` hides it in every
repr) and is not in any file this suite writes. Unbilled scenarios get none. Do not run pytest
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
| Full interactive launch in a window and window close (parked `killed`) | `interactive-window-close` (billed); unbilled twin with a stub: `as-stub-lifecycle` |
| Headless escalation with promotion and continuation (spike D5) | `promotion-escalation`; unbilled parts in `promotion-subchecks` |
| Resume; fork into a throwaway; in-place fork holder | `resume-continue`, `fork-throwaway`, `fork-inplace-holder` |
| Teardown with real wtctl | `teardown-wtctl-real`, `as-stub-lifecycle` |
| `jobsctl cancel --force` through `launch --wait` | `jobsctl-cancel-force` (skips without jobsctl) |
| `claude` resumable after kill-pane, SIGTERM idle and mid-turn (D1 to D3); minimal environment (D4) | `resumable-after-kill-pane`, `resumable-after-sigterm`, `minimal-environment` |
| Spike regressions B1 and A8 | `termctl-sighup-leader`, `termctl-open-pid`, `as-stub-lifecycle` |
| The promotion sub-checks | `promotion-subchecks`, `termctl-open-pid`, `termctl-keep-on-exit`, `promotion-escalation` |

### The #36 comments

| Comment (PR) | Where it is checked |
|---|---|
| #8 fakes: claude-session | argv forms: existing `fidelity-argv`; shapes of `start`, `status`, `list`, `stop`, `forget`, `reconcile`, exit codes, outcome file mode and single write: `cs-pending-run`, `cs-dry-run-shapes`, `cs-stop-grace`; exec signals and exit 15: `cs-stop-grace`; hooks, finalizers under a real run: `headless-launch-wait`. Not covered: option abbreviations (see `fidelity-argv`), the order of refusals when several apply, `--actor` caps, hook splitting and parallelism, the trust dialog's `starting` state, exec killed by SIGKILL exit code (`liveness-kills` records it). |
| #8 termctl / tmux | `termctl-open-pid`, `termctl-keep-on-exit`, `termctl-focus`, `tmux-real-server`. Not covered: the `locked` error (needs a concurrent label command), `--keep-on-exit` on tmux 3.2 (host has 3.5a). |
| #8 wtctl | `wtctl-create-list-show`, `wtctl-collisions-reuse`, `wtctl-remove-status`, `wtctl-adopt-forget-holders` |
| #9 session backend | `cs-pending-run`, `cs-dry-run-shapes`. Not covered: the outcome-file refusal under the caller cwd for isolated runs, a launch-failed start's not-started outcome. |
| #14 relay | `headless-launch-wait` (real hook payload, `first_working_at`), `liveness-kills` (late delivery). Not covered: a hook whose stderr closed early, an event for a session whose label names another agent, redelivery of a dropped `malformed_payload`. |
| #15 self-label | status fields: `cs-pending-run`, `headless-launch-wait`; `verify-secret`: `cs-pending-run` (the self-label flow itself needs an agent inside a session: not automated). |
| #20, #21, #22, #23, #25 wtctl | the `wtctl-*` scenarios; `as-stub-lifecycle`; `fork-inplace-holder`, `teardown-wtctl-real`. A `remove` killed on timeout mid-delete, exit 1 `partial` (needs a failure injected into git) and the `creating` state: not covered. |
| #26, #27 window | `termctl-open-pid`, `as-stub-lifecycle`, `interactive-window-close`. Not covered: post-open hooks slower than 60 s, a user's real termctl config (rules, hooks), WSL entries like `=C:`, the marker loop against an attached client. |
| #28 attach | `termctl-focus` (no client, unknown client, unknown id). The `switched` answer needs an attached client: not automated. |
| #74 here | `placement-here-terminal`: recorded, not automated (needs a human at a terminal). |
| #75 liveness | 1: `cs-pending-run`; 2: `cs-start-killed`; 3 and 4: `liveness-kills`, `as-dead-pane-reconcile`; 5: `as-stub-lifecycle`, `fork-inplace-holder`, `resume-continue`; 6 (foreign sessions) and 7 (`forget --holder` of `creating`/`partial`): the `missing`/`unlinked` half in `wtctl-adopt-forget-holders`; the rest not covered. |
| #78 overrides | `cs-dry-run-shapes` (claude-session half), `wtctl-remove-status` (wtctl half). `termctl open --override-escalation` with a changed hook and the slow `pre-teardown` hook: not covered. |
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

- tmux: an unknown command without a server fails at the connect (`error connecting to <sock>`),
  not as `unknown command`.
- wtctl `create --dry-run` over a record the holder already has returns `reused: true`
  (design.md says a dry run never returns `reused`): `wtctl-create-list-show` fails on it.
- wtctl, explicit `--branch` held by another holder under the same `--name`: refusal reason
  `path_exists` (the path check runs first), not `branch_in_use`; under another `--name` the
  same holder gets its existing record back (`reused: true`, the record keeps the old name).
- wtctl `baseline check` change: `error.code` is `refused` and `error.reason` is
  `repo_config_changed` (with `details.changed_classes`).
- claude-session `verify-secret` of an unknown session answers like a wrong secret
  (`11`, `{"match": false}`), not exit 10; an id prefix needs 4 characters after `cs-`.
- claude-session takes `--json` after the command (a leading `--json` is exit 2).
- `stop` of a SIGTERM-ignoring child took 11.6 s with the default `kill_grace` of 10 s, inside
  agent-spawn's `DEFAULT_GRACE` plus margin.
- agent-spawn accepts only `pre-launch` and `post-launch` as `--hook` names; the other hooks
  come from config.

Everything else the `tools` wave checks passed. The per-scenario details are in the test
docstrings; run with `LIVE_OBSERVATIONS=live.jsonl` to get the observed values.

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
