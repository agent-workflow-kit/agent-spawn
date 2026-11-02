# agent-spawn: implementation plan

Status: planned, 2026-11-02. Implements [interface.md](interface.md) (normative) using the internal design in [design.md](design.md). Work is tracked as GitHub issues in this repository; each issue names its stage below. When an issue is picked up, its implementer expands it into bite-sized test-first steps (write the failing test, see it fail, implement, see it pass, commit) before writing code.

## Conventions for every stage

- Test-first. Every behavior starts as a failing test.
- Every subprocess call (claude-session, termctl, wtctl, `crontab`), signal, pidfd, `prctl`, socket peer-credential read, `/proc` read and clock read goes through `src/agent_spawn/gateway.py` (and `procs.py` on top of it); unit tests replace them with fakes.
- Fake binaries for `claude-session`, `termctl`, `wtctl` and `tmux` live in `tests/fakebin/`. Their canned outputs are validated against the siblings' schemas copied into `tests/sibling-schemas/<tool>/<version>/` and pinned to their interface versions (claude-session, termctl, wtctl 0.1.0-design). Fakes log their invocations so tests assert exact argv (flags, actor, decision ids, `--hook`, `--outcome-file`, `--label agent=<id>`).
- Process tests run real processes (stand-ins for `exec` and `claude`, a private tmux server `tmux -S <tmp>/sock -f /dev/null`) against a temporary `XDG_STATE_HOME` and `XDG_CONFIG_HOME`; they run in CI and are not `live`. Only tests that need real Claude Code, the real sibling tools, jobsctl or macOS are `live`.
- Every `--json` output gets a golden-file test validated against `schemas/` (a `referencing` registry keyed by `$id`); every hook payload, log and audit record written in a test is validated too.
- Every spike finding that design.md "Live verification" lists as relied on (spikes A–D) becomes a regression test in the stage that relies on it.
- Every lifecycle transition is a compare-and-set under the agent lease (interface.md, Lifecycle transitions); tests race two processes on each one.
- `make verify` passes before every commit. PR titles are Conventional Commit headers; no `!` before 1.0.
- Runtime dependencies: `pyyaml`, `pydantic`; `textual` only in the `tui` extra. Test-only: `jsonschema`, `referencing`.

## Module map

| Module | Stage | Responsibility |
|---|---|---|
| `cli.py` | 0 | argparse surface (the full, frozen command and flag set), global flags, human-only gate, dispatch |
| `output.py` | 0 | result objects, error envelopes (`lower_error`), exit codes, `--result-file` (atomic, no-follow) |
| `config.py` | 0 | YAML + `config.d` merge, Pydantic models, defaults equal to `config.example.yaml`, `AGENT_SPAWN_*` overrides, stored absolute config path |
| `ids.py`, `labels.py`, `validate.py` | 0 | `ag-` ids and prefix resolution; label grammar, multiset, reserved keys; names, durations, actors, decision ids |
| `gateway.py` | 0 | the only boundary for subprocesses, signals, sockets, `/proc`, clocks |
| `procs.py` | 0 | process identity (PID, start time, boot id; zombie = dead), pidfd with the `ctypes` `syscall` fallback (kqueue on macOS), `pidfd_send_signal`, double fork with PID pipe and acknowledgement, signal forwarding helpers |
| `store.py` | 0 | `state.db` schema, additive migrations, schema-version checks, leases (agent, delivery, reconcile), compare-and-set helpers, outbox writes |
| `audit.py` | 0 | `log.jsonl`, `audit.jsonl`, `export` |
| `session_backend.py` | 1 | the agent session role over claude-session: `start`/`start --resume [--fork]`, dry-run classification, `status`, `stop`, `forget`, `label`, `verify-secret`, `list --label`, `reconcile`; outcome-file handling |
| `supervise.py` | 1–3 | run loop, started polling, outcome table, automatic report resume, chain rule, ephemeral teardown, marker application, outbox draining; host mode (stage 3) |
| `relay.py` | 1 | `_session-event`: records events only |
| `hooks.py` | 1 | hook processes, selectors, payloads, delivery lease, per-invocation hooks |
| `launch.py`, `resume.py` | 1–4 | launch and resume orders, pre-start checks, resume failures, resume without a session, forks |
| `third_party.py` | 1–2 | classification and the cwd rule (interface.md, Third-party runs) |
| `worktrees.py` | 2 | wtctl calls, records of a cwd, holds, deterministic retry |
| `teardown.py`, `prune.py` | 2 | teardown order with named steps, prune |
| `term.py` | 3 | termctl calls (open, dry-run preflight, mark/unmark, focus, close, labels, list `--dead`) |
| `handoff.py` | 3 | peer-checked one-shot socket (listener and client) |
| `stop.py` | 1, 3 | `stop` action selection |
| `liveness.py`, `reconcile.py` | 4 | inline liveness checks and `reconcile` |
| `cron.py` | 4 | `cron install`/`uninstall` |
| `tui/` | 6 | `browse` (Textual, `tui` extra) |

## Stage 0: Foundations

**Goal**: CLI skeleton, config, grammars, ids, labels, state store, logging, output and the process-identity layer, with no agent behavior yet.
**Success Criteria**: `agent-spawn --version` works; every command parses its full flag set and exits 2 on usage errors with the error envelope; human-only flags exit 2 without a TTY or `--actor human:*`; config loads with drop-ins and fails fast with `config_invalid` naming the key; `state.db` opens in WAL mode, migrates additively and refuses a newer schema (`schema_newer`); leases are taken over from dead holders; `--result-file` is written atomically without following links.
**Tests**: argparse coverage of every command and flag from interface.md; id prefix resolution (2/10/21, tombstones excluded); label grammar, 64-label cap, reserved `agent`/`kind`, no-op add/remove; duration grammar incl. bare-number strings; config precedence and invalid-config messages; migration and `schema_newer`; lease takeover (dead PID, reused PID with another start time, zombie); process identity parsing of `/proc/<pid>/stat` with spaces and `)` in `comm`; pidfd and the `syscall` fallback; error-envelope and result-file golden files.
**Status**: Not Started

## Stage 1: Headless agents without worktrees (0.1)

**Goal**: A headless agent with `--worktree none` can be launched, waited on, resumed, forked, stopped, labeled, listed and torn down; the review-run and gatectl-style flows work end to end against fakes.
**Success Criteria**: launch and resume follow interface.md "Launch order" and "Resume order" for headless placement (detached supervisor, `--wait` waiting and forwarding SIGTERM, `--wait-timeout` 22 leaving the agent running); classification by claude-session's dry run and the post-start check; the cwd rule for session-dir runs; started detection by polling `phase: session`; the outcome table, automatic report resume and chain rule; the relay records events and the supervisor delivers `post-state-change`, `post-launch`, `post-escalate`, `post-park` through the outbox; agent-spawn's `pre-launch` hook; `wait`, `list`, `show`, `ack`, `label add|remove` (incl. self-labeling with `verify-secret`), `stop` (session and supervisor actions), `teardown` without a worktree, `export`.
**Tests**: golden `--json` for every command in scope; exact claude-session argv (relay hook with `--state-dir`/`--config` and shell quoting, outcome file, `--label agent=<id>`, explicit `--mode`); classification cases (carried `--reads-third-party`, resume without `--kind`, fork of a `respond` source, dry-run failure starts nothing); post-start mismatch cancels the pending run; the full outcome table incl. split `killed` and `not-started` rows; chain rule for a vetoed automatic report resume; `first_working_at`; outbox at-least-once delivery and redelivery after a killed supervisor; SIGTERM chain launcher → supervisor → `exec` stand-in → child (process test, spike B5/C4); detached supervisor redirects stdio and never holds the caller's pipe (spike C3); `wait` never forwards signals; self-labeling refusals (tainted, third-party run, wrong session, bad secret file).
**Status**: Not Started

## Stage 2: Worktrees, throwaways and teardown (0.2)

**Goal**: Agents get, hold and release wtctl worktrees; third-party runs are confined to session dirs and throwaways; teardown and prune remove work only when nothing would be lost. The ticket-work and review-respond flows work against fakes.
**Success Criteria**: `--worktree new|<path>` per launch step 4 with deterministic retry (`--name <agent_id>`, `wtctl list` lookup, `bootstrap_failed` record kept); records of a cwd and holds for every run, including forks and `--worktree none --cwd`; the cwd rule (`third_party_cwd`, `third_party_run`) per interface.md "Third-party runs"; `show` reports `repo_config_changed` from `wtctl baseline check`; teardown order with named steps, `worktree_partial`, multiple holds, dead-pane step skipped without windows; `prune`; forks into throwaways (`--fork --cwd`) with the fork record inserted at allocation and released on failure.
**Tests**: exact wtctl argv (holder, actor, `--yes`, `--merged-head` full sha, `--force` only from a human); retry after a crash between `create` and the record write finds the record; `creating` record parks with the `wtctl reconcile` hint; third-party run with a worktree cwd refused before allocation; trusted run in a throwaway refused (launch, resume, fork, automatic report resume, promotion); symlinked cwd resolved by realpath; wtctl missing with no stored wtctl path versus with one; teardown refusals passed through with `lower_error`; `partial` keeps `tearing_down`; held record refuses the original's teardown until the fork is torn down; prune age from `last_live_at` or `created`, protections, continuation of interrupted teardowns.
**Status**: Not Started

## Stage 3: Windows (0.3)

**Goal**: Interactive agents run in tmux windows through termctl, receive the launcher's environment over the peer-checked handoff, show the attention marker, and can be promoted from headless; `--placement here` runs in the current terminal.
**Success Criteria**: interface.md "Environment handoff" (listen before `termctl open`, accept after `pid`, peer PID and start time, `pid: null`/`gone`, framing, 4 MiB cap, deadlines); window-hosted `supervise` forwards SIGHUP/SIGTERM, ignores SIGINT/SIGQUIT, never writes to its terminal, exits 0 after bookkeeping; markers applied by the supervisor from relay records; `attach`; promotion through `supervise --host` with the preflight before `start`; dead promotion panes closed; `--placement here` implies `--wait` and supports `--result-file`; `stop` of a window agent.
**Tests**: process tests against a private tmux server: handoff success, thief connection rejected, wrong start time rejected, early connection queued in the backlog, `pid: null`, slow `post-open` hook, payload over 4 MiB, deadline (spike A); closing the window parks `killed` without attention and closes the chain within seconds (spike B2/B3); SIGKILLed supervisor leaves `supervisor_lost` (spike B9); `pane_pid` binding (spike A8); terminal variables taken from the pane; marker set on `needs-input`/`idle` and cleared on `working` only for the current run's window; termctl label cap (30 agent labels after `kind=` and `agent=`); `attach` `switched`/`attach`/`no_window`/details.status; preflight failure exits 12 before any hook or allocation.
**Status**: Not Started

## Stage 4: Lifecycle robustness and operations (0.4)

**Goal**: Every crash, interruption, race and reboot is recovered without signaling anything automatically; escalations can be overridden by a human; agent-spawn installs into cron and survives upgrades.
**Success Criteria**: interface.md "Lifecycle transitions", "Liveness", "Resume failures", "Resume without a session", "Escalation overrides" (routing to `wtctl create`, `termctl open`, claude-session), `pre-teardown` hook, `reconcile` (incl. `claude-session reconcile`, label re-sync, tombstone and file housekeeping, `fork_abandoned`, `ambiguous_session`), `session_missing`, `cron install|uninstall`, upgrades with live agents.
**Tests**: two concurrent resumes, resume versus teardown, resume versus `stop`: exactly one wins; launcher killed before and after the handoff; supervisor killed between runs; crash between `start` and the `csession_id` write recovered by `list --label agent=<id>`; a launching agent whose launcher died with a pending run is left to `never_adopted`; liveness never calls `claude-session stop`; read-only commands never deliver hooks; escalation override routed to exactly one tool and refused without a target; reconcile idempotent; cron block install/replace/uninstall; an older `_session-event` against a newer schema.
**Status**: Not Started

## Stage 5: Live suite and macOS (0.5)

**Goal**: Every relied-on third-party behavior is re-checked against real Claude Code, tmux, the real sibling tools and jobsctl; macOS paths are checked.
**Success Criteria**: a `live` suite (deselected by default, never in CI) covering: full interactive launch in a window and window close; headless escalation with promotion and continuation; resume and fork into a throwaway; teardown with real wtctl; `jobsctl cancel --force` through a job running `launch --wait`; `claude` resumable after kill-pane and SIGTERM (spike D). The macOS checklist (headless, `--placement here`, `--worktree none`, kqueue waits, `LOCAL_PEERPID` handoff) passes manually and is recorded in design.md "Live verification".
**Tests**: the live scenarios above, each with `claude --model haiku` and small bounds.
**Status**: Not Started

## Stage 6: TUI (0.6)

**Goal**: `agent-spawn browse` lets a human triage agents; every action maps to a CLI command.
**Success Criteria**: views and keys of interface.md "TUI"; `--focus`; `--result-file` records `attach`/`resume` intents without performing them; exits 2 with the install hint without Textual; runs as a child of another TUI.
**Tests**: Textual pilot tests for each key (attach, resume with `--mode interactive`, ack, stop, teardown with the TUI's own confirmation, wait, filter, refresh); result-file intents validated against `schemas/cli/browse-result.json`; untrusted strings displayed escaped.
**Status**: Not Started

## Version roadmap

- **0.1** (stage 1): headless agents without worktrees; enough for review-run and gatectl-style callers.
- **0.2** (stage 2): worktrees, throwaways, teardown and prune; enough for ticket-work (headless) and review-respond.
- **0.3** (stage 3): windows, handoff, markers, attach, promotion, `--placement here`; enough for interactive ticket-work and ticket-create.
- **0.4** (stage 4): crash, race and reboot recovery, escalation overrides, cron; feature-complete for every workflow consumer and the harness.
- **0.5** (stage 5): live suite and macOS checks.
- **0.6** (stage 6): `browse`.
- **1.0**: stages 0–6 done; the v1 review, ticket and response slices have run through agent-spawn without interface changes; the live suite and the macOS checklist passed; the interface is declared stable (a deliberate breaking-change commit per architecture section 18.3).
- **1.x** (additive): a cgroup- or scope-based detached supervisor option (`systemd-run --user --scope`) for hosts with `KillUserProcesses=yes`; agent-spawn `pre-resume` hook if a policy needs one; a second termctl placement target when termctl supports panes.
- **2.0**: a second agent session backend (binding C, architecture section 3.5), or any non-additive state or interface change.
