# agent-spawn: design

Status: Seeded from awkit architecture v1.4; not yet designed.

## Role and layer

Layer 2 (composite). Launch, resume, park and tear down Claude agents: worktree + terminal + Claude session. Works fully standalone (architecture C2).

## Responsibilities (architecture §5.2)

Launch, resume, park and tear down agents: worktree + terminal + Claude session.

- Commands: `launch`, `resume`, `list [--label k=v] [--needs-attention]`, `show`, `attach`, `ack <agent_id>`, `label add/remove`, `teardown`, `prune`, `reconcile`, `browse`; internal: `supervise`.
- Uses: claude-session; termctl (interactive window placement only); wtctl (only when a worktree or a local clone is needed).
- Notes: Flags and defaults are in section 9.2. Authoritative label store for agents. Owns post-exit policies through its supervise wrapper. Does not use jobsctl. Never notifies directly: it records attention as data (`needs_attention`) and fires hooks (`post-escalate`, `post-park`, `post-state-change`). Agent lifecycle is separate from window lifecycle (section 9). Records repo-config hashes at launch (section 13.2 rule 10) and reports `repo_config_changed` in `show --json`. `resume --fork` records the fork as a new agent (section 9.4).

Section numbers above refer to `../awkit-docs/architecture.md`.

## Uses

- claude-session: binding TBD (A/B/C), decided in this design session
- termctl (interactive window placement only): binding TBD (A/B/C), decided in this design session
- wtctl (only when a worktree or a local clone is needed): binding TBD (A/B/C), decided in this design session

## Cross-cutting sections to read

In `../awkit-docs/architecture.md` (plus `../awkit-docs/design-guide.md` throughout):

- §3.2: shareability: termctl only for interactive window placement, wtctl only when needed; a headless `--worktree none` launch needs neither
- §8: the Claude session model (ids, kinds and per-call overrides in §8.1; `exec` in §8.2; outcomes in §8.3; states in §8.4; resume, fork and taint in §8.5)
- §9 (all): agent lifecycle: this tool's primary contract (summary below)
- §10: key flows A-D, all of which launch or resume agents through this tool
- §12: autonomy and gates; §12.4 `--yes` (teardown may pass `--yes` to `wtctl remove` but never `--force`)
- §13: security: permissions and environment (§13.1), repo-config hashes and git safety (§13.2 rule 10), credentials (§13.3)
- §15: notifications: `post-state-change`, `post-escalate`, `post-park` are the recommended producers

## Lifecycle contracts owned (architecture §9, summary)

Read architecture §9 in full; this is an index, not a substitute.

- **§9.1 Agents and windows:** agent ≠ window; statuses *live*, *parked*, *torn down* (tombstones; `resume` of a torn-down agent exits 10); authoritative label store (mirrors labels onto the session, records `job=<id>` from the env var named by `job_id_env`, default `JOBSCTL_JOB_ID`); self-labeling via `label add --session <id>` authenticated with the session secret and limited to allowlisted keys (default `pr`); attention recorded as `needs_attention` (`unreported`, `escalated`, `blocked`, `failed`, `lost`) with hooks `post-escalate`, `post-park`, `post-state-change`.
- **§9.2 Launch, hosting, waiting and jobs:** repo-config baseline at launch; launch flags (`--kind`, `--skill`, `--plugin-dir`, `--prompt-file`/`--prompt`, `--repo`, `--branch`, `--ref`, `--worktree new|<path>|none`, `--cwd <path>|session-dir`, `--mode`, `--placement window|here`, `--ephemeral`, `--wait`, `--label`, plus §8.1 per-call overrides); defaults `--mode interactive`, `--placement window`, `--worktree none`; process hosting per mode; signal handling and the SIGTERM cancellation chain; `--wait` exits with the outcome code; never uses jobsctl.
- **§9.3 Supervision and post-exit policies:** status prefix via `termctl rename`; done; unreported exits (headless: one automatic resume, then escalate; interactive: park); reported failed; killed; promotion headless → interactive; held; lost.
- **§9.4 Resume, attach and acknowledge:** `resume` passes through `claude-session resume` flags; `--fork` creates a new agent (labels only from `--label`, `forked_from=<agent_id>`, same repo-config baseline, never owns the worktree); exit 21 on ambiguous selectors, 20 `live`, 10 torn down; `attach`; `ack`.
- **§9.5 Teardown and prune:** removes only worktrees it created; always refuses live sessions (also when any agent in the worktree is live); refuses `repo_config_changed` before any git command; passes `--yes` (and a human's `--force`) to `wtctl remove`; `--merged-head`; `prune --older-than` with `prune.protect_labels` (default `[owner]`).
- **§9.6 Placement and nesting:** configurable placement end state (v1: one shared session, a window per agent); always starts top-level `claude` processes.

## Constraints (decided)

From architecture §20: Stores each agent's window id and base name (the status prefix is applied to the base name); `supervise` polls `claude-session status` for the status prefix; tmux-hosted `supervise` loads an env file from agent-spawn's state dir (mode 0600, only kind-allowed variables, never `secret_env` or `token_env` variables, deleted immediately after loading); repo-config hashes are recorded at launch and compared before teardown and on `show`; tombstones of torn-down agents are kept for N days (configurable).

## Open topics

From architecture §20:

- placement config
- parked cleanup
- PATH/env
- how secret variables a kind allows reach a tmux-hosted `supervise` without touching disk or argv
- how a human clears a `repo_config_changed` refusal after inspecting the change

## Decisions

| Decision | Rationale | Date |
|---|---|---|

## Questions for the design session

1. What does placement config look like (v1: one shared session, a window per agent; end state per §9.6)?
2. How are long-parked agents cleaned up beyond `prune --older-than` (defaults, protections)?
3. How are PATH and the environment assembled for `supervise` and the session?
4. How do secret variables a kind allows reach a tmux-hosted `supervise` without touching disk or argv?
5. How does a human clear a `repo_config_changed` refusal after inspecting the change?
