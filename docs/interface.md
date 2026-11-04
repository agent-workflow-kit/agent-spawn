# agent-spawn: interface

Status: interface 0.1.0-design, 2026-11-02. Written before implementation (architecture section 18.1).

**This document is the normative source** for agent-spawn's behavior. [design.md](design.md) explains how it is implemented, and the architecture (sections 9 and 20) links here instead of restating these rules. Where another document differs, this one wins.

Normative sections: [Agent statuses](#agent-statuses), [Launch order](#launch-order), [Resume order](#resume-order), [Environment handoff](#environment-handoff), [Supervision](#supervision), [Outcome table](#outcome-table), [Promotion](#promotion), [State relay](#state-relay), [Signals](#signals), [Third-party runs](#third-party-runs), [Teardown order](#teardown-order), [Liveness](#liveness), [`stop`](#stop), [Output by exit code](#output-by-exit-code).

Conventions: design guide sections 4 to 10. JSON Schemas (draft 2020-12) are in [schemas/](../schemas/); `$ref`s between them are relative (from `cli/` and `hooks/` e.g. `../defs.json#/$defs/agent`), so validate with a registry holding every schema. Every command that prints data supports `--json`; with `--json`, errors are printed on stdout as an error envelope ([schemas/cli/error.json](../schemas/cli/error.json)).

agent-spawn uses three sibling tools, each through its documented CLI and `--json` output (binding A): claude-session (always), termctl (only for windows) and wtctl (for worktrees, and to find the wtctl records of a run's cwd). Their interfaces are normative for their own behavior; this document cites their sections by name.

**Every lower-tool call** agent-spawn makes carries these, stated once here: every `claude-session start` (new, `--resume`, `--fork`) is preceded by its [pre-start checks](#third-party-runs) and carries the relay hook (`--hook post-state-change=<relay argv>`, [State relay](#state-relay)), `--label agent=<agent_id>`, an outcome file (`--outcome-file`, [Files](#files)) and an explicit `--mode`; every `termctl open` carries `--json`, `--actor` and the decision ids; every call to a command that accepts `--actor` and `--decision-id` carries the actor and decision ids of [Global flags](#global-flags). Every path agent-spawn compares or passes as a cwd is a realpath.

## Concepts

- **Agent**: one Claude Code session launched by agent-spawn, together with its optional worktree and window, identified by an `agent_id`. An agent maps to exactly one claude-session session (`csession_id`) once its session is allocated. A fork is a new agent.
- **Run**: one claude-session run of the agent's session. Every run is supervised by `supervise`.
- **Adopted**: claude-session's adoption compare-and-set for the run succeeded, so the run has a wrapper (`claude-session exec`). claude-session's `pre-launch` hooks and launch validation run after adoption.
- **Started**: claude-session reports the run's `phase: session` (`claude` was spawned; `claude-session status <csession_id> --json` `run.phase`, or the run's outcome object). A run waiting at Claude Code's trust dialog is started. The supervisor observes it by polling ([Supervision](#supervision)); relay records are only an early wake-up. Clearing `needs_attention`, the `launching` → `live` transition and `post-launch` use **started**.
- **Supervisor**: the `agent-spawn supervise` process of the agent's current run chain (the run, an automatic report resume and a promotion). There is at most one per agent.
- **Window**: a termctl managed pane hosting the supervisor (interactive, `--placement window`) or a promoted run (`supervise --host`). Windows are disposable; the agent outlives them. Every pane agent-spawn opens is labeled `agent=<agent_id>`. Each run records its own window, or none.
- **Worktree**: a wtctl worktree record the agent holds: created (`--worktree new`, owned), adopted (`--worktree <path>`), or held because a run's cwd lies inside it ([Holding](#records-of-a-cwd)); always with the agent's id as a holder. The agent object's `holds` lists every record the agent holds now; its `worktree` is the record it created or adopted at launch, else the nearest record it holds for its cwd.
- **Third-party-reading run**: a run that claude-session's dry run classifies as reading third-party content ([Third-party runs](#third-party-runs)). Every other run is **trusted**.

### Identifiers

| Id | Grammar | Notes |
|---|---|---|
| `agent_id` | `^ag-[a-z2-7]{8}$` | Allocated by `launch` (and by `resume --fork`); never reused, also after tombstone deletion. |
| `csession_id` | claude-session's | Recorded from `claude-session start`'s success output or its exit-12 `launch_failed` envelope (both carry `csession_id` and `run`), or by the [liveness](#liveness) recovery; never by the relay. |
| `term_id` | termctl's (`tc-…`) | Stored per run with `socket_path` and the pane PID. |
| `wt_record_id` | wtctl's (`wt-…`) | The record created or adopted at launch step 4, stored from every wtctl result that names it ([Launch order](#launch-order) step 4). Records held for a cwd are stored as holds. |

Input: the `ag-` prefix is optional; a unique prefix of at least 4 characters after it is accepted; fewer exits 2, no match exits 10, several exit 21 (`candidates`). Prefixes never match torn-down agents; a full id of a torn-down agent is shown by `show`, returns its last outcome from `wait` ([`wait`](#wait)), and makes every other command exit 10 with `details.status: torn_down` (callers retrying `teardown` treat this as done).

### Names

An agent's **name** is the display name of its windows: 1 to 64 Unicode code points, no control characters (C0, DEL, C1), not containing `#[` (termctl's name rules). `--name` is validated against them (exit 2). The name is `--name`, else `--worktree-name`, else `<kind> <first 6 characters of the id after ag->`, where `<kind>` is the explicit `--kind` or the word `agent`. It is fixed when the record is allocated ([Launch order](#launch-order) step 3) and is never null. agent-spawn never derives a name from labels or from third-party text. The name is stored and used for every window of the agent (resume, promotion); `resume --name` replaces it.

### Labels

A label is `key=value`, split at the first `=`; key `^[a-z0-9][a-z0-9_.-]{0,63}$`; value 0 to 256 characters from `[A-Za-z0-9._:/#@+=,-]`; at most 64 pairs per agent (exit 2 beyond, except for `job=`, [Labels on agents](#labels-on-agents)). Labels are a multiset (design guide section 9): adding a pair the agent already has, or removing a pair it does not have, is a no-op. agent-spawn assigns no meaning to any key; it writes `job=<id>` ([Labels on agents](#labels-on-agents)) and passes `agent=<agent_id>` and `kind=<run kind>` to lower tools ([Labels on lower tools](#labels-on-lower-tools)). The keys `agent` and `kind` are reserved: `--label`, `label add` and `label remove` with either key exit 2.

`job=<id>` identifies only retries of the same job. A workflow finds its agent by its own stable labels plus its own records (the `agent_id` it stored), not by `job=` alone; a caller that abandons a failed launch tears the agent down.

### Records of a cwd

A run's cwd is first resolved with `realpath`; every containment rule below and in [Launch order](#launch-order) uses realpaths, and agent-spawn passes the realpath as `--cwd`. A path A **contains** a path B when A equals B or is a path-component ancestor of it.

- **Records**: every live wtctl record (worktree or throwaway, any state that is not a tombstone) whose recorded `path` contains the cwd, from one `wtctl list --json` call (record fields `id`, `path`, `object_type`, `state`, `holders`). Nested records (possible only for adopted records) are all included, and every rule applies to each of them.
- **Lookup errors fail closed**: a non-zero `wtctl list` exit or unparseable output refuses the command before anything changes (exit with wtctl's code, `tool: wtctl`).
- **wtctl availability**: config `wtctl_cmd: null` means there is no wtctl on this machine (standalone use): the records of every cwd are empty, a third-party-reading run needs `--cwd session-dir` (any other cwd exits 20 `third_party_cwd`), and the worktree flags (`--worktree` other than `none`, `--branch`, `--base`, `--ref`, `--detach`, `--worktree-name`, `--no-bootstrap`, `--copy-env`) exit 2. Otherwise wtctl is resolved from `wtctl_cmd`, else from the absolute wtctl path stored with the agent; when a lookup of the records of a cwd is needed and wtctl cannot be resolved, the command exits 12 `tool_missing` before anything changes. There is no other standalone fallback.

The records of a cwd drive these rules:

- **Holding.** An agent never runs inside a wtctl worktree record it does not hold. Before each run's `claude-session start` (launch, resume, fork), the agent holds every worktree record of its cwd that it does not hold yet: `wtctl adopt <record path> --holder <agent_id> --json` (`existing: true` means the holder was added to the existing record); a hold failure maps as a failed `adopt` in [Launch order](#launch-order) step 4. Throwaways are never held. Holds are recorded on the agent (`holds`; `owned: false` unless it created the record); teardown releases every one of them ([Teardown order](#teardown-order)).
- **Third-party-reading runs** must run in their own session dir or at the root of a throwaway; **trusted runs** are refused inside any throwaway ([Third-party runs](#third-party-runs)).

## Agent statuses

| Status | Meaning |
|---|---|
| `launching` | a launch, or a resume, is in progress and its run is not started yet; the launching process's PID and start time are recorded |
| `live` | the current run is started and its supervisor is alive, or its session is live |
| `parked` | no live session; resumable (includes finished headless agents and failed launches) |
| `tearing_down` | `teardown` is in progress or was interrupted (`teardown_step` records how far it got) |
| `torn_down` | tombstone: worktree removed or released, session forgotten; kept `tombstone_days` |

Transitions: `launching` → `live` when the run is started; `launching` → `parked` on a launch or resume failure ([Launch order](#launch-order), [Resume failures](#resume-failures)) or when the run ends before it started; `live` → `parked` when the supervisor has applied the [outcome table](#outcome-table) (or the [liveness](#liveness) check did so for a dead supervisor); `parked` → `launching` on `resume`; `parked` → `tearing_down` → `torn_down` by `teardown`; `tearing_down` → `parked` when teardown is vetoed, escalated or refused at a step that changed nothing ([Teardown order](#teardown-order)). Every park, by any process, sets `promotion_pending: false` in the same transaction.

**`needs_attention`** is null or one of `unreported`, `escalated`, `blocked`, `failed`, `timeout`, `lost`, `launch_failed`, `session_missing`. Applying the [outcome table](#outcome-table) sets it to the table's value ("none" clears it). It is also set by a failed launch or resume, by [liveness](#liveness) (`lost`, `session_missing`) and by a refused, failed or launch-phase-ended supervisor-initiated run ([chain runs that stop early](#supervision)); it is cleared when a later run of the agent is started, by `teardown` and by `ack`. `post-escalate` fires each time it changes to a non-null value (setting the value it already has fires nothing).

**`supervisor_lost`** is a separate flag: true while the agent is `live`, its supervisor is dead and its session is still live ([Liveness](#liveness)). Liveness sets `needs_attention: lost` only on the transition into `supervisor_lost`; `ack` clears `needs_attention` but not `supervisor_lost`, which clears when the run is final.

**`park_reason`** records why the agent last parked: `done`, `blocked`, `escalated`, `failed`, `unreported`, `killed`, `timeout`, `lost`, `vetoed`, `launch_failed`, `session_missing`. **`escalated_by`** names the tool whose hook escalation parked the agent (`wtctl`, `termctl`, `claude-session` for a launch-phase `escalated` run); it is non-null only when a hook escalation parked the agent, so it is null after a `needs-input` park ([Escalation overrides](#escalation-overrides)).

### Lifecycle transitions

Every lifecycle entry is a compare-and-set under the agent's lease. The liveness check that precedes it, which may call lower tools, runs without the lease; the compare-and-set then re-reads the row under the lease. No lower-tool call or hook runs while the lease is held.

| Entry | Compare-and-set | Loser |
|---|---|---|
| `resume` (plain) | `parked` → `launching`, recording the launcher's PID and start time and `expect_run_after` (the session's current run number, null without a session); after the liveness check, before any other lower-tool call | exit 20 `live` (status `launching` or `live`) or `tearing_down` |
| `resume --fork` | inserts the fork's record (status `launching`, [Resume order](#resume-order) step 3) | (not contended: a new id) |
| `teardown` | `parked` → `tearing_down` with `teardown_step: started` and the teardown process's PID and start time; before `pre-teardown` | exit 20 `live` or `tearing_down` |
| supervisor, run started | `launching` → `live` | (not contended) |
| supervisor, ephemeral `done` | `live` → `tearing_down` (`teardown_step: started`, the supervisor recorded as the teardown process) | (not contended: the agent is live) |
| park after a failed placement | `status = launching` → `parked` | does not park (another process already did) |
| `ack`, `label`, `stop` | read and write under the lease | none |

A `tearing_down` agent whose recorded teardown process is dead may be entered again by `teardown` or `prune` (a retry, [Teardown order](#teardown-order)); one whose teardown process is alive refuses with 20 `tearing_down`.

## Launch order

`launch` performs these steps in order. "Park" means: set status `parked` with the given `needs_attention` and `park_reason`, record the `post-park` event (and `post-escalate` when `needs_attention` changed to a non-null value), and exit with the code stated; the error envelope carries `agent_id`.

1. **Usage checks and preflight** (no agent record is created), in this order:
   - exit 2: flags and grammars; human-only flags without a TTY or `--actor human:*` ([Human-only flags](#global-flags)); `--worktree new` without `--repo`; `--branch`/`--base`/`--ref`/`--detach`/`--worktree-name`/`--no-bootstrap`/`--copy-env` without `--worktree new`; `--cwd` absolute with `--worktree new`; `--cwd session-dir` with `--worktree new` or `--worktree <path>`; `--cwd` whose realpath lies outside the worktree's realpath with `--worktree <path>`; a worktree flag with config `wtctl_cmd: null` ([Records of a cwd](#records-of-a-cwd)); `--placement here` in interactive mode without a TTY on stdin and stdout; `--wait-timeout` with `--placement here`; `--prompt` together with `--prompt-file`; a headless launch with neither `--skill` nor a prompt; a reserved label key; a `--hook` name other than `pre-launch` or `post-launch`; the config.
   - **Tool resolution** (exit 12, `tool_missing`, naming the install command): claude-session always; termctl when a window will be opened; wtctl when `--worktree` is not `none`, and otherwise whenever the lookup of the [records of a cwd](#records-of-a-cwd) needs it (unless config `wtctl_cmd` is null; its availability rule).
   - **Classification** ([Third-party runs](#third-party-runs)): the `agent_id` is allocated in memory and the prompt (`--prompt` text, or the `--prompt-file`, including `-`) and every `--input-file` are copied into `<state>/prompts/<agent_id>/` ([Files](#files)); later steps pass these copies, never the caller's paths, and any exit before step 3 deletes the directory. Then claude-session's dry run classifies the run (third-party-reading or trusted) and gives its `repo` and, without a worktree and without `--cwd`, its `cwd` (the kind's default, possibly `session-dir`).
   - **Pre-start checks 2 to 4** ([Third-party runs](#third-party-runs)): the records of the cwd (`--worktree <path>` whose realpath lies inside a throwaway record then exits 2, as wtctl `adopt` would after allocation); for a third-party-reading run the cwd rule (`third_party_cwd`; with `--worktree new` or `--worktree <path>` it always refuses) and the result-file check; for a trusted run the throwaway check (`third_party_run`). With `--worktree new` the cwd does not exist yet: a trusted run's records are looked up at step 4 (wtctl never creates a worktree inside a throwaway).
   - **Window preflight** (when a window will be opened): `termctl open --dry-run --json` with the same `--name`, `--label`s and `--placement` as the real open (and `--cwd` when the directory exists; never for `session-dir`, whose directory does not exist yet), `-- <absolute agent-spawn path> supervise`, runs termctl's tmux checks. termctl exit 2 (for example `not_supported`) exits 2 (`code: usage`, `tool: termctl`); any other failure exits 12 (`code: failed`, `tool: termctl`, `lower_code` termctl's code such as `tmux_missing`, `tmux_too_old` or `server_unavailable`, hint "use `--placement here` or `--mode headless`").
2. **`pre-launch` hooks** ([Hooks](#hooks)): veto exits 11, escalation exits 13; nothing is created and no agent record exists. `launch --override-escalation` overrides these hooks only.
3. **Allocate** the agent record with the id of step 1: status `launching`, the launcher's PID and start time, the resolved launch settings and [name](#names), the actor and decision ids (used by every later internal call, [Global flags](#global-flags)), the absolute paths of claude-session, termctl and wtctl and of the config file in use, the per-invocation `--hook` entries, and the labels plus `job=<id>` ([Labels on agents](#labels-on-agents)).
4. **Worktree.**
   - `--worktree new`: `wtctl create --repo <id> --holder <agent_id> --name <worktree name> [--branch] [--base] [--ref] [--detach] [--no-bootstrap] [--copy-env] --json`, where the worktree name is `--worktree-name`, else the `agent_id` (deterministic, so a resume finds the record, [Resume order](#resume-order) step 5).
   - `--worktree <path>`: `wtctl adopt <path> --holder <agent_id> [--repo <id>] --json`.
   - `wt_record_id` is stored from every result that names the record: success and `bootstrap_failed` (`details.record`).
   - On a non-zero exit of `wtctl create`, park and exit with wtctl's code (envelope `tool: wtctl`, wtctl's error in `lower_error`, [Error envelope](#error-envelope)): 11 → `park_reason: vetoed`, no attention; 13 → `escalated` (`escalated_by: wtctl`); 12 `busy` → `park_reason: launch_failed`, **no** `needs_attention`, `details.retryable: true`; 12 `bootstrap_failed` → `launch_failed` (the record is stored; a resume continues at step 5, after a human may have run `wtctl bootstrap`); a record left `creating` by a crash → `launch_failed` with the hint "run `wtctl reconcile`"; any other → `launch_failed`.
   - A failed `wtctl adopt` (`--worktree <path>`, or a hold below) parks and exits with wtctl's code the same way: 12 `busy` → `park_reason: launch_failed`, **no** `needs_attention`, `details.retryable: true`; any other exit → `launch_failed`. (`adopt` fires no hooks and has no `--override-escalation`, so it never exits 11 or 13.)
   - Resolve the cwd: the worktree root, or `--cwd` resolved inside it; with `--worktree none`, `--cwd`, else the classification plan's `cwd` (the kind's default, possibly `session-dir`); then its realpath. With `--worktree new`, a trusted run's pre-start checks 2 and 4 run now (a lookup failure parks `launch_failed` and exits with wtctl's code).
5. **Holds** (pre-start check 5, [Third-party runs](#third-party-runs)): agent-spawn [holds](#records-of-a-cwd) every worktree record of the cwd it does not hold yet; a hold failure parks as a failed `adopt` in step 4.
6. **Session.** `claude-session start --json` with: the claude-session launch flags given ([Launch flags](#launch-flags)); `--repo` (when known); `--cwd` (the realpath of step 4, or `session-dir`); `--label` for every agent label plus `agent=<agent_id>`; the relay hook; the outcome file; `--prompt-file` and `--input-file` with the copies of step 1. The `csession_id` and run are recorded from the success output or from the exit-12 `launch_failed` envelope, and the copies are deleted once a session was allocated. On success agent-spawn reads `claude-session status <csession_id> --json` once, records the run's kind, `reads_third_party`, the session's `tainted` and the agent's `repo`, and runs the classification check ([Third-party runs](#third-party-runs); a mismatch it cannot accept, or a failed `status`, exits 12 `launch_failed` with `details.cause` `classification_mismatch` or `classification_unknown`). On a non-zero exit: park and pass the code through (`tool: claude-session`): 12 → `launch_failed` (`code: launch_failed` when claude-session reported `launch_failed`, with its `end_reason` and `hint` in `lower_error`; else `code: failed`); 20 → `launch_failed`; any other non-zero exit (1, 2, 10, 21) → `launch_failed` with `needs_attention: launch_failed`, exiting with claude-session's code.
7. **Place** the supervisor:
   - interactive, `--placement window`: listen on the [handoff](#environment-handoff) socket, then `termctl open --name <name> --cwd <window cwd> --label … [--placement <term-placement>] --keep-on-exit failed --json [--override-escalation] -- <supervise argv>`, where `<supervise argv>` is `<absolute agent-spawn path> supervise <agent_id> --handoff <socket> --handoff-timeout <seconds> --state-dir <absolute state dir> --config <absolute config path>` ([Internal commands](#internal-commands)); record `term_id`, `socket_path`, and the pane PID (the supervisor) with its start time on the run; complete the handoff. termctl exit 11 → park `vetoed` (no attention); 13 → park `escalated` (`escalated_by: termctl`); any other failure, or a handoff failure (exit 12, `handoff_failed`) → park `launch_failed`.
   - interactive, `--placement here`: run the supervisor in this process (implied `--wait`).
   - headless: start a detached supervisor ([Supervision](#supervision)): double fork; the grandchild writes its PID, start time and boot id to a pipe, the launcher records them under the agent's lease and writes an acknowledgement. After the acknowledgement the grandchild closes every inherited descriptor except stdin, stdout and stderr (already redirected) and execs `<absolute agent-spawn path> supervise <agent_id> --state-dir <absolute state dir> --config <absolute config path>` (no handoff; the environment is inherited). exec keeps the PID and the start time, so the recorded identity stays valid; the new process opens the state database only after exec and spawns `exec` after its startup check. A pipe failure, or a pipe closed before the acknowledgement, is a failed detached fork.
   - The **window cwd** (`<window cwd>`, also in [Promotion](#promotion) step 2): the run's cwd; for `session-dir` the realpath of `<csession_dir>/work`, from `start`'s output (`csession_dir`) or `claude-session status`. Every lower-tool `--cwd` other than `claude-session start`'s uses this path.
   - After a placement failure the launcher runs `claude-session stop <csession_id>` on the pending run only when no supervisor exists (`termctl open` failed, the handoff never completed, or the detached fork failed); then no process can adopt the run. Exactly one process parks the agent (the compare-and-set of [Lifecycle transitions](#lifecycle-transitions)).
8. **Adopt and start.** The supervisor runs `claude-session exec --session-id <csession_id>`, which adopts the run, and records `exec`'s PID and start time. When the run is started ([Supervision](#supervision) step 2), the supervisor sets status `live`, clears `needs_attention` and records `post-launch`. A claude-session `pre-launch` veto or escalation, or a launch validation failure at adoption, ends the run in the launch phase and is handled by the [outcome table](#outcome-table) like any other outcome.
9. **Return.** The launcher observes agent-spawn's own state (written by the supervisor), running the [liveness](#liveness) check on each poll and using the supervisor's pidfd only as an early wake-up. Without `--wait`: once the run is started, exit 0 with the [result object](#objects) (`run: null`); if the run is final first, exit with its [outcome code](#outcome-table) and the result object when it was adopted, else (end reason `canceled` or `never_adopted`) with the error envelope (`code: launch_failed`, exit 12). Before adoption the wait is bounded by claude-session's pending-run expiry (`never_adopted`); after adoption by claude-session's `pre-launch` hook timeouts and launch validations, after which the run is started or final. With `--wait` (and with `--placement here`): when the agent is no longer live, no promotion is pending and the supervisor has exited, exit with the outcome code of the last run and the result object; `--wait-timeout` expiring first exits 22 (`timed_out: true`), leaving the agent running.

### Launch flags

Passed through to `claude-session start` unchanged (claude-session's interface, section Launch flags, is normative for them): `--kind`, `--mode`, `--skill`, `--plugin-dir`, `--model`, `--timeout`, `--max-turns`, `--permission-profile`, `--sandbox`, `--setting-sources`, `--reads-third-party`, `--env-allow`, `--env-allowlist-only`, `--auto-exit`/`--no-auto-exit`; on resume also `--reset` and `--allow-tainted`. `--prompt-file` and `--input-file` are passed as private copies (step 1). `--mode` defaults to `interactive` (claude-session's default) and is always passed explicitly. `--kind` on resume follows claude-session's carry rule: a kind given now becomes the previous run's kind for the next resume; agent-spawn never re-passes a stored kind (except the [resume without a session](#resume-without-a-session), which re-runs the launch). `--actor` and `--decision-id` are [global flags](#global-flags). `--override-escalation` is agent-spawn's own ([Escalation overrides](#escalation-overrides)).

agent-spawn's own launch flags:

| Flag | Default | Meaning |
|---|---|---|
| `--repo <repo-id>` | none | Required with `--worktree new`; passed to wtctl and claude-session. |
| `--worktree new\|<abs path>\|none` | `none` | [Launch order](#launch-order) step 4. A relative path exits 2. |
| `--branch <name>`, `--base <branch>`, `--ref <sha>`, `--detach`, `--worktree-name <name>`, `--no-bootstrap`, `--copy-env` | wtctl's defaults | Passed to `wtctl create` (`--worktree-name` as `--name`, default the `agent_id`); only with `--worktree new`. |
| `--cwd <path>\|session-dir` | with a worktree: its root; without: the classification plan's `cwd` (the kind's default, possibly `session-dir`) | With `--worktree new`, a relative path resolved inside the worktree (absolute exits 2). With `--worktree <path>`, its realpath must stay inside the worktree's (exit 2). `session-dir` only without a worktree (exit 2). Without a worktree: passed through. |
| `--prompt <text>` | none | Trusted text only, for humans; copied to a private file and passed as `--prompt-file`. Never used for untrusted text (design guide section 4.1). |
| `--prompt-file <path>`, `--input-file <path>` | none | Copied at classification (step 1); the copies are passed. |
| `--placement window\|here` | `window` | Interactive only; ignored with a warning for headless. `here` implies `--wait`. |
| `--term-placement <name>` | none (termctl's rules) | Passed to `termctl open --placement`; stored. |
| `--name <text>` | [Names](#names) | Window name; stored. |
| `--label key=value` | none | Repeatable; adds labels. Keys `agent` and `kind` exit 2. |
| `--promote` / `--no-promote` | config | [Promotion](#promotion); the explicit choice is stored (`promote`: true, false or null). |
| `--ephemeral` | off | Tear down after a run that ended `done` ([Supervision](#supervision)); stored. The session directory is forgotten with it, so `run.report.artifact.path` no longer exists afterwards: a caller that needs the artifact does not use `--ephemeral`, and tears down after copying it. Never inherited by a fork. |
| `--wait` | off | Block until the agent is no longer live, no promotion is pending and the supervisor has exited; exit with the outcome code. |
| `--wait-timeout <duration>` | none | With `--wait`: give up waiting (exit 22); the agent keeps running. Not with `--placement here` (exit 2). |
| `--result-file <path>` | none | [Result file](#result-file). |
| `--hook <name>=<cmd>` | none | Per-invocation agent-spawn hook (`pre-launch`, `post-launch`); [Hooks](#hooks). Never passed to claude-session. |
| `--override-escalation` | off | Human-only; [Escalation overrides](#escalation-overrides). Audited. |
| `--dry-run` | off | [`launch`](#launch). |

Durations: a positive number of seconds (fractions allowed), or `<number><unit>` with unit `ms`, `s`, `m`, `h` or `d`; in flags and the environment a bare number string means seconds. Zero exits 2.

## Resume order

`resume (<agent_id> | --select k=v...) [--fork]` with resume's own flags ([`resume`](#resume)):

1. **Usage checks** (exit 2): flags and grammars; human-only flags as in launch; `--repo`, the worktree flags (`--worktree`, `--branch`, `--base`, `--ref`, `--detach`, `--worktree-name`, `--no-bootstrap`, `--copy-env`) and `--ephemeral` are refused; `--cwd` only with `--fork`; `--prompt` together with `--prompt-file`; `--placement here` without a TTY on stdin and stdout; `--wait-timeout` with `--placement here`; `--hook pre-launch` (agent-spawn's `pre-launch` does not gate a resume; claude-session's `pre-launch` gates every resume and fork); a reserved label key. There is no skill or prompt requirement.
2. **Resolve** the selector: an id, or `--select` pairs (ANDed, agents that are not torn down; none exits 10, several 21). A torn-down agent exits 10.
3. **Lifecycle entry.** The [liveness](#liveness) check, then:
   - plain resume: the compare-and-set `parked` → `launching` ([Lifecycle transitions](#lifecycle-transitions)); a loser exits 20 (`live`, or `tearing_down`). An agent whose session is reported missing exits 10 (`details.status: session_missing`).
   - `--fork`: the source may be live (a fork never changes its source); a `tearing_down` source exits 20 `tearing_down`; a source without a session exits 12 `no_run`. The fork's record is inserted: a new `agent_id` (reserved like any agent id), status `launching`, no session, `forked_from`, the launcher's PID and start time, the source's repo, name, `term_placement` and `promote`, never `ephemeral`.
4. **Resolve the run.**
   - Mode: `--mode`, else the mode of the agent's (a fork: the source's) last run that was not a promotion; always passed explicitly to claude-session. A human answering a headless agent passes `--mode interactive` (the TUI's `r` does). Placement: `--placement`, else `window` (never an inherited `here`). `--name`, `--term-placement` and `--promote`/`--no-promote` replace the stored values.
   - Cwd: plain resume, the agent's recorded cwd. Fork: `--cwd`; else, when the source's cwd is its session's `work/` directory (recorded symbolically as `session-dir`, or equal to the source's `csession_dir`/`work`), `--cwd session-dir` (the fork's own `work/`; claude-session refuses such a fork without `--cwd`, `fork_cwd_unsupported`); else the source's recorded cwd, passed explicitly.
   - Skill: plain resume only: when the session has never had a started run (every run ended with `phase: launch`) and no `--skill` is given, the stored launch `--skill` is passed (claude-session carries the undelivered prompt and inputs, but not the skill). A fork passes only the `--skill` given.
   - `--override-escalation` without a target exits 2 ([Escalation overrides](#escalation-overrides)).
   - Prompt and input files are copied into `<state>/prompts/<agent_id>/` (the fork's id for a fork).
   - **Tool resolution** as in launch step 1, by the resolved mode and placement.
   - **Classification**: claude-session's dry run of the planned `start --resume` ([Third-party runs](#third-party-runs); an agent without a session: [Resume without a session](#resume-without-a-session)); its `would_refuse` (`live`, `tainted`, ...) exits 20 with that reason (`tool: claude-session`). Then pre-start checks 2 to 4 as in launch step 1 (for an agent without a recorded cwd, after step 5; a third-party-reading run with a stored worktree is refused at once, [Resume without a session](#resume-without-a-session)).
   - **Window preflight** as in launch step 1, by the resolved mode and placement.
5. **Worktree** (only for an agent without a session whose launch failed at or before step 4 of [Launch order](#launch-order); [Resume without a session](#resume-without-a-session)):
   - `wt_record_id` set: `wtctl show <wt_record_id> --json`: `active` → skip `create`; a tombstone → exit 12 `worktree_gone`; `creating` → park `launch_failed` with the hint "run `wtctl reconcile`"; another live state → `worktree_state` ([Resume without a session](#resume-without-a-session)).
   - `wt_record_id` unset: `wtctl list --repo <id> --json`, then the live record whose `name` equals the planned worktree name and whose `holders` include this agent: `active` → store `wt_record_id`, skip `create`; `creating` → park `launch_failed` with the hint "run `wtctl reconcile`"; another live state → `worktree_state`; none → launch step 4 with the stored settings. (wtctl refuses an explicit `--name` collision with `branch_exists` or `path_exists` before any reuse, so agent-spawn finds its record itself.)
6. **Holds**: Holding (pre-start check 5, [Third-party runs](#third-party-runs)) for the run's cwd, as in launch step 5. A fork takes its holds under the fork's id before its `start`.
7. **Session.**
   - Agent without a session: launch steps 6 to 9 with the stored settings and the overrides of [Resume without a session](#resume-without-a-session).
   - Plain resume: `claude-session start --resume <csession_id> --mode <mode> --json` with the given pass-throughs, `--label` for the newly given labels and `job=` ([Labels on agents](#labels-on-agents)), the relay hook, an outcome file and the prompt copies.
   - `--fork`: `claude-session start --resume <source csession_id> --fork --mode <mode> --cwd <cwd> --json` with the pass-throughs, `--label` for every fork label (the given `--label`s, plus `job=` only from an explicit `--label job=`) plus `agent=<fork agent_id>`, the relay hook, an outcome file and the prompt copies. A failure before a fork session exists (any refusal or failure of steps 4 to 6, or a refused `start`) releases the holds the fork took (`wtctl forget <record> --holder <fork id>`; `not_a_holder` counts as released), deletes its prompt copies and deletes the fork's record (the id stays reserved), then exits with the failure's code (`agent_id: null`, `tool` of the failing tool); no pending run exists then, so nothing is canceled. `start` exit 12 `launch_failed` (a fork session was allocated) parks the fork `launch_failed` with its `csession_id`; its next resume is a plain resume of the fork.
   - After `start`: the copies are deleted once a session was allocated; the run's kind, `reads_third_party`, the session's `tainted` and the agent's `repo` are recorded and the classification check runs, as in launch step 6.
8. **Place**, **start** and **return** as in [Launch order](#launch-order) steps 7 to 9. A window is opened only in interactive mode with placement `window`.

A resume clears `needs_attention` when its run is started. agent-spawn's `pre-launch` hooks never run on resume.

### Resume without a session

An agent without a session is always a launch whose session was never allocated (a fork without a session never remains, [Liveness](#liveness)). Its resume re-runs the launch:

- **Classification** (step 4): the dry run is the launch's `claude-session start ... --dry-run --json` (no `--resume`), built from the stored launch settings and pass-throughs plus the flags given now; `--cwd` when the directory exists, else `--repo`.
- **Mode**: `--mode`, else the stored launch mode. **Skill**: `--skill`, else the stored launch `--skill`.
- **Prompt and inputs**: a newly given `--prompt`/`--prompt-file` replaces the stored prompt copy, and newly given `--input-file`s replace the stored input set (claude-session's carry rule); otherwise the stored copies are passed. A failed resume never deletes the copies; they are deleted only once a session is allocated, or at teardown.
- **Order**: the classification at step 4 runs before any wtctl call of step 5. A run classified third-party-reading whose stored worktree mode is `new` or `<path>` is refused there with exit 20 `third_party_cwd` (first [Resume failures](#resume-failures) row), before `wtctl create` or `adopt`.
- **Cwd**: when the agent has no recorded cwd, it is resolved after step 5 exactly as in launch step 4 (the worktree root, or the stored `--cwd` inside it; with `--worktree none`, the stored `--cwd` or the plan's `cwd`) and recorded before the holds. The pre-start checks that need the cwd (the records, the cwd rule, the throwaway check and the result-file check) run then; with a worktree the third-party cwd rule always refuses, as at launch.
- **Step 5 states**: a live record (from `wtctl show <wt_record_id>`, or the name lookup when `wt_record_id` is unset) in a state other than `active` or `creating` (`unlinked`, `missing`, `clone_missing`, `partial`) parks `launch_failed` with `needs_attention: launch_failed` and exits 12 (`code: failed`, `tool: agent-spawn`, `details.cause: worktree_state`, `details.retryable: false`, hint "`wtctl show <id>`; a human repairs or forgets the record"). A tombstone stays exit 12 `worktree_gone` ([Resume failures](#resume-failures), first row).
- **Session** (step 7): launch steps 6 to 9 with the stored settings and the overrides above.
- **`resume --dry-run`**: the plan is built from this dry run.

### Resume failures

| Failure | Agent afterwards | `needs_attention` |
|---|---|---|
| Any failure before a run of this resume exists: refusals (20 `live`, `tainted`, `tearing_down`, `third_party_cwd`, `third_party_run`, claude-session's `would_refuse`), 10 for the agent itself, usage and tool errors, the window preflight, the classification dry run, the records lookup, wtctl 12 `busy` of the holds or of step 5 (exit 12, `details.retryable: true`), other wtctl failures of the holds or of step 5 not in the next row, claude-session `start` exits 1, 2, 10 (other than the session missing), 11, 20 and 21, a launcher interrupted before a session existed | plain resume: the pre-resume row is restored in one transaction (launcher fields cleared, `updated` unchanged), no events, no `post-park`; a fork: no fork exists (step 7) | unchanged |
| claude-session `start` exit 12 (a run was allocated: it is recorded in `runs[]` with its `not-started` outcome, updating `run`, `last_outcome` and `last_end_reason`); a placement or handoff failure; wtctl 11 or 13 of step 5's `create`; a `creating` record or another non-`active` live state in step 5 ([Resume without a session](#resume-without-a-session)) | `parked`, `park_reason: launch_failed` (termctl and wtctl 11/13 as in launch: `vetoed`, `escalated`) | `launch_failed` (replacing the previous value; `post-escalate`), none for `vetoed` |
| the classification check after `start` cancels the run ([Third-party runs](#third-party-runs): a mismatch it cannot accept, or a failed `status`) | `parked`, `park_reason: launch_failed`, `details.cause` `classification_mismatch` or `classification_unknown` | `launch_failed` |
| claude-session exit 10 for the agent's session | `parked`, `park_reason: session_missing`; exit 10 with `details.status: session_missing` | `session_missing` |
| the run ends in its launch phase | the [outcome table](#outcome-table) | per the table |

For `--fork`, every failure before its session exists is the first row (no fork remains, step 7).

### Escalation overrides

`--override-escalation` is human-only and audited. It is passed only to the tool that escalated, on its next call:

- `launch --override-escalation`: overrides agent-spawn's own `pre-launch` hooks (exit 13 from any of them counts as allow; a veto still vetoes). There is no agent yet, so nothing else is overridden.
- `resume --override-escalation`, by the agent's `escalated_by`: `wtctl` → `wtctl create --override-escalation` (step 5); `termctl` → `termctl open --override-escalation`; `claude-session` → `claude-session start --resume --override-escalation`, only when the session's last run ended `escalated`. Without a target (`escalated_by` null, as after a `needs-input` park, or `claude-session` while the last run did not end `escalated`) the flag exits 2.
- `resume --fork --override-escalation` ignores the source's `escalated_by`: the override is passed only to the fork's `termctl open` (the call that can have escalated on the human's previous attempt, which left no fork; `wtctl adopt` never escalates), never to claude-session (which refuses `--override-escalation` with `--fork`).
- `teardown --override-escalation`: overrides agent-spawn's `pre-teardown` hooks and is passed to `wtctl remove`.

Every pre-hook payload of an overriding invocation has `override_escalation: true`, so a policy can veto it. The override is not bound to the escalation the human saw ([Residual risks](#residual-risks)).

## Environment handoff

A tmux window inherits the tmux server's environment, not the caller's. agent-spawn therefore hands its own environment to the process in a window over a one-shot socket; nothing is written to disk, argv or tmux's environment.

1. Before `termctl open`, the launching process (the launcher, or the supervisor for a promotion) creates a fresh directory `<state>/handoff/<random>/` (mode 0700), writes an `owner` file in it (its PID, process start time and boot id; `reconcile` removes a directory whose owner is dead) and listens on a Unix stream socket `s` in it. The socket path must fit the platform limit (exit 12, `handoff_failed`, otherwise).
2. The window runs `<absolute agent-spawn path> supervise <agent_id> --handoff <socket path> --handoff-timeout <seconds> --state-dir <absolute state dir> --config <absolute config path>` (with `--host` for a promotion, [Internal commands](#internal-commands)).
3. The launching process `accept`s only after `termctl open --json` returned; connections made earlier wait in the backlog. `pid: null` or `state: gone` in termctl's result fails the handoff at once (`handoff_failed`, `details.cause`: `no_pane_pid` or `pane_gone`). Otherwise it records the start time of process `pid` and answers only the connection whose peer uid is the user's, whose peer PID equals `pid` and whose process start time equals the recorded one; on Linux it reads the peer with `SO_PEERCRED`, on macOS with `LOCAL_PEERPID` and `getpeereid`. Every other connection is closed without data and logged (`handoff_rejected`). The peer check is the authentication.
4. Framing: the window process connects and sends nothing; the launching process writes one JSON object, `{"env": {name: value, ...}}` (its own environment minus `TERM`, `TERM_PROGRAM`, `TERM_PROGRAM_VERSION`, `COLORTERM`, `TMUX` and `TMUX_PANE`), at most 4 MiB, terminated by `shutdown(SHUT_WR)`. Both sides then close; the launching process removes the socket and its directory (a missing file counts as removed).
5. The window process builds its environment from `env` plus those six variables from its own (pane) environment, then continues ([Supervision](#supervision)). A missing, empty, oversized or malformed payload makes it exit 1 without starting anything, logging `handoff_failed`. Its deadline to connect and read is `--handoff-timeout` plus 60 s (termctl's post-open hooks may delay the launcher); on expiry it exits 1 with `handoff_failed` in its log.
6. The launching process's deadline is `handoff_timeout` (default `30s`) from `termctl open` returning; on expiry it removes the socket and the launch fails ([Launch order](#launch-order) step 7).

agent-spawn does not filter the environment: what reaches `claude` is decided by claude-session per kind (allowlist, `secret_env`, `claude_auth_env`). A window-hosted agent therefore gets the environment of whoever launched or resumed it, exactly as a headless or `--placement here` agent would. Under jobsctl this is the job's environment, including `--pass-env` values. Internal processes take the state dir and config from their argv, never from `XDG_STATE_HOME` or `XDG_CONFIG_HOME`.

## Supervision

`agent-spawn supervise <agent_id> [--handoff <socket> --handoff-timeout <seconds>] [--host] --state-dir <path> --config <path>` is internal and the only supervisor of the agent's runs. On start, in this order: install the [signal](#signals) handlers; (window) complete the handoff and adopt the received environment; open the state dir given in argv; verify that the agent's record names this process as its supervisor (PID, start time, boot id) and that the agent is `launching` or `live` (`--host`: that a promotion is pending). A mismatch exits 2 and is logged. Then:

1. Write nothing to the terminal; log to `<state>/supervise/<agent_id>.log`.
2. Run `claude-session exec --session-id <csession_id>` as a child (with the terminal as its stdin, stdout and stderr in interactive mode; `/dev/null` and the log otherwise). Record `exec`'s PID and start time. Poll `claude-session status <csession_id> --json` every second until `run.phase` is `session` or the run is final (a relay record is an early wake-up). When the run is started, set `launching` → `live` (compare-and-set; a later run of the chain: status stays `live`), clear `needs_attention` and record `post-launch` for that run (also for an automatic report resume and a promotion). While it lives, every second, the supervisor applies the window markers of its current run from the recorded state (for an interactive run with a window: `needs-input` or `idle` → `termctl mark <term_id> attention --socket <socket_path>`; `working` → `termctl unmark <term_id> --socket <socket_path>`; termctl makes repeats no-ops; a failure is a warning) and delivers the agent's outbox ([Hook delivery](#hook-delivery)).
3. When `exec` exits, read the run's outcome object from the outcome file; if it does not parse, or its `csession_id` or `run` differs from the run this supervisor adopted, fall back to `claude-session status <csession_id> --run <n> --json`. A null `first_working_at` is filled from that run's `recent_events` (`claude-session status <csession_id> --run <n> --json`, the run's first event with `to_activity: working`), else stays null. Record the run in the agent's history and apply the [outcome table](#outcome-table):
   - **automatic report resume** (headless `exited-unreported`, once per run chain, when `auto_report_resume` is true and the end reason is not `no_session_start`): the [pre-start checks](#third-party-runs), then `claude-session start --resume <csession_id> --mode headless --prompt-file <fixed prompt> --json` (prompt: "Your last run ended without a status report. File your status report now with claude-session report."), the classification check, then back to step 2 with the new run. The auto-resumed run goes through the outcome table and may be promoted once.
   - **promotion** ([Promotion](#promotion)), then wait for the promoted run;
   - otherwise park.

   The automatic report resume and promotion are skipped when the run's end reason is `stopped`, `forwarded_term`, `forwarded_hup` or `forwarded_int`, or when the supervisor received SIGTERM or SIGHUP during this run chain; the agent then parks per the outcome table (a reported run keeps its outcome).

   **Chain runs that stop early.** A supervisor-initiated run (automatic report resume or promotion) that is refused or fails before or at its `start` (pre-start checks, classification dry run, `start`, the classification check, the window open or the handoff), or that ends in its launch phase (`not-started` with any end reason, `vetoed`, or `escalated` with `phase: launch`), parks per the [outcome table](#outcome-table) row of the run that triggered it: `needs_attention`, `park_reason`, `escalated_by` and `last_outcome` come from that row and run; the chain run, when it exists, is recorded in `runs[]` with its own outcome; `post-park` and (when `needs_attention` changes) `post-escalate` carry `chain_failure`: `{tool, reason, run, end_reason}`, where `tool` is the tool that refused or failed (`agent-spawn` for its own checks), `reason` the refusal reason or error code, `run` the chain run's number (else null) and `end_reason` its end reason when it ran (else null). `chain_failure` is null for every other park. The [liveness](#liveness) rule for a dead supervisor applies the same rule.
4. **Ephemeral** (`--ephemeral` and a `done` outcome), in this order: record the outcome → compare-and-set `live` → `tearing_down` (`teardown_step: started`), with the supervisor still recorded as the teardown process → [teardown](#teardown-order) from its step 3 under automation rules (no `--force`, no human-only flags; the run is final, so step 2's checks pass for this agent) → `post-teardown` on success; on a refusal, status `parked` (`park_reason: done`) and `post-park` with `teardown_refused: true`, with a warning in the log → the supervisor exits. Waiters return after the supervisor exits.
5. Exit 0 (a non-zero exit means `supervise` itself failed; its window then stays open as a dead pane, `--keep-on-exit failed`).

**Hosting.** The supervisor runs in a window (interactive, `--placement window`), in the foreground of the launching process (`--placement here` only), or detached (every headless run): double fork, `setsid`, stdin `/dev/null`, stdout and stderr to the supervisor log, so it never holds the caller's pipes open; after the launcher acknowledged its PID, the grandchild closes every other inherited descriptor and execs `<absolute agent-spawn path> supervise <agent_id> --state-dir <absolute state dir> --config <absolute config path>` (same PID and start time; no handoff, the environment is inherited), which spawns `exec` after its startup check ([Launch order](#launch-order) step 7). `launch|resume --wait` of a headless agent waits on the detached supervisor (pidfd; kqueue on macOS) and forwards SIGTERM to it ([Signals](#signals)).

**No `exec` child.** While the supervisor has no `exec` child (before it spawned the first one, after a run ended and before the next `start` returned, or while a promotion's window is being opened), a SIGTERM or SIGHUP makes the supervisor cancel its own pending run, if any (`claude-session stop <csession_id>`; no other process can adopt it), skip the automatic resume and promotion, park (before a first run: per the [outcome table](#outcome-table) row of the canceled run, `killed` without attention; else per the previous run's outcome) and exit.

**Host mode** (`supervise --host`, a promotion window): it receives the supervisor's environment, runs the printed `claude-session exec --session-id` argv as a child, forwards SIGHUP and SIGTERM to it, ignores SIGINT and SIGQUIT, never writes to its terminal, and exits 0 after `exec` exits (so promotion windows close). It does no bookkeeping; the supervisor does.

## Outcome table

Applied when a run ends without being promoted (claude-session's outcome codes; its interface, section Outcome mapping, is normative for the outcomes). The agent's `last_outcome` and `last_end_reason` are the run's; applying the table sets `needs_attention` to the table's value ("none" clears it).

| Run outcome (exit) | Agent | `needs_attention` | `park_reason` |
|---|---|---|---|
| `done` (0) | parked, or torn down with `--ephemeral` | none | `done` |
| `blocked` (13) | parked | `blocked` | `blocked` |
| `needs-input` (13) | parked | `escalated` | `escalated` (`escalated_by: null`) |
| `failed` (18) | parked | `failed` | `failed` |
| `exited-unreported` (14), headless | automatic report resume once (not for end reason `no_session_start`), after which the resumed run's row applies; without a resume: parked | `unreported` | `unreported` |
| `exited-unreported` (14), interactive | parked | `unreported` | `unreported` |
| `killed` (15), end reason `stopped`, `forwarded_term`, `forwarded_hup` or `forwarded_int` | parked | none | `killed` |
| `killed` (15), end reason `signal_<name>` or `internal_error` | parked | `failed` | `killed` |
| `timeout` (16) | parked | `timeout` | `timeout` |
| `lost` (17) | parked | `lost` | `lost` |
| `not-started` (12), end reason `canceled` or `launch_interrupted` (a deliberate stop or cancel) | parked | none | `killed` |
| `not-started` (12), any other end reason (`never_adopted`, launch validations) | parked | `launch_failed` | `launch_failed` |
| `vetoed` (11) | parked | none | `vetoed` |
| `escalated` (13, phase `launch`) | parked | `escalated` | `escalated` (`escalated_by: claude-session`) |

A session-phase 13 (`blocked`, `needs-input`) is first offered to [Promotion](#promotion). A launcher interrupted before a session existed parks a launch `killed` without attention (`details.cause: interrupted`) and returns a resumed agent to `parked` unchanged ([Resume failures](#resume-failures)). `launch --wait`, `resume --wait` and `wait` exit with the outcome code of the agent's last run (0 for an ephemeral `done`).

## Promotion

A headless run that ends with a session-phase exit 13 (a `blocked` or `needs-input` report) is **promoted** when the agent's `promote` setting resolves to true and the run chain has not been promoted yet. `promote` is resolved per run kind at promotion time: the stored explicit choice (`--promote`/`--no-promote`), else config `repos.<repo>.kinds.<run kind>.promote`, else `kinds.<run kind>.promote`, default false. From the decision until the promoted run ends or the promotion fails, the agent's `promotion_pending` is true.

1. The [pre-start checks](#third-party-runs) and the window preflight (`termctl open --dry-run`, as in [Launch order](#launch-order) step 1). Then the supervisor runs `claude-session start --resume <csession_id> --mode interactive --json` and the classification check. A refusal or failure parks per the [chain rule](#supervision) (a preflight failure creates no pending run).
2. It opens a window: [handoff](#environment-handoff) socket, then `termctl open --name <name> --cwd <window cwd> --label … [--placement] --keep-on-exit failed --json -- <absolute agent-spawn path> supervise <agent_id> --host --handoff <socket> --handoff-timeout <seconds> --state-dir <absolute state dir> --config <absolute config path>`; records `term_id`, `socket_path` and the pane PID (the host process, [Supervision](#supervision)) with its start time on the promoted run.
3. It waits for the host pane PID to exit (pidfd; kqueue on macOS), then reads the promoted run's outcome file, with `claude-session status <csession_id> --run <n> --json` as the fallback, and applies the outcome table to the promoted run (which is never promoted again); a promoted run that ended in its launch phase parks per the [chain rule](#supervision). The promoted run's `mode` and `placement` are `interactive` and `window`.

If termctl or tmux is missing at the real open, `termctl open` fails (11 and 13 included), or the handoff fails, the pending run is canceled (`claude-session stop`) and the agent parks per the [chain rule](#supervision) (the original run's row, `escalated` or `blocked`, with `chain_failure` set). A promoted run keeps the run's kind profile: third-party-reading kinds stay sandboxed, in `dontAsk`, with their allowlist.

## State relay

agent-spawn learns about session state changes from claude-session's per-invocation hook `post-state-change=<relay argv>`, passed on every `start` ([Concepts](#concepts)), which claude-session delivers at least once, in order per session, including late delivery after the wrapper died. The relay argv is `<absolute agent-spawn path> --state-dir <absolute state dir> --config <absolute config path> _session-event`, every element quoted with POSIX shell quoting (claude-session runs the hook with `sh -c`).

`_session-event` (internal) reads the claude-session payload on stdin and, in one transaction, records only:

1. the event, deduplicated by `(csession_id, event_id)` (claude-session's key); an event whose `object_id` differs from the recorded `csession_id` of the agent named by its `agent=<agent_id>` label, or for an agent whose `csession_id` is unset (only the `starting` event of an in-flight `start` can arrive then), or naming no known agent, is logged and ignored;
2. the agent's `run`, `session_state`, `activity` and `needs_input_reason` only when the event's `run` is at least the agent's recorded `run`; the run's own fields (`first_working_at`, its first `working` activity, and the last state, activity and `needs_input_reason` on its run row) regardless;
3. agent-spawn's `post-state-change` outbox event ([Hooks](#hooks)).

It then commits and exits 0, always (failures are logged, never returned to claude-session). It calls no lower tool and runs no hooks: the supervisor applies window markers and delivers the outbox ([Supervision](#supervision), [Hook delivery](#hook-delivery)). It runs in claude-session's hook environment and uses only the state dir and config named in its argv and the absolute tool paths stored with the agent. It never creates or migrates a database: a missing `state.db` is logged to stderr and it exits 0; with an older database schema it processes the event when the schema is readable, else logs `schema_older` and exits 0 ([Upgrades](#upgrades)). `agent-spawn reconcile` runs `claude-session reconcile` first, which delivers events that were not delivered.

## Signals

Normative. "Forward" means: send once to the exact recorded PID, verified by PID, process start time and boot id (on Linux through a pidfd opened before the start-time check). Handlers are idempotent: a second identical signal is not forwarded again.

| Process | SIGTERM | SIGHUP | SIGINT | SIGQUIT |
|---|---|---|---|---|
| `supervise` with an `exec` child (any hosting) | forward to `exec` (the chain then ends: no automatic report resume or promotion) | forward to `exec` (a closed window signals only the pane's session leader, which is `supervise`; the chain then ends) | interactive: ignore (the terminal delivers it to `claude`); headless: forward to `exec` | ignore |
| `supervise` during a promotion (promoted run in its window) | forward to the host pane PID | forward to the host pane PID | ignore | ignore |
| `supervise` with no `exec` child (before the first, between runs, while a promotion window opens) | cancel its own pending run, skip the automatic resume and promotion, park, exit ([Supervision](#supervision)) | as SIGTERM | interactive: ignore; headless: as SIGTERM | ignore |
| `supervise --host` | forward to `exec` | forward to `exec` | ignore | ignore |
| launcher (`launch`/`resume`) before its run is started, no supervisor yet (the handoff did not complete, or the detached supervisor has not reported its PID) | stop the in-flight lower-tool call (SIGTERM to its exact PID), close the handoff socket, cancel the pending run if one exists (`claude-session stop`), park ([Outcome table](#outcome-table); a fork without a session: delete its record, as in [Resume order](#resume-order) step 7), exit 143 with the error envelope (`code: launch_failed`, `details.cause: interrupted`) | as SIGTERM, exit 130 | as SIGTERM, exit 130 | default |
| launcher before its run is started, a verified supervisor exists | forward SIGTERM to the supervisor and keep waiting (with or without `--wait`) until the agent parked and the supervisor exited; then output and exit as normal ([Output by exit code](#output-by-exit-code)), never 130 or 143 | as SIGTERM | as SIGTERM | default |
| `launch`/`resume --wait` after its run started (the process that started the run chain; supervisor in a window or detached) | forward to the supervisor, keep waiting for the outcome | stop waiting, exit 130; the agent is unaffected | stop waiting, exit 130; the agent is unaffected | default |
| `launch`/`resume --placement here` (supervisor in-process) | the launcher rows until `exec` is spawned, then as `supervise` | as SIGTERM | as SIGTERM | ignore |
| `wait` | stop waiting, exit 143; never forwards | stop waiting, exit 130 | stop waiting, exit 130 | default |

`supervise` never exits on a signal before `exec` has exited and the bookkeeping is done, never writes to its terminal (a write after a hangup fails with EIO), and sends SIGCONT along with each forwarded SIGTERM (a stopped child would otherwise never act on it). A supervisor that forwards to `exec` during claude-session's launch phase lets claude-session end the run `not-started` (`launch_interrupted`, or `canceled` when a stop was requested), which parks without attention. **Cancellation chain:** jobsctl's shim → the job command → `agent-spawn launch|resume` (the process that started the run chain, recorded with its PID and start time, in every phase) → the supervisor → `exec` → `claude`, each forwarding SIGTERM to the exact next PID; a forwarded SIGTERM ends the run `killed` (`forwarded_term`) or, before it started, `not-started`, and the agent parks without attention. `wait` is an observer and never forwards; its pidfd on the supervisor is only a wake-up optimization. `--wait-timeout`, SIGINT or SIGHUP of the waiter, or a SIGKILL of the waiter leave the agent running. agent-spawn completes forwarding and bookkeeping within a few seconds; jobsctl queues that run agents should have a `kill_grace` of at least 30 seconds.

## Third-party runs

agent-spawn never runs a third-party-reading session in a worktree, and never runs a trusted session in a throwaway: a third-party-reading run's cwd is its own session dir or equals the root path of a live wtctl throwaway record (and lies inside no worktree record); a trusted run's cwd lies inside no throwaway record. A subdirectory of a throwaway is refused for a third-party-reading run: another session sharing the throwaway could replace a component of that path with a symlink while the run is live, whereas the root's parent (wtctl's throwaway root) is not session-writable. wtctl defines throwaways as never trusted (wtctl's interface, Records). wtctl's own repo-config baseline checks (at `remove`, and before a tool-performed push) are unchanged; agent-spawn runs no baseline check of its own before a run. These rules use the [records of a cwd](#records-of-a-cwd).

**Classification.** Before every `claude-session start` it makes (launch, plain resume, fork, automatic report resume, promotion), agent-spawn runs the same command as a dry run: `claude-session start [--resume <id> [--fork]] <the same launch flags and pass-throughs> --dry-run --json`, without `--hook`, `--hook-timeout` and `--outcome-file`, with the private prompt copies, and with `--cwd` when the directory exists (for `--worktree new`, before the worktree exists, `--cwd` is omitted and `--repo` passed; classification does not depend on the cwd). The run is **third-party-reading** when the plan's `profile.reads_third_party` is true or its `taints` is true (the profile includes a flag-implied or carried `--reads-third-party`). claude-session's dry run exits 0, 12 (`would_fail` set) or 20 (`would_refuse` set) with a plan, and agent-spawn uses the plan in all three cases: `would_refuse` refuses the command with exit 20 and that reason (`tool: claude-session`) before anything changes; `would_fail` is not acted on (the real `start` reports it). Any other result (exit 2 or 10, an error envelope, unparseable output) fails the command with claude-session's code (`tool: claude-session`) before anything changes, so no run starts. The plan's `run_kind` and `repo` are recorded with the run.

**Pre-start checks.** Before every `claude-session start` agent-spawn makes (launch, plain resume, fork, automatic report resume, promotion), in this order:

1. the classification dry run;
2. the [records of the cwd](#records-of-a-cwd) (one `wtctl list --json`, also when the run's cwd is unchanged, as in a run chain);
3. a **third-party-reading** run: the **cwd rule**: its cwd must be its own session dir (`session-dir`, or equal to the realpath of its own `<csession_dir>/work`), or equal (as realpaths) the root path of a live throwaway record and lie inside no worktree record; otherwise exit 20 `third_party_cwd` (`tool: agent-spawn`, message "third-party-reading runs run only in their session dir or at a throwaway's root"); a subdirectory of a throwaway is refused too. With `--worktree new` or `--worktree <path>` the cwd is a worktree, so the launch is always refused. Then the result-file check ([Result file](#result-file));
4. a **trusted** run: the **throwaway check**: a cwd inside any live throwaway record exits 20 `third_party_run` (`tool: agent-spawn`, message "throwaways are for third-party-reading runs only");
5. Holding of every worktree record of the cwd ([Records of a cwd](#records-of-a-cwd); throwaways are never held).

Where each check runs: a launch runs checks 1 to 4 at [launch](#launch-order) step 1, before anything is allocated (with `--worktree new`, a trusted run's checks 2 and 4 at step 4, once the cwd exists), and check 5 at step 5; a resume runs checks 1 to 4 at [resume](#resume-order) step 4 (an agent without a recorded cwd: after step 5) and check 5 at step 6; a supervisor-initiated run runs all five in the supervisor before its `start`, and a refusal or failure parks per the [chain rule](#supervision) (`chain_failure.reason`, for example `third_party_run`). A refusal is logged `third_party_refused` (`details.reason`). A dry run performs checks 1 to 4 read-only and takes no lease.

**Classification check.** After every `start` returns (the run is pending and no supervisor exists yet, so nothing has executed), agent-spawn reads `claude-session status <csession_id> --json` and compares `run.reads_third_party` (or `tainted_by_run` equal to this run) with the classification:

- They agree: continue.
- They differ, and the run's cwd satisfies the rule for its actual classification (a third-party-reading run in its session dir, or a trusted run outside every throwaway record): when the actual classification is third-party-reading, the result-file check ([Result file](#result-file)) runs now; if it passes, continue and log `classification_mismatch` at level `warn`; if it fails, handle the run as in the next case.
- They differ otherwise (a run classified trusted that reads third-party content outside its session dir or whose `--result-file` fails the third-party rule, or a run classified third-party-reading that is trusted inside a throwaway): cancel the pending run (`claude-session stop`; no process can adopt it yet), park `launch_failed` (attention set) with `details.cause: classification_mismatch`, log `classification_mismatch` and exit 12 (`code: launch_failed`, `tool: agent-spawn`). A supervisor-initiated run is canceled the same way and parks per the [chain rule](#supervision) (`chain_failure.reason: classification_mismatch`).
- `status` fails or prints unparseable output: cancel the pending run, park `launch_failed` with `details.cause: classification_unknown`, log `classification_unknown` and exit 12 (`code: launch_failed`, `tool: agent-spawn`). A supervisor-initiated run parks per the chain rule (`chain_failure.reason: classification_unknown`).

**`show`** reports `third_party_run` (whether the agent's cwd lies inside a live throwaway record, so trusted runs of the agent are refused) and `repo_config_changed` from `wtctl baseline check` of the agent's own worktree record, for information ([`show`](#show)).

These rules cover only runs launched through agent-spawn ([Residual risks](#residual-risks)).

## Teardown order

`teardown <agent_id> [--merged-head <sha>] [--force] [--override-escalation] [--hook NAME=CMD]... [--dry-run] [--actor <actor>] [--decision-id <id>]... [--json]`:

1. **Resolve and check usage.** A full id of a torn-down agent exits 10 with `details.status: torn_down` (done). `--merged-head` must be a full 40-hex sha (exit 2 otherwise); `--hook` names other than `pre-teardown` and `post-teardown` exit 2.
2. **Lifecycle entry and live checks** (exit 20, `reason: live`; nothing overrides them, not even `--force`): the [liveness](#liveness) check; the agent is live or `launching`, or a promotion is pending; the agent has a session whose current run `claude-session status` shows pending or live. Then the compare-and-set `parked` → `tearing_down` (`teardown_step: started`); a `tearing_down` agent whose teardown process is dead is a **retry** and continues where `teardown_step` shows (its teardown process is alive: exit 20 `tearing_down`).
3. **`pre-teardown` hooks**: veto 11, escalation 13 (re-run with `--override-escalation`, human-only); either returns the agent to `parked` unchanged. A retry runs them only when `teardown_step` is `started` (they never completed), else skips them and logs `teardown_retry`. Then the live checks of step 2 run again (a refusal returns the agent to `parked`, exit 20 `live`). Marker `hooks_done`.
4. **Worktree** (skipped when the agent holds no record, or when `teardown_step` shows it done). Records are processed in this order: every held record in stored order, then the owned record last:
   - held (adopted, held for a cwd, or an in-place fork): `wtctl forget <record> --holder <agent_id> --json`. If the agent's `adopt` had found an existing record with no other holders, releasing the agent's hold forgets that record (wtctl removes no files);
   - owned (created by the agent): `wtctl remove <record> --yes --holder <agent_id> [--merged-head <sha>] [--force] [--override-escalation] --json`.

   wtctl exit 0, or exit 10 with `details.state` `removed`, `forgotten` or `gone`, completes a record, and its hold row records its completion (`released`); when every hold row is released the marker is `worktree_done`. wtctl exit 1 `partial` (the removal deleted part of the tree) sets `worktree_partial` and stores the id of the record that went partial: the agent stays `tearing_down` and the command exits 12 `partial` (`tool: wtctl`). While the marker is `worktree_partial`, a retry (or `prune`) first runs `wtctl show <that record> --json`: a tombstone (`removed`, `gone`, `forgotten`) → that record is complete and teardown continues; anything else → the agent stays `tearing_down`, exit 12 `partial` (it never returns to `parked`; `prune` skips it, `reconcile` reports `teardown_incomplete`, and a human finishes with `wtctl remove`). Otherwise any other result (refusals such as `held`, `dirty`, `unpushed`, `in_use`, `repo_config_changed`, `unlinked`, `locked`; exit 12 or 13) changed nothing for that record: the agent returns to `parked` and the command exits with wtctl's code (`tool: wtctl`, wtctl's error, with its `checks` and `processes`, in `lower_error`). Records released earlier stay released, and the next run's Holding re-adds them (when the release forgot an adopted record, the re-adoption makes a new record, whose id replaces the old one on the hold row and, if it was the launch record, in `worktree.record_id`). Because the owned record is last, an agent never returns to `parked` with its worktree removed.
5. **Windows** (skipped when no run of the agent had a window): for each distinct socket path of the agent's runs' windows, `termctl list --label agent=<agent_id> --dead --socket <socket_path> --json`, then `termctl close <term_id> --socket <socket_path>` for each dead pane listed. Exit 10 and exit 20 `running` are no-ops; any other failure is a warning. Marker `windows_done`.
6. **Session** (skipped without a session, or when it is reported missing): `claude-session forget <csession_id>` (exit 10 counts as done). A failure (e.g. 20 `live` from an orphaned child) stops the teardown with the marker unchanged (status stays `tearing_down`, exit with claude-session's code; for 20 `live` the hint "`agent-spawn stop <agent_id>`"); a retry continues here. Marker `session_done`.
7. **Tombstone**: status `torn_down`, `needs_attention` cleared, the agent's prompt copies and outcome files deleted, `post-teardown`, log record; an audit record when a worktree was removed or `--force` was given.

`teardown_step` is a named marker: `started` after the compare-and-set of step 2, then `hooks_done`, `worktree_partial`, `worktree_done`, `windows_done`, `session_done` (null when not tearing down), so a retried teardown continues where it stopped. "Changed nothing, back to `parked`" applies only while the marker is not `worktree_partial`. `--force` is human-only and is passed only to `wtctl remove`, where it overrides `held`, `dirty`, `unpushed` and a failed fetch; it never overrides `live`, `in_use`, `repo_config_changed`, `unlinked` or `locked`. `--merged-head` is trusted by wtctl as given ([Residual risks](#residual-risks)). `--dry-run` performs steps 1 and 2 without the compare-and-set and, for the owned record, `wtctl status <record> --holder <agent_id> [--merged-head <sha>] --json`, and prints the plan (one entry per held and owned record); it exits 20 whenever `would_refuse` is set (agent-spawn's `live` or `tearing_down`, or a refusal `wtctl status` predicts), else 0.

## Liveness

`list`, `show`, `wait`, `launch|resume` (on each poll), `resume`, `teardown`, `stop`, `prune` and `reconcile` check the agents they touch inline:

- the supervisor (including a host process), `exec` and the launcher are checked by PID, process start time and boot id; a zombie counts as dead;
- `launching` with **no supervisor recorded**: a live launcher → nothing changes; a dead launcher → the rules for no `csession_id`, an interrupted resume and a pending run below;
- `launching` with no supervisor recorded, a dead launcher and no `csession_id`: `claude-session list --label agent=<agent_id> --json`; exactly one match is recorded as the agent's session (a crash between `start` and its recording), then the rules below apply; more than one is reported (`reconcile` change `ambiguous_session`) and the agent parks `launch_failed` without a session; none found: an agent with `forked_from` set is an **abandoned fork**: its holds are released (`wtctl forget <record> --holder <agent_id>`; `not_a_holder` and exit 10 count as released; any other failure is a warning, and the record and its holds stay for the next check), its prompt copies are deleted and its record is deleted (the id stays reserved), logged `fork_abandoned` and reported as `reconcile` change `fork_abandoned`; any other agent parks `launch_failed`;
- `launching` with no supervisor recorded and a dead launcher, after a plain resume whose run does not exist (the session's current run is not newer than `expect_run_after`): the pre-resume park fields are restored (status `parked`, the previous `park_reason`, `needs_attention` and `last_outcome`) without hooks, logged `interrupted`;
- a pending run whose supervisor is dead (any status) is left to claude-session's own expiry (`never_adopted`) and reported (`reconcile` change `pending_run`); it never sets `supervisor_lost`. Once a run is final: a run agent-spawn did not start as the first run of a launch or resume (unknown to agent-spawn, or recorded `auto_resumed` or `promoted`) that ended in its launch phase gets the [outcome table](#outcome-table) row of the last recorded final run that triggered it (the [chain rule](#supervision)); any other final run gets its own row;
- `launching` with a **recorded supervisor**: the supervisor's liveness alone decides, whether or not the launcher is alive. Alive → nothing changes. Dead → by the session's current run: pending → left to claude-session's expiry (the pending-run rule above); adopted but still `phase: launch` → the agent stays `launching` and nothing is set; `phase: session` and live → the started transition (and `post-launch`) is applied, then the `live`-with-dead-supervisor rule below (`supervisor_lost`); final → the outcome table (as for `live`, without the automatic report resume or promotion);
- `live` with a dead supervisor: read the session's current run with `claude-session status <csession_id> --json` (which itself runs claude-session's liveness check). If the run is final, apply the outcome table without the automatic report resume or promotion (they belong to a supervisor) and park. If the session is still live (an orphaned `claude`, or a wrapper that outlived its supervisor), the agent stays `live` with `supervisor_lost: true` (`needs_attention: lost` on the transition into it); `agent-spawn stop` is the remedy;
- claude-session reports the agent's session not found (exit 10): park with `park_reason` and `needs_attention` `session_missing`.

Liveness never calls `claude-session stop` and never signals. `list`, `show` and `wait` record the changes they make, but leave delivery of the resulting hook events to the next mutating command, the supervisor or `reconcile` ([Hook delivery](#hook-delivery)). In every dry run the liveness check only reports: it parks nothing, sets no `session_missing` and runs no hooks.

## Commands

### `launch`

```text
agent-spawn launch [launch flags] [--json]
```

[Launch order](#launch-order). Output: the [result object](#objects) ([schemas/cli/launch.json](../schemas/cli/launch.json)) when a run created by this invocation was adopted, else the error envelope ([Output by exit code](#output-by-exit-code)). With `--dry-run`: step 1 only (usage checks, tool resolution, classification and pre-start checks 2 to 4 read-only, the window preflight), plus read-only plans from the lower tools (`wtctl create --dry-run --json` for `--worktree new`; `termctl placements --label … --json` for a window); no hooks run, no prompt copy is kept and nothing is created; prints [schemas/cli/launch-dry-run.json](../schemas/cli/launch-dry-run.json) (`dry_run`, `resume`, `agent_settings`, `reads_third_party`, `session_plan`, `would_refuse`, `tools`, `worktree_plan`, `term_plan`, `session_start_argv`, `hooks`, `warnings`). Its `would_refuse` is `third_party_cwd`, `third_party_run` or claude-session's `would_refuse`. A dry run never takes a lease. Every dry run exits 20 whenever `would_refuse` is set (also printing the plan), else 0 (or 2, 12 for usage and tool errors, with the error envelope).

Exit codes: 0; 1; 2; 10 (via wtctl or claude-session); 11; 12; 13; 14–18 (the run's outcome); 20; 21 (via wtctl); 22; 130; 143.

### `resume`

```text
agent-spawn resume (<agent_id> | --select k=v...) [--fork] [--cwd <path>] [--mode <mode>] [--placement window|here]
    [--kind <kind>] [--skill <skill>] [--prompt <text> | --prompt-file <path>] [--input-file <path>]...
    [claude-session pass-throughs] [--reset <setting>]... [--allow-tainted] [--override-escalation]
    [--name <text>] [--term-placement <name>] [--label k=v]... [--promote | --no-promote]
    [--wait] [--wait-timeout <duration>] [--result-file <path>] [--hook post-launch=CMD]...
    [--dry-run] [--json]
```

[Resume order](#resume-order). `--select k=v` (repeatable, ANDed) selects the agent; `--label` always adds labels. Output as `launch` ([schemas/cli/resume.json](../schemas/cli/resume.json)); with `--fork`, `agent` is the new fork. `--dry-run` performs steps 1 to 4 with the same read-only checks and prints [schemas/cli/launch-dry-run.json](../schemas/cli/launch-dry-run.json) with `resume: true` and `would_refuse` (`live`, `tearing_down`, `third_party_cwd`, `third_party_run`, or claude-session's `would_refuse` such as `tainted`), built from agent-spawn's own checks and the one classification dry run (of `claude-session start --resume`, or the launch's for an agent without a session, [Resume without a session](#resume-without-a-session)); exit 20 when it is set.

Exit codes: 0; 1; 2; 10 (agent, session or via wtctl); 11; 12; 13; 14–18 (the run's outcome); 20 (`live`, `tearing_down`, `third_party_cwd`, `third_party_run`, or a lower tool's reason); 21; 22; 130; 143.

### `wait`

```text
agent-spawn wait <agent_id> [--timeout <duration>] [--result-file <path>] [--json]
```

Blocks until the agent is not `live` or `launching`, no promotion is pending and its supervisor has exited (each poll runs the [liveness](#liveness) check; the supervisor's pidfd is only a wake-up optimization), then exits with the outcome code of the last run and prints the [result object](#objects) ([schemas/cli/wait.json](../schemas/cli/wait.json)). A `launching` agent is waited on. A parked agent returns at once with its last outcome; a parked agent that never had a run exits 12 (`no_run`). A `tearing_down` or torn-down agent (also given by its full id) returns its last outcome (exit: its outcome code) once no supervisor of it is alive; exit 10 only for an unknown id or a tombstone without a run. `--timeout` expiring first exits 22 (`timed_out: true`, `run: null`). `wait` never forwards a signal ([Signals](#signals)).

Exit codes: 0; 1; 2; 10; 11; 12 (`no_run`, or the run's outcome); 13; 14–18; 21; 22; 130; 143.

### `list`

```text
agent-spawn list [--label k=v]... [--needs-attention] [--status <status>]... [--session-state <state>]...
    [--kind <kind>] [--repo <repo-id>] [--mode interactive|headless] [--forked-from <agent_id>]
    [--since <time|duration>] [--json]
```

Agents ordered by `created` descending, ties by `agent_id` ([schemas/cli/list.json](../schemas/cli/list.json): `{"agents": [<agent>...]}`). Without `--status`, torn-down agents are excluded. `--label` is ANDed; `--status` and `--session-state` (claude-session states, matched against the relayed `session_state`) are ORed; `--kind` matches the launch kind or the last run's kind; `--mode` matches the last run's mode; `--since` matches `updated`. Runs the [liveness](#liveness) check for live and launching agents; it never delivers hooks and never calls claude-session for the session fields (they come from relay records).

### `show`

```text
agent-spawn show <agent_id> [--json]
```

Prints `{"agent", "runs", "session", "supervisor", "third_party_run", "repo_config_changed", "warnings"}` ([schemas/cli/show.json](../schemas/cli/show.json)): the agent object; `runs`, the agent's run history (each with `first_working_at` and its window); `session`, claude-session `status --json`'s run summary (`state`, `needs_input_reason`, `activity`, `run`, `report_status`, `sandbox_active`, `tainted`; null without a session or when claude-session reports it not found, which also parks the agent `session_missing`); `supervisor` (`{pid, alive, exec_pid}`); `third_party_run` (whether the agent's cwd lies inside a live wtctl throwaway record, so trusted runs of the agent are refused, [Third-party runs](#third-party-runs); null when the cwd is not known yet or the records lookup failed, with a warning); `repo_config_changed` (`{changed: bool, changed_classes, items}` from `wtctl baseline check` of the agent's own worktree record, the agent object's `worktree`; informational; null without a worktree record, or when the check could not run, with a warning). A full id of a torn-down agent shows its tombstone (`session`, `supervisor` and `repo_config_changed` null).

### `attach`

```text
agent-spawn attach <agent_id> [--client <client>] [--no-attach] [--json]
```

- A live agent whose current run has a window: `termctl focus <term_id> --socket <socket_path> --json [--client] [--no-attach]`. `action: switched`: exit 0. `action: attach`: without `--json`, agent-spawn replaces itself with the returned `argv` (the user's terminal attaches); with `--json`, it prints the termctl result and exits 0.
- An agent that is not live: exit 10 with `details.status` (`parked`, `launching`, `tearing_down`, `torn_down`, or null for an unknown id); for `parked`, hint `agent-spawn resume <agent_id>`.
- A live agent whose current run has no window (headless, or `--placement here`): exit 12, `no_window`, hint "headless agents may be promoted; read the session with `claude-session transcript <csession_id>`".

Output: [schemas/cli/attach.json](../schemas/cli/attach.json) (`agent_id`, `action`, `client`, `argv`, `term_id`, `socket_path`). termctl's failures pass through (`tool: termctl`).

### `ack`

`agent-spawn ack <agent_id> [--actor] [--json]`: clears `needs_attention` without resuming (a no-op if none is set: `changed: false`); it does not clear `supervisor_lost`; logged (`ack`, with the actor), records `post-state-change`. Output: `{"agent", "changed"}` ([schemas/cli/ack.json](../schemas/cli/ack.json)).

### `label add` / `label remove`

```text
agent-spawn label add <agent_id> key=value... [--dry-run] [--json]
agent-spawn label remove <agent_id> key=value... [--dry-run] [--json]
agent-spawn label add --session <csession_id> key=value... [--json]
```

Edits the agent's labels (at least one pair; exit 2 otherwise; keys `agent` and `kind` exit 2). Adding a pair the agent already has, or removing one it does not have, is a no-op. Changes are mirrored to the session (`claude-session label add|remove`) and, when the agent has a live window, to the pane (`termctl label add|remove`; a change to a pair that is not among the first 30 stored agent labels is not mirrored to the pane, a warning, never queued, since termctl allows at most 32 labels per pane). A mirror failure is a warning and queues the change for re-sync ([Labels on lower tools](#labels-on-lower-tools)). Output: `{"agent_id", "labels", "added"|"removed", "changed", "warnings", "dry_run"}` ([schemas/cli/label-add.json](../schemas/cli/label-add.json), [schemas/cli/label-remove.json](../schemas/cli/label-remove.json)); `added`/`removed` list the pairs actually changed (possibly none, then `changed: false`).

**Self-labeling** (`--session`), run by an agent inside its session:

1. `<csession_id>` must equal `CSESSION_ID` (full id), and `CSESSION_DIR` must be set (exit 2 otherwise).
2. Every key must be in config `self_label_keys` (default `[pr]`; exit 2 otherwise). Only `add` is allowed.
3. agent-spawn never creates a database in this mode: a missing `state.db` exits 12 (`failed`, hint naming `XDG_STATE_HOME` and `XDG_CONFIG_HOME`, which the session's environment must carry when they are not the defaults).
4. `claude-session status <csession_id> --json`: a `tainted` session or a current run with `reads_third_party: true` exits 11. `CSESSION_DIR` must equal the reported `csession_dir` (compared by realpath; exit 11 otherwise).
5. agent-spawn opens `$CSESSION_DIR/secret` without following symlinks; the file must be a regular file owned by the user, mode 0600 and at most 128 bytes (exit 11 otherwise).
6. It runs `claude-session verify-secret <csession_id>` with the secret on stdin; a mismatch (exit 11) exits 11.
7. It adds the pairs to the agent whose session is `<csession_id>` (exit 10 if none) and mirrors them.

Supported case: trusted, unsandboxed kinds whose permission profile allows `Bash(agent-spawn label add --session:*)` and whose environment carries `XDG_STATE_HOME` and `XDG_CONFIG_HOME` when they are not the defaults.

### `stop`

```text
agent-spawn stop <agent_id> [--grace <duration>] [--actor <actor>] [--decision-id <id>]... [--dry-run] [--json]
```

Ends a live agent's current run chain deliberately. Not human-only; always audited (`stop`, with the actor). After the [liveness](#liveness) check it takes the first action that applies:

- `torn_down` or unknown: exit 10.
- `parked`: a no-op, `already_parked: true` (`action: none`), exit 0.
- `tearing_down`: when its session's current run is pending or live (for example an orphaned child that made teardown step 6 fail), `claude-session stop` as below (`action: session_stop`); else a no-op, `already_parked: true` (`action: none`), exit 0.
- the agent has a pending or live run (`csession_id` set and `claude-session status` shows its current run pending or live, including a promotion's pending run and `supervisor_lost`): `claude-session stop <csession_id> [--grace D] --actor --decision-id --json` (`action: session_stop`); its exits 10 and 20 pass through.
- a supervisor with no pending run (between runs): SIGTERM to the verified supervisor (`action: signal_supervisor`).
- `launching` with no session or no pending run yet (before `start` returned): SIGTERM to the verified launcher (`action: signal_launcher`), which ends as in [Signals](#signals).

It then waits for the bookkeeping (the agent parked and its supervisor exited, or the liveness check parked it; for a `tearing_down` agent, the run final), at most the grace (claude-session's `kill_grace` when not given) plus 30 s. A record that disappears during the wait (a fork whose launcher deleted it before its session existed, or an [abandoned fork](#liveness)) counts as done: exit 0, `agent: null`, `removed: true`. `stop` records nothing on the agent: it parks per the [outcome table](#outcome-table). A stopped run parks `killed` (`stopped`) without attention; a canceled pending run or an interrupted launch parks `killed` without attention; a run that had already reported keeps its report's outcome (claude-session's `stop`); a supervisor stopped between runs parks per the previous run's outcome; a stopped run is never resumed automatically or promoted. Output: `{"agent", "action", "session_stop", "already_parked", "removed", "dry_run", "warnings"}` ([schemas/cli/stop.json](../schemas/cli/stop.json)); `agent` is null exactly when `removed` is true (then `action` is `signal_launcher`); `session_stop` is claude-session's `stop` result verbatim (null for the other actions). `--dry-run` reports the action without signaling anything (`session_stop`: claude-session's `stop --dry-run` result for `session_stop`). Exit codes: 0; 1; 2; 10 (also claude-session's); 12 (still live after the wait, like claude-session's `stop`; or a lower-tool failure); 20 (claude-session's refusal); 21.

### `teardown`

[Teardown order](#teardown-order). Output: `{"agent", "worktrees": [{"record_id", "path", "owned", "action", "wtctl"}], "session_forgotten", "windows_closed", "warnings"}` ([schemas/cli/teardown.json](../schemas/cli/teardown.json)): one `worktrees` entry per record the agent held, in processing order (`[]` without any); `action` is `removed` (the owned record), `forgotten` (the release forgot the record), `released` (the hold was dropped and the record stays) or `none` (already done on a retry); `wtctl` is wtctl's result verbatim (untrusted) or null. `windows_closed` lists the closed panes' `term_id`s. `--dry-run`: [schemas/cli/teardown-dry-run.json](../schemas/cli/teardown-dry-run.json) (`dry_run`, `agent_id`, `would_refuse`, `worktrees`: `[{"record_id", "path", "owned", "action", "wtctl_status"}]`, where `wtctl_status` is `wtctl status --json` of the owned record and null for held ones), printed on exit 0 and on exit 20 (whenever `would_refuse` is set).

Exit codes: 0; 1; 2; 10; 11; 12 (including `partial`); 13; 20; 21.

### `prune`

```text
agent-spawn prune [--older-than <duration>] [--include-owned] [--include-attention] [--dry-run] [--json]
```

Tears down parked agents whose age (`last_live_at`, else `created`) exceeds `--older-than` (else config `prune.older_than`; with neither, prune does nothing and warns), and continues `tearing_down` agents older than the threshold whose teardown process is dead. Skipped: agents with a label matching `prune.protect_labels` (keys or `key=value` pairs; default `[owner]`) unless `--include-owned`; agents with `needs_attention` set unless `--include-attention`; agents whose `teardown_step` is `worktree_partial` and whose record is not a tombstone ([Teardown order](#teardown-order) step 4). Each teardown runs under automation rules (never `--force`). Output: `{"results": [{"agent_id", "result": "torn_down"|"refused"|"skipped"|"failed", "reason", "tool", "details"}], "warnings", "dry_run"}` ([schemas/cli/prune.json](../schemas/cli/prune.json)); exit 0 whatever the per-agent results (2 on usage errors).

### `reconcile`

```text
agent-spawn reconcile [--dry-run] [--json]
```

Safe under concurrent invocation (lease `reconcile`; a second caller exits 0 with `already_running: true`). In order:

1. `claude-session reconcile --json` (a failure is a warning).
2. The [liveness](#liveness) check for every `launching` and `live` agent.
3. Undelivered agent-spawn hook events are delivered ([Hook delivery](#hook-delivery)).
4. Agents with `teardown_step` set are reported (`teardown_incomplete`), not continued.
5. Pending label mirror changes are re-synced to sessions and live panes ([Labels on lower tools](#labels-on-lower-tools)).
6. Tombstones older than `tombstone_days` are deleted (the id stays reserved); handoff directories whose `owner` is dead, `prompts/<id>/` directories that have no agent record and are older than one hour, and leftover prompt copies and outcome files of torn-down agents are removed.

Output: `{"changes": [{"agent_id", "kind", "details"}], "warnings", "already_running", "dry_run"}` ([schemas/cli/reconcile.json](../schemas/cli/reconcile.json)); `kind` is `parked`, `supervisor_lost`, `launch_failed`, `session_missing`, `pending_run`, `ambiguous_session`, `teardown_incomplete`, `fork_abandoned`, `labels_resynced`, `events_delivered` or `tombstone_deleted`. `--dry-run` reports without changing anything. agent-spawn's reconcile does not run `wtctl reconcile`.

### `cron install` / `cron uninstall`

```text
agent-spawn cron install [--interval <minutes>] [--env-file <path>] [--dry-run] [--json]
agent-spawn cron uninstall [--dry-run] [--json]
```

Writes or removes the managed crontab block `# agent-spawn managed`, which runs `agent-spawn reconcile --quiet` every `--interval` minutes (1–59, default 10) with an explicit `PATH` including `~/.local/bin` and the directory of the absolute agent-spawn path, optionally prefixed by sourcing `--env-file` (owned by the user, not group- or world-writable; else exit 2). Output as claude-session's ([schemas/cli/cron-install.json](../schemas/cli/cron-install.json), [schemas/cli/cron-uninstall.json](../schemas/cli/cron-uninstall.json)). No `crontab` command exits 12 (`failed`). With this block, claude-session's own cron entry is unnecessary.

### `export`

`agent-spawn export [--since <time|duration>] [--action <action>]... [--json]`: audit records, oldest first ([schemas/cli/export.json](../schemas/cli/export.json)).

### `browse`

`agent-spawn browse [--focus <agent_id>] [--result-file <path>]`: [TUI](#tui).

### Internal commands

`supervise <agent_id> [--host] [--handoff <socket> --handoff-timeout <seconds>] --state-dir <path> --config <path>` ([Supervision](#supervision); `--host` for a promotion window, [Promotion](#promotion)) and `--state-dir <path> --config <path> _session-event` ([State relay](#state-relay)). Not for direct use; each refuses (exit 2) when the agent's record does not expect it. Every internal argv carries the absolute state dir and the resolved absolute config path (the default path when no file existed at launch; a missing file then means built-in defaults); internal processes never derive the state dir from `XDG_STATE_HOME`.

## Global flags

`--json`, `--config <path>`, `--verbose`, `--quiet`, `--version`, `--yes` (accepted and a no-op: agent-spawn's own steps never prompt; teardown is the decision to remove and passes `--yes` to `wtctl remove`), `--actor <actor>` (`^(human|auto):[A-Za-z0-9._@-]{1,64}$`; default `human:<me.name>` when stdin is a TTY, else `auto:unknown`), `--decision-id <id>` (repeatable; `^[A-Za-z0-9._:@/-]{1,128}$`, at most 32). `--actor` and `--decision-id` are passed to every lower-tool call that accepts them. `--dry-run` is accepted by `launch`, `resume`, `label add/remove`, `stop`, `teardown`, `prune`, `reconcile` and `cron install/uninstall`.

**Internal actor.** Every lower-tool call made by `supervise`, `prune` and `reconcile` uses `--actor auto:agent-spawn` and the decision ids stored with the launch (or the resume that started the run chain), never a TTY-derived human actor.

**Human-only flags**: `--force` (teardown), `--override-escalation` (launch, resume, teardown) and `--allow-tainted` (resume) are accepted only when stdin is a TTY or with an explicit `--actor human:<user>`; otherwise exit 2. An explicit non-human `--actor` (such as `auto:<name>`) is refused even when stdin is a TTY. Automation never passes them. The gate is advisory ([Residual risks](#residual-risks)).

## Exit codes

| Code | Meaning in agent-spawn |
|---|---|
| 0 | success; for waiting commands: the last run reported done (or, without `--wait`, the run started) |
| 1 | internal error |
| 2 | usage error, invalid config, human-only flag without a human actor |
| 10 | agent not found or torn down; the agent's session is missing (`details.status: session_missing`); a lower tool's not-found passed through |
| 11 | a `pre-launch`/`pre-teardown` hook vetoed; a lower tool's veto passed through; self-labeling check failed; for waiting commands: the run's launch was vetoed |
| 12 | expected failure: `tool_missing`, `handoff_failed`, `no_window`, `no_run`, `worktree_gone`, `partial`, `failed`, `schema_newer`, `launch_failed` (including `details.cause` `classification_mismatch` and `classification_unknown`); `failed` with `details.cause: worktree_state`; a lower tool's expected failure (`code: failed`, its code in `lower_code`); for waiting commands: the run never started |
| 13 | a pre-hook escalated; a lower tool's escalation or confirmation required; for waiting commands: the run ended blocked or needs-input (not promoted), or its launch escalated |
| 14–18 | waiting commands: exited-unreported, killed, timeout, lost, failed |
| 20 | refused; `reason` is agent-spawn's `live`, `tearing_down`, `third_party_cwd`, `third_party_run`, or a lower tool's reason passed through (`tainted`, `repo_config_changed` from wtctl at teardown, `held`, `dirty`, `unpushed`, `in_use`, `unlinked`, `locked`, `main_worktree`, `branch_in_use`, `branch_exists`, `path_exists`, `not_owned`, ...) |
| 21 | ambiguous id prefix or `--select` selector; a lower tool's ambiguity passed through |
| 22 | `--wait-timeout` or `wait --timeout` expired while the agent was live |
| 130 | interrupted by SIGINT or SIGHUP (a waiter: the agent is unaffected; or a launcher interrupted before any supervisor existed) |
| 143 | interrupted by SIGTERM (`wait`, or a launcher interrupted before any supervisor existed) |

## Output by exit code

With `--json`:

| Exit | Command | Stdout |
|---|---|---|
| 0, 11–18, 22 | `launch`, `resume` when a run created by this invocation was adopted; `wait` with a run | the [result object](#objects) (for 22: `timed_out: true`, `run: null`) |
| 0 | other commands | the command's schema |
| 20 | `launch`/`resume --dry-run`, `teardown --dry-run` | the dry-run plan with `would_refuse` (every dry run exits 20 exactly when `would_refuse` is set) |
| 10, 11, 12, 13, 20, 21 | `launch`, `resume` (with or without `--wait`) when no run of this invocation was adopted; every other command | error envelope (with `agent_id` when an agent record exists) |
| 1, 2 | any | error envelope |
| 130, 143 | a launcher interrupted before any supervisor existed | error envelope (`code: launch_failed`, `details.cause: interrupted`) |
| 130, 143 | a waiter (`launch`/`resume --wait` after the start, `wait`) | nothing |

Callers treat "an error envelope, or a result whose `phase` is `launch`" as "the launch failed".

**Terminal text.** Error lines and warnings on stderr are single lines: control characters (C0 including newline and tab, DEL, C1), Unicode format characters (category Cf, such as bidi controls and zero-width characters), U+2028, U+2029 and lone surrogates are shown as backslash escapes. Multi-line human output (usage, human-readable results) keeps newline and tab and escapes the rest the same way.

### Result file

`--result-file <path>` on `launch`, `resume` and `wait` receives exactly what `--json` would print (result object or error envelope), whatever the exit code, written atomically with mode 0600. This includes usage errors (exit 2): a usage error found after parsing uses the parsed path; for argparse's own errors the path is taken from a pre-scan of the raw argv (tokens before `--`, `--result-file <path>` or `--result-file=<path>` after the command), and a path that cannot be resolved or written is a warning. This rule covers only these three commands; the file of `browse --result-file` is the intent file defined in [TUI](#tui). The command resolves the parent directory's realpath once, when it checks the path, and stores it. The write opens that realpath component by component from `/` with `O_NOFOLLOW|O_DIRECTORY` (a symlinked component fails the write, with a warning), then creates a temporary file relative to it with `O_CREAT|O_EXCL|O_NOFOLLOW` and renames it into place. **Third-party path rule.** For a third-party-reading run, a path whose parent's realpath lies inside the run's cwd, inside the throwaway record whose root is that cwd, or inside the run's session directory (`<csession_dir>`, whatever the cwd; known on resume and fork from `claude-session status`, and on launch once `start` returned) exits 2. The rule applies in every case: at [pre-start check](#third-party-runs) 3, and in the classification check when a run classified trusted turns out to be third-party-reading (a failure there cancels the run and parks `classification_mismatch`). `wait --result-file` applies the same rule (exit 2) when the agent's run in progress, else its latest run, is third-party-reading. When nothing would be printed (an interrupted waiter), nothing is written. A write failure is a warning on stderr and never changes the exit code. It is accepted with every placement; with `--placement here` it is the way to get the result (stdout is the terminal).

### Error envelope

`{"error": {"code", "message", "reason", "agent_id", "tool", "lower_code", "lower_error", "details", "candidates", "hook", "hook_stderr", "decision_ids"}}` ([schemas/cli/error.json](../schemas/cli/error.json)):

- `code` is agent-spawn's (`usage`, `config_invalid`, `not_found`, `vetoed`, `escalated`, `confirmation_required`, `refused`, `ambiguous`, `tool_missing`, `handoff_failed`, `no_window`, `no_run`, `worktree_gone`, `partial`, `launch_failed`, `failed`, `schema_newer`, `internal`); `reason` is set with `refused`.
- `tool` names the tool whose failure is passed through (`agent-spawn`, `claude-session`, `termctl`, `wtctl`) and `lower_code` that tool's error code. `lower_error` is that tool's whole `error` object, verbatim and untrusted (its `details`, wtctl's `checks`, `items` and `processes`, claude-session's `end_reason`, `hint`, `csession_id` and `run`, a lower tool's hook stderr); it is null when `tool` is `agent-spawn` and no lower call was involved. `tool` is `agent-spawn` for `third_party_cwd`, `third_party_run`, `classification_mismatch` and `classification_unknown`; wtctl's `repo_config_changed` refusal of `remove` at teardown passes through with `tool: wtctl`.
- `details` holds agent-spawn's own context only: `status` (`torn_down`, `session_missing`, or the agent's status for `attach`: `parked`, `launching`, `tearing_down`), `cause` (`interrupted`, `no_pane_pid`, `pane_gone`, `classification_mismatch`, `classification_unknown`, `worktree_state`), `retryable`, `hint`.
- `hook` and `hook_stderr` (untrusted, cut to 4096 bytes) with `vetoed` and `escalated` from agent-spawn's own hooks; `agent_id` is null when no agent record exists (also a refused or failed fork).

Lower-tool mapping: exit 10, 11, 13 and 20 keep their meaning (`not_found`, `vetoed`, `escalated` or `confirmation_required`, `refused` with the lower `reason`); exit 12 → `code: failed` (`launch_failed` when claude-session's `start` reported `launch_failed`), `lower_code` set; exit 21 → `code: ambiguous` with `candidates: []` and the lower ids in `lower_error`; wtctl exit 1 `partial` → agent-spawn exit 12, `code: partial`; termctl exit 2 of the window preflight → exit 2, `code: usage`.

## Objects

**Agent object** ([schemas/defs.json](../schemas/defs.json) `$defs/agent`): `agent_id`, `status`, `needs_attention`, `park_reason`, `escalated_by`, `kind` (launch kind), `run_kind` (last run's), `mode` and `placement` (the last run's; a promoted run is `interactive`/`window`; before a run, the launch's), `term_placement`, `name`, `repo` (`--repo`, else the classification plan's, confirmed by `status` after `start`), `cwd` (a realpath, or `session-dir` for a session's own `work/`), `labels`, `csession_id`, `run` (current or last run number), `session_state`, `activity`, `needs_input_reason` (from relay records), `forked_from`, `worktree` (`{record_id, path, owned}` or null), `holds` (`[{record_id, path, owned}]`, the records the agent holds now; empty when none), `window` (`{term_id, socket_path}` of the current or last run, null when that run had none; for any mode), `promote` (the explicit choice or null), `promotion_pending`, `ephemeral`, `last_outcome`, `last_end_reason`, `supervisor_lost`, `tainted` (the session's taint from `claude-session status` after each `start`; null without a session), `teardown_step` (a named marker, [Teardown order](#teardown-order)), `created`, `last_live_at`, `updated`, `torn_down_at`, `warnings`.

**Outcome object** (`$defs/outcome`): claude-session's outcome object for one final run of the agent (claude-session's interface, section Outcome object, is normative for its fields) plus `agent_id`, `promoted` (the run was a promotion), `auto_resumed` (the run was the automatic report resume) and `first_working_at` (the run's first `working` activity, from relay records or, when the relay missed it, from `status` `recent_events`; else null). Each `runs[]` entry of `show` carries it too.

**Result object** (`launch`, `resume`, `wait`): `{"agent", "phase", "outcome", "end_reason", "exit_code", "run", "first_working_at", "timed_out", "warnings"}` (design guide section 4.2): `run` is the outcome object of the run whose outcome-table row was applied (for a chain run that stopped early, the triggering run, [chain rule](#supervision); null while the run is live, without `--wait` after the start, or when timed out), and the waiter's exit code is that run's outcome code; `phase`, `outcome`, `end_reason` and `exit_code` are copied from `run` at the top level (null when `run` is null); `first_working_at` is the `first_working_at` of the first run this invocation started (`wait`: of the run chain's first run), null if it never reached `working`; it is the **delivery signal**: a chain whose first run reached `working` received its prompt (a later run's own value, such as the automatic report resume's, is not); `timed_out` is true only with exit 22.

## Hooks

Configured under `hooks:` (design guide section 8: `cmd` via `sh -c` in its own process group, JSON on stdin, timeout default 30 s, `async: true` only on post-hooks) or per invocation with `--hook NAME=CMD` (`pre-launch`, `post-launch` on `launch`; `post-launch` on `resume`; `pre-teardown`, `post-teardown` on `teardown`; fixed 30 s timeout, never async). `--hook` is agent-spawn's own and is never passed to claude-session.

| Hook | When | Veto/escalate | Payload |
|---|---|---|---|
| `pre-launch` | [Launch order](#launch-order) step 2, before anything is created (never on resume) | yes: veto exits 11, escalation exits 13 (`--override-escalation` re-run) | [schemas/hooks/pre-launch.json](../schemas/hooks/pre-launch.json): the resolved launch settings (kind, mode, placement, repo, worktree mode, branch, cwd, name, labels including `job=`, skill, has_prompt, input names, plugin dirs, promote, ephemeral, wait), `settings` (the full stored launch settings: worktree mode, `ref`, `base`, `branch`, `detach`, `no_bootstrap`, `copy_env`, `worktree_name`, `wait_timeout`, ...), `pass_throughs` (the claude-session flags given, by name, with their values: `sandbox`, `permission_profile`, `env_allowlist_only`, `setting_sources`, ...; never prompt text), `run_kind` (the classification plan's), `reads_third_party` (the classification), `actor`, `decision_ids`, `override_escalation` |
| `pre-teardown` | [Teardown order](#teardown-order) step 3 | yes: veto 11, escalation 13 | [schemas/hooks/pre-teardown.json](../schemas/hooks/pre-teardown.json): the agent object, `merged_head`, `force`, `actor`, `decision_ids`, `override_escalation` |
| `post-launch` | a run of the agent was started (launch, resume, automatic report resume, promotion) | no | [schemas/hooks/post-launch.json](../schemas/hooks/post-launch.json): the agent object, `run`, `resumed`, `fork`, `promoted`, `auto_resumed` |
| `post-state-change` | every relayed session state change and every change of the agent's `status` or `needs_attention` | no | [schemas/hooks/post-state-change.json](../schemas/hooks/post-state-change.json) |
| `post-escalate` | `needs_attention` changed to a non-null value | no | [schemas/hooks/post-escalate.json](../schemas/hooks/post-escalate.json): the agent object, `needs_attention`, the last outcome object, `chain_failure` |
| `post-park` | the agent became parked | no | [schemas/hooks/post-park.json](../schemas/hooks/post-park.json): the agent object, `reason` (= `park_reason`), the last outcome object, `teardown_refused` (for an ephemeral agent), `chain_failure` ([chain rule](#supervision)) |
| `post-teardown` | the agent became a tombstone | no | [schemas/hooks/post-teardown.json](../schemas/hooks/post-teardown.json): the agent object, `worktrees` (`[{record_id, path, owned, action}]`), `merged_head`, `force` |

Every payload has the top-level fields `tool` (`agent-spawn`), `hook`, `object_type` (`agent`), `object_id` (the agent id; null in `pre-launch`), `repo` (or null), `labels`, `kind`, `run_kind`, `reads_third_party` (of the last run, or null before a run; the classification in `pre-launch`), `tainted` (the session's taint; null without a session, always null in `pre-launch`), `mode`, `status`, `needs_attention`, `event_id` (agent-spawn's, unique per event; deduplicate on it) and `ts`.

**`post-state-change`** adds the agent object (`agent`) and `change` (`session` or `agent`); for `session` changes claude-session's fields `csession_id`, `run`, `state`, `previous_state`, `activity`, `needs_input_reason`, `final`, `cause`, `report` (`status`, `summary`, `reason`; summary and reason are untrusted text), `session_event_id`, `event_ts`, `phase`, `outcome`, `end_reason`, `exit_code`, `message` and `sandbox_active` (`outcome`, `end_reason` and `exit_code` are null until the event is final); for `agent` changes `previous_status` and `previous_needs_attention`.

**Redaction.** For every post-hook, notification templates exclude `report.summary` and `report.reason` (in `post-state-change`, and in the outcome objects of the other payloads) when `reads_third_party` or `tainted` is true (architecture section 15).

**Selectors** (optional on every config entry, ANDed): `kinds` (the run kind is one of), `exclude_kinds`, `labels` (the agent had all of these pairs), and for `post-state-change` only, `states` (the session state is one of; an entry with `states` fires only for `change: session` events) and `statuses` (the agent status is one of).

**Execution**: hooks run with the environment of the process that fires them plus `AGENT_SPAWN_HOOK`, `AGENT_SPAWN_ID` (the agent id; empty in `pre-launch`) and `AGENT_SPAWN_OBJECT_TYPE=agent`; cwd is agent-spawn's state dir. Hooks are fired only by the CLI and the supervisor (the relay fires none), so they get the caller's environment; hook `cmd`s should use absolute paths, and notification hooks should set `async: true`. A `decision_id` in a pre-hook's JSON stdout is recorded on the agent (log record of the step; audit records where the step is audited). Pre-hook outcomes follow the design guide (exit 13 escalates, any other non-zero exit vetoes, a timeout vetoes). Post-hook failures are logged and never change the command's result.

### Hook delivery

agent-spawn's post-hook events are recorded in an outbox (`events`) in the same transaction as the change they describe, and delivered at least once, in order per agent, deduplicated by `event_id`. Delivery is serialized by a per-agent lease `delivery:<agent_id>`: its holder delivers every undelivered event of the agent in `event_id` order; a process that cannot take the lease leaves its events to the holder, the next mutating command, the supervisor (which drains the outbox every second) or `reconcile`. The read-only commands `list`, `show` and `wait` never deliver. Per-invocation `--hook` entries of `launch` and `resume` are stored with the run and delivered by whichever process delivers that event.

## Config

`~/.config/agent-spawn/config.yaml` plus `config.d/*.yaml` (lexical order, merged after it; maps merged, `hooks` lists appended, other lists replaced). Schema: [schemas/config.json](../schemas/config.json); every key with its default is in [config.example.yaml](../config.example.yaml). Invalid config exits 2 (`config_invalid`) naming the key.

- **Path.** `--config <path>` names the config file (a relative path is resolved against the cwd). The default is `$XDG_CONFIG_HOME/agent-spawn/config.yaml` when `XDG_CONFIG_HOME` is absolute, else `$HOME/.config/agent-spawn/config.yaml` when `HOME` is absolute, else the same under the home directory of the effective user's passwd entry; with none of these absolute, `config_invalid`. The path in use, stored with agents and passed to internal processes, is always absolute.
- **Missing files.** A missing default file means built-in defaults. A missing file named by `--config` is `config_invalid` naming the path; internal commands (whose `--config` is the path recorded at launch, possibly a default path that did not exist) treat a missing file as built-in defaults. A config file, config directory or `config.d` that is a dangling symlink is `config_invalid`.
- **Drop-ins.** `config.d` is the directory beside the config file in use, including one named by `--config`. Only `*.yaml` files are read (not `.yml`); names starting with `.` are ignored; the order is the lexical order of the file names. Symlinks are followed; an entry that is not a regular file after that (a directory, a FIFO, a dangling symlink) is `config_invalid`. Each file is at most 1 MiB, and `config.d` holds at most 256 drop-ins.
- **YAML dialect.** One document per file, read with YAML 1.2 core-schema scalars: booleans are only `true`/`True`/`TRUE`/`false`/`False`/`FALSE`, null is only `null`/`Null`/`NULL`/`~`/empty, integers are decimal (`010` is ten), floats are the 1.2 core form; every other plain scalar (`yes`, `on`, `0x10`, `1:30`, `1_0`, dates) is a string. Anchors, aliases, merge keys and explicit tags (`!!int`, `!local`, ...) are refused, and so are duplicate keys; mapping keys must be strings. A file is validated on its own (errors name the file) and again after merging. Integer keys (`tombstone_days`) take integers only, never a number with a fractional part such as `30.0`. Error messages name the file, line or key, never the value.
- **Merging.** A map merges into a map key by key; a `hooks.<hook>` list is appended to when both values are lists; any other value, lists included, replaces the earlier one (so `wtctl_cmd: null` in a drop-in replaces a list). Keys whose schema type has no null (for example `repos.<repo-id>.kinds.<kind>.promote` or a hook entry's selectors) refuse an explicit `null`; omit the key instead.

| Key | Default | Meaning |
|---|---|---|
| `me.name` | OS username | default actor `human:<me.name>` |
| `claude_session_cmd` | `[claude-session]` | argv of claude-session; resolved to an absolute path on `PATH` at launch and stored with the agent |
| `termctl_cmd` | `[termctl]` | argv of termctl |
| `wtctl_cmd` | `[wtctl]` | argv of wtctl; `null`: no wtctl on this machine (standalone use, [Records of a cwd](#records-of-a-cwd)). Otherwise an unresolvable wtctl exits 12 `tool_missing` whenever the records of a cwd are needed |
| `job_id_env` | `JOBSCTL_JOB_ID` | variable whose value becomes the label `job=<value>` ([Labels on agents](#labels-on-agents)) |
| `self_label_keys` | `[pr]` | keys an agent may add to itself |
| `handoff_timeout` | `30s` | [Environment handoff](#environment-handoff); bounds only the handoff |
| `auto_report_resume` | `true` | the automatic report resume for headless unreported runs |
| `tombstone_days` | `30` | |
| `prune.older_than` | `null` | default for `prune --older-than` (`null`: prune does nothing without the flag) |
| `prune.protect_labels` | `[owner]` | label keys or `key=value` pairs that protect an agent from `prune` |
| `kinds.<kind>.promote` | `false` | [Promotion](#promotion) |
| `repos.<repo-id>.kinds.<kind>.promote` | none | per-repo override (absent: `kinds.<kind>.promote` applies) |
| `hooks.<hook>` | `[]` | `{cmd, timeout, async, kinds, exclude_kinds, labels, states, statuses}` entries |

Precedence: flags > per-repo override > environment > config > built-in defaults (design guide section 5). Kind names follow claude-session's grammar; repo ids its repo-id grammar. Internal processes load the config file recorded at launch and use the state dir recorded at launch (`--config` and `--state-dir` in their argv, [Internal commands](#internal-commands)).

## Environment variables

- Config overrides `AGENT_SPAWN_<KEY>` for top-level scalar keys only: `AGENT_SPAWN_HANDOFF_TIMEOUT`, `AGENT_SPAWN_TOMBSTONE_DAYS`, `AGENT_SPAWN_JOB_ID_ENV`, `AGENT_SPAWN_AUTO_REPORT_RESUME`. Lists and maps cannot be set from the environment. A bare number string for a duration means seconds. A set but empty variable is invalid; `AGENT_SPAWN_AUTO_REPORT_RESUME` is exactly `true` or `false`; `AGENT_SPAWN_TOMBSTONE_DAYS` is plain ASCII digits. An invalid value is `config_invalid` naming the variable, never its value.
- Read: the variable named by `job_id_env`; `CSESSION_ID` (self-labeling, and the `job=` rule); `CSESSION_DIR` (self-labeling only); `XDG_CONFIG_HOME`, `HOME` (the default config path, [Config](#config)), `XDG_STATE_HOME` (by the CLI only; internal processes take the state and config paths from argv).
- Hook variables: `AGENT_SPAWN_HOOK`, `AGENT_SPAWN_ID`, `AGENT_SPAWN_OBJECT_TYPE`; never config keys.
- agent-spawn reads no credentials. The environment it hands to windows ([Environment handoff](#environment-handoff)) may contain the caller's secrets; it never stores, logs or prints them.

## Labels

### Labels on agents

agent-spawn is the authoritative label store for agents. Labels given with `--label` at launch, resume and fork are added. `job=<value>` is added at launch and at a plain resume when the variable named by `job_id_env` is set and its value fits the value grammar (an invalid value is ignored with a warning), except:

- when `CSESSION_ID` is set (the caller runs inside an agent session), `job=` is taken only from an explicit `--label job=…`;
- a fork gets `job=` only from an explicit `--label job=…`;
- when adding `job=` would exceed the 64-pair cap, it is skipped with a warning (never a failure).

`job=` values accumulate, one per job that launched or resumed the agent ([Labels](#labels)).

### Labels on lower tools

| Tool | Labels |
|---|---|
| claude-session | at launch: the agent's labels plus `agent=<agent_id>`; at a fork: every fork label plus `agent=<fork agent_id>`; at a plain resume: the newly given labels and `job=`. Later changes mirrored with `label add/remove` |
| termctl | `kind=<run kind>`, `agent=<agent_id>`, then the first 30 agent labels in stored order (the rest dropped with a warning; termctl allows at most 32 per pane); later changes to those first 30 mirrored while the window lives (a change to any other pair is not mirrored, a warning) |
| wtctl | none; the agent's id is a holder of the record |

**Mirror re-sync.** A mirror change that failed is kept in a per-agent pending-mirror queue. `reconcile` re-syncs by adding missing agent labels and removing only the pairs agent-spawn removed (from that queue). Pane re-sync covers only the first 30 stored agent labels. It never touches `kind=`, `agent=` or labels agent-spawn did not add.

## TUI

`agent-spawn browse` (Textual): a list of agents (status, `needs_attention`, kind, name, labels, age) with filters (`/` label filter, status, attention only), a detail pane with the data of `show`, and actions that run the CLI: `a` attach, `r` resume (interactive, in a window), `k` ack, `s` stop (confirmed in the TUI's own dialog), `t` teardown (confirmed in the TUI's own dialog, then `--yes --actor human:<user>`), `w` wait. Keys follow the design guide (`q`, `?`, `/`, `enter`, `r` refresh is `R`). `--focus <agent_id>` opens on that agent. Without `--result-file`, the TUI performs `attach` and `resume` itself. With `--result-file <path>`, `a` and `r` only record the intent and exit: the file receives `{"intent": "attach"|"resume"|"back", "selected": [<agent_id>...]}` ([schemas/cli/browse-result.json](../schemas/cli/browse-result.json); `attach` carries exactly one id, `back` may carry none); nothing is written without the flag.

## Files

State directory `$XDG_STATE_HOME/agent-spawn/` (default `~/.local/state/agent-spawn/`) for the CLI; internal processes use the state dir given in their argv. Mode 0700; files 0600:

- `state.db` (private, SQLite WAL: agents, runs (each with its window), holds, the hook outbox, the pending-mirror queue), `log.jsonl` ([schemas/log.json](../schemas/log.json)), `audit.jsonl` ([schemas/audit.json](../schemas/audit.json));
- `supervise/<agent_id>.log`: the supervisor's own log (no secrets, no prompt text);
- `handoff/<random>/`: a transient socket `s` and its `owner` file ([Environment handoff](#environment-handoff));
- `prompts/<agent_id>/` (mode 0700): copies of the prompt and input files, made at classification (launch step 1, resume step 4) and deleted once a `claude-session start` allocated a session, at the tombstone step of teardown, or when the command exits before its agent record exists (`reconcile` removes leftovers older than one hour);
- `outcomes/<agent_id>/`: claude-session outcome files of the agent's runs (one random token per run), deleted at the tombstone step of teardown (step 7); `reconcile` removes leftovers of torn-down agents. `wait` on a tombstone reads `state.db`.

## Audit and log

- `audit.jsonl` records `teardown` (when it removed a worktree or used `--force`), `stop`, `launch_escalation_override`, `resume_escalation_override`, `teardown_escalation_override`, `resume_tainted` (a resume with `--allow-tainted`; claude-session audits it as well), each with `ts`, `actor`, `decision_id` (array), `target` (the agent id), `action`, `inputs` (`labels`, `repo`, `worktree`, `csession_id`, `merged_head`, `force`) and `result`. `export` prints them.
- `log.jsonl` records `launch`, `resume`, `fork`, `adopt`, `holder_added`, `status_change`, `attention_set`, `attention_cleared`, `ack`, `stop`, `label_change`, `relay_event`, `relay_event_ignored`, `promotion`, `auto_report_resume`, `third_party_refused` (`details.reason`: `third_party_cwd` or `third_party_run`), `classification_mismatch`, `classification_unknown`, `fork_abandoned`, `teardown`, `teardown_retry`, `prune`, `reconcile_change`, `handoff_failed`, `handoff_rejected`, `signal_sent` (with the exact PID), `interrupted`, `hook_delivered`, `hook_failed`, `hook_timeout`, `hook_veto`, `hook_escalate`, `schema_older`, `warning`, `migration`; each with `ts`, `level`, `event`, `agent_id` (or null) and event-specific fields. The log never contains environment values, secrets, prompt text, summaries or reasons.

## Upgrades

Database migrations are additive within a schema major version. Every command checks the schema first: an older database is migrated, a newer one refused (`schema_newer`, exit 12). `_session-event` never migrates: with an older schema it processes the event when the schema is readable, else logs `schema_older` and exits 0. Upgrade agent-spawn while no agents are live: windows and hooks run the absolute agent-spawn path recorded at launch.

## Residual risks

- **Environment in windows.** The handoff keeps secrets off disk, argv and tmux, but the supervisor's environment is readable through `/proc/<pid>/environ` by any process of the same user that is not sandboxed (as for any process). Sandboxed sessions cannot read it.
- **Prompt text at rest.** Prompt and input copies stay in `<state>/prompts/<agent_id>/` (0600) from classification until a session is allocated; a launch that fails after allocation and before its session keeps them until it is resumed or torn down, and a crashed command's copies until `reconcile` removes them.
- **Human-only flags are advisory.** `--force`, `--override-escalation` and `--allow-tainted` require a TTY or a self-declared `--actor human:*`; they stop automation that follows the rules, not a hostile caller.
- **Overrides are stateless.** `--override-escalation` is not bound to the escalation the human saw (as in termctl): a re-run with changed settings is overridden too. Pre-hook payloads carry `override_escalation: true`, so a policy can veto overriding runs.
- **Unauthenticated state.** Labels, agent-spawn's state, relay events and outcome files are not authenticated against unsandboxed processes of the same user, including trusted agents (which can run `agent-spawn label add` on any agent or write the state directory). Consumers verify links against authoritative sources (the forge, wtctl records, their own state) before destructive or external actions: for example a `pr=` label against the agent's branch, and their own forks by recorded agent ids rather than labels.
- **`--merged-head` is trusted.** wtctl compares the worktree's HEAD with the sha given and does not verify that it was merged, so a wrong sha lets `teardown` remove unpushed work. agent-spawn only checks the grammar.
- **SIGQUIT in a window.** Until claude-session's `exec` ignores SIGQUIT, Ctrl-\ in a window ends the run `lost` or `killed` without a forwarded signal.
- **A SIGKILLed waiter.** A `launch|resume --wait` killed with SIGKILL (for example by jobsctl after `kill_grace`) cannot forward; the agent runs to completion (its detached supervisor is reparented to the shim's subreaper, then init) and parks normally.
- **Relay events lost to an old schema.** An event that `_session-event` cannot read after an upgrade (`schema_older`) is lost; the supervisor's own final bookkeeping still records the final state.
- **Orphans.** A supervisor killed with SIGKILL cannot forward; the kernel hangs up `exec` and `claude` when the pane closes, which normally ends them. A `claude` that survives is reported (`supervisor_lost`) and ended with `agent-spawn stop`.
- **Runs started outside agent-spawn.** The cwd rule and the throwaway check cover only runs launched through agent-spawn: a third-party-reading run started directly with `claude-session start` (for example in a worktree) is not refused, and a trusted session started directly in a throwaway is not refused.
- **Unsandboxed third-party runs.** A third-party-reading run whose sandbox was lowered or is unavailable (claude-session warns and reports `sandbox_active`) can write outside its session dir or throwaway, including into worktrees, and processes it leaves behind can keep writing after it parks (architecture section 13.2 (a)).
- **Trusted cwd above a throwaway.** Only a cwd inside a throwaway record is refused. A trusted agent rooted above one (for example at the throwaway root or its parent) that works in it can load planted Claude project config (a nested `CLAUDE.md`). Do not run trusted agents at or above wtctl's throwaway root.
- **Trusted cwd in a third-party-writable location.** A trusted run started with `--cwd` inside some session's `work/` directory, or inside another location a third-party-reading run can write (for example a kind's `write_dirs`), is not refused. agent-spawn's rule is only that third-party-reading runs run in session dirs and throwaway roots and trusted runs never in a throwaway; it does not guarantee that the two never share a location.
- **A throwaway directory that outlives its record.** The throwaway check uses wtctl's live records. A directory at a throwaway path whose record is removed or `gone` (for example one that reappears later) is not a record, and trusted runs there are not refused.
- **Delivery signal after a relay failure.** `first_working_at` falls back to claude-session's `recent_events`, which is bounded; a run whose relay events were lost may report null although it reached `working`.
- **Other session-writable paths.** agent-spawn checks only the cwd and the `--result-file` path of a third-party-reading run. claude-session's other session-writable paths (a kind's `write_dirs`, profile sandbox write additions, a writable `--add-dir`) are not checked: a third-party kind configured with a writable path inside a worktree can change it. The built-in third-party kinds have none.
- **Promotion windows in untrusted directories.** Claude Code's trust dialog fires no hooks; a window whose cwd Claude Code has never trusted waits in `starting` until the user answers. Trust the worktree root (e.g. wtctl's `~/wtctl`) once.
- **Logout.** A detached supervisor stays in the login session's cgroup; with systemd's `KillUserProcesses=yes`, logging out ends it (and the session). `awkit doctor` checks this.
- **`Ctrl-Z` in a window.** The window's process group is orphaned, so job-control stops are discarded; a `claude` that stops itself is resumed with SIGCONT only by a forwarded SIGTERM.

## Platform notes

- Linux and WSL2: fully supported (pidfd via `os.pidfd_open`, or a `syscall` fallback where the Python build lacks it; `SO_PEERCRED`).
- macOS: required for shareable use (headless, `--placement here`, `--worktree none`); waiting on a non-child uses kqueue `EVFILT_PROC`/`NOTE_EXIT`, the handoff peer check uses `LOCAL_PEERPID` and `getpeereid` (both still to verify).
- The absolute agent-spawn path recorded at launch must exec agent-spawn in place (a console script or a wrapper that `exec`s, never one that forks): the handoff checks that the peer is the pane's own process.
- tmux 3.2 or newer through termctl; claude-session and wtctl at their interface versions 0.1.0 or newer.
