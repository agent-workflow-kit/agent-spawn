# agent-spawn

Launch, resume, park and tear down Claude Code agents: optional worktree + optional terminal window + Claude session.

## Status

Designed (interface 0.1.0-design); not implemented yet. The interface is [docs/interface.md](docs/interface.md) (normative) with [schemas/](schemas/); decisions and live verification are in [docs/design.md](docs/design.md); the implementation plan is [docs/plan.md](docs/plan.md).

## Layer

L2 (composite). Works fully standalone; it does not need the `awkit` harness. A headless launch without a worktree needs only claude-session.

## What it does

- Launches an agent: optionally creates or adopts a git worktree (wtctl), starts a Claude session (claude-session), and runs it interactively in a tmux window (termctl), in the current terminal, or headless.
- Supervises every run (`supervise`): records the outcome, parks the agent, marks it as needing attention when it needs you, resumes a headless run once when it ended without a report, and optionally promotes a headless run that needs input into an interactive window. `stop` ends a live agent deliberately.
- Agents outlive windows: a parked agent is resumed (`resume`), forked into a new agent (`resume --fork`), waited on (`wait`) and finally torn down (`teardown`), which removes the worktree it created only when nothing would be lost.
- Runs third-party-reading sessions only in their own session directory or a wtctl throwaway, and refuses trusted runs in any throwaway.
- Authoritative label store for agents; mirrors labels onto sessions and windows.
- Never notifies directly: it records `needs_attention` and fires hooks (`post-state-change`, `post-escalate`, `post-park`, ...) that you wire to a notifier.
- Does not use jobsctl; workflow tools run `agent-spawn launch|resume --wait` as jobs.

## Commands

| Command | Purpose |
|---|---|
| `agent-spawn launch [flags]` | Create and start an agent |
| `agent-spawn resume (<agent_id> \| --select k=v) [--fork]` | Resume a parked agent, or fork an agent |
| `agent-spawn wait <agent_id>` | Wait for an agent and exit with its outcome code |
| `agent-spawn list [filters]` / `show <agent_id>` | Inspect agents |
| `agent-spawn attach <agent_id>` | Focus or attach to a live agent's window |
| `agent-spawn ack <agent_id>` | Clear `needs_attention` |
| `agent-spawn stop <agent_id>` | End a live agent's run (it parks per its outcome; a stopped run parks `killed`) |
| `agent-spawn label add\|remove <agent_id> k=v...` | Edit labels (`label add --session` lets an agent label itself) |
| `agent-spawn teardown <agent_id> [--merged-head SHA]` | Remove or release the worktree, forget the session, keep a tombstone |
| `agent-spawn prune --older-than D` | Tear down long-parked agents (opt-in) |
| `agent-spawn reconcile` / `cron install\|uninstall` | Repair state; schedule reconcile |
| `agent-spawn export` | Audit records |
| `agent-spawn browse` | TUI |

Exit codes follow the awkit design guide (0, 1, 2, 10–21, session outcomes 14–18); agent-spawn adds 22 (a wait timed out), 130 and 143 (interrupted). `launch`, `resume` and `wait` accept `--result-file` to receive the JSON result in a file (for `--placement here`, whose stdout is the terminal). See [docs/interface.md](docs/interface.md#exit-codes).

## Dependencies

| Tool | Binding | Needed for | Install |
|---|---|---|---|
| claude-session | A (hard; the agent session role) | always | `uv tool install git+https://github.com/agent-workflow-kit/claude-session` |
| termctl | A | interactive agents in windows, promotion, `attach` | `uv tool install git+https://github.com/agent-workflow-kit/termctl` |
| wtctl | A | `--worktree new\|<path>`, worktrees and throwaways a cwd lies in, repo-config status in `show`, teardown of worktrees | `uv tool install git+https://github.com/agent-workflow-kit/wtctl` |

A missing sibling fails with an error naming its install command; it is checked only when needed.

## Config and environment

`~/.config/agent-spawn/config.yaml` (every key in [config.example.yaml](config.example.yaml)). Environment overrides `AGENT_SPAWN_<KEY>` for scalar keys (`handoff_timeout`, `tombstone_days`, `job_id_env`, `auto_report_resume`). agent-spawn reads no credentials; a window-hosted agent receives the launching process's environment over a peer-checked one-shot socket, never through a file.

## Install

```sh
uv tool install git+https://github.com/agent-workflow-kit/agent-spawn
```

## Documentation

- [Interface](docs/interface.md), [design](docs/design.md), [plan](docs/plan.md)
- [awkit architecture](../awkit-docs/architecture.md) ([published](https://github.com/agent-workflow-kit/awkit-docs/blob/main/architecture.md))
- [awkit design guide](../awkit-docs/design-guide.md) ([published](https://github.com/agent-workflow-kit/awkit-docs/blob/main/design-guide.md))

## License

MIT — see LICENSE.
