# agent-spawn

Launch, resume, park and tear down Claude agents: worktree + terminal + Claude session.

## Status

Pre-design: interface not yet specified; see [docs/design.md](docs/design.md).

## Layer

L2 (composite). Works fully standalone; it does not need the `awkit` harness.

## What it does

- Launches agents (optional worktree via wtctl, optional tmux window via termctl, Claude session via claude-session) and supervises them with post-exit policies (`supervise`).
- Authoritative label store for agents; agent lifecycle (live, parked, torn down) is separate from window lifecycle.
- Never notifies directly: records attention as data (`needs_attention`) and fires hooks (`post-escalate`, `post-park`, `post-state-change`).
- Does not use jobsctl.
- Records repo-config hashes at launch and reports `repo_config_changed` in `show --json`; `resume --fork` records the fork as a new agent.

## Indicative commands

Indicative, finalized in [docs/interface.md](docs/interface.md).

- `agent-spawn launch`
- `agent-spawn resume`
- `agent-spawn list [--label k=v] [--needs-attention]`
- `agent-spawn show`
- `agent-spawn attach`
- `agent-spawn ack <agent_id>`
- `agent-spawn label add/remove`
- `agent-spawn teardown`
- `agent-spawn prune`
- `agent-spawn reconcile`
- `agent-spawn browse`
- `agent-spawn supervise  (internal)`

## Dependencies

- claude-session: binding: TBD (A/B/C), decided in this tool's design session
- termctl (interactive window placement only): binding: TBD (A/B/C), decided in this tool's design session
- wtctl (only when a worktree or a local clone is needed): binding: TBD (A/B/C), decided in this tool's design session

## Install

```sh
uv tool install git+https://github.com/agent-workflow-kit/agent-spawn
```

## Documentation

- [awkit architecture](../awkit-docs/architecture.md) ([published](https://github.com/agent-workflow-kit/awkit-docs/blob/main/architecture.md))
- [awkit design guide](../awkit-docs/design-guide.md) ([published](https://github.com/agent-workflow-kit/awkit-docs/blob/main/design-guide.md))

## License

MIT — see LICENSE.
