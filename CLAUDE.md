# agent-spawn: instructions for Claude sessions

**Naming.** Tools have no shared prefix; see design guide §1 naming rules.

## What this is

`agent-spawn` is a Layer 2 (composite) awkit tool: launch, resume, park and tear down Claude agents: worktree + terminal + Claude session.
It must work fully standalone: no dependency on the `awkit` harness, a bus, a shared library or a
shared contracts package.

## Read first

1. `../awkit-docs/architecture.md`: this tool's §5.2 entry and its §20 entry (constraints and open
   topics), plus the cross-cutting sections listed in `docs/design.md`.
2. `../awkit-docs/design-guide.md`: the conventions every awkit tool follows.
3. `docs/design.md`: this tool's design doc.

## Rules

- Open source: no confidential information (no company names, hostnames, ticket project keys,
  custom-field IDs, people's names, repo or org names). Organization specifics belong in the
  user's config.
- Never import sibling repos. Read their docs and schemas by path only. Copying a sibling's
  schemas into `tests/` pinned to a version is allowed.
- No upward awareness: never know about, or behave differently for, tools that call this one.
  Behavior may differ only by explicit arguments.
- Interface first: write `docs/interface.md` and `schemas/` before implementation.
- Semver. A breaking interface change is, before 1.0, a `feat` commit whose title names the break (never `!` before 1.0), and from 1.0 on a `!` commit; the title becomes the changelog line (details go in the PR body). Either way, update `awkit-docs/interface-registry.md`. Never hand-edit CHANGELOG.md or the version.
- Run `make verify` before every commit.
- All subprocess and HTTP calls go through `src/agent_spawn/gateway.py`, so tests can substitute fakes
  (fake binaries live in `tests/fakebin/`).
- Never run destructive git commands (reset, checkout --, clean, restore, force-push) and never
  kill processes by pattern; signal only exact PIDs this tool started (verified by PID plus start
  time) or process groups it created for hooks.
- Untrusted or agent-generated text travels only in files or stdin, never argv or env.
- Minimize dependencies; justify each one in `docs/design.md`.

## Workflow

1. Start with a design session that resolves the open topics in `docs/design.md` and records
   decisions there.
2. Write `docs/interface.md` and `schemas/` (the interface spec), and register it in
   `../awkit-docs/interface-registry.md`.
3. Implement test-first.

## Repository workflow

- Work in a git worktree on a branch named `<type>/<short-slug>`, never in the main checkout (design guide section 17).
- PR titles are Conventional Commit headers; they become the squash commit, the version bump and the changelog line (architecture section 18.3). Do not use `!` before 1.0 (a breaking change before then is a `feat` whose title names the break).
- Never use `--no-verify`. Check PR titles, bodies and comments with `guard text` where the private-content guard is installed (architecture section 18.7).
- Never merge a PR without the maintainer's explicit instruction for that PR; merges go through `awkit-ci`'s `scripts/pr-merge` (architecture section 18.6).
- Never hand-edit `CHANGELOG.md` or the version in `pyproject.toml`; the release job owns them.
