# Fake binaries: fidelity notes

The fakes in this directory stand in for `claude-session`, `termctl`, `wtctl` and `tmux` in agent-spawn's tests. They are a test aid, not a re-implementation: where a fake is simpler than the real program it says so here, and anything not checked against the real program is listed under "Unverified".

## How a fake behaves

- Standard-library Python only, run through the stubs in this directory (`claude-session`, `termctl`, `wtctl`, `tmux`) with `python3` from `PATH`. A fake refuses to run (exit 70) unless `FAKEBIN_LOG` is set, so it cannot be used by accident outside the `fakebin` fixture.
- Every call is logged to `FAKEBIN_LOG` (JSON lines): an `invoke` record before anything happens (argv, cwd, PID, a few environment variables) and an `exit` record (exit code, signals handled, stdin read). A call killed by an unhandled signal has no `exit` record.
- argv is parsed with the real tool's grammar: value flags take the next token even when it starts with `-` (getopt), `--flag=value` is equivalent, a boolean flag rejects `=value`, unknown flags are usage errors, and value grammars (labels, actors, decision ids, durations, ids, names) are the ones in the sibling interfaces. An argv the real tool refuses with exit 2 is refused with exit 2 and, with `--json`, the tool's error envelope.
- Canned outputs are built from the argv (and, for `claude-session`, `termctl` and `wtctl`, from a small state file next to the log), so the output of one call is consistent with the next. Every output is validated in tests against the schemas copied to `tests/sibling-schemas/<tool>/0.1.0-design/`.
- A test scripts responses with `fakebin.script(tool, command, ...)`: `exit`, `stdout`, `merge` (into the canned output), `error` (the error envelope), `stderr`, `sleep`, `hang`, `on_signal`, `outcome`, `secret`. A scripted `error` is raised before the fake changes its state, as a refusing real tool changes nothing.
- Anything a fake does not implement exits 1 with `error.code: fake_unsupported`, a code that is in no sibling schema, so a test cannot mistake it for the real program.

## Verified against the real programs

`tests/live/test_fidelity.py` (marker `live`, never in CI) re-checks these against the real programs:

- the exit code and error code of every row of `tests/fakebin/fidelity_cases.py` marked verified (claude-session 0.38.4 and termctl 0.25.0 argument parsing; 55 rows on 2026-11-14), including `--` handling, getopt value consumption, `--json=x`, human-only flags, the grammars and `verify-secret`;
- tmux 3.5a: `-V` output, the "no server" error for `list-sessions` on a socket that does not exist, the unknown-option usage text, `;` as a command separator and `\;` as a literal, and the `new-window -P -F` output shape.

## Known simplifications

claude-session:

- Implemented: `start` (with `--resume`, `--fork`, `--dry-run`), `exec` (new and `--session-id` adoption), `resume`, `stop`, `wait`, `status`, `list`, `label add|remove`, `forget`, `reconcile`, `verify-secret`. Not implemented (`fake_unsupported`): `report`, `transcript`, `export`, `kinds`, `cron`, `hook-event`.
- There is no prompt positional or `--` prompt: the interface passes a prompt with `--prompt-file`, and the fake refuses a stray positional. If a caller was told "a prompt needs `--`", that applies to `claude` itself, not to `claude-session`.
- A full session id is accepted even if the fake never saw it (the real tool answers 10); only prefixes are resolved against sessions created by `start`.
- `exec` runs no `claude`: it finishes at once with the scripted outcome (`done` by default), writes the outcome file (temporary file, rename, mode 0600; the real tool also walks parents with `O_NOFOLLOW`), and prints the outcome object. A scripted `hang` ends on SIGTERM or SIGHUP as `killed` (exit 15) with the outcome file written, like the real wrapper after forwarding the signal; SIGINT is not special-cased.
- Hooks, taint, third-party isolation, config files, launch-order validations (sandbox, cwd, paths), `--input-file` copies and the `expires_at` clock are not modeled. `--dry-run` plans are fixed shapes (an empty `origins`, `settings`, `hooks` and so on). Human output is not emulated (nothing is printed without `--json`, except `--version`).
- Interactive `exec`/`resume` need a TTY on stdin and stdout, as in the real tool; tests use a pty to cover that path.

termctl:

- Implemented: `open` (and `--dry-run`), `show`, `exists`, `list`, `rename`, `mark`, `unmark`, `focus`, `close`, `label add|remove`. `placements` is `fake_unsupported`.
- No tmux is involved. Panes live in a state file per socket path; a pane never dies by itself and every pane is `running`, `focus` always "switches" `/dev/pts/0`, and a marker's text is its name (the real text comes from config). Hooks, locking, the audit log, config, the tmux version check and the socket safety checks are not modeled; the "no server" state is "no pane was ever opened on this socket".
- `open` checks that argv[0] is an executable (using the `PATH` of `--env PATH=` when given), as the real command check does.
- The interface says argv without `--` exits 2; the real termctl 0.25.0 opened a shell pane instead (and a `--` with nothing after it exits 2). The fake follows the interface. This is a documentation/implementation difference in termctl, not something to rely on either way.

wtctl:

- Implemented: `create` (worktree and throwaway, including the retry-after-crash `reused` result and per-repo unique names), `list`, `show`, `status`, `remove`, `forget` (with and without `--holder`), `label add|remove`, `locate`, `baseline check`, `reconcile`, `prune`. Not implemented (`fake_unsupported`): `adopt`, `bootstrap`, `extract`, `baseline accept`, `clone`, `export`, `cron`.
- No git is involved. `create` makes an empty directory under `<XDG_STATE_HOME>/wtctl-fake/`; `remove` deletes only that directory. Dirty, unpushed, locked, in-use, integrity and baseline checks always pass; scripting a refusal is how a test gets one. Bootstrap never runs (`state: none`). `remove` without a TTY needs `--yes` (exit 13), `--force` and plain `forget` need a human actor, and holders other than `--holder` refuse with `held`.
- The real wtctl was a scaffold when this was written (every command exited 2 without an error envelope), so there is no live comparison; the fake follows `docs/interface.md` of wtctl and the pinned schemas only.

tmux:

- Only the client options, `-V`, `;` splitting and the "no server" failure are modeled; every other command needs a scripted reply. For real tmux behavior use the `private_tmux` fixture (a real server on `-S <tmp>/sock -f /dev/null` with a private `HOME`).

## Unverified

- claude-session: the real 0.38.4 in the build environment failed with an internal `ModuleNotFoundError` for every call that gets past argument parsing, so the human-only flag checks, `--actor` grammar, a missing `--prompt-file` and everything that needs state or config were compared with the interface document only (rows marked `False` in `fidelity_cases.py`). The shapes of `start`, `exec`, `status`, `stop`, `wait`, `list`, `forget` and `reconcile` outputs are the schema's, not observed.
- The sibling schemas are pinned to interface `0.1.0-design` and copied from the sibling commits recorded in each `PIN.json`; the real programs' released versions are far ahead of the fakes' `--version` (`0.1.0`).
- Signal behavior of the real `exec`, `termctl close` and `wtctl remove` is taken from their interface documents; none was exercised.
- Anything the fakes do not log or model (timing, concurrency between real sibling processes, lease contention) is not tested by them.
