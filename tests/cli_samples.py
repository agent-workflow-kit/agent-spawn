"""The frozen command surface of docs/interface.md, as data shared by the CLI tests.

``FLAGS[command]`` maps every flag the command takes (global flags excluded) to a sample value
(``None`` for a switch). ``MIN_ARGV[command]`` is the smallest argv that parses. The tests assert
that the parser takes exactly these flags, so adding or dropping one is a deliberate change here.
"""

from __future__ import annotations

PASS_THROUGHS: dict[str, str | None] = {
    "--kind": "implement",
    "--skill": "review",
    "--plugin-dir": "/plugins",
    "--model": "haiku",
    "--timeout": "10m",
    "--max-turns": "5",
    "--permission-profile": "strict",
    "--sandbox": "required",
    "--setting-sources": "user,project",
    "--reads-third-party": None,
    "--env-allow": "HOME",
    "--env-allowlist-only": "PATH",
    "--auto-exit": None,
    "--no-auto-exit": None,
}

LAUNCH: dict[str, str | None] = {
    **PASS_THROUGHS,
    "--mode": "headless",
    "--repo": "host/group/project",
    "--worktree": "new",
    "--branch": "feature",
    "--base": "main",
    "--ref": "0" * 40,
    "--detach": None,
    "--worktree-name": "wt",
    "--no-bootstrap": None,
    "--copy-env": None,
    "--cwd": "session-dir",
    "--prompt": "do it",
    "--prompt-file": "-",
    "--input-file": "/tmp/in.txt",
    "--placement": "here",
    "--term-placement": "left",
    "--name": "my agent",
    "--label": "pr=1",
    "--promote": None,
    "--no-promote": None,
    "--ephemeral": None,
    "--wait": None,
    "--wait-timeout": "5m",
    "--result-file": "/tmp/r.json",
    "--hook": "pre-launch=true",
    "--override-escalation": None,
    "--dry-run": None,
}

RESUME: dict[str, str | None] = {
    **PASS_THROUGHS,
    "--select": "k=v",
    "--fork": None,
    "--cwd": "session-dir",
    "--mode": "interactive",
    "--placement": "window",
    "--prompt": "go on",
    "--prompt-file": "/tmp/p.txt",
    "--input-file": "/tmp/in.txt",
    "--reset": "model",
    "--allow-tainted": None,
    "--override-escalation": None,
    "--name": "n",
    "--term-placement": "left",
    "--label": "a=b",
    "--promote": None,
    "--no-promote": None,
    "--wait": None,
    "--wait-timeout": "1h",
    "--result-file": "/tmp/r.json",
    "--hook": "post-launch=true",
    "--dry-run": None,
}

FLAGS: dict[str, dict[str, str | None]] = {
    "launch": LAUNCH,
    "resume": RESUME,
    "wait": {"--timeout": "30s", "--result-file": "/tmp/r.json"},
    "list": {
        "--label": "k=v",
        "--needs-attention": None,
        "--status": "parked",
        "--session-state": "idle",
        "--kind": "review",
        "--repo": "host/group/project",
        "--mode": "headless",
        "--forked-from": "ag-abcdefgh",
        "--since": "1d",
    },
    "show": {},
    "attach": {"--client": "/dev/pts/1", "--no-attach": None},
    "ack": {},
    "label add": {"--session": "cs-abcdefgh", "--dry-run": None},
    "label remove": {"--dry-run": None},
    "stop": {"--grace": "5s", "--dry-run": None},
    "teardown": {
        "--merged-head": "a" * 40,
        "--force": None,
        "--override-escalation": None,
        "--hook": "pre-teardown=true",
        "--dry-run": None,
    },
    "prune": {
        "--older-than": "7d",
        "--include-owned": None,
        "--include-attention": None,
        "--dry-run": None,
    },
    "reconcile": {"--dry-run": None},
    "cron install": {"--interval": "10", "--env-file": "/tmp/env", "--dry-run": None},
    "cron uninstall": {"--dry-run": None},
    "export": {"--since": "1d", "--action": "stop"},
    "browse": {"--focus": "ag-abcdefgh", "--result-file": "/tmp/r.json"},
    "supervise": {"--host": None, "--handoff": "/tmp/s", "--handoff-timeout": "30"},
    "_session-event": {},
}

# The smallest valid argv of each command (positionals only; internal commands carry their
# mandatory absolute --state-dir and --config).
INTERNAL_GLOBALS = ["--state-dir", "/state", "--config", "/state/config.yaml"]
MIN_ARGV: dict[str, list[str]] = {
    "launch": ["launch", "--skill", "s", "--mode", "headless"],
    "resume": ["resume", "ag-abcdefgh"],
    "wait": ["wait", "ag-abcdefgh"],
    "list": ["list"],
    "show": ["show", "ag-abcdefgh"],
    "attach": ["attach", "ag-abcdefgh"],
    "ack": ["ack", "ag-abcdefgh"],
    "label add": ["label", "add", "ag-abcdefgh", "pr=1"],
    "label remove": ["label", "remove", "ag-abcdefgh", "pr=1"],
    "stop": ["stop", "ag-abcdefgh"],
    "teardown": ["teardown", "ag-abcdefgh"],
    "prune": ["prune"],
    "reconcile": ["reconcile"],
    "cron install": ["cron", "install"],
    "cron uninstall": ["cron", "uninstall"],
    "export": ["export"],
    "browse": ["browse"],
    "supervise": [*INTERNAL_GLOBALS, "supervise", "ag-abcdefgh"],
    "_session-event": [*INTERNAL_GLOBALS, "_session-event"],
}

GLOBAL_OPTIONS = {
    "-h",
    "--help",
    "--json",
    "--config",
    "--verbose",
    "--quiet",
    "--yes",
    "--actor",
    "--decision-id",
    "--state-dir",
}
