"""Fake ``wtctl`` (interface 0.1.0-design).

Implemented commands: ``create`` (worktrees and throwaways, with the retry-after-crash ``reused``
path), ``adopt``, ``list``, ``show``, ``status``, ``remove``, ``forget``, ``label add|remove``,
``locate``, ``baseline check``, ``reconcile`` and ``prune``. ``bootstrap``, ``extract``,
``baseline accept``, ``clone``, ``export`` and ``cron`` exit 1 with ``fake_unsupported``.
Records live in ``<FAKEBIN_LOG>.records.json``. ``create`` makes the directory (an empty one,
no git) under ``<state>/wtctl-fake/``; ``remove`` deletes only a directory the fake created.
See ``tests/fakebin/FIDELITY.md``.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

from . import grammars as g
from .core import (
    BOOL,
    FIXED_TS,
    MULTI,
    VALUE,
    Context,
    Exit,
    Parsed,
    base32_id,
    counter,
    first_positional,
    locked,
    parse_args,
    peek_counter,
    read_json,
    require_pattern,
    run_fake,
)

VERSION = "0.1.0"
GLOBAL = {
    "--json": BOOL,
    "--config": VALUE,
    "--verbose": BOOL,
    "--quiet": BOOL,
    "--dry-run": BOOL,
    "--yes": BOOL,
    "--actor": VALUE,
    "--decision-id": MULTI,
}
SPECS: dict[str, dict[str, str]] = {
    "create": {
        **GLOBAL,
        "--repo": VALUE,
        "--branch": VALUE,
        "--detach": BOOL,
        "--base": VALUE,
        "--ref": VALUE,
        "--name": VALUE,
        "--holder": VALUE,
        "--label": MULTI,
        "--no-bootstrap": BOOL,
        "--copy-env": BOOL,
        "--hook": MULTI,
        "--override-escalation": BOOL,
        "--throwaway": BOOL,
        "--source": VALUE,
        "--depth": VALUE,
    },
    "adopt": {
        **GLOBAL,
        "--repo": VALUE,
        "--name": VALUE,
        "--holder": VALUE,
        "--label": MULTI,
    },
    "list": {
        **GLOBAL,
        "--repo": VALUE,
        "--label": MULTI,
        "--throwaways": BOOL,
        "--worktrees": BOOL,
        "--older-than": VALUE,
        "--state": MULTI,
        "--all": BOOL,
    },
    "show": GLOBAL,
    "status": {**GLOBAL, "--merged-head": VALUE, "--holder": VALUE},
    "remove": {
        **GLOBAL,
        "--merged-head": VALUE,
        "--keep-branch": BOOL,
        "--holder": VALUE,
        "--force": BOOL,
        "--hook": MULTI,
        "--override-escalation": BOOL,
    },
    "forget": {**GLOBAL, "--holder": VALUE},
    "label": GLOBAL,
    "locate": GLOBAL,
    "baseline": GLOBAL,
    "reconcile": GLOBAL,
    "prune": {
        **GLOBAL,
        "--throwaways": BOOL,
        "--older-than": VALUE,
        "--if-configured": BOOL,
        "--label": MULTI,
    },
}
UNSUPPORTED = ("bootstrap", "extract", "clone", "export", "cron")
CHECK_NAMES = (
    "integrity locked baseline in_use held fetch tracked untracked ignored nested_repos stash "
    "unpushed_head unpushed_branches"
).split()
LIVE_STATES = ("creating", "active", "unlinked", "missing", "clone_missing", "partial")
ALL_STATES = (*LIVE_STATES, "gone", "removed", "forgotten")
MAX_LABELS = 32


# --- state ---------------------------------------------------------------------------------


def _store_path() -> Path:
    return Path(os.environ["FAKEBIN_LOG"] + ".records.json")


def load_records() -> dict[str, dict[str, Any]]:
    return read_json(_store_path(), "records file", dict, {})


def update_records(change: Callable[[dict[str, dict[str, Any]]], Any]) -> Any:
    path = _store_path()
    with locked(path):
        data: dict[str, dict[str, Any]] = read_json(path, "records file", dict, {})
        result = change(data)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data))
        os.replace(tmp, path)
        return result


def fake_root() -> str:
    base = os.environ.get("XDG_STATE_HOME") or os.path.join(os.path.expanduser("~"), ".local/state")
    return os.path.join(base, "wtctl-fake")


def is_human(ctx: Context, p: Parsed) -> bool:
    actor = p.get("--actor")
    if actor is not None:
        return actor.startswith("human:")
    return ctx.stdin_is_tty()


def check_common(ctx: Context, p: Parsed) -> None:
    if (actor := p.get("--actor")) is not None:
        require_pattern(actor, g.ACTOR, "--actor")
    ids = p.multi("--decision-id")
    if len(ids) > 32:
        ctx.usage("at most 32 --decision-id values")
    for decision in ids:
        require_pattern(decision, g.DECISION_ID, "--decision-id")


# --- records -------------------------------------------------------------------------------


def valid_branch(name: str) -> bool:
    """``git check-ref-format --branch`` (the rules that matter for a name given on argv)."""
    if name in ("", "@") or name.startswith("-") or name.startswith("/") or name.endswith("/"):
        return False
    if name.endswith(".") or name.endswith(".lock") or ".." in name or "@{" in name:
        return False
    if re.search(r"[\x00-\x20\x7f~^:?*\[\\]", name) or "//" in name:
        return False
    return not any(part.startswith(".") or part.endswith(".lock") for part in name.split("/"))


def derive_name(branch: str | None, fallback: str) -> str:
    if not branch:
        return fallback
    name = re.sub(r"[^A-Za-z0-9._-]", "-", branch)
    name = re.sub(r"-+", "-", name).lstrip("-.")[:64]
    return name or fallback


def check_object(status: str = "pass") -> dict[str, Any]:
    return {"status": status, "details": []}


def held_check(holders: list[str]) -> dict[str, Any]:
    return {"status": "fail", "details": [{"holder": h} for h in holders]}


def checks_object() -> dict[str, Any]:
    return {name: check_object() for name in CHECK_NAMES}


def resolve(ctx: Context, raw: str, *, tombstones: bool = False) -> dict[str, Any]:
    """Find a record by id (4 to 8 characters after an optional ``wt-``), full id or path."""
    records = load_records()
    if raw.startswith("/"):
        real = os.path.realpath(raw)
        for record in records.values():
            if record["state"] in LIVE_STATES and os.path.realpath(record["path"]) == real:
                return record
        ctx.fail(10, "not_found", f"no record for path {raw!r}")
    text = raw.lower()
    body = text[3:] if text.startswith("wt-") else text
    if not re.fullmatch(r"[a-z2-7]{4,8}", body):
        ctx.usage(f"invalid record id or path: {raw!r} (a path must be absolute)")
    if len(body) == 8:
        record = records.get(f"wt-{body}")
        if record is None:
            ctx.fail(10, "not_found", f"no record {raw!r}")
        if record["state"] not in LIVE_STATES and not tombstones:
            ctx.fail(
                10,
                "not_found",
                f"record {raw!r} is {record['state']}",
                details={"state": record["state"], "removed_at": record["removed_at"]},
            )
        return record
    live = sorted(
        i for i, r in records.items() if i.startswith(f"wt-{body}") and r["state"] in LIVE_STATES
    )
    if not live:
        ctx.fail(10, "not_found", f"no record matches {raw!r}")
    if len(live) > 1:
        ctx.fail(
            21,
            "ambiguous",
            f"{raw!r} matches several records",
            candidates=[{"id": i, "path": records[i]["path"]} for i in live],
        )
    return records[live[0]]


# --- dispatch ------------------------------------------------------------------------------


def dispatch(ctx: Context) -> tuple[int, Any, str]:
    argv = ctx.argv
    before = argv[: argv.index("--")] if "--" in argv else argv
    if "--version" in before:
        raise Exit(0, f"wtctl {os.environ.get('FAKEBIN_VERSION', VERSION)}\n")
    ctx.json_mode = "--json" in before
    command, rest = _split(ctx, argv)
    if command in UNSUPPORTED:
        ctx.fail(1, "fake_unsupported", f"the fake does not implement {command!r}")
    if command not in SPECS:
        ctx.usage(f"unknown command: {command!r}")
    if any(token in ("-h", "--help") for token in rest):
        raise Exit(0, f"usage: wtctl {command} ...\n")
    if command in ("label", "baseline"):
        # The action is the first positional once the global value flags (--config, --actor,
        # --decision-id) are skipped: `--config X label add ...` is a normal spelling.
        at = first_positional(rest, {"--config", "--actor", "--decision-id"})
        sub = "" if at is None else rest[at]
        allowed = ("add", "remove") if command == "label" else ("check", "accept")
        if sub not in allowed:
            ctx.usage(f"{command} needs one of {', '.join(allowed)}")
        assert at is not None
        del rest[at]
        command = f"{command} {sub}"
        if command == "baseline accept":
            ctx.fail(1, "fake_unsupported", "the fake does not implement 'baseline accept'")
    p = parse_args(rest, SPECS[command.split()[0]])
    ctx.json_mode = p.has("--json")
    if p.has("--verbose") and p.has("--quiet"):
        ctx.usage("--verbose and --quiet are mutually exclusive")
    check_common(ctx, p)
    return HANDLERS[command](ctx, p)


def _split(ctx: Context, argv: list[str]) -> tuple[str, list[str]]:
    value_flags = {name for spec in SPECS.values() for name, kind in spec.items() if kind != BOOL}
    i = 0
    while i < len(argv):
        token = argv[i]
        if token == "--":
            break
        if token.startswith("-") and token != "-":
            name, eq, _ = token.partition("=")
            if name in value_flags and not eq:
                i += 1
        else:
            return token, argv[:i] + argv[i + 1 :]
        i += 1
    ctx.usage("a command is required")


def do_create(ctx: Context, p: Parsed) -> tuple[int, Any, str]:
    if p.positionals:
        ctx.usage(f"unexpected argument: {p.positionals[0]!r}")
    repo = p.get("--repo")
    if repo is None:
        ctx.usage("create needs --repo")
    require_pattern(repo, g.REPO_ID, "--repo")
    throwaway = p.has("--throwaway")
    branch = p.get("--branch")
    if p.has("--branch") and p.has("--detach"):
        ctx.usage("--branch and --detach are mutually exclusive")
    if branch is not None and not valid_branch(branch):
        ctx.usage(f"invalid --branch: {branch!r} (git check-ref-format --branch)")
    if (ref := p.get("--ref")) is not None:
        require_pattern(ref, g.SHA, "--ref")
    if (name := p.get("--name")) is not None:
        require_pattern(name, g.WT_NAME, "--name")
    holder = p.get("--holder")
    if holder is not None:
        require_pattern(holder, g.HOLDER, "--holder")
    labels = p.multi("--label")
    if len(set(labels)) > MAX_LABELS:
        ctx.usage("at most 32 --label values")
    for label in labels:
        require_pattern(label, g.LABEL, "label")
    for hook in p.multi("--hook"):
        hook_name, eq, cmd = hook.partition("=")
        if hook_name not in ("pre-create", "post-create") or not eq or not cmd:
            ctx.usage(f"invalid --hook: {hook!r}")
    if p.has("--override-escalation") and not is_human(ctx, p):
        ctx.usage("--override-escalation needs a human actor")
    if throwaway:
        if ref is None:
            ctx.usage("create --throwaway needs --ref")
        for flag in (
            "--branch",
            "--detach",
            "--base",
            "--name",
            "--holder",
            "--no-bootstrap",
            "--copy-env",
        ):
            if p.has(flag):
                ctx.usage(f"{flag} cannot be used with --throwaway")
    else:
        for flag in ("--source", "--depth"):
            if p.has(flag):
                ctx.usage(f"{flag} needs --throwaway")
    if (source := p.get("--source")) is not None and source not in ("auto", "local", "remote"):
        ctx.usage(f"invalid --source: {source!r}")
    depth = p.get("--depth")
    if depth is not None and not re.fullmatch(r"[1-9][0-9]*", depth):
        ctx.usage("--depth must be a positive integer")
    dry = p.has("--dry-run")

    def create(data: dict[str, dict[str, Any]]) -> tuple[dict[str, Any], bool]:
        # wtctl's Create order: a refusal (branch_in_use, branch_exists, path_exists) that names
        # an active record the holder holds is a retry after a crash and returns that record
        # (`reused`); a dry run answers the same (wtctl interface.md, Create order, "Retry after a
        # crash": "also under --dry-run"). An explicit --name is never suffixed.
        def refuse_or_reuse(existing: dict[str, Any], reason: str, **extra: Any) -> Any:
            if reason != "branch_exists":
                if existing["state"] == "active" and holder in existing["holders"]:
                    return existing, True  # retry after a crash: the record, unchanged
            ctx.fail(
                20,
                "refused",
                f"{reason}: {existing['name']!r}",
                reason=reason,
                **extra,
            )

        if branch is not None:
            for existing in data.values():
                if (
                    existing["state"] in LIVE_STATES
                    and existing["repo"] == repo
                    and existing["branch"] == branch
                ):
                    return refuse_or_reuse(
                        existing,
                        "branch_in_use",
                        holder_id=existing["id"],
                        holder_path=existing["path"],
                    )
        # a dry run reports the id a real create would take, and takes none
        record_id = base32_id("wt-", peek_counter("wt") if dry else counter("wt"))
        taken = {
            r["name"] for r in data.values() if r["state"] in LIVE_STATES and r["repo"] == repo
        }
        wanted = name or derive_name(branch, record_id[3:])
        if name is not None and name in taken:
            holder_record = next(
                r
                for r in data.values()
                if r["state"] in LIVE_STATES and r["repo"] == repo and r["name"] == name
            )
            if branch is None and not p.has("--detach"):
                # the template-built branch collides first: never reused
                ctx.fail(
                    20,
                    "refused",
                    f"the branch of the name {name!r} exists",
                    reason="branch_exists",
                    details={"tip": holder_record["head_at_create"]},
                )
            return refuse_or_reuse(holder_record, "path_exists", holder_id=holder_record["id"])
        final, n = wanted, 1
        while final in taken:
            n += 1
            final = f"{wanted[: 64 - len(str(n)) - 1]}-{n}"
        path = os.path.join(fake_root(), "throwaways" if throwaway else "worktrees", record_id[3:])
        if not throwaway:
            # {root}/{host}/{path}/{name}: the host's ':' becomes '_' (wtctl interface, Locations)
            host, _, project = repo.partition("/")
            path = os.path.join(fake_root(), "worktrees", host.replace(":", "_"), project, final)
        number = (
            int(
                record_id[3:].translate(
                    str.maketrans(
                        "abcdefghijklmnopqrstuvwxyz234567", "0123456789abcdefghijklmnopqrstuv"
                    )
                ),
                32,
            )
            % 100000
        )
        sha = ref or "a" * 40
        record: dict[str, Any] = {
            "id": record_id,
            "object_type": "throwaway" if throwaway else "worktree",
            "owned": None if throwaway else True,
            "state": "active",
            "repo": repo,
            "clone_path": f"/fake/clones/{repo}",
            "common_dir": None if throwaway else f"/fake/clones/{repo}/.git",
            "admin_dir": None if throwaway else f"/fake/clones/{repo}/.git/worktrees/{final}",
            "common_id": None if throwaway else "1:1",
            "dir_id": f"1:{number + 10}",
            "base_dir": os.path.dirname(path),
            "root_id": "1:2" if throwaway else None,
            "path": path,
            "name": None if throwaway else final,
            "branch": None if throwaway else branch,
            "created_branch": None if throwaway else bool(branch),
            "base": None if throwaway else p.get("--base"),
            "base_sha": None if throwaway else "b" * 40,
            "head_at_create": None if throwaway else sha,
            "ref": ref,
            "source": (source if source in ("local", "remote") else "local") if throwaway else None,
            "depth": (int(depth) if depth else 1) if throwaway else None,
            "fetch_url": f"https://{repo}.git",
            "remote": None if throwaway else "origin",
            "labels": sorted(set(labels)),
            "holders": [holder] if holder else [],
            "baseline_digest": None if throwaway else "c" * 64,
            "bootstrap": None
            if throwaway
            else {"state": "none", "steps": [], "created_paths_count": 0},
            "created": FIXED_TS,
            "updated": FIXED_TS,
            "missing_since": None,
            "removed_at": None,
            "last_error": None,
        }
        if not dry:
            os.makedirs(path, exist_ok=True)
            data[record_id] = record
        return record, False

    ctx.check_scripted_error("create")
    record, reused = update_records(create)
    failed_step = ctx.response("create").get("bootstrap_failed")
    if failed_step is not None and not dry and not reused:
        # A failed bootstrap step keeps the worktree and the record (active) and exits 12.
        ctx.fail(
            12,
            "bootstrap_failed",
            "a bootstrap step failed",
            details={
                "record": record,
                "step": {
                    "argv": [str(failed_step)],
                    "exit": 1,
                    "duration_ms": 0,
                    "output_tail": "",
                },
            },
        )
    if record["object_type"] == "throwaway":
        action = None
        bootstrap = None
    else:
        action = None if reused else ("new" if branch else "detached")
        bootstrap = (
            None
            if reused
            else {"state": "none", "steps": [], "created_paths_count": 0, "tracked_changes": []}
        )
    out = {
        "record": record,
        "branch_action": action,
        "bootstrap": bootstrap,
        "reused": reused,
        "warnings": [],
        "dry_run": dry and not reused,
    }
    return 0, out, "create"


def do_adopt(ctx: Context, p: Parsed) -> tuple[int, Any, str]:
    if len(p.positionals) != 1:
        ctx.usage("adopt takes exactly one path")
    raw = p.positionals[0]
    if not raw.startswith("/"):
        ctx.usage(f"adopt needs an absolute path, got {raw!r}")
    if (repo := p.get("--repo")) is not None:
        require_pattern(repo, g.REPO_ID, "--repo")
    if (name := p.get("--name")) is not None:
        require_pattern(name, g.WT_NAME, "--name")
    holder = p.get("--holder")
    if holder is not None:
        require_pattern(holder, g.HOLDER, "--holder")
    labels = p.multi("--label")
    for label in labels:
        require_pattern(label, g.LABEL, "label")
    real = os.path.realpath(raw)
    root = os.path.realpath(fake_root())
    if real == root or real.startswith(root + os.sep + "throwaways"):
        ctx.usage("the path lies in wtctl's state dir or throwaway root")
    dry = p.has("--dry-run")

    def adopt(data: dict[str, dict[str, Any]]) -> tuple[dict[str, Any], bool]:
        for existing in data.values():
            if existing["state"] in LIVE_STATES and os.path.realpath(existing["path"]) == real:
                state = existing["state"]
                if state == "missing":  # the path is not a git worktree any more
                    ctx.fail(10, "not_found", f"{raw!r} is not a git worktree")
                if state == "creating":
                    ctx.fail(
                        20,
                        "refused",
                        "the record is being created",
                        reason="path_exists",
                        holder_id=existing["id"],
                    )
                if state != "active":  # unlinked, clone_missing, partial
                    ctx.fail(20, "refused", f"the record is {state}", reason="unlinked")
                if holder is not None and holder not in existing["holders"] and not dry:
                    existing["holders"] = sorted(
                        [*existing["holders"], holder]
                    )  # sorted, as wtctl lists them
                return dict(existing), True
        if not os.path.isdir(real):
            ctx.fail(10, "not_found", f"{raw!r} is not a git worktree")
        record_id = base32_id("wt-", peek_counter("wt") if dry else counter("wt"))
        base = re.sub(r"[^A-Za-z0-9._-]", "-", os.path.basename(real)).lstrip("-.")[:64]
        wanted = name or base or record_id[3:]
        the_repo = repo or "example.invalid/fake/adopted"
        taken = {
            r["name"] for r in data.values() if r["state"] in LIVE_STATES and r["repo"] == the_repo
        }
        final, n = wanted, 1
        while final in taken:
            n += 1
            final = f"{wanted[: 64 - len(str(n)) - 1]}-{n}"
        record: dict[str, Any] = {
            "id": record_id,
            "object_type": "worktree",
            "owned": False,
            "state": "active",
            "repo": the_repo,
            "clone_path": f"/fake/clones/{the_repo}",
            "common_dir": f"/fake/clones/{the_repo}/.git",
            "admin_dir": f"/fake/clones/{the_repo}/.git/worktrees/{final}",
            "common_id": "1:1",
            "dir_id": "1:7",
            "base_dir": None,
            "root_id": None,
            "path": real,
            "name": final,
            "branch": None,
            "created_branch": False,
            "base": None,
            "base_sha": None,
            "head_at_create": "a" * 40,
            "ref": None,
            "source": None,
            "depth": None,
            "fetch_url": f"https://{the_repo}.git",
            "remote": "origin",
            "labels": sorted(set(labels)),
            "holders": [holder] if holder else [],
            "baseline_digest": "c" * 64,
            "bootstrap": {"state": "none", "steps": [], "created_paths_count": 0},
            "created": FIXED_TS,
            "updated": FIXED_TS,
            "missing_since": None,
            "removed_at": None,
            "last_error": None,
        }
        if not dry:
            data[record_id] = record
        return record, False

    ctx.check_scripted_error("adopt")
    record, existing = update_records(adopt)
    return 0, {"record": record, "existing": existing, "dry_run": dry}, "adopt"


def do_list(ctx: Context, p: Parsed) -> tuple[int, Any, str]:
    if p.positionals:
        ctx.usage("list takes no positional arguments")
    if p.has("--throwaways") and p.has("--worktrees"):
        ctx.usage("--throwaways and --worktrees are mutually exclusive")
    if (repo := p.get("--repo")) is not None:
        require_pattern(repo, g.REPO_ID, "--repo")
    if (older := p.get("--older-than")) is not None:
        require_pattern(older, g.WT_DURATION, "--older-than")
    for label in p.multi("--label"):
        require_pattern(label, g.LABEL, "label")
    wanted = set(p.multi("--label"))
    states = p.multi("--state")
    for state in states:
        if state not in ALL_STATES:
            ctx.usage(f"invalid --state: {state!r}")
    records = [
        r
        for r in load_records().values()
        if (r["state"] in states if states else r["state"] in LIVE_STATES)
        and (repo is None or r["repo"] == repo)
        and wanted <= set(r["labels"])
        and (not p.has("--throwaways") or r["object_type"] == "throwaway")
        and (not p.has("--worktrees") or r["object_type"] == "worktree")
    ]
    records.sort(key=lambda r: (r["created"], r["id"]), reverse=True)
    return 0, {"records": records, "foreign": []}, "list"


def do_show(ctx: Context, p: Parsed) -> tuple[int, Any, str]:
    if len(p.positionals) != 1:
        ctx.usage("show takes exactly one record id or path")
    return 0, resolve(ctx, p.positionals[0], tombstones=True), "show"


def content_refusal(record: dict[str, Any]) -> tuple[str, str] | None:
    """The content refusal a test planted on a record (``fake_dirty`` / ``fake_unpushed``) as
    ``(reason, failing check)``; real wtctl reports the check names in ``force_overrides``."""
    if record.get("fake_dirty"):
        return "dirty", "tracked"
    if record.get("fake_unpushed"):
        return "unpushed", "unpushed_head"
    return None


def status_object(record: dict[str, Any], *, dry_run: bool | None = None) -> dict[str, Any]:
    held = bool(record["holders"])
    content = content_refusal(record)
    reason = "held" if held else (content[0] if content else None)
    overrides = (["held"] if held else []) + ([content[1]] if content else [])
    out: dict[str, Any] = {
        "id": record["id"],
        "object_type": record["object_type"],
        "would_remove": reason is None,
        "reason": reason,
        # an adopted worktree is removed by a human only
        "human_required": record["object_type"] == "worktree" and record.get("owned") is False,
        "checks": checks_object(),
        "force_overrides": overrides,
        "head": record["head_at_create"],
        "current_branch": record["branch"],
        "warnings": [],
    }
    if held:
        out["checks"]["held"] = held_check(record["holders"])
    if dry_run is not None:
        out["dry_run"] = dry_run
    return out


def do_status(ctx: Context, p: Parsed) -> tuple[int, Any, str]:
    if len(p.positionals) != 1:
        ctx.usage("status takes exactly one record id or path")
    if (head := p.get("--merged-head")) is not None:
        require_pattern(head, g.SHA, "--merged-head")
    if (holder := p.get("--holder")) is not None:
        require_pattern(holder, g.HOLDER, "--holder")
    record = resolve(ctx, p.positionals[0])
    if holder is not None and set(record["holders"]) <= {holder}:
        record = {**record, "holders": []}
    return 0, status_object(record), "status"


def do_remove(ctx: Context, p: Parsed) -> tuple[int, Any, str]:
    if len(p.positionals) != 1:
        ctx.usage("remove takes exactly one record id or path")
    if (head := p.get("--merged-head")) is not None:
        require_pattern(head, g.SHA, "--merged-head")
    holder = p.get("--holder")
    if holder is not None:
        require_pattern(holder, g.HOLDER, "--holder")
    if p.has("--force") and not is_human(ctx, p):
        ctx.usage("--force needs a human actor")
    if p.has("--override-escalation") and not is_human(ctx, p):
        ctx.usage("--override-escalation needs a human actor")
    for hook in p.multi("--hook"):
        hook_name, eq, cmd = hook.partition("=")
        if hook_name not in ("pre-remove", "post-remove") or not eq or not cmd:
            ctx.usage(f"invalid --hook: {hook!r}")
    record = resolve(ctx, p.positionals[0])
    others = [h for h in record["holders"] if h != holder]
    forced = p.has("--force")
    if others and not forced:
        ctx.fail(
            20,
            "refused",
            "the record is held",
            reason="held",
            checks={**checks_object(), "held": held_check(others)},
            details={"holders": record["holders"]},
        )
    content = content_refusal(record)
    if content is not None and not forced:
        ctx.fail(
            20,
            "refused",
            f"the worktree is {content[0]}",
            reason=content[0],
            checks=checks_object(),
        )
    if p.has("--dry-run"):
        return 0, status_object({**record, "holders": others}, dry_run=True), "remove"
    if record["object_type"] == "worktree" and not p.has("--yes") and not ctx.stdin_is_tty():
        ctx.fail(13, "confirmation_required", "no terminal and no --yes; nothing changed")

    def remove(data: dict[str, dict[str, Any]]) -> dict[str, Any]:
        current = data[record["id"]]
        current.update(state="removed", removed_at=FIXED_TS, holders=[])
        return dict(current)

    ctx.check_scripted_error("remove")
    removed = update_records(remove)
    root = os.path.realpath(fake_root())
    if os.path.realpath(removed["path"]).startswith(root + os.sep):
        shutil.rmtree(removed["path"], ignore_errors=True)
    kept = "keep_branch" if p.has("--keep-branch") else "not_created"
    out = {
        "record": removed,
        "checks": checks_object(),
        "overridden": (["held"] if others and forced else [])
        + ([content[1]] if content is not None and forced else []),
        "branch_deleted": False,
        "branch_kept_reason": kept if removed["object_type"] == "worktree" else None,
        "warnings": [],
    }
    return 0, out, "remove"


def do_forget(ctx: Context, p: Parsed) -> tuple[int, Any, str]:
    if len(p.positionals) != 1:
        ctx.usage("forget takes exactly one record id or path")
    holder = p.get("--holder")
    if holder is not None:
        require_pattern(holder, g.HOLDER, "--holder")
    elif not is_human(ctx, p):
        ctx.fail(13, "confirmation_required", "plain forget is a human-only command")
    record = resolve(ctx, p.positionals[0])
    if record["object_type"] == "throwaway":
        ctx.usage("a throwaway is removed, not forgotten")
    warnings: list[str] = []
    dry = p.has("--dry-run")

    def forget(data: dict[str, dict[str, Any]]) -> dict[str, Any]:
        current = data[record["id"]]
        if holder is None:
            current.update(state="forgotten", removed_at=FIXED_TS, holders=[])
        elif holder in current["holders"]:
            if not dry:
                current["holders"] = [h for h in current["holders"] if h != holder]
                # The record becomes forgotten only when this call removed the last holder and
                # the record is adopted; an owned record is never forgotten through --holder.
                if not current["holders"] and current["owned"] is False:
                    current.update(state="forgotten", removed_at=FIXED_TS)
        else:
            warnings.append("not_a_holder")
        return dict(current)

    ctx.check_scripted_error("forget")
    updated = update_records(forget)
    return 0, {"record": updated, "warnings": warnings, "dry_run": dry}, "forget"


def label_handler(sub: str) -> Callable[[Context, Parsed], tuple[int, Any, str]]:
    def handler(ctx: Context, p: Parsed) -> tuple[int, Any, str]:
        if len(p.positionals) < 2:
            ctx.usage(f"label {sub} needs a record and at least one key=value")
        for pair in p.positionals[1:]:
            require_pattern(pair, g.LABEL, "label")
        record = resolve(ctx, p.positionals[0])
        dry = p.has("--dry-run")

        def change(data: dict[str, dict[str, Any]]) -> list[str]:
            current = set(data[record["id"]]["labels"])
            wanted = set(p.positionals[1:])
            result = sorted(current | wanted if sub == "add" else current - wanted)
            if len(result) > MAX_LABELS:
                ctx.usage("a record holds at most 32 labels")
            if not dry:
                data[record["id"]]["labels"] = result
            return result

        ctx.check_scripted_error(f"label {sub}")
        labels = update_records(change)
        return 0, {"id": record["id"], "labels": labels, "dry_run": dry}, f"label {sub}"

    return handler


def do_locate(ctx: Context, p: Parsed) -> tuple[int, Any, str]:
    if len(p.positionals) != 1:
        ctx.usage("locate takes exactly one repo id")
    repo = p.positionals[0]
    require_pattern(repo, g.REPO_ID, "repo id")
    out = {
        "repo": repo,
        "clone_path": f"/fake/clones/{repo}",
        "common_dir": f"/fake/clones/{repo}/.git",
        "source": "config",
        "fetch_url": f"https://{repo}.git",
        "default_branch": "main",
    }
    return 0, out, "locate"


def do_baseline_check(ctx: Context, p: Parsed) -> tuple[int, Any, str]:
    if len(p.positionals) != 1:
        ctx.usage("baseline check takes exactly one record id or path")
    record = resolve(ctx, p.positionals[0])
    if record["object_type"] == "throwaway":
        ctx.usage("a throwaway has no baseline")
    digest = record["baseline_digest"]
    return (
        0,
        {"id": record["id"], "changed": False, "digest": digest, "recorded_digest": digest},
        "baseline check",
    )


def do_reconcile(ctx: Context, p: Parsed) -> tuple[int, Any, str]:
    if p.positionals:
        ctx.usage("reconcile takes no arguments")
    out = {"changes": [], "reports": [], "already_running": False, "dry_run": p.has("--dry-run")}
    return 0, out, "reconcile"


def do_prune(ctx: Context, p: Parsed) -> tuple[int, Any, str]:
    if p.positionals:
        ctx.usage("prune takes no arguments")
    if not p.has("--throwaways"):
        ctx.usage("prune needs --throwaways")
    older = p.get("--older-than")
    if older is not None:
        require_pattern(older, g.WT_DURATION, "--older-than")
    elif not p.has("--if-configured"):
        ctx.usage("prune needs --older-than or --if-configured")
    for label in p.multi("--label"):
        require_pattern(label, g.LABEL, "label")
    out = {"removed": [], "skipped": [], "already_running": False, "dry_run": p.has("--dry-run")}
    return 0, out, "prune"


HANDLERS: dict[str, Callable[[Context, Parsed], tuple[int, Any, str]]] = {
    "create": do_create,
    "adopt": do_adopt,
    "list": do_list,
    "show": do_show,
    "status": do_status,
    "remove": do_remove,
    "forget": do_forget,
    "label add": label_handler("add"),
    "label remove": label_handler("remove"),
    "locate": do_locate,
    "baseline check": do_baseline_check,
    "reconcile": do_reconcile,
    "prune": do_prune,
}


def main(argv: list[str]) -> int:
    return run_fake("wtctl", argv, dispatch)
