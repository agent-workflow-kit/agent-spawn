"""Self-labeling: ``label add --session <csession_id> key=value...``, run inside a session.

The caller is an agent session, so everything about it is distrusted until proven (interface.md,
Self-labeling). Every check fails closed and runs in the documented order:

1. the id equals ``CSESSION_ID`` and ``CSESSION_DIR`` is set (exit 2);
2. every key is in ``self_label_keys`` (exit 2);
3. an existing ``state.db`` (never created: exit 12);
4. ``claude-session status``: not tainted, current run not third-party-reading, the directory
   matches ``CSESSION_DIR`` (exit 11);
5. ``$CSESSION_DIR/secret`` opened without following links, a regular file of this user, mode
   0600, at most 128 bytes (exit 11);
6. ``claude-session verify-secret`` with the secret on stdin (exit 11 on a mismatch);
7. the pairs are added to the agent whose session this is (exit 10 if none) and mirrored.

Messages never contain the secret, the directory contents or any text read from the session.
"""

from __future__ import annotations

import os
import stat
from typing import Any

from . import gateway as gw
from .agent_lookup import backend_for, open_existing, store_failed
from .annotate import agent_lease, log_quietly, mirror_labels, print_warnings
from .commands.base import Invocation
from .labels import Label, parse_user_label
from .output import CliError, Outcome, PayloadKind
from .session_backend import SESSION_ID
from .store import StoreError
from .validate import GrammarError

__all__ = ["MAX_SECRET_BYTES", "read_secret", "self_label"]

MAX_SECRET_BYTES = 128
_SECRET_NAME = "secret"


def refused(why: str) -> CliError:
    """A failed self-labeling check: exit 11 (a fixed message, never untrusted text)."""
    return CliError("vetoed", f"self-labeling refused: {why}", exit_code=11)


def read_secret(directory: str, uid: int, expected: tuple[int, int]) -> str:
    """The session secret at ``<directory>/secret``, or a refusal (exit 11).

    ``directory`` is a realpath. The file is opened with ``O_NOFOLLOW`` (and ``O_NONBLOCK``, so a
    FIFO cannot hang the open) relative to an open directory descriptor; the checks run on the
    descriptor with ``fstat`` after the open, and the read is bounded. ``expected`` is the
    ``(st_dev, st_ino)`` of the directory taken when the path was checked, before this open.
    """
    dir_fd = -1
    fd = -1
    try:
        try:
            dir_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            fd = os.open(
                _SECRET_NAME,
                os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC,
                dir_fd=dir_fd,
            )
        except OSError:
            raise refused("the session secret cannot be opened") from None
        # The descriptor must be the directory that was checked before the open (a swap after
        # that check would otherwise open another session's secret).
        try:
            opened = os.fstat(dir_fd)
        except OSError:
            raise refused("the session directory cannot be checked") from None
        if (opened.st_dev, opened.st_ino) != expected:
            raise refused("the session directory changed while it was opened")
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise refused("the session secret is not a regular file")
        if info.st_uid != uid:
            raise refused("the session secret is not owned by this user")
        if stat.S_IMODE(info.st_mode) != 0o600:
            raise refused("the session secret does not have mode 0600")
        if info.st_size > MAX_SECRET_BYTES:
            raise refused("the session secret is too large")
        try:
            data = os.read(fd, MAX_SECRET_BYTES + 1)
        except OSError:
            raise refused("the session secret cannot be read") from None
        if not data or len(data) > MAX_SECRET_BYTES:
            raise refused("the session secret is empty or too large")
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError:
            raise refused("the session secret is not text") from None
    finally:
        for descriptor in (fd, dir_fd):
            if descriptor >= 0:
                os.close(descriptor)


def _lower_failed(command: str, error: CliError) -> CliError:
    """A failed lower-tool call as a fixed message: no lower text, same exit code."""
    return CliError("failed", f"claude-session {command} failed", exit_code=error.exit_code)


def _check_agent(row: dict[str, Any], session_id: str) -> None:
    """The checks on the agent record that a self-label needs (run before and under the lease)."""
    if row["status"] == "torn_down" or row.get("csession_id") != session_id:
        raise CliError.of("not_found", "no agent has this session")
    if row["status"] == "tearing_down":
        raise refused("the agent is being torn down")
    if row.get("tainted") is True:
        raise refused("the agent is recorded as tainted")


def _realpath(path: str) -> str:
    try:
        return os.path.realpath(path, strict=True)
    except (OSError, ValueError):
        raise refused("the session directory cannot be resolved") from None


def self_label(inv: Invocation) -> Outcome:
    args = inv.args
    assert inv.config is not None
    # 1. who the caller says it is
    session_id: str = args.session
    from_env = os.environ.get("CSESSION_ID")
    session_dir = os.environ.get("CSESSION_DIR")
    if SESSION_ID.fullmatch(session_id) is None:
        raise CliError.usage("--session needs a full session id")
    if from_env is None or from_env != session_id:
        raise CliError.usage("--session must equal CSESSION_ID (the full id)")
    if not session_dir:
        raise CliError.usage("CSESSION_DIR is not set: self-labeling runs inside an agent session")
    # 2. only the allowed keys, add only (the parser offers no remove with --session)
    try:
        pairs: list[Label] = list(dict.fromkeys(parse_user_label(text) for text in args.pairs))
    except GrammarError as exc:
        raise CliError.usage(str(exc)) from None
    allowed = set(inv.config.self_label_keys)
    for pair in pairs:
        if pair.key not in allowed:
            raise CliError.usage(f"label key {pair.key!r} is not in self_label_keys")
    # 3. never create a database
    store = open_existing(inv, migrate=False)
    if store is None:
        raise CliError(
            "failed",
            "no state.db: self-labeling never creates one",
            details={
                "hint": "the session's environment must carry XDG_STATE_HOME and "
                "XDG_CONFIG_HOME when they are not the defaults"
            },
        )
    warnings: list[str] = []
    try:
        with store:
            # 4. claude-session's view of the session (any failure refuses: fail closed)
            backend = backend_for(inv)
            try:
                status = backend.status(session_id)
            except CliError as error:  # the lower tool's text may carry the directory
                raise _lower_failed("status", error) from None
            if status.csession_id != session_id:
                raise refused("claude-session answered for another session")
            if status.tainted:
                raise refused("the session is tainted")
            if status.run.reads_third_party:
                raise refused("the current run reads third-party content")
            real_dir = _realpath(session_dir)
            if real_dir != _realpath(status.csession_dir):
                raise refused("CSESSION_DIR is not this session's directory")
            try:
                dir_info = os.lstat(real_dir)
            except OSError:
                raise refused("the session directory cannot be resolved") from None
            # 5. the secret file
            secret = read_secret(
                real_dir, gw.get_gateway().getuid(), (dir_info.st_dev, dir_info.st_ino)
            )
            # 6. verified by claude-session; the secret travels on stdin only
            try:
                verified = backend.verify_secret(session_id, secret)
            except CliError as error:  # the lower tool's text may carry the secret
                raise _lower_failed("verify-secret", error) from None
            if verified is not True:
                raise refused("the session secret does not match")
            # 7. the agent whose session this is
            matches = [
                row
                for row in store.list_agents()
                if row.get("csession_id") == session_id and row["status"] != "torn_down"
            ]
            if not matches:
                raise CliError.of("not_found", "no agent has this session")
            if len(matches) > 1:
                raise CliError("internal", "several agents claim this session")
            _check_agent(matches[0], session_id)
            agent_id = matches[0]["agent_id"]
            with agent_lease(store, agent_id) as lease:
                # the row was read before the lease: read it again and repeat the checks
                row = store.get_agent(agent_id)
                if row is None:
                    raise CliError.of("not_found", "no agent has this session")
                _check_agent(row, session_id)
                try:
                    change = store.edit_labels(lease, agent_id, "add", pairs)
                except GrammarError as exc:
                    raise CliError.usage(str(exc)) from None
                final = store.get_labels(agent_id).texts()
            if change.changed:
                log_quietly(
                    inv,
                    warnings,
                    "label_change",
                    agent_id=agent_id,
                    csession_id=session_id,
                    details={"op": "add", "labels": [str(p) for p in change.applied]},
                )
                mirror_labels(inv, store, row, "add", change.applied, warnings)
    except StoreError as error:
        raise store_failed(error) from None
    print_warnings(inv, warnings)
    payload: dict[str, Any] = {
        "agent_id": agent_id,
        "labels": final,
        "added": [str(pair) for pair in change.applied],
        "changed": change.changed,
        "warnings": warnings,
        "dry_run": False,
    }
    return Outcome(0, payload, PayloadKind.COMMAND, f"{agent_id}: labels updated")
