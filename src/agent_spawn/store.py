"""The state store: ``state.db`` (SQLite, WAL), leases, compare-and-set helpers and the outbox.

Rules every method follows:

* Untrusted text only ever travels as a bound parameter. Column names come from the fixed specs
  in ``store_schema``; a name outside them is a ``ValueError``, never interpolated.
* Every write is one ``BEGIN IMMEDIATE`` transaction, so a change and its outbox events commit
  together or not at all. ``transaction()`` joins an enclosing transaction.
* A lifecycle entry is a compare-and-set under the agent's lease (interface.md, Lifecycle
  transitions). The lease is verified again inside the write transaction.
* Clocks, ``/proc`` reads and sleeping go through the gateway and ``procs``.

Leases are keyed by holder identity (PID, start time, boot id). A lease whose holder is dead
(no such process, another start time or boot id, or a zombie) is taken over at once; a live
holder is never preempted, however long it holds. An unparseable process entry is an error,
never "dead".
"""

from __future__ import annotations

import os
import re
import sqlite3
import stat
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from . import procs
from .gateway import Gateway, get_gateway
from .ids import IdEntry, new_agent_id
from .labels import Label, LabelChange, LabelSet
from .output import CliError
from .procs import ProcessIdentity
from .store_schema import (
    AGENT_COLUMNS,
    MIGRATIONS,
    RUN_COLUMNS,
    Migration,
    ValueDecodeError,
    decode_value,
    encode_value,
)

__all__ = [
    "AgentMissing",
    "CasLost",
    "IdConflict",
    "Lease",
    "LeaseHeld",
    "LeaseLost",
    "OutboxEvent",
    "SchemaOlder",
    "Store",
    "StoreBusy",
    "StoreCorrupt",
    "StoreError",
    "StoreMissing",
    "default_state_dir",
]

DB_NAME = "state.db"
DIR_MODE = 0o700
FILE_MODE = 0o600
DEFAULT_BUSY_TIMEOUT = 5.0
RELAY_EVENT_DAYS = 7
_SIDECARS = ("-wal", "-shm", "-journal")
_AGENT_ID = re.compile(r"ag-[a-z2-7]{8}")
_LEASE_KEY = re.compile(r"(?:agent|delivery):ag-[a-z2-7]{8}|reconcile")
_BUSY_CODES = frozenset({5, 6})  # SQLITE_BUSY, SQLITE_LOCKED
_IMMUTABLE = frozenset({"agent_id", "created"})
_ALLOCATION_ATTEMPTS = 32


class StoreError(Exception):
    """The store cannot do what was asked (not a lost race)."""


class StoreMissing(StoreError):
    """``state.db`` does not exist and the caller must not create it."""


class StoreCorrupt(StoreError):
    """The database or a stored value is not what agent-spawn wrote."""


class StoreBusy(StoreError):
    """Another process held the write lock for longer than the busy timeout."""


class SchemaOlder(StoreError):
    """The database schema is older and this open may not migrate it."""

    def __init__(self, found: int, expected: int) -> None:
        super().__init__(f"state.db has schema {found}, this agent-spawn needs {expected}")
        self.found = found
        self.expected = expected


class IdConflict(StoreError):
    """The agent id was allocated by someone else first: allocate another."""

    def __init__(self, agent_id: str) -> None:
        super().__init__(f"agent id {agent_id} is already allocated")
        self.agent_id = agent_id


class AgentMissing(StoreError):
    def __init__(self, agent_id: str) -> None:
        super().__init__(f"no agent record {agent_id}")
        self.agent_id = agent_id


class LeaseHeld(StoreError):
    """A live process holds the lease."""

    def __init__(self, key: str, holder: ProcessIdentity) -> None:
        super().__init__(f"lease {key} is held by pid {holder.pid}")
        self.key = key
        self.holder = holder


class LeaseLost(StoreError):
    """The lease passed to a write is no longer held by its holder."""


class CasLost(StoreError):
    """A compare-and-set found another status (or other expected value): the loser."""

    def __init__(self, agent_id: str, expected: tuple[str, ...], current: dict[str, Any] | None):
        actual = None if current is None else current["status"]
        super().__init__(f"agent {agent_id} is {actual}, expected {'|'.join(expected) or 'any'}")
        self.agent_id = agent_id
        self.expected = expected
        self.current = current

    @property
    def status(self) -> str | None:
        return None if self.current is None else str(self.current["status"])


@dataclass(frozen=True)
class Lease:
    key: str
    holder: ProcessIdentity
    took_over_from: ProcessIdentity | None = field(default=None, compare=False)


@dataclass(frozen=True)
class OutboxEvent:
    """An agent-spawn hook event, written in the transaction of the change it describes."""

    event: str
    payload: Mapping[str, Any] = field(default_factory=dict)


def default_state_dir(environ: Mapping[str, str] | None = None) -> Path:
    """``$XDG_STATE_HOME/agent-spawn``, default ``~/.local/state/agent-spawn`` (CLI only)."""
    env = os.environ if environ is None else environ
    base = env.get("XDG_STATE_HOME", "")
    if not base or not os.path.isabs(base):
        home = env.get("HOME", "")
        base = os.path.join(home if home else os.path.expanduser("~"), ".local", "state")
    return Path(base) / "agent-spawn"


def _map(exc: sqlite3.Error) -> Exception:
    code = getattr(exc, "sqlite_errorcode", None)
    if code is not None and (code & 0xFF) in _BUSY_CODES:
        return StoreBusy(f"state.db is busy: {exc}")
    if isinstance(exc, sqlite3.DatabaseError) and not isinstance(
        exc, sqlite3.IntegrityError | sqlite3.OperationalError
    ):
        return StoreCorrupt(f"state.db is unreadable: {exc}")
    return StoreError(f"state.db error: {exc}")


def _row(row: sqlite3.Row, columns: Mapping[str, str], *keys: str) -> dict[str, Any]:
    out: dict[str, Any] = {key: row[key] for key in keys}
    for name, kind in columns.items():
        try:
            out[name] = decode_value(kind, row[name])
        except ValueDecodeError as exc:
            raise StoreCorrupt(f"column {name}: {exc}") from exc
    return out


def _check_columns(changes: Mapping[str, object], columns: Mapping[str, str]) -> None:
    for name in changes:
        if name not in columns or name in _IMMUTABLE:
            raise ValueError(f"cannot write column {name!r}")


class Store:
    """An open ``state.db``. One instance per process and thread."""

    def __init__(
        self,
        conn: sqlite3.Connection,
        path: Path,
        gateway: Gateway,
        *,
        identity: ProcessIdentity | None = None,
    ) -> None:
        self._conn = conn
        self.path = path
        self.gateway = gateway
        self._identity = identity
        self._in_txn = False

    # -- opening -------------------------------------------------------------------------

    @classmethod
    def open(
        cls,
        state_dir: str | os.PathLike[str],
        *,
        create: bool = True,
        migrate: bool = True,
        gateway: Gateway | None = None,
        busy_timeout: float = DEFAULT_BUSY_TIMEOUT,
        migrations: Sequence[Migration] = MIGRATIONS,
        identity: ProcessIdentity | None = None,
    ) -> Store:
        """Open (and with ``create``, make) the state dir and ``state.db``.

        A newer schema exits 12 ``schema_newer`` before anything is written. An older one is
        migrated when ``migrate``; otherwise ``SchemaOlder`` (the relay never migrates). A
        missing database with ``create=False`` is ``StoreMissing``.
        """
        gw = gateway or get_gateway()
        path = _prepare_files(Path(state_dir), create=create, uid=gw.getuid())
        try:
            conn = sqlite3.connect(
                str(path),
                timeout=busy_timeout,
                isolation_level=None,
                check_same_thread=True,
            )
        except (sqlite3.Error, ValueError) as exc:
            raise StoreError(f"cannot open {path}: {exc}") from exc
        conn.row_factory = sqlite3.Row
        store = cls(conn, path, gw, identity=identity)
        try:
            store._setup(migrate=migrate, migrations=migrations)
        except BaseException:
            conn.close()
            raise
        return store

    def _setup(self, *, migrate: bool, migrations: Sequence[Migration]) -> None:
        wanted = len(migrations)
        try:
            self._conn.execute("PRAGMA foreign_keys = ON")
            version = self._read_version()
            if version is not None and version > wanted:
                raise CliError.of(
                    "schema_newer",
                    f"state.db has schema {version}; this agent-spawn understands up to "
                    f"{wanted}: upgrade agent-spawn",
                )
            mode = self._conn.execute("PRAGMA journal_mode").fetchone()[0]
            if str(mode).lower() != "wal":
                if version is not None and version < wanted and not migrate:
                    raise SchemaOlder(version, wanted)
                mode = self._conn.execute("PRAGMA journal_mode = WAL").fetchone()[0]
                if str(mode).lower() != "wal":
                    raise StoreError(f"cannot enable WAL for {self.path} (journal_mode {mode})")
            self._conn.execute("PRAGMA synchronous = FULL")
        except sqlite3.Error as exc:
            raise _map(exc) from exc
        if version is None or version < wanted:
            if not migrate:
                raise SchemaOlder(0 if version is None else version, wanted)
            self._migrate(migrations)
        _fix_sidecar_modes(self.path)

    def _read_version(self) -> int | None:
        """The schema version; ``None`` if empty; ``StoreCorrupt`` for a foreign database."""
        tables = {
            row[0]
            for row in self._conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
        if not tables:
            return None
        if "meta" not in tables:
            raise StoreCorrupt(f"{self.path} is not an agent-spawn database")
        row = self._conn.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
        if row is None or not str(row[0]).isascii() or not str(row[0]).isdigit():
            raise StoreCorrupt(f"{self.path} has no valid schema version")
        return int(row[0])

    def _migrate(self, migrations: Sequence[Migration]) -> None:
        wanted = len(migrations)
        with self.transaction():
            # Re-read under the write lock: another process may have migrated meanwhile.
            version = self._read_version() or 0
            if version > wanted:
                raise CliError.of(
                    "schema_newer",
                    f"state.db has schema {version}; this agent-spawn needs {wanted}",
                )
            for step in range(version, wanted):
                migrations[step](self._conn)
                self._conn.execute(
                    "INSERT INTO meta (key, value) VALUES ('schema_version', ?) "
                    "ON CONFLICT (key) DO UPDATE SET value = excluded.value",
                    (str(step + 1),),
                )

    def schema_version(self) -> int:
        return self._read_version() or 0

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> Store:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- transactions --------------------------------------------------------------------

    @contextmanager
    def transaction(self) -> Iterator[None]:
        """``BEGIN IMMEDIATE`` .. ``COMMIT``; joins an enclosing transaction."""
        if self._in_txn:
            yield
            return
        try:
            self._conn.execute("BEGIN IMMEDIATE")
        except sqlite3.Error as exc:
            raise _map(exc) from exc
        self._in_txn = True
        try:
            yield
            self._conn.execute("COMMIT")
        except BaseException as exc:
            try:
                self._conn.execute("ROLLBACK")
            except sqlite3.Error:
                pass  # a failed COMMIT may have rolled back already
            if isinstance(exc, sqlite3.Error):
                raise _map(exc) from exc
            if isinstance(exc, UnicodeEncodeError):
                raise StoreError(f"text cannot be stored: {exc}") from exc
            raise
        finally:
            self._in_txn = False

    def _now(self) -> float:
        return self.gateway.wall_time()

    # -- AgentIndex (ids.py) -------------------------------------------------------------

    def entries_with_prefix(self, prefix: str) -> list[IdEntry]:
        """Stored records (tombstones included) whose id starts with ``prefix``: one range scan
        of the primary key."""
        try:
            rows = self._conn.execute(
                "SELECT agent_id, status FROM agents WHERE agent_id >= ? AND agent_id < ? "
                "ORDER BY agent_id",
                (prefix, prefix + "\U0010ffff"),
            ).fetchall()
        except sqlite3.Error as exc:
            raise _map(exc) from exc
        return [IdEntry(row["agent_id"], row["status"] == "torn_down") for row in rows]

    def is_reserved(self, agent_id: str) -> bool:
        try:
            row = self._conn.execute(
                "SELECT 1 FROM allocated_ids WHERE agent_id = ? "
                "UNION ALL SELECT 1 FROM agents WHERE agent_id = ? LIMIT 1",
                (agent_id, agent_id),
            ).fetchone()
        except sqlite3.Error as exc:
            raise _map(exc) from exc
        return row is not None

    # -- leases --------------------------------------------------------------------------

    def holder_identity(self) -> ProcessIdentity:
        """This process's identity (PID, start time, boot id), read once."""
        if self._identity is None:
            found = procs.read_identity(self.gateway.getpid(), gateway=self.gateway)
            if found is None:
                raise StoreError("cannot read this process's identity")
            self._identity = found
        return self._identity

    def acquire_lease(
        self,
        key: str,
        *,
        holder: ProcessIdentity | None = None,
        wait: float = 0.0,
        poll: float = 0.05,
    ) -> Lease:
        """Take lease ``key`` for ``holder`` (default this process).

        Free, or held by a dead process: taken at once (``Lease.took_over_from`` names the dead
        holder). Held by a live process, this one included: ``LeaseHeld`` once ``wait`` seconds
        of retrying are over.
        """
        if _LEASE_KEY.fullmatch(key) is None:
            raise ValueError(f"invalid lease key {key!r}")
        me = holder or self.holder_identity()
        deadline = self.gateway.monotonic() + max(0.0, wait)
        while True:
            try:
                return self._try_acquire(key, me)
            except LeaseHeld:
                if self.gateway.monotonic() >= deadline:
                    raise
                self.gateway.sleep(poll)

    def _try_acquire(self, key: str, me: ProcessIdentity) -> Lease:
        encoded = encode_value("proc", me)
        with self.transaction():
            row = self._conn.execute("SELECT holder FROM leases WHERE key = ?", (key,)).fetchone()
            if row is None:
                self._conn.execute(
                    "INSERT INTO leases (key, holder, acquired) VALUES (?, ?, ?)",
                    (key, encoded, self._now()),
                )
                return Lease(key, me)
            try:
                current = decode_value("proc", row["holder"])
            except ValueDecodeError as exc:
                raise StoreCorrupt(f"lease {key} has an unreadable holder: {exc}") from exc
            assert isinstance(current, ProcessIdentity)
            # A ProcessIdentityError (unparseable /proc entry) propagates: never "dead".
            if current == me or procs.is_alive(current, gateway=self.gateway):
                raise LeaseHeld(key, current)
            self._conn.execute(
                "UPDATE leases SET holder = ?, acquired = ? WHERE key = ?",
                (encoded, self._now(), key),
            )
            return Lease(key, me, took_over_from=current)

    def release_lease(self, lease: Lease) -> bool:
        """Release; false when the lease was no longer this holder's (nothing is deleted)."""
        with self.transaction():
            cursor = self._conn.execute(
                "DELETE FROM leases WHERE key = ? AND holder = ?",
                (lease.key, encode_value("proc", lease.holder)),
            )
            return cursor.rowcount == 1

    @contextmanager
    def lease(
        self, key: str, *, holder: ProcessIdentity | None = None, wait: float = 0.0
    ) -> Iterator[Lease]:
        taken = self.acquire_lease(key, holder=holder, wait=wait)
        try:
            yield taken
        finally:
            self.release_lease(taken)

    def lease_holder(self, key: str) -> ProcessIdentity | None:
        row = self._conn.execute("SELECT holder FROM leases WHERE key = ?", (key,)).fetchone()
        if row is None:
            return None
        try:
            value = decode_value("proc", row["holder"])
        except ValueDecodeError as exc:
            raise StoreCorrupt(f"lease {key} has an unreadable holder: {exc}") from exc
        assert isinstance(value, ProcessIdentity)
        return value

    def _require_lease(self, lease: Lease, key: str) -> None:
        if lease.key != key:
            raise ValueError(f"needs lease {key}, got {lease.key}")
        row = self._conn.execute("SELECT holder FROM leases WHERE key = ?", (key,)).fetchone()
        if row is None or row["holder"] != encode_value("proc", lease.holder):
            raise LeaseLost(f"lease {key} is no longer held by pid {lease.holder.pid}")

    # -- agents --------------------------------------------------------------------------

    def get_agent(self, agent_id: str) -> dict[str, Any] | None:
        try:
            return self._get_agent(agent_id)
        except sqlite3.Error as exc:
            raise _map(exc) from exc

    def _get_agent(self, agent_id: str) -> dict[str, Any] | None:
        row = self._conn.execute("SELECT * FROM agents WHERE agent_id = ?", (agent_id,)).fetchone()
        return None if row is None else _row(row, AGENT_COLUMNS, "agent_id")

    def list_agents(self, *, statuses: Sequence[str] | None = None) -> list[dict[str, Any]]:
        try:
            if statuses is None:
                rows = self._conn.execute("SELECT * FROM agents ORDER BY created, agent_id")
            else:
                marks = ", ".join("?" for _ in statuses)
                rows = self._conn.execute(
                    f"SELECT * FROM agents WHERE status IN ({marks}) ORDER BY created, agent_id",  # noqa: S608
                    tuple(statuses),
                )
            return [_row(row, AGENT_COLUMNS, "agent_id") for row in rows.fetchall()]
        except sqlite3.Error as exc:
            raise _map(exc) from exc

    def find_agents(self, labels: Sequence[Label], *, include_torn_down: bool = False) -> list[str]:
        """Ids of the agents holding every ``labels`` pair (ANDed), in id order."""
        pairs = tuple(dict.fromkeys(labels))
        where = "" if include_torn_down else "AND a.status != 'torn_down'"
        try:
            if not pairs:
                rows = self._conn.execute(
                    f"SELECT a.agent_id FROM agents a WHERE 1 = 1 {where} ORDER BY a.agent_id"  # noqa: S608
                )
            else:
                clause = " OR ".join("(l.key = ? AND l.value = ?)" for _ in pairs)
                params: list[object] = [x for pair in pairs for x in (pair.key, pair.value)]
                rows = self._conn.execute(
                    "SELECT a.agent_id FROM agents a JOIN agent_labels l USING (agent_id) "  # noqa: S608
                    f"WHERE ({clause}) {where} GROUP BY a.agent_id "
                    "HAVING count(*) = ? ORDER BY a.agent_id",
                    (*params, len(pairs)),
                )
            return [row[0] for row in rows.fetchall()]
        except sqlite3.Error as exc:
            raise _map(exc) from exc

    def insert_agent(
        self,
        agent_id: str,
        fields: Mapping[str, object],
        *,
        labels: LabelSet | None = None,
        events: Sequence[OutboxEvent] = (),
    ) -> dict[str, Any]:
        """Insert the record (launch step 3, a fork) and reserve the id for ever, in one
        transaction. ``IdConflict`` when the id was allocated before (live, tombstoned or
        deleted): allocate again."""
        if _AGENT_ID.fullmatch(agent_id) is None:
            raise ValueError(f"invalid agent id {agent_id!r}")
        _check_columns(fields, AGENT_COLUMNS)
        if "name" not in fields or "status" not in fields:
            raise ValueError("an agent record needs a name and a status")
        now = self._now()
        values: dict[str, object] = {"created": now, "updated": now, **fields}
        names = list(values)
        with self.transaction():
            cursor = self._conn.execute(
                "INSERT OR IGNORE INTO allocated_ids (agent_id, allocated) VALUES (?, ?)",
                (agent_id, now),
            )
            if cursor.rowcount != 1:
                raise IdConflict(agent_id)
            self._conn.execute(
                f"INSERT INTO agents (agent_id, {', '.join(names)}) "  # noqa: S608
                f"VALUES (?{', ?' * len(names)})",
                (agent_id, *(encode_value(AGENT_COLUMNS[n], values[n]) for n in names)),
            )
            if labels is not None:
                for pos, pair in enumerate(labels.pairs):
                    self._conn.execute(
                        "INSERT INTO agent_labels (agent_id, pos, key, value) VALUES (?, ?, ?, ?)",
                        (agent_id, pos, pair.key, pair.value),
                    )
            self._add_events(agent_id, events)
            created = self._get_agent(agent_id)
        assert created is not None
        return created

    def insert_new_agent(
        self,
        fields: Mapping[str, object],
        *,
        proposed: str | None = None,
        labels: LabelSet | None = None,
        events: Sequence[OutboxEvent] = (),
        random_bytes: Callable[[int], bytes] = os.urandom,
    ) -> dict[str, Any]:
        """Insert with ``proposed`` (the id launch step 1 allocated in memory) or, when it was
        taken meanwhile, a fresh id. The caller compares the returned ``agent_id`` with
        ``proposed`` and moves anything it keyed by the old id (the prompt copies)."""
        candidate = proposed
        for _ in range(_ALLOCATION_ATTEMPTS):
            if candidate is None:
                candidate = new_agent_id(self, random_bytes=random_bytes)
            try:
                return self.insert_agent(candidate, fields, labels=labels, events=events)
            except IdConflict:
                candidate = None
        raise StoreError("could not allocate an unused agent id")

    def compare_and_set(
        self,
        lease: Lease,
        agent_id: str,
        *,
        expect_status: str | Sequence[str],
        changes: Mapping[str, object],
        expect: Mapping[str, object] | None = None,
        events: Sequence[OutboxEvent] = (),
        touch: bool = True,
    ) -> dict[str, Any]:
        """Apply ``changes`` only if the agent's status is in ``expect_status`` (and every
        ``expect`` column equals its value), under the agent's lease, with ``events`` in the
        same transaction. Returns the new record; ``CasLost`` (carrying the current record)
        for the loser."""
        wanted = (expect_status,) if isinstance(expect_status, str) else tuple(expect_status)
        return self._write(
            lease, agent_id, changes, wanted=wanted, expect=expect, events=events, touch=touch
        )

    def update_agent(
        self,
        lease: Lease,
        agent_id: str,
        changes: Mapping[str, object],
        *,
        events: Sequence[OutboxEvent] = (),
        touch: bool = True,
    ) -> dict[str, Any]:
        """Write ``changes`` under the agent's lease whatever the status (``ack``, ``stop``,
        bookkeeping). ``AgentMissing`` when the record is gone."""
        return self._write(
            lease, agent_id, changes, wanted=(), expect=None, events=events, touch=touch
        )

    def _write(
        self,
        lease: Lease,
        agent_id: str,
        changes: Mapping[str, object],
        *,
        wanted: tuple[str, ...],
        expect: Mapping[str, object] | None,
        events: Sequence[OutboxEvent],
        touch: bool,
    ) -> dict[str, Any]:
        _check_columns(changes, AGENT_COLUMNS)
        _check_columns(expect or {}, AGENT_COLUMNS)
        values = dict(changes)
        if touch:
            values["updated"] = self._now()
        if not values:
            raise ValueError("nothing to write")
        sets = ", ".join(f"{name} = ?" for name in values)
        params: list[object] = [encode_value(AGENT_COLUMNS[n], v) for n, v in values.items()]
        params.append(agent_id)
        conditions = ["agent_id = ?"]
        if wanted:
            conditions.append(f"status IN ({', '.join('?' for _ in wanted)})")
            params.extend(wanted)
        for name, value in (expect or {}).items():
            if value is None:
                conditions.append(f"{name} IS NULL")
            else:
                conditions.append(f"{name} = ?")
                params.append(encode_value(AGENT_COLUMNS[name], value))
        with self.transaction():
            self._require_lease(lease, f"agent:{agent_id}")
            cursor = self._conn.execute(
                f"UPDATE agents SET {sets} WHERE {' AND '.join(conditions)}",  # noqa: S608
                params,
            )
            if cursor.rowcount != 1:
                current = self._get_agent(agent_id)
                if current is None:
                    raise AgentMissing(agent_id) if not wanted else CasLost(agent_id, wanted, None)
                raise CasLost(agent_id, wanted, current)
            self._add_events(agent_id, events)
            updated = self._get_agent(agent_id)
        assert updated is not None
        return updated

    def delete_agent(self, lease: Lease, agent_id: str) -> bool:
        """Delete the record with its runs, holds, labels and events; the id stays reserved."""
        with self.transaction():
            self._require_lease(lease, f"agent:{agent_id}")
            cursor = self._conn.execute("DELETE FROM agents WHERE agent_id = ?", (agent_id,))
            return cursor.rowcount == 1

    # -- labels --------------------------------------------------------------------------

    def get_labels(self, agent_id: str) -> LabelSet:
        try:
            rows = self._conn.execute(
                "SELECT key, value FROM agent_labels WHERE agent_id = ? ORDER BY pos", (agent_id,)
            ).fetchall()
        except sqlite3.Error as exc:
            raise _map(exc) from exc
        return LabelSet(tuple(Label(row["key"], row["value"]) for row in rows))

    def edit_labels(
        self,
        lease: Lease,
        agent_id: str,
        op: Literal["add", "add_run", "remove"],
        labels: Sequence[Label],
        *,
        events: Sequence[OutboxEvent] = (),
    ) -> LabelChange:
        """Read-modify-write the agent's labels under its lease. The 64-pair cap is checked
        against the stored set (``GrammarError`` beyond it, nothing written). Changed pairs are
        queued in ``label_mirror_pending``; a pair added and removed before mirroring cancels."""
        with self.transaction():
            self._require_lease(lease, f"agent:{agent_id}")
            if self._get_agent(agent_id) is None:
                raise AgentMissing(agent_id)
            current = self.get_labels(agent_id)
            if op == "add":
                change = current.add(labels)
            elif op == "add_run":
                change = current.add_run(labels)
            else:
                change = current.remove(labels)
            direction = "remove" if op == "remove" else "add"
            if direction == "remove":
                for pair in change.applied:
                    self._conn.execute(
                        "DELETE FROM agent_labels WHERE agent_id = ? AND key = ? AND value = ?",
                        (agent_id, pair.key, pair.value),
                    )
            else:
                row = self._conn.execute(
                    "SELECT coalesce(max(pos), -1) FROM agent_labels WHERE agent_id = ?",
                    (agent_id,),
                ).fetchone()
                for offset, pair in enumerate(change.applied, start=row[0] + 1):
                    self._conn.execute(
                        "INSERT INTO agent_labels (agent_id, pos, key, value) VALUES (?, ?, ?, ?)",
                        (agent_id, offset, pair.key, pair.value),
                    )
            for pair in change.applied:
                opposite = self._conn.execute(
                    "DELETE FROM label_mirror_pending "
                    "WHERE agent_id = ? AND key = ? AND value = ? AND op != ?",
                    (agent_id, pair.key, pair.value, direction),
                )
                if opposite.rowcount == 0:
                    self._conn.execute(
                        "INSERT OR REPLACE INTO label_mirror_pending (agent_id, key, value, op) "
                        "VALUES (?, ?, ?, ?)",
                        (agent_id, pair.key, pair.value, direction),
                    )
            if change.changed:
                self._conn.execute(
                    "UPDATE agents SET updated = ? WHERE agent_id = ?", (self._now(), agent_id)
                )
                self._add_events(agent_id, events)
        return change

    def mirror_pending(self, agent_id: str) -> list[tuple[Label, str]]:
        rows = self._conn.execute(
            "SELECT key, value, op FROM label_mirror_pending WHERE agent_id = ? ORDER BY rowid",
            (agent_id,),
        ).fetchall()
        return [(Label(row["key"], row["value"]), row["op"]) for row in rows]

    def mirror_done(self, lease: Lease, agent_id: str, items: Sequence[tuple[Label, str]]) -> None:
        """Drop queue entries the session or pane received (only entries still equal)."""
        with self.transaction():
            self._require_lease(lease, f"agent:{agent_id}")
            for pair, op in items:
                self._conn.execute(
                    "DELETE FROM label_mirror_pending "
                    "WHERE agent_id = ? AND key = ? AND value = ? AND op = ?",
                    (agent_id, pair.key, pair.value, op),
                )

    # -- runs and holds ------------------------------------------------------------------

    def put_run(
        self, lease: Lease, agent_id: str, run: int, fields: Mapping[str, object]
    ) -> dict[str, Any]:
        """Create run ``run`` or update the given fields of it."""
        _check_columns(fields, RUN_COLUMNS)
        with self.transaction():
            self._require_lease(lease, f"agent:{agent_id}")
            if self._get_agent(agent_id) is None:
                raise AgentMissing(agent_id)
            self._conn.execute(
                "INSERT OR IGNORE INTO agent_runs (agent_id, run) VALUES (?, ?)", (agent_id, run)
            )
            if fields:
                sets = ", ".join(f"{name} = ?" for name in fields)
                self._conn.execute(
                    f"UPDATE agent_runs SET {sets} WHERE agent_id = ? AND run = ?",  # noqa: S608
                    (*(encode_value(RUN_COLUMNS[n], v) for n, v in fields.items()), agent_id, run),
                )
            stored = self._get_run(agent_id, run)
        assert stored is not None
        return stored

    def _get_run(self, agent_id: str, run: int) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT * FROM agent_runs WHERE agent_id = ? AND run = ?", (agent_id, run)
        ).fetchone()
        return None if row is None else _row(row, RUN_COLUMNS, "agent_id", "run")

    def get_run(self, agent_id: str, run: int) -> dict[str, Any] | None:
        try:
            return self._get_run(agent_id, run)
        except sqlite3.Error as exc:
            raise _map(exc) from exc

    def get_runs(self, agent_id: str) -> list[dict[str, Any]]:
        try:
            rows = self._conn.execute(
                "SELECT * FROM agent_runs WHERE agent_id = ? ORDER BY run", (agent_id,)
            ).fetchall()
        except sqlite3.Error as exc:
            raise _map(exc) from exc
        return [_row(row, RUN_COLUMNS, "agent_id", "run") for row in rows]

    def put_hold(
        self, lease: Lease, agent_id: str, record_id: str, path: str, *, owned: bool
    ) -> bool:
        """Record a held worktree record; false when the agent already holds it."""
        with self.transaction():
            self._require_lease(lease, f"agent:{agent_id}")
            row = self._conn.execute(
                "SELECT coalesce(max(pos), -1) FROM agent_holds WHERE agent_id = ?", (agent_id,)
            ).fetchone()
            cursor = self._conn.execute(
                "INSERT OR IGNORE INTO agent_holds (agent_id, record_id, path, owned, pos) "
                "VALUES (?, ?, ?, ?, ?)",
                (agent_id, record_id, path, int(owned), row[0] + 1),
            )
            return cursor.rowcount == 1

    def release_hold(self, lease: Lease, agent_id: str, record_id: str) -> None:
        with self.transaction():
            self._require_lease(lease, f"agent:{agent_id}")
            self._conn.execute(
                "UPDATE agent_holds SET released = 1 WHERE agent_id = ? AND record_id = ?",
                (agent_id, record_id),
            )

    def replace_hold_record(
        self, lease: Lease, agent_id: str, old_id: str, new_id: str, path: str
    ) -> None:
        """A re-adoption created a new record: the row keeps its place and takes the new id
        (and the agent's ``wt_record_id`` follows when it named the old one)."""
        with self.transaction():
            self._require_lease(lease, f"agent:{agent_id}")
            self._conn.execute(
                "UPDATE agent_holds SET record_id = ?, path = ? "
                "WHERE agent_id = ? AND record_id = ?",
                (new_id, path, agent_id, old_id),
            )
            self._conn.execute(
                "UPDATE agents SET wt_record_id = ?, wt_path = ? "
                "WHERE agent_id = ? AND wt_record_id = ?",
                (new_id, path, agent_id, old_id),
            )

    def get_holds(self, agent_id: str, *, include_released: bool = False) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT record_id, path, owned, released FROM agent_holds WHERE agent_id = ? "
            "AND (? OR released = 0) ORDER BY pos",
            (agent_id, int(include_released)),
        ).fetchall()
        return [
            {
                "record_id": r["record_id"],
                "path": r["path"],
                "owned": bool(r["owned"]),
                "released": bool(r["released"]),
            }
            for r in rows
        ]

    # -- relay dedup and the outbox ------------------------------------------------------

    def record_relay_event(self, csession_id: str, event_id: str) -> bool:
        """Record the claude-session event key; false when it was recorded before. Joins an
        enclosing transaction, so the relay's one transaction can use it."""
        with self.transaction():
            cursor = self._conn.execute(
                "INSERT OR IGNORE INTO relay_events (csession_id, event_id, received) "
                "VALUES (?, ?, ?)",
                (csession_id, event_id, self._now()),
            )
            return cursor.rowcount == 1

    def prune_relay_events(self) -> int:
        cutoff = self._now() - RELAY_EVENT_DAYS * 86400
        with self.transaction():
            return self._conn.execute(
                "DELETE FROM relay_events WHERE received < ?", (cutoff,)
            ).rowcount

    def add_events(self, agent_id: str, events: Sequence[OutboxEvent]) -> None:
        """Queue events alone (the relay, which holds no lease); use the ``events`` argument of
        a write to commit them with a change."""
        with self.transaction():
            self._add_events(agent_id, events)

    def _add_events(self, agent_id: str, events: Sequence[OutboxEvent]) -> None:
        now = self._now()
        for item in events:
            self._conn.execute(
                "INSERT INTO events (agent_id, event, payload, created) VALUES (?, ?, ?, ?)",
                (agent_id, item.event, encode_value("json", dict(item.payload)), now),
            )

    def pending_events(self, agent_id: str, *, limit: int = 1000) -> list[dict[str, Any]]:
        """Undelivered events of the agent, oldest ``event_id`` first."""
        try:
            rows = self._conn.execute(
                "SELECT event_id, event, payload, created FROM events "
                "WHERE agent_id = ? AND delivered = 0 ORDER BY event_id LIMIT ?",
                (agent_id, limit),
            ).fetchall()
            return [
                {
                    "event_id": r["event_id"],
                    "agent_id": agent_id,
                    "event": r["event"],
                    "payload": decode_value("json", r["payload"]),
                    "created": r["created"],
                }
                for r in rows
            ]
        except sqlite3.Error as exc:
            raise _map(exc) from exc
        except ValueDecodeError as exc:
            raise StoreCorrupt(f"event payload: {exc}") from exc

    def mark_delivered(self, lease: Lease, agent_id: str, event_id: int) -> bool:
        """Mark one event delivered; needs the agent's delivery lease."""
        with self.transaction():
            self._require_lease(lease, f"delivery:{agent_id}")
            cursor = self._conn.execute(
                "UPDATE events SET delivered = 1 WHERE event_id = ? AND agent_id = ? "
                "AND delivered = 0",
                (event_id, agent_id),
            )
            return cursor.rowcount == 1


def _prepare_files(state_dir: Path, *, create: bool, uid: int) -> Path:
    """Make or check the state dir (0700) and ``state.db`` (0600) without following links.

    The state dir itself may be a symlink (the user chose that); ``state.db`` and its
    ``-wal``/``-shm``/``-journal`` files may not, and everything must belong to this user.
    """
    try:
        directory = Path(os.path.realpath(state_dir))
        if not directory.exists():
            if not create:
                raise StoreMissing(f"no state directory {directory}")
            os.makedirs(directory, mode=DIR_MODE, exist_ok=True)
        info = os.stat(directory)
        if not stat.S_ISDIR(info.st_mode):
            raise StoreError(f"{directory} is not a directory")
        if info.st_uid != uid:
            raise StoreError(f"{directory} is not owned by this user")
        if stat.S_IMODE(info.st_mode) != DIR_MODE:
            os.chmod(directory, DIR_MODE)
        db = directory / DB_NAME
        for suffix in ("",) + _SIDECARS:
            _check_leaf(Path(str(db) + suffix), uid)
        if not db.exists():
            if not create:
                raise StoreMissing(f"no database {db}")
            try:
                fd = os.open(db, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, FILE_MODE)
            except FileExistsError:
                pass  # another process created it first
            else:
                os.close(fd)
            _check_leaf(db, uid)
        if stat.S_IMODE(os.lstat(db).st_mode) != FILE_MODE:
            os.chmod(db, FILE_MODE)
        return db
    except (OSError, ValueError) as exc:
        raise StoreError(f"cannot prepare state directory {state_dir}: {exc}") from exc


def _check_leaf(path: Path, uid: int) -> None:
    try:
        info = os.lstat(path)
    except FileNotFoundError:
        return
    if stat.S_ISLNK(info.st_mode):
        raise StoreError(f"{path} is a symbolic link: refusing to follow it")
    if not stat.S_ISREG(info.st_mode):
        raise StoreError(f"{path} is not a regular file")
    if info.st_uid != uid:
        raise StoreError(f"{path} is not owned by this user")


def _fix_sidecar_modes(db: Path) -> None:
    for suffix in _SIDECARS:
        side = Path(str(db) + suffix)
        try:
            if stat.S_IMODE(os.lstat(side).st_mode) != FILE_MODE:
                os.chmod(side, FILE_MODE)
        except OSError:
            pass  # gone already (WAL checkpoint) or not ours to fix
