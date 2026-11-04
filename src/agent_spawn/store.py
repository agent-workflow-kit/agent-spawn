"""The state store: ``state.db`` (SQLite, WAL), leases, compare-and-set helpers and the outbox.

Rules every method follows:

* Untrusted text only ever travels as a bound parameter. Column names come from the fixed specs
  in ``store_schema``; a name outside them is a ``ValueError``, never interpolated.
* Every write is one ``BEGIN IMMEDIATE`` transaction, so a change and its outbox events commit
  together or not at all. A nested ``transaction()`` is a savepoint: a caught inner failure
  undoes only the inner work.
* A lifecycle entry is a compare-and-set under the agent's lease (interface.md, Lifecycle
  transitions), through the named helpers (``enter_resume``, ``enter_teardown``, ``mark_live``,
  ``park`` ...). The lease is verified again inside the write transaction.
* Every sqlite3 error, and text that cannot be encoded or decoded, surfaces as a ``StoreError``.
* Clocks, ``/proc`` reads and sleeping go through the gateway and ``procs``.

Leases are keyed by holder identity (PID, start time, boot id). A lease whose holder is dead
(another boot id, no such process, another start time, or a zombie) is taken over at once; a
live holder is never preempted, however long it holds, and a lease is not reentrant. Liveness is
decided outside the write lock, and the takeover happens only if the stored holder is unchanged.
An unparseable process entry or holder is an error, never "dead".
"""

from __future__ import annotations

import errno
import os
import re
import sqlite3
import stat
import urllib.parse
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
    "TRANSITIONS",
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
_AUTOCHECKPOINT = 1000  # SQLite's default wal_autocheckpoint (pages)
RELEASE_DEADLINE = 60.0
_OPEN_FLAGS = os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC

# interface.md "Agent statuses" and "Lifecycle transitions": the statuses each status may move
# to. A retried teardown re-enters ``tearing_down`` over a dead teardown process.
TRANSITIONS: Mapping[str, frozenset[str]] = {
    "parked": frozenset({"launching", "tearing_down"}),
    "launching": frozenset({"live", "parked"}),
    "live": frozenset({"parked", "tearing_down"}),
    "tearing_down": frozenset({"tearing_down", "parked", "torn_down"}),
    "torn_down": frozenset(),
}
# The park fields ``enter_resume`` saves and ``restore_pre_resume`` puts back (plus ``updated``).
_PRE_RESUME_FIELDS = (
    "park_reason",
    "needs_attention",
    "escalated_by",
    "last_outcome",
    "name",
    "term_placement",
    "promote",
)


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


def _translate(exc: BaseException) -> BaseException:
    """The store error for a low-level failure; anything else unchanged."""
    if isinstance(exc, sqlite3.Error):
        return _map(exc)
    if isinstance(exc, UnicodeEncodeError):
        return StoreError(f"text cannot be stored: {exc.reason}")
    if isinstance(exc, UnicodeDecodeError):
        return StoreCorrupt(f"stored text is not valid UTF-8: {exc.reason}")
    return exc


@contextmanager
def _mapped() -> Iterator[None]:
    """Run reads (or statements) with every sqlite3 and text-codec error mapped."""
    try:
        yield
    except (sqlite3.Error, UnicodeError) as exc:
        mapped = _translate(exc)
        if mapped is exc:
            raise
        raise mapped from exc


def _decode_text(raw: bytes) -> str:
    """Strict UTF-8 for every TEXT value read (a bad byte is ``UnicodeDecodeError``, mapped to
    ``StoreCorrupt``; sqlite3's own message would quote the stored bytes)."""
    return raw.decode("utf-8")


def _connect(target: str, busy_timeout: float, *, uri: bool = False) -> sqlite3.Connection:
    try:
        conn = sqlite3.connect(
            target, timeout=busy_timeout, isolation_level=None, check_same_thread=True, uri=uri
        )
    except (sqlite3.Error, ValueError) as exc:
        raise StoreError(f"cannot open {target}: {exc}") from exc
    conn.row_factory = sqlite3.Row
    conn.text_factory = _decode_text
    return conn


def _read_version(conn: sqlite3.Connection, path: Path) -> int | None:
    """The schema version; ``None`` if empty; ``StoreCorrupt`` for a foreign database."""
    tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    if not tables:
        return None
    if "meta" not in tables:
        raise StoreCorrupt(f"{path} is not an agent-spawn database")
    row = conn.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
    if row is None or not str(row[0]).isascii() or not str(row[0]).isdigit():
        raise StoreCorrupt(f"{path} has no valid schema version")
    return int(row[0])


def _check_version(version: int | None, wanted: int, *, migrate: bool) -> None:
    """Refuse a newer schema (exit 12) and, without ``migrate``, an older or empty one."""
    if version is not None and version > wanted:
        raise CliError.of(
            "schema_newer",
            f"state.db has schema {version}; this agent-spawn understands up to "
            f"{wanted}: upgrade agent-spawn",
        )
    if (version is None or version < wanted) and not migrate:
        raise SchemaOlder(0 if version is None else version, wanted)


def _hold_checkpoints(conn: sqlite3.Connection, *, hold: bool) -> None:
    """Stop (or restore) this connection's WAL checkpoints: on close and automatic ones."""
    with _mapped():
        conn.setconfig(sqlite3.SQLITE_DBCONFIG_NO_CKPT_ON_CLOSE, hold)
        conn.execute(f"PRAGMA wal_autocheckpoint = {0 if hold else _AUTOCHECKPOINT}")


def _peek_version(path: Path, file_id: tuple[int, int], busy_timeout: float) -> int | None:
    """Read the schema version on a read-only connection, which can never write the database
    or checkpoint its WAL (a newer database stays exactly as the newer agent-spawn left it)."""
    conn = _connect(f"file:{urllib.parse.quote(str(path))}?mode=ro", busy_timeout, uri=True)
    try:
        with _mapped():
            version = _read_version(conn, path)
        _check_same_file(path, file_id)
        return version
    finally:
        conn.close()


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
        self._depth = 0  # open transaction levels: 0 none, 1 BEGIN, more are savepoints

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

        The schema version is read first on a read-only connection: a newer schema exits 12
        ``schema_newer`` and an older (or empty) one without ``migrate`` is ``SchemaOlder``, both
        before anything is written (no journal-mode change, no checkpoint). With ``migrate`` an
        older schema is migrated. A missing database with ``create=False`` is ``StoreMissing``.
        ``state.db`` and its sidecars must be regular, single-link files of this user, never
        symlinks; a database replaced while it is opened is refused.
        """
        gw = gateway or get_gateway()
        uid = gw.getuid()
        path, file_id = _prepare_files(Path(state_dir), create=create, uid=uid)
        _check_version(_peek_version(path, file_id, busy_timeout), len(migrations), migrate=migrate)
        conn = _connect(str(path), busy_timeout)
        store = cls(conn, path, gw, identity=identity)
        try:
            # A newer agent-spawn may migrate between the peek and this open: until this
            # connection's own version check passes, it must not checkpoint (on close or commit).
            _hold_checkpoints(conn, hold=True)
            store._setup(
                file_id=file_id,
                uid=uid,
                migrate=migrate,
                migrations=migrations,
                busy_timeout=busy_timeout,
            )
        except BaseException:
            conn.close()
            raise
        return store

    def _setup(
        self,
        *,
        file_id: tuple[int, int],
        uid: int,
        migrate: bool,
        migrations: Sequence[Migration],
        busy_timeout: float,
    ) -> None:
        wanted = len(migrations)
        with _mapped():
            self._conn.execute("PRAGMA foreign_keys = ON")
            version = _read_version(self._conn, self.path)
        _check_same_file(self.path, file_id)
        _check_version(version, wanted, migrate=migrate)
        _hold_checkpoints(self._conn, hold=False)
        self._enable_wal(busy_timeout)
        with _mapped():
            self._conn.execute("PRAGMA synchronous = FULL")
        if version is None or version < wanted:
            self._migrate(migrations)
        _fix_sidecar_modes(self.path, uid)

    def _enable_wal(self, busy_timeout: float) -> None:
        """Switch to WAL, retrying while concurrent first opens hold the lock; a database that
        another process already switched is accepted as it is."""
        deadline = self.gateway.monotonic() + busy_timeout
        delay = 0.01
        while True:
            try:
                with _mapped():
                    mode = str(self._conn.execute("PRAGMA journal_mode").fetchone()[0]).lower()
                    if mode != "wal":
                        mode = str(
                            self._conn.execute("PRAGMA journal_mode = WAL").fetchone()[0]
                        ).lower()
                if mode == "wal":
                    return
                if self.gateway.monotonic() >= deadline:
                    raise StoreError(f"cannot enable WAL for {self.path} (journal_mode {mode})")
            except StoreBusy:
                if self.gateway.monotonic() >= deadline:
                    raise
            self.gateway.sleep(delay)
            delay = min(delay * 2, 0.2)

    def _migrate(self, migrations: Sequence[Migration]) -> None:
        wanted = len(migrations)
        with self.transaction():
            # Re-read under the write lock: another process may have migrated meanwhile.
            version = _read_version(self._conn, self.path) or 0
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
        with _mapped():
            return _read_version(self._conn, self.path) or 0

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> Store:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- transactions --------------------------------------------------------------------

    @contextmanager
    def transaction(self) -> Iterator[None]:
        """``BEGIN IMMEDIATE`` .. ``COMMIT``. Nested, a ``SAVEPOINT``: an inner failure rolls
        back only the inner work (the outer transaction decides the rest). Errors are mapped
        the same way at every level."""
        nested = self._depth > 0
        savepoint = f"nested_{self._depth}"
        with _mapped():
            self._conn.execute(f"SAVEPOINT {savepoint}" if nested else "BEGIN IMMEDIATE")
        self._depth += 1
        try:
            yield
            with _mapped():
                self._conn.execute(f"RELEASE {savepoint}" if nested else "COMMIT")
        except BaseException as exc:
            try:
                if nested:
                    self._conn.execute(f"ROLLBACK TO {savepoint}")
                    self._conn.execute(f"RELEASE {savepoint}")
                else:
                    self._conn.execute("ROLLBACK")
            except sqlite3.Error:
                pass  # a failed COMMIT may have rolled back already
            mapped = _translate(exc)
            if mapped is exc:
                raise
            raise mapped from exc
        finally:
            self._depth -= 1

    @property
    def in_transaction(self) -> bool:
        return self._depth > 0

    def _no_transaction(self, what: str) -> None:
        if self._depth:
            raise StoreError(f"{what} inside an open transaction (it would hold the write lock)")

    def _now(self) -> float:
        return self.gateway.wall_time()

    # -- AgentIndex (ids.py) -------------------------------------------------------------

    def entries_with_prefix(self, prefix: str) -> list[IdEntry]:
        """Stored records (tombstones included) whose id starts with ``prefix``: one range scan
        of the primary key."""
        with _mapped():
            rows = self._conn.execute(
                "SELECT agent_id, status FROM agents WHERE agent_id >= ? AND agent_id < ? "
                "ORDER BY agent_id",
                (prefix, prefix + "\U0010ffff"),
            ).fetchall()
        return [IdEntry(row["agent_id"], row["status"] == "torn_down") for row in rows]

    def is_reserved(self, agent_id: str) -> bool:
        with _mapped():
            row = self._conn.execute(
                "SELECT 1 FROM allocated_ids WHERE agent_id = ? "
                "UNION ALL SELECT 1 FROM agents WHERE agent_id = ? LIMIT 1",
                (agent_id, agent_id),
            ).fetchone()
        return row is not None

    # -- leases --------------------------------------------------------------------------

    def holder_identity(self) -> ProcessIdentity:
        """This process's identity (PID, start time, boot id), read once."""
        if self._identity is None:
            try:
                found = procs.read_identity(self.gateway.getpid(), gateway=self.gateway)
            except procs.ProcessIdentityError as exc:
                raise StoreError(f"cannot read this process's identity: {exc}") from exc
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
        holder). Held by another live process: ``LeaseHeld`` once ``wait`` seconds of retrying
        are over. Held by ``holder`` itself: ``LeaseHeld`` at once (leases are not reentrant).
        Never inside an open transaction (``StoreError``).
        """
        self._no_transaction("acquire_lease")
        if _LEASE_KEY.fullmatch(key) is None:
            raise ValueError(f"invalid lease key {key!r}")
        me = holder or self.holder_identity()
        deadline = self.gateway.monotonic() + max(0.0, wait)
        while True:
            try:
                return self._try_acquire(key, me)
            except LeaseHeld as held:
                if held.holder == me or self.gateway.monotonic() >= deadline:
                    raise
                self.gateway.sleep(poll)

    def _try_acquire(self, key: str, me: ProcessIdentity) -> Lease:
        """Read the holder and decide its liveness without the write lock (on macOS that runs
        tools with timeouts); take over in a write transaction only if the stored holder text
        is still the one judged dead, else judge again."""
        encoded = encode_value("proc", me)
        while True:
            with _mapped():
                row = self._conn.execute(
                    "SELECT holder FROM leases WHERE key = ?", (key,)
                ).fetchone()
            if row is None:
                with self.transaction():
                    inserted = self._conn.execute(
                        "INSERT OR IGNORE INTO leases (key, holder, acquired) VALUES (?, ?, ?)",
                        (key, encoded, self._now()),
                    ).rowcount
                if inserted == 1:
                    return Lease(key, me)
                continue  # someone took it meanwhile: judge the new holder
            stored = row["holder"]
            current = _decode_holder(key, stored)
            if current == me or self._holder_alive(current):
                raise LeaseHeld(key, current)
            with self.transaction():
                taken = self._conn.execute(
                    "UPDATE leases SET holder = ?, acquired = ? WHERE key = ? AND holder = ?",
                    (encoded, self._now(), key, stored),
                ).rowcount
            if taken == 1:
                return Lease(key, me, took_over_from=current)

    def _holder_alive(self, holder: ProcessIdentity) -> bool:
        """Another boot id is dead without reading ``/proc``; an unreadable or unparseable
        process entry is a ``StoreError``, never "dead"."""
        try:
            if holder.boot_id != procs.read_boot_id(gateway=self.gateway):
                return False
            return procs.is_alive(holder, gateway=self.gateway)
        except procs.ProcessIdentityError as exc:
            raise StoreError(f"cannot tell whether lease holder pid {holder.pid} is alive") from exc

    def release_lease(self, lease: Lease) -> bool:
        """Release; false when the lease was no longer this holder's (nothing is deleted).

        A busy database is retried with backoff (gateway clock) for up to ``RELEASE_DEADLINE``
        seconds, since a lease that is never released locks its key out for the holder's
        lifetime; then ``StoreBusy``. Never inside an open transaction (``StoreError``)."""
        self._no_transaction("release_lease")
        encoded = encode_value("proc", lease.holder)
        deadline = self.gateway.monotonic() + RELEASE_DEADLINE
        delay = 0.05
        while True:
            try:
                with self.transaction():
                    deleted = self._conn.execute(
                        "DELETE FROM leases WHERE key = ? AND holder = ?", (lease.key, encoded)
                    ).rowcount
                return deleted == 1
            except StoreBusy:
                if self.gateway.monotonic() >= deadline:
                    raise
                self.gateway.sleep(delay)
                delay = min(delay * 2, 1.0)

    @contextmanager
    def lease(
        self, key: str, *, holder: ProcessIdentity | None = None, wait: float = 0.0
    ) -> Iterator[Lease]:
        """Hold ``key`` around the body. An exception of the body always wins: a release that
        fails then is attached to it as a note, never raised instead of it."""
        taken = self.acquire_lease(key, holder=holder, wait=wait)
        try:
            yield taken
        except BaseException as exc:
            try:
                self.release_lease(taken)
            except Exception as failed:  # noqa: BLE001 - reported on the body's exception
                exc.add_note(f"releasing lease {key} failed too: {failed!r}")
            raise
        self.release_lease(taken)

    def lease_holder(self, key: str) -> ProcessIdentity | None:
        with _mapped():
            row = self._conn.execute("SELECT holder FROM leases WHERE key = ?", (key,)).fetchone()
        return None if row is None else _decode_holder(key, row["holder"])

    def _require_lease(self, lease: Lease, key: str) -> None:
        if lease.key != key:
            raise ValueError(f"needs lease {key}, got {lease.key}")
        row = self._conn.execute("SELECT holder FROM leases WHERE key = ?", (key,)).fetchone()
        if row is None or row["holder"] != encode_value("proc", lease.holder):
            raise LeaseLost(f"lease {key} is no longer held by pid {lease.holder.pid}")

    # -- agents --------------------------------------------------------------------------

    def get_agent(self, agent_id: str) -> dict[str, Any] | None:
        with _mapped():
            return self._get_agent(agent_id)

    def _get_agent(self, agent_id: str) -> dict[str, Any] | None:
        row = self._conn.execute("SELECT * FROM agents WHERE agent_id = ?", (agent_id,)).fetchone()
        return None if row is None else _row(row, AGENT_COLUMNS, "agent_id")

    def list_agents(self, *, statuses: Sequence[str] | None = None) -> list[dict[str, Any]]:
        with _mapped():
            if statuses is None:
                rows = self._conn.execute("SELECT * FROM agents ORDER BY created, agent_id")
            else:
                marks = ", ".join("?" for _ in statuses)
                rows = self._conn.execute(
                    f"SELECT * FROM agents WHERE status IN ({marks}) ORDER BY created, agent_id",  # noqa: S608
                    tuple(statuses),
                )
            return [_row(row, AGENT_COLUMNS, "agent_id") for row in rows.fetchall()]

    def find_agents(self, labels: Sequence[Label], *, include_torn_down: bool = False) -> list[str]:
        """Ids of the agents holding every ``labels`` pair (ANDed), in id order."""
        pairs = tuple(dict.fromkeys(labels))
        where = "" if include_torn_down else "AND a.status != 'torn_down'"
        with _mapped():
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
                try:
                    candidate = new_agent_id(self, random_bytes=random_bytes)
                except RuntimeError as exc:
                    raise StoreError(f"could not allocate an agent id: {exc}") from exc
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
        for the loser, ``AgentMissing`` when there is no record.

        A ``status`` in ``changes`` must be reachable from every expected status
        (``TRANSITIONS``), else ``ValueError``. Prefer the named lifecycle helpers."""
        wanted = (expect_status,) if isinstance(expect_status, str) else tuple(expect_status)
        if not wanted:
            raise ValueError("compare_and_set needs at least one expected status")
        _check_transition(wanted, changes.get("status"))
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
        bookkeeping). ``AgentMissing`` when the record is gone. ``status`` is not writable
        here (``ValueError``): status changes go through ``compare_and_set`` or the lifecycle
        helpers, which check ``TRANSITIONS``."""
        if "status" in changes:
            raise ValueError("update_agent cannot write status: use compare_and_set or a helper")
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
        expect_not: Mapping[str, object] | None = None,
    ) -> dict[str, Any]:
        _check_columns(changes, AGENT_COLUMNS)
        _check_columns(expect or {}, AGENT_COLUMNS)
        _check_columns(expect_not or {}, AGENT_COLUMNS)
        values = dict(changes)
        if values.get("status") == "parked":
            # interface.md, Agent statuses: every park, by any process, clears it.
            if values.get("promotion_pending") is True:
                raise ValueError("a park always sets promotion_pending to false")
            values["promotion_pending"] = False
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
        for name, value in (expect_not or {}).items():
            if value is None:
                conditions.append(f"{name} IS NOT NULL")
            else:
                conditions.append(f"{name} IS NOT ?")
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
                    raise AgentMissing(agent_id)
                raise CasLost(agent_id, wanted, current)
            self._add_events(agent_id, events)
            updated = self._get_agent(agent_id)
        assert updated is not None
        return updated

    # -- lifecycle entries (interface.md, Lifecycle transitions) -------------------------

    def enter_resume(
        self,
        lease: Lease,
        agent_id: str,
        *,
        launcher: ProcessIdentity,
        expect_run_after: int | None,
        changes: Mapping[str, object] | None = None,
        events: Sequence[OutboxEvent] = (),
    ) -> dict[str, Any]:
        """Plain resume: ``parked`` → ``launching``, recording the launcher and
        ``expect_run_after`` and saving the park fields for ``restore_pre_resume``."""
        with self.transaction():
            self._require_lease(lease, f"agent:{agent_id}")
            current = self._get_agent(agent_id)
            if current is None:
                raise AgentMissing(agent_id)
            saved = {name: current[name] for name in (*_PRE_RESUME_FIELDS, "updated")}
            return self.compare_and_set(
                lease,
                agent_id,
                expect_status="parked",
                changes={
                    **(changes or {}),
                    "status": "launching",
                    "launcher": launcher,
                    "expect_run_after": expect_run_after,
                    "pre_resume": saved,
                },
                events=events,
            )

    def restore_pre_resume(self, lease: Lease, agent_id: str) -> dict[str, Any]:
        """A resume that failed (or was interrupted) before its run existed: ``launching`` →
        ``parked`` with the saved park fields and ``updated`` put back and the launcher fields
        cleared; no events. ``CasLost`` when the agent is not a resume in ``launching``."""
        with self.transaction():
            self._require_lease(lease, f"agent:{agent_id}")
            current = self._get_agent(agent_id)
            if current is None:
                raise AgentMissing(agent_id)
            saved = current["pre_resume"]
            if current["status"] != "launching" or saved is None:
                raise CasLost(agent_id, ("launching",), current)
            names = (*_PRE_RESUME_FIELDS, "updated")
            if not isinstance(saved, dict) or set(saved) != set(names):
                raise StoreCorrupt(f"agent {agent_id} has an unreadable pre-resume row")
            return self._write(
                lease,
                agent_id,
                {
                    **saved,
                    "status": "parked",
                    "launcher": None,
                    "expect_run_after": None,
                    "pre_resume": None,
                },
                wanted=("launching",),
                expect=None,
                events=(),
                touch=False,
            )

    def enter_teardown(
        self,
        lease: Lease,
        agent_id: str,
        *,
        teardown_proc: ProcessIdentity,
        retry_from: ProcessIdentity | None = None,
        events: Sequence[OutboxEvent] = (),
    ) -> dict[str, Any]:
        """``teardown``: ``parked`` → ``tearing_down`` with ``teardown_step: started`` and
        the teardown process. With ``retry_from`` (the recorded teardown process, which the
        caller found dead before taking the lease), a retry: ``tearing_down`` stays, only the
        teardown process changes, and only if ``retry_from`` is still the recorded one."""
        if retry_from is not None:
            return self.compare_and_set(
                lease,
                agent_id,
                expect_status="tearing_down",
                expect={"teardown_proc": retry_from},
                changes={"teardown_proc": teardown_proc},
                events=events,
            )
        return self.compare_and_set(
            lease,
            agent_id,
            expect_status="parked",
            changes={
                "status": "tearing_down",
                "teardown_step": "started",
                "teardown_proc": teardown_proc,
                "teardown_partial_record": None,
            },
            events=events,
        )

    def mark_live(
        self,
        lease: Lease,
        agent_id: str,
        *,
        changes: Mapping[str, object] | None = None,
        events: Sequence[OutboxEvent] = (),
    ) -> dict[str, Any]:
        """The run is started: ``launching`` → ``live``."""
        return self.compare_and_set(
            lease,
            agent_id,
            expect_status="launching",
            changes={**(changes or {}), "status": "live"},
            events=events,
        )

    def enter_ephemeral_teardown(
        self,
        lease: Lease,
        agent_id: str,
        *,
        supervisor: ProcessIdentity,
        events: Sequence[OutboxEvent] = (),
    ) -> dict[str, Any]:
        """The supervisor of an ephemeral agent whose run is ``done``: ``live`` →
        ``tearing_down`` (``teardown_step: started``), the supervisor recorded as the
        teardown process; only for an ephemeral agent that ``supervisor`` supervises."""
        return self.compare_and_set(
            lease,
            agent_id,
            expect_status="live",
            expect={"ephemeral": True, "supervisor": supervisor},
            changes={
                "status": "tearing_down",
                "teardown_step": "started",
                "teardown_proc": supervisor,
                "teardown_partial_record": None,
            },
            events=events,
        )

    def park(
        self,
        lease: Lease,
        agent_id: str,
        *,
        park_reason: str,
        needs_attention: str | None,
        escalated_by: str | None = None,
        from_status: str | Sequence[str] = ("launching", "live"),
        changes: Mapping[str, object] | None = None,
        events: Sequence[OutboxEvent] = (),
    ) -> dict[str, Any]:
        """``launching``/``live`` → ``parked`` with ``park_reason``, ``needs_attention`` and
        ``escalated_by`` (non-null only for a hook escalation); ``promotion_pending`` false."""
        wanted = (from_status,) if isinstance(from_status, str) else tuple(from_status)
        if not wanted or not set(wanted) <= {"launching", "live"}:
            raise ValueError(f"park is from launching or live, not {wanted!r}")
        return self.compare_and_set(
            lease,
            agent_id,
            expect_status=wanted,
            changes={
                **(changes or {}),
                "status": "parked",
                "park_reason": park_reason,
                "needs_attention": needs_attention,
                "escalated_by": escalated_by,
            },
            events=events,
        )

    def park_after_failed_placement(
        self,
        lease: Lease,
        agent_id: str,
        *,
        park_reason: str,
        needs_attention: str | None,
        escalated_by: str | None = None,
        changes: Mapping[str, object] | None = None,
        events: Sequence[OutboxEvent] = (),
    ) -> dict[str, Any]:
        """``status = launching`` → ``parked``: written only by the process that wins it (a
        loser does not park, ``CasLost``)."""
        return self.park(
            lease,
            agent_id,
            park_reason=park_reason,
            needs_attention=needs_attention,
            escalated_by=escalated_by,
            from_status="launching",
            changes=changes,
            events=events,
        )

    def abort_teardown(
        self, lease: Lease, agent_id: str, *, events: Sequence[OutboxEvent] = ()
    ) -> dict[str, Any]:
        """A teardown vetoed, escalated or refused at a step that changed nothing:
        ``tearing_down`` → ``parked`` with the teardown fields cleared. Never once the marker
        is ``worktree_partial`` (``CasLost``): that agent stays ``tearing_down``."""
        return self._write(
            lease,
            agent_id,
            {
                "status": "parked",
                "teardown_step": None,
                "teardown_proc": None,
                "teardown_partial_record": None,
            },
            wanted=("tearing_down",),
            expect=None,
            expect_not={"teardown_step": "worktree_partial"},
            events=events,
            touch=True,
        )

    def tombstone(
        self, lease: Lease, agent_id: str, *, events: Sequence[OutboxEvent] = ()
    ) -> dict[str, Any]:
        """Teardown's last step: ``tearing_down`` → ``torn_down``, ``needs_attention`` and
        ``teardown_step`` cleared, ``torn_down_at`` now."""
        return self.compare_and_set(
            lease,
            agent_id,
            expect_status="tearing_down",
            changes={
                "status": "torn_down",
                "needs_attention": None,
                "teardown_step": None,
                "promotion_pending": False,
                "torn_down_at": self._now(),
            },
            events=events,
        )

    def delete_agent(self, lease: Lease, agent_id: str) -> bool:
        """Delete the record with its runs, holds, labels, mirror queue and events (undelivered
        included); the id stays reserved."""
        with self.transaction():
            self._require_lease(lease, f"agent:{agent_id}")
            cursor = self._conn.execute("DELETE FROM agents WHERE agent_id = ?", (agent_id,))
            return cursor.rowcount == 1

    def _require_agent(self, agent_id: str) -> None:
        if self._get_agent(agent_id) is None:
            raise AgentMissing(agent_id)

    # -- labels --------------------------------------------------------------------------

    def get_labels(self, agent_id: str) -> LabelSet:
        with _mapped():
            rows = self._conn.execute(
                "SELECT key, value FROM agent_labels WHERE agent_id = ? ORDER BY pos", (agent_id,)
            ).fetchall()
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
        against the stored set (``GrammarError`` beyond it, nothing written). Each changed pair
        is queued in ``label_mirror_pending`` with its latest op (an earlier op of the pair is
        replaced, never cancelled: it may already be in flight)."""
        with self.transaction():
            self._require_lease(lease, f"agent:{agent_id}")
            self._require_agent(agent_id)
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
                self._queue_mirror(agent_id, pair, direction, replace=True)
            if change.changed:
                self._conn.execute(
                    "UPDATE agents SET updated = ? WHERE agent_id = ?", (self._now(), agent_id)
                )
                self._add_events(agent_id, events)
        return change

    def _queue_mirror(self, agent_id: str, pair: Label, op: str, *, replace: bool) -> None:
        # The queue is ordered by rowid: a replaced pair moves to the end, as the latest change.
        verb = "INSERT OR REPLACE" if replace else "INSERT OR IGNORE"
        self._conn.execute(
            f"{verb} INTO label_mirror_pending (agent_id, key, value, op) VALUES (?, ?, ?, ?)",  # noqa: S608
            (agent_id, pair.key, pair.value, op),
        )

    def mirror_pending(self, agent_id: str) -> list[tuple[Label, str]]:
        with _mapped():
            rows = self._conn.execute(
                "SELECT key, value, op FROM label_mirror_pending WHERE agent_id = ? ORDER BY rowid",
                (agent_id,),
            ).fetchall()
        return [(Label(row["key"], row["value"]), row["op"]) for row in rows]

    def mirror_done(self, lease: Lease, agent_id: str, items: Sequence[tuple[Label, str]]) -> None:
        """Record pushes the session or pane received. An entry is dropped only when its op is
        the one pushed. Then, since the stored labels are the truth, a push that disagrees with
        them (an add of a pair the agent no longer has, a remove of one it has again: pushes
        that landed out of order) makes sure the corrective op is queued."""
        with self.transaction():
            self._require_lease(lease, f"agent:{agent_id}")
            self._require_agent(agent_id)
            for pair, op in items:
                if op not in ("add", "remove"):
                    raise ValueError(f"invalid mirror op {op!r}")
                self._conn.execute(
                    "DELETE FROM label_mirror_pending "
                    "WHERE agent_id = ? AND key = ? AND value = ? AND op = ?",
                    (agent_id, pair.key, pair.value, op),
                )
                present = (
                    self._conn.execute(
                        "SELECT 1 FROM agent_labels WHERE agent_id = ? AND key = ? AND value = ?",
                        (agent_id, pair.key, pair.value),
                    ).fetchone()
                    is not None
                )
                if present != (op == "add"):
                    self._queue_mirror(
                        agent_id, pair, "add" if present else "remove", replace=False
                    )

    # -- runs and holds ------------------------------------------------------------------

    def put_run(
        self, lease: Lease, agent_id: str, run: int, fields: Mapping[str, object]
    ) -> dict[str, Any]:
        """Create run ``run`` or update the given fields of it."""
        _check_columns(fields, RUN_COLUMNS)
        with self.transaction():
            self._require_lease(lease, f"agent:{agent_id}")
            self._require_agent(agent_id)
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
        with _mapped():
            return self._get_run(agent_id, run)

    def get_runs(self, agent_id: str) -> list[dict[str, Any]]:
        with _mapped():
            rows = self._conn.execute(
                "SELECT * FROM agent_runs WHERE agent_id = ? ORDER BY run", (agent_id,)
            ).fetchall()
            return [_row(row, RUN_COLUMNS, "agent_id", "run") for row in rows]

    def put_hold(
        self, lease: Lease, agent_id: str, record_id: str, path: str, *, owned: bool
    ) -> bool:
        """Record a held worktree record; false when the agent already holds it."""
        with self.transaction():
            self._require_lease(lease, f"agent:{agent_id}")
            self._require_agent(agent_id)
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
            self._require_agent(agent_id)
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
            self._require_agent(agent_id)
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
        with _mapped():
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
        with _mapped():
            rows = self._conn.execute(
                "SELECT event_id, event, payload, created FROM events "
                "WHERE agent_id = ? AND delivered = 0 ORDER BY event_id LIMIT ?",
                (agent_id, limit),
            ).fetchall()
        try:
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


def _check_transition(wanted: tuple[str, ...], target: object) -> None:
    for status in wanted:
        if status not in TRANSITIONS:
            raise ValueError(f"unknown status {status!r}")
    if target is None:
        return
    for status in wanted:
        if target not in TRANSITIONS[status]:
            raise ValueError(f"{status} -> {target} is not a lifecycle transition")


def _decode_holder(key: str, raw: object) -> ProcessIdentity:
    try:
        value = decode_value("proc", raw)
    except ValueDecodeError as exc:
        raise StoreCorrupt(f"lease {key} has an unreadable holder: {exc}") from exc
    if not isinstance(value, ProcessIdentity):
        raise StoreCorrupt(f"lease {key} has no holder")
    return value


def _prepare_files(state_dir: Path, *, create: bool, uid: int) -> tuple[Path, tuple[int, int]]:
    """Make or check the state dir (0700) and ``state.db`` (0600) without following links.

    The state dir itself may be a symlink (the user chose that); ``state.db`` and its
    ``-wal``/``-shm``/``-journal`` files may not, and each must be a regular file of this user
    with a single link. Each is checked with ``lstat``; only a file whose mode is loose is
    opened (``O_NOFOLLOW``), checked again with ``fstat`` and tightened with ``fchmod``
    through that descriptor (closing a descriptor drops every POSIX lock this process holds
    on the file, so a file with the right mode is never opened). A sidecar that SQLite deleted
    meanwhile is skipped. Returns the path and the database's ``(st_dev, st_ino)``, checked
    again after SQLite opened the path.
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
        for suffix in _SIDECARS:
            _check_leaf(Path(str(db) + suffix), uid, sidecar=True)
        file_id = _check_leaf(db, uid, sidecar=False)
        if file_id is None:
            if not create:
                raise StoreMissing(f"no database {db}")
            fd = _open_leaf(db, create=True)
            if fd is not None:
                os.close(fd)  # just made (or made by a racer): no connection holds it yet
            file_id = _check_leaf(db, uid, sidecar=False)
            if file_id is None:
                raise StoreError(f"{db} disappeared while it was created")
        return db, file_id
    except (OSError, ValueError) as exc:
        raise StoreError(f"cannot prepare state directory {state_dir}: {exc}") from exc


def _open_leaf(path: Path, *, create: bool = False) -> int | None:
    """Open without following a symlink (and without blocking on a FIFO); ``None`` when the
    file does not exist. ``create`` makes it 0600 unless another process made it first."""
    try:
        if create:
            try:
                return os.open(path, _OPEN_FLAGS | os.O_CREAT | os.O_EXCL, FILE_MODE)
            except FileExistsError:
                pass  # another process created it first
        return os.open(path, _OPEN_FLAGS)
    except FileNotFoundError:
        return None
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise StoreError(f"{path} is a symbolic link: refusing to follow it") from exc
        raise


def _check_leaf(path: Path, uid: int, *, sidecar: bool) -> tuple[int, int] | None:
    """Check a leaf by ``lstat`` (regular, this user's, one link, not a symlink) and tighten a
    loose mode to 0600. ``None`` when it does not exist, or is a sidecar SQLite deleted while
    it was checked (normal on a last close or WAL switch). Returns ``(st_dev, st_ino)``."""
    try:
        info = os.lstat(path)
    except FileNotFoundError:
        return None
    if stat.S_ISLNK(info.st_mode):
        raise StoreError(f"{path} is a symbolic link: refusing to follow it")
    _check_info(info, path, uid)
    if stat.S_IMODE(info.st_mode) == FILE_MODE:
        return info.st_dev, info.st_ino
    fd = _open_leaf(path)
    if fd is None:
        if sidecar:
            return None
        raise StoreError(f"{path} disappeared while it was checked")
    try:
        info = os.fstat(fd)
        if info.st_nlink == 0 and sidecar:
            return None  # SQLite deleted it between our lstat and open
        _check_info(info, path, uid)
        os.fchmod(fd, FILE_MODE)
        return info.st_dev, info.st_ino
    finally:
        os.close(fd)


def _check_info(info: os.stat_result, path: Path, uid: int) -> None:
    if not stat.S_ISREG(info.st_mode):
        raise StoreError(f"{path} is not a regular file")
    if info.st_uid != uid:
        raise StoreError(f"{path} is not owned by this user")
    if info.st_nlink != 1:
        raise StoreError(f"{path} has {info.st_nlink} hard links: refusing it")


def _check_same_file(path: Path, file_id: tuple[int, int]) -> None:
    """SQLite opened ``path`` by name: refuse it unless it is still the file that was checked."""
    try:
        info = os.lstat(path)
    except OSError as exc:
        raise StoreError(f"{path} disappeared while it was opened: {exc}") from exc
    if (info.st_dev, info.st_ino) != file_id or not stat.S_ISREG(info.st_mode):
        raise StoreError(f"{path} was replaced while it was opened: refusing it")


def _fix_sidecar_modes(db: Path, uid: int) -> None:
    """Check the sidecars SQLite created, by ``lstat`` only: the connection is open, and
    opening and closing a sidecar would drop its locks on the ``-shm``. SQLite gives them the
    database's mode, so a loose one is refused (the next open tightens it before SQLite
    opens the database); one a checkpoint removed is skipped."""
    for suffix in _SIDECARS:
        side = Path(str(db) + suffix)
        try:
            info = os.lstat(side)
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise StoreError(f"cannot check {side}: {exc}") from exc
        if stat.S_ISLNK(info.st_mode):
            raise StoreError(f"{side} is a symbolic link: refusing to follow it")
        _check_info(info, side, uid)
        if stat.S_IMODE(info.st_mode) != FILE_MODE:
            raise StoreError(
                f"{side} has mode {stat.S_IMODE(info.st_mode):04o}, not "
                f"{FILE_MODE:04o}: refusing it"
            )
