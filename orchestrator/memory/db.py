"""
SQLite connection for memory. One module-level connection, opened lazily on
first real use so a session that never touches memory pays nothing.

`sqlite-vec` provides the `vec0` virtual table for nearest-neighbour search. It
is a loadable extension with prebuilt wheels, so memory needs no server process
and works offline like everything else here. The vector width is config
(`memory.embedding.dimension`) and is baked into each vec0 table at CREATE time,
which is why the width is also recorded in `meta` and checked before every write.

Schema creation and upgrades live in migrate.py. This module owns the
connection: opening it, quarantining a corrupt file instead of crashing on it,
and the process-wide lock + `transaction()` that make a multi-step write atomic
without letting another thread's commit land in the middle of it.
"""

from __future__ import annotations

import contextlib
import pathlib
import sqlite3
import sys
import threading
import time
from collections.abc import Iterator

import sqlite_vec

from config import CONFIG

from . import migrate

_DB_PATH = pathlib.Path(__file__).parent.parent / CONFIG.memory.db_path
_conn: sqlite3.Connection | None = None

# One connection is shared by every thread (asyncio.to_thread callers), so every
# store function takes this lock. It is re-entrant: a transaction() holds it for
# its whole body and the store calls inside simply re-acquire it.
LOCK = threading.RLock()
_tx_state = threading.local()


class MemoryUnavailableError(Exception):
    """Any failure opening or using the memory DB (disk full, file locked, the
    extension failing to load, an embedding width that no longer matches
    config...). One exception type so recall and the writer can catch it and
    degrade to "no memory this turn" instead of failing the turn, the same
    pattern LLMUnreachableError and STTError use for their backends."""


def in_transaction() -> bool:
    """True only on the thread currently inside `transaction()`."""
    return getattr(_tx_state, "depth", 0) > 0


@contextlib.contextmanager
def transaction() -> Iterator[sqlite3.Connection]:
    """All-or-nothing group of store calls. Store functions skip their own
    commit while this is open; an exception rolls everything back, so a crash
    mid-write leaves the database exactly as it was."""
    with LOCK:
        conn = get_connection()
        if in_transaction():  # nested: the outer block owns commit/rollback
            _tx_state.depth += 1
            try:
                yield conn
            finally:
                _tx_state.depth -= 1
            return
        conn.execute("BEGIN IMMEDIATE")
        _tx_state.depth = 1
        try:
            yield conn
        except BaseException:
            conn.rollback()
            raise
        else:
            conn.commit()
        finally:
            _tx_state.depth = 0


def commit(conn: sqlite3.Connection) -> None:
    """Commit unless a transaction() block owns the commit."""
    if not in_transaction():
        conn.commit()


def _looks_corrupt(exc: BaseException) -> bool:
    text = str(exc).lower()
    return isinstance(exc, sqlite3.DatabaseError) and any(
        marker in text for marker in ("malformed", "not a database", "disk image", "corrupt")
    )


def _quarantine(path: pathlib.Path) -> pathlib.Path:
    """Moves a corrupt DB (and its WAL/SHM sidecars) aside, never deleting it,
    so the owner can still try to recover it by hand."""
    stamp = int(time.time())
    target = path.with_name(f"{path.name}.corrupt-{stamp}")
    for suffix in ("", "-wal", "-shm"):
        src = pathlib.Path(str(path) + suffix)
        if src.exists():
            src.rename(pathlib.Path(str(target) + suffix))
    return target


def _open_connection(path: str | pathlib.Path | None = None, dimension: int | None = None) -> sqlite3.Connection:
    """`path`/`dimension` are only overridden by tests (a throwaway temp file);
    real callers take the config defaults. Opens, verifies, migrates."""
    db_path = pathlib.Path(path) if path is not None else _DB_PATH
    dim = dimension if dimension is not None else CONFIG.memory.embedding.dimension

    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    try:
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        verdict = conn.execute("PRAGMA quick_check(1)").fetchone()
        if verdict and verdict[0] != "ok":
            raise sqlite3.DatabaseError(f"database disk image is malformed ({verdict[0]})")
        migrate.run(conn, dim)
    except BaseException:
        conn.close()
        raise
    return conn


def open_with_recovery(path: str | pathlib.Path | None = None, dimension: int | None = None) -> sqlite3.Connection:
    """Like `_open_connection`, but a corrupt file is quarantined and replaced
    with a fresh one instead of taking memory down for good. Anything that is not
    corruption (locked file, missing extension, full disk) still raises, because
    quarantining a healthy file would be data loss."""
    db_path = pathlib.Path(path) if path is not None else _DB_PATH
    try:
        return _open_connection(db_path, dimension)
    except sqlite3.DatabaseError as exc:
        if not _looks_corrupt(exc):
            raise
        moved = _quarantine(db_path)
        print(
            f"[yuri] WARNING: memory.db was corrupt ({exc}). Moved it to {moved} and starting "
            "with an empty memory. Nothing was deleted.",
            file=sys.stderr,
            flush=True,
        )
        return _open_connection(db_path, dimension)


def get_connection() -> sqlite3.Connection:
    global _conn
    with LOCK:
        if _conn is None:
            try:
                _conn = open_with_recovery()
            except Exception as exc:  # noqa: BLE001 -- see MemoryUnavailableError
                raise MemoryUnavailableError(
                    f"couldn't open/initialize the memory DB at {_DB_PATH}: {exc}"
                ) from exc
        return _conn


def check_embedding_dimension_matches() -> None:
    """Called once at startup. A vec0 table keeps the width it was created with;
    if `memory.embedding.dimension` changes later, the old tables silently keep
    the old width and a later insert fails deep inside sqlite-vec. Comparing the
    declared widths to config here turns that into a clear message at startup.
    Writes are independently refused by store._check_dimension()."""
    conn = get_connection()
    configured_dim = CONFIG.memory.embedding.dimension
    expected_fragment = f"float[{configured_dim}]"
    for table in ("episode_vectors", "facts_vec"):
        row = conn.execute("SELECT sql FROM sqlite_master WHERE name = ?", (table,)).fetchone()
        if row is None:
            continue
        declared_sql = row[0] or ""
        if expected_fragment not in declared_sql:
            print(
                f"[yuri] WARNING: memory.db's {table} table doesn't declare {expected_fragment}, "
                f"but config.yaml's memory.embedding.dimension is {configured_dim} -- the embedding "
                f"model was changed after memory already existed. Table SQL: {declared_sql!r}. "
                "Memory writes that carry an embedding are refused until this is resolved: revert "
                f"the model/dimension, or move {_DB_PATH} aside to start fresh.",
                file=sys.stderr,
                flush=True,
            )
            return
