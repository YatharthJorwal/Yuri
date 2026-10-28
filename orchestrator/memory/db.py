"""
SQLite connection + schema for Phase 3 memory. One module-level connection,
same "lazy singleton" shape stt.py already uses for its model -- opened on
first real use, not at import time, so a session that never triggers any
memory read/write (e.g. the very first turn of a fresh DB) doesn't pay for
it up front. Unlike stt.py's model, though, opening a SQLite file + running
`CREATE TABLE IF NOT EXISTS` is cheap enough that "lazy" here is really just
"don't touch disk before config is loaded", not a real perf concern.

`sqlite-vec` (https://github.com/asg017/sqlite-vec) is a loadable SQLite
extension providing a `vec0` virtual table for nearest-neighbor search over
float vectors -- pure-C, no GPU/network dependency, ships prebuilt wheels
for Windows/Linux/macOS via `pip install sqlite-vec`, so this doesn't need
its own server process (unlike the LLM/TTS/STT backends) and works offline
same as everything else in this project has to.

Embedding dimension is config, not hardcoded (`memory.embedding.dimension`
in config.yaml) -- it has to match whatever embedding model is configured,
and `vec0` tables declare their vector width at CREATE time, so the
episode_vectors table is created with that exact width baked in. If the
embedding model is ever swapped for one with a different dimension, the
existing table won't match -- see `recall_dimension_mismatch()` below,
checked once at startup rather than left to surface as a cryptic sqlite-vec
error mid-turn.
"""

from __future__ import annotations

import pathlib
import sqlite3
import sys

import sqlite_vec

from config import CONFIG

_DB_PATH = pathlib.Path(__file__).parent.parent / CONFIG.memory.db_path
_conn: sqlite3.Connection | None = None


class MemoryUnavailableError(Exception):
    """Wraps any failure opening/initializing the memory DB (disk full,
    file locked by another process, sqlite-vec extension failing to load,
    a stored embedding dimension that no longer matches config, etc.) --
    one exception type so recall.py/consolidation.py can catch it and
    degrade to "no memory this turn" instead of the turn itself failing,
    same pattern LLMUnreachableError/TTSUnreachableError/STTError already
    use for their own backends."""


def _open_connection(path: str | pathlib.Path | None = None, dimension: int | None = None) -> sqlite3.Connection:
    """`path`/`dimension` are only ever overridden by test_memory.py, so
    tests can point at a throwaway temp file instead of the real
    config-driven DB (`data/memory.db`) -- real callers (get_connection()
    below) always take the config defaults."""
    db_path = pathlib.Path(path) if path is not None else _DB_PATH
    dim = dimension if dimension is not None else CONFIG.memory.embedding.dimension

    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS facts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS episodes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            summary TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        )
        """
    )
    # Phase 9's persistent conversation-log panel -- a plain verbatim
    # transcript (one row per turn: what the user said, what she actually
    # said back), completely separate from facts/episodes above. Deliberately
    # NOT read by recall.py or written by consolidation.py -- this is for
    # the user's own review (the log panel's "clear log" button), not
    # something fed back into her memory/context. See store.py's
    # add_transcript_turn/get_transcript_log/clear_transcript_log.
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS transcript_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_text TEXT NOT NULL,
            assistant_text TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        )
        """
    )
    # vec0 virtual table's rowid IS the episode id -- inserted explicitly
    # as `rowid` in store.py, not auto-assigned, so a lookup in one table
    # always has a matching rowid in the other. float[N] fixes the vector
    # width at creation time; see the module docstring on what happens if
    # config's dimension ever stops matching an already-created table.
    conn.execute(
        f"CREATE VIRTUAL TABLE IF NOT EXISTS episode_vectors USING vec0(embedding float[{dim}])"
    )
    conn.commit()
    return conn


def get_connection() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        try:
            _conn = _open_connection()
        except Exception as exc:  # noqa: BLE001 -- deliberately broad, see class docstring
            raise MemoryUnavailableError(
                f"couldn't open/initialize the memory DB at {_DB_PATH}: {exc}"
            ) from exc
    return _conn


def check_embedding_dimension_matches() -> None:
    """Called once at orchestrator startup (app.py). vec0 bakes its vector
    width into the table at CREATE time -- if `memory.embedding.dimension`
    in config.yaml is ever changed after episodes already exist (e.g.
    swapping nomic-embed-text's 768 dims for a different model), the old
    table stays at the old width silently; inserting a differently-sized
    embedding later raises deep inside sqlite-vec instead of somewhere
    obvious. Checking a real vec0 table's declared width against config
    right at startup turns that into a clear, actionable message instead
    of a mid-turn crash the first time someone actually swaps embedding
    models."""
    conn = get_connection()
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE name = 'episode_vectors'"
    ).fetchone()
    if row is None:
        return  # table doesn't exist yet -- nothing to mismatch against
    declared_sql = row[0] or ""
    configured_dim = CONFIG.memory.embedding.dimension
    expected_fragment = f"float[{configured_dim}]"
    if expected_fragment not in declared_sql:
        print(
            f"[luna] WARNING: memory.db's episode_vectors table doesn't "
            f"declare {expected_fragment}, but config.yaml's "
            f"memory.embedding.dimension is {configured_dim} -- looks like "
            "the embedding model was changed after episodes already "
            f"existed. Table SQL: {declared_sql!r}. New episodes will "
            "likely fail to insert until this is resolved (delete "
            f"{_DB_PATH} to start fresh, losing existing memory, or revert "
            "the embedding model/dimension back to what the table was "
            "actually created with).",
            file=sys.stderr,
            flush=True,
        )
