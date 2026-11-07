"""
Schema versioning for memory.db, plus the opt-in importer for Luna's old DB.

`run()` is called every time a connection opens. It is additive and idempotent:
it only ever adds columns and tables, never rewrites or drops data, and it
records the version in `meta(schema_version)` so a second call is a no-op. A
fresh file goes through the same path as a v1 file (create the v1 tables, then
migrate), so there is exactly one code path to trust.

Version 1 = Luna's schema (facts, episodes, transcript_log, episode_vectors).
Version 2 = provenance, supersede pointers, salience, derived profile card,
            audit events, and a vector table for facts (docs/phases/PHASE-3-MEMORY.md).
"""

from __future__ import annotations

import sqlite3
import struct
import sys

SCHEMA_VERSION = 2

# Sentinels stored in `superseded_by`. Positive values are the id of the row
# that replaced this one; these are the "no replacement" reasons.
FORGOTTEN = -1   # the user asked for it to be forgotten
DECAYED = -2     # fell out of search through decay (episodes only)

_V2_FACT_COLUMNS = (
    ("subject", "TEXT"),
    ("source", "TEXT NOT NULL DEFAULT 'writer_pass'"),
    ("confidence", "REAL NOT NULL DEFAULT 0.7"),
    ("pinned", "INTEGER NOT NULL DEFAULT 0"),
    ("created_ts", "REAL"),
    ("last_seen_at", "REAL"),
    ("decayed_at", "REAL"),
    ("superseded_by", "INTEGER"),
)
_V2_EPISODE_COLUMNS = (
    ("salience", "REAL NOT NULL DEFAULT 0.5"),
    ("created_ts", "REAL"),
    ("last_used_at", "REAL"),
    ("decayed_at", "REAL"),
    ("session_id", "TEXT"),
    ("turns", "INTEGER"),
    ("tokens", "INTEGER"),
    ("superseded_by", "INTEGER"),
)
_V2_TRANSCRIPT_COLUMNS = (("temp", "INTEGER NOT NULL DEFAULT 0"),)


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}


def _add_columns(conn: sqlite3.Connection, table: str, columns: tuple[tuple[str, str], ...]) -> None:
    existing = _columns(conn, table)
    for name, ddl in columns:
        if name not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {ddl}")


def _create_v1_tables(conn: sqlite3.Connection, dim: int) -> None:
    conn.execute(
        "CREATE TABLE IF NOT EXISTS facts (id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "content TEXT NOT NULL, created_at TEXT NOT NULL DEFAULT (datetime('now')))"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS episodes (id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "summary TEXT NOT NULL, created_at TEXT NOT NULL DEFAULT (datetime('now')))"
    )
    # Verbatim log for the user's own review panel. The writer reads it too
    # (it is the durable scratch buffer), but only rows with temp = 0.
    conn.execute(
        "CREATE TABLE IF NOT EXISTS transcript_log (id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "user_text TEXT NOT NULL, assistant_text TEXT NOT NULL, "
        "created_at TEXT NOT NULL DEFAULT (datetime('now')))"
    )
    conn.execute(f"CREATE VIRTUAL TABLE IF NOT EXISTS episode_vectors USING vec0(embedding float[{dim}])")


def get_version(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
    return int(row[0]) if row else 1


def run(conn: sqlite3.Connection, dim: int) -> None:
    """Bring `conn` to SCHEMA_VERSION inside one transaction."""
    conn.commit()
    conn.execute("BEGIN IMMEDIATE")
    try:
        conn.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)")
        _create_v1_tables(conn, dim)
        if get_version(conn) < 2:
            _add_columns(conn, "facts", _V2_FACT_COLUMNS)
            _add_columns(conn, "episodes", _V2_EPISODE_COLUMNS)
            _add_columns(conn, "transcript_log", _V2_TRANSCRIPT_COLUMNS)
            # Rows that already exist predate provenance: stamp them from the
            # old text timestamp so decay has something numeric to work with.
            conn.execute(
                "UPDATE facts SET created_ts = COALESCE(created_ts, CAST(strftime('%s', created_at) AS REAL)), "
                "last_seen_at = COALESCE(last_seen_at, CAST(strftime('%s', created_at) AS REAL)), "
                "decayed_at = COALESCE(decayed_at, CAST(strftime('%s', created_at) AS REAL))"
            )
            conn.execute(
                "UPDATE episodes SET created_ts = COALESCE(created_ts, CAST(strftime('%s', created_at) AS REAL)), "
                "decayed_at = COALESCE(decayed_at, CAST(strftime('%s', created_at) AS REAL))"
            )
        # These are all IF NOT EXISTS, so they also repair a half-migrated file.
        conn.execute(f"CREATE VIRTUAL TABLE IF NOT EXISTS facts_vec USING vec0(embedding float[{dim}])")
        conn.execute(
            "CREATE TABLE IF NOT EXISTS profile_card (id INTEGER PRIMARY KEY CHECK (id = 1), "
            "text TEXT NOT NULL, updated_at REAL NOT NULL)"
        )
        conn.execute(
            "CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL NOT NULL, "
            "kind TEXT NOT NULL, actor TEXT NOT NULL, detail TEXT NOT NULL)"
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_events_kind ON events(kind, id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_facts_live ON facts(superseded_by)")
        conn.execute("INSERT OR IGNORE INTO meta (key, value) VALUES ('embedding_dim', ?)", (str(dim),))
        conn.execute(
            "INSERT INTO meta (key, value) VALUES ('schema_version', ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (str(SCHEMA_VERSION),),
        )
        conn.commit()
    except BaseException:
        conn.rollback()
        raise


# ---------------------------------------------------------------------------
# Luna importer -- never runs automatically (`python -m memory --import-luna`).
# ---------------------------------------------------------------------------

IMPORT_FACT_CONFIDENCE = 0.6
IMPORT_EPISODE_SALIENCE = 0.4


def import_luna(luna_db_path: str, *, dim: int | None = None) -> dict[str, int]:
    """Copies Luna's facts and episodes into the current memory.db.

    Facts arrive as `source="import"`, confidence 0.6 (no vector yet; the writer
    backfills them). Episodes get fresh ids and salience 0.4; their embeddings
    are copied only if the dimension matches, otherwise the episode text is kept
    and left out of semantic search. Luna's file is opened read-only."""
    import sqlite_vec

    from . import db, store

    src = sqlite3.connect(f"file:{luna_db_path}?mode=ro", uri=True)
    try:
        src.enable_load_extension(True)
        sqlite_vec.load(src)
        src.enable_load_extension(False)
        facts = [r[0] for r in src.execute("SELECT content FROM facts ORDER BY id")]
        episodes = src.execute("SELECT id, summary FROM episodes ORDER BY id").fetchall()
        vectors: dict[int, bytes] = {}
        try:
            vectors = {int(r[0]): bytes(r[1]) for r in src.execute("SELECT rowid, embedding FROM episode_vectors")}
        except sqlite3.DatabaseError:
            pass  # no vector table in the old file; text-only import
    finally:
        src.close()

    want = dim if dim is not None else int(store.get_meta("embedding_dim", "0") or 0)
    counts = {"facts": 0, "episodes": 0, "episodes_without_vector": 0}
    with db.transaction():
        for content in facts:
            if store.add_fact(content, source="import", confidence=IMPORT_FACT_CONFIDENCE) is not None:
                counts["facts"] += 1
        for old_id, summary in episodes:
            raw = vectors.get(int(old_id))
            vector = list(struct.unpack(f"{len(raw) // 4}f", raw)) if raw else None
            if vector is not None and len(vector) != want:
                vector = None
            store.add_episode(summary, vector, salience=IMPORT_EPISODE_SALIENCE)
            counts["episodes"] += 1
            if vector is None:
                counts["episodes_without_vector"] += 1
        store.log_event("import", "migrate", f"luna facts={counts['facts']} episodes={counts['episodes']}")
    print(f"[yuri] imported from Luna: {counts}", file=sys.stderr)
    return counts

