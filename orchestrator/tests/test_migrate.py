"""Schema v2 migration, corruption quarantine, dimension refusal, transactions
and the Luna importer. All against real SQLite + sqlite-vec."""

from __future__ import annotations

import sqlite3
import struct

import pytest
import sqlite_vec

from memory import db, migrate, store
from memory_helpers import fake_vector


def _v1_database(path, dim=4):
    """A database exactly as Luna / pre-Phase-3 Yuri wrote it."""
    conn = sqlite3.connect(str(path))
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.execute("CREATE TABLE facts (id INTEGER PRIMARY KEY AUTOINCREMENT, content TEXT NOT NULL, created_at TEXT NOT NULL DEFAULT (datetime('now')))")
    conn.execute("CREATE TABLE episodes (id INTEGER PRIMARY KEY AUTOINCREMENT, summary TEXT NOT NULL, created_at TEXT NOT NULL DEFAULT (datetime('now')))")
    conn.execute("CREATE TABLE transcript_log (id INTEGER PRIMARY KEY AUTOINCREMENT, user_text TEXT NOT NULL, assistant_text TEXT NOT NULL, created_at TEXT NOT NULL DEFAULT (datetime('now')))")
    conn.execute(f"CREATE VIRTUAL TABLE episode_vectors USING vec0(embedding float[{dim}])")
    conn.execute("INSERT INTO facts (content, created_at) VALUES ('likes pizza', '2026-01-02 03:04:05')")
    conn.execute("INSERT INTO episodes (summary, created_at) VALUES ('fixed lipsync', '2026-01-02 03:04:05')")
    conn.execute("INSERT INTO episode_vectors (rowid, embedding) VALUES (1, ?)", (struct.pack("4f", *fake_vector(0.3)),))
    conn.execute("INSERT INTO transcript_log (user_text, assistant_text) VALUES ('hi', 'hello')")
    conn.commit()
    return conn


def _columns(conn, table):
    return [r[1] for r in conn.execute(f"PRAGMA table_info({table})")]


def test_v1_database_migrates_in_place_without_losing_rows(tmp_path):
    path = tmp_path / "old.db"
    _v1_database(path).close()

    conn = db._open_connection(path=path, dimension=4)
    try:
        assert migrate.get_version(conn) == 2
        row = conn.execute("SELECT content, source, confidence, pinned, superseded_by, created_ts FROM facts").fetchone()
        assert row[:5] == ("likes pizza", "writer_pass", 0.7, 0, None)
        assert row[5] and row[5] > 1.7e9  # stamped from the old text timestamp
        assert conn.execute("SELECT summary, salience FROM episodes").fetchone() == ("fixed lipsync", 0.5)
        assert conn.execute("SELECT temp FROM transcript_log").fetchone() == (0,)
        # the old episode vector is still searchable
        assert conn.execute("SELECT count(*) FROM episode_vectors").fetchone() == (1,)
        for table in ("facts_vec", "profile_card", "events", "meta"):
            assert conn.execute("SELECT 1 FROM sqlite_master WHERE name = ?", (table,)).fetchone()
    finally:
        conn.close()


def test_migration_is_idempotent(tmp_path):
    path = tmp_path / "idem.db"
    _v1_database(path).close()
    db._open_connection(path=path, dimension=4).close()
    conn = db._open_connection(path=path, dimension=4)
    try:
        cols = _columns(conn, "facts")
        assert len(cols) == len(set(cols))
        assert conn.execute("SELECT count(*) FROM facts").fetchone() == (1,)
        assert migrate.get_version(conn) == 2
    finally:
        conn.close()


def test_fresh_database_lands_on_v2_with_dimension_recorded(memory_db):
    assert migrate.get_version(memory_db) == 2
    assert store.get_meta("embedding_dim") == "4"


def test_corrupt_file_is_quarantined_not_deleted(tmp_path, capsys):
    path = tmp_path / "memory.db"
    garbage = b"this is not a sqlite database at all" * 40
    path.write_bytes(garbage)

    conn = db.open_with_recovery(path=path, dimension=4)
    try:
        moved = list(tmp_path.glob("memory.db.corrupt-*"))
        assert len(moved) == 1 and moved[0].read_bytes() == garbage
        assert migrate.get_version(conn) == 2  # fresh, usable
        assert "corrupt" in capsys.readouterr().err.lower()
    finally:
        conn.close()


def test_only_corruption_triggers_quarantine():
    assert db._looks_corrupt(sqlite3.DatabaseError("database disk image is malformed"))
    assert db._looks_corrupt(sqlite3.DatabaseError("file is not a database"))
    assert not db._looks_corrupt(sqlite3.OperationalError("database is locked"))
    assert not db._looks_corrupt(RuntimeError("malformed"))


def test_embedding_of_the_wrong_width_is_refused_before_any_write(memory_db):
    fact_id = store.add_fact("likes pizza")
    with pytest.raises(db.MemoryUnavailableError, match="dimensions"):
        store.set_fact_vector(fact_id, [1.0, 0.0, 0.0])  # 3 != 4
    with pytest.raises(db.MemoryUnavailableError):
        store.add_episode("won't be stored", [1.0, 0.0])
    assert store.get_episode_rows() == []


def test_transaction_rolls_back_everything_on_error(memory_db):
    with pytest.raises(RuntimeError):
        with db.transaction():
            store.add_fact("will vanish")
            store.set_meta("k", "v")
            raise RuntimeError("boom")
    assert store.get_all_facts() == []
    assert store.get_meta("k") is None


def test_transaction_commits_once_and_nests(memory_db):
    with db.transaction():
        store.add_fact("a")
        with db.transaction():
            store.add_fact("b")
    assert sorted(store.get_all_facts()) == ["a", "b"]
    assert not db.in_transaction()


def test_luna_import_is_additive_and_marks_provenance(memory_db, tmp_path):
    luna = tmp_path / "luna.db"
    conn = _v1_database(luna)
    conn.execute("INSERT INTO episodes (summary) VALUES ('no vector for me')")
    conn.commit()
    conn.close()
    store.add_fact("already here")

    counts = migrate.import_luna(str(luna))

    assert counts == {"facts": 1, "episodes": 2, "episodes_without_vector": 1}
    rows = {f["content"]: f for f in store.get_fact_rows()}
    assert rows["likes pizza"]["source"] == "import" and rows["likes pizza"]["confidence"] == 0.6
    assert "already here" in rows
    assert all(e["salience"] == 0.4 for e in store.get_episode_rows())
    assert store.search_episodes(fake_vector(0.3), 2) == ["fixed lipsync"]  # only the one with a vector
    assert store.get_events("import")
