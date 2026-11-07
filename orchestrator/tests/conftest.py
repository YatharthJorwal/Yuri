import pytest

from memory import db


@pytest.fixture
def memory_db(tmp_path, monkeypatch):
    """Real sqlite-vec connection on a throwaway file (dim=4), swapped in for
    db.get_connection so every memory module uses it."""
    conn = db._open_connection(path=tmp_path / "test_memory.db", dimension=4)
    monkeypatch.setattr(db, "_conn", conn)
    monkeypatch.setattr(db, "get_connection", lambda: conn)
    yield conn
    conn.close()
