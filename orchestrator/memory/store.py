"""
Facts, episodes, events, the profile card and the transcript buffer: plain CRUD
plus the vec0 nearest-neighbour queries. Everything here is synchronous (sqlite3
has no async driver and these are fast local-disk calls); async callers run them
through `asyncio.to_thread`.

Nothing in this module deletes a fact or an episode. "Forgetting" and
"replacing" both set `superseded_by` and drop the row's vector, so the row stays
readable (and `restore_fact` can bring it back) while search and recall never see
it again. `superseded_by` is NULL for a live row, the id of the replacement for a
superseded one, and one of migrate.FORGOTTEN / migrate.DECAYED otherwise.

Every public function takes db.LOCK, so a `db.transaction()` on one thread cannot
be interleaved with another thread's reads or commits.
"""

from __future__ import annotations

import functools
import json
import math
import struct
import time
from typing import Any

from . import db
from .migrate import DECAYED, FORGOTTEN

DEFAULT_CONFIDENCE = {"user_stated": 1.0, "correction": 1.0, "writer_pass": 0.7, "import": 0.6}


def _locked(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with db.LOCK:
            return fn(*args, **kwargs)

    return wrapper


# ---------------------------------------------------------------------------
# vectors
# ---------------------------------------------------------------------------


def _pack(vector: list[float]) -> bytes:
    """sqlite-vec stores and queries vectors as raw little-endian float32 bytes."""
    return struct.pack(f"{len(vector)}f", *vector)


def _unit(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vector))
    return [x / norm for x in vector] if norm > 0 else list(vector)


def _prep(vector: list[float]) -> bytes:
    """Unit-normalise, then pack. With unit vectors L2 distance maps exactly to
    cosine similarity (see `similarity`), which is what the recall floors use."""
    return _pack(_unit(vector))


def similarity(distance: float) -> float:
    """Cosine similarity from the L2 distance between two unit vectors."""
    return max(-1.0, min(1.0, 1.0 - (distance * distance) / 2.0))


def _check_dimension(conn, vector: list[float]) -> None:
    row = conn.execute("SELECT value FROM meta WHERE key = 'embedding_dim'").fetchone()
    if row is not None and int(row[0]) != len(vector):
        raise db.MemoryUnavailableError(
            f"embedding has {len(vector)} dimensions but memory.db was created for {row[0]}. "
            "The embedding model changed after memory existed; refusing to write rather than "
            "corrupt the index. Revert memory.embedding, or move memory.db aside to start fresh."
        )


# ---------------------------------------------------------------------------
# meta / events / profile card
# ---------------------------------------------------------------------------


@_locked
def get_meta(key: str, default: str | None = None) -> str | None:
    row = db.get_connection().execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return row[0] if row else default


@_locked
def set_meta(key: str, value: str) -> None:
    conn = db.get_connection()
    conn.execute(
        "INSERT INTO meta (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, str(value)),
    )
    db.commit(conn)


@_locked
def log_event(kind: str, actor: str, detail: str | dict | list = "") -> None:
    """Append-only audit trail: what changed memory, who did it, and why."""
    text = detail if isinstance(detail, str) else json.dumps(detail, separators=(",", ":"), sort_keys=True)
    conn = db.get_connection()
    conn.execute(
        "INSERT INTO events (ts, kind, actor, detail) VALUES (?, ?, ?, ?)", (time.time(), kind, actor, text)
    )
    db.commit(conn)


@_locked
def get_events(kind: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
    conn = db.get_connection()
    if kind is None:
        rows = conn.execute("SELECT id, ts, kind, actor, detail FROM events ORDER BY id DESC LIMIT ?", (limit,))
    else:
        rows = conn.execute(
            "SELECT id, ts, kind, actor, detail FROM events WHERE kind = ? ORDER BY id DESC LIMIT ?",
            (kind, limit),
        )
    return [{"id": r[0], "ts": r[1], "kind": r[2], "actor": r[3], "detail": r[4]} for r in rows.fetchall()]


@_locked
def get_profile_card() -> str | None:
    row = db.get_connection().execute("SELECT text FROM profile_card WHERE id = 1").fetchone()
    return row[0] if row else None


@_locked
def set_profile_card(text: str) -> None:
    conn = db.get_connection()
    conn.execute(
        "INSERT INTO profile_card (id, text, updated_at) VALUES (1, ?, ?) "
        "ON CONFLICT(id) DO UPDATE SET text = excluded.text, updated_at = excluded.updated_at",
        (text.strip(), time.time()),
    )
    db.commit(conn)


# ---------------------------------------------------------------------------
# facts
# ---------------------------------------------------------------------------

_FACT_COLUMNS = "id, content, subject, source, confidence, pinned, created_ts, last_seen_at, superseded_by"


def _fact_dict(row) -> dict[str, Any]:
    return {
        "id": row[0],
        "content": row[1],
        "subject": row[2],
        "source": row[3],
        "confidence": row[4],
        "pinned": bool(row[5]),
        "created_ts": row[6],
        "last_seen_at": row[7],
        "superseded_by": row[8],
    }


@_locked
def add_fact(
    content: str,
    *,
    source: str = "writer_pass",
    confidence: float | None = None,
    subject: str | None = None,
    pinned: bool = False,
) -> int | None:
    """Inserts a live fact and returns its id (None for blank text)."""
    content = content.strip()
    if not content:
        return None
    if confidence is None:
        confidence = DEFAULT_CONFIDENCE.get(source, 0.7)
    now = time.time()
    conn = db.get_connection()
    cursor = conn.execute(
        "INSERT INTO facts (content, subject, source, confidence, pinned, created_ts, last_seen_at, decayed_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (content, subject, source, confidence, int(pinned), now, now, now),
    )
    db.commit(conn)
    return cursor.lastrowid


@_locked
def set_fact_vector(fact_id: int, embedding: list[float]) -> None:
    conn = db.get_connection()
    _check_dimension(conn, embedding)
    conn.execute("DELETE FROM facts_vec WHERE rowid = ?", (fact_id,))
    conn.execute("INSERT INTO facts_vec (rowid, embedding) VALUES (?, ?)", (fact_id, _prep(embedding)))
    db.commit(conn)


@_locked
def facts_missing_vectors(limit: int = 50) -> list[tuple[int, str]]:
    """Live facts with no vector row, e.g. written while the embedder was down
    or imported from Luna. The writer pass backfills them."""
    rows = db.get_connection().execute(
        "SELECT id, content FROM facts WHERE superseded_by IS NULL "
        "AND id NOT IN (SELECT rowid FROM facts_vec) ORDER BY id LIMIT ?",
        (limit,),
    )
    return [(r[0], r[1]) for r in rows.fetchall()]


@_locked
def get_all_facts(limit: int | None = None) -> list[str]:
    return [content for _id, content in get_all_facts_with_ids(limit)]


@_locked
def get_all_facts_with_ids(limit: int | None = None) -> list[tuple[int, str]]:
    """Live facts only, newest first. `id DESC` breaks ties because
    `created_at` has one-second resolution and a pass writes several at once."""
    query = "SELECT id, content FROM facts WHERE superseded_by IS NULL ORDER BY created_at DESC, id DESC"
    if limit is not None:
        query += f" LIMIT {int(limit)}"
    return [(r[0], r[1]) for r in db.get_connection().execute(query).fetchall()]


@_locked
def get_fact_rows(*, include_superseded: bool = False, limit: int | None = None) -> list[dict[str, Any]]:
    query = f"SELECT {_FACT_COLUMNS} FROM facts"
    if not include_superseded:
        query += " WHERE superseded_by IS NULL"
    query += " ORDER BY created_at DESC, id DESC"
    if limit is not None:
        query += f" LIMIT {int(limit)}"
    return [_fact_dict(r) for r in db.get_connection().execute(query).fetchall()]


@_locked
def get_fact(fact_id: int) -> dict[str, Any] | None:
    row = db.get_connection().execute(f"SELECT {_FACT_COLUMNS} FROM facts WHERE id = ?", (fact_id,)).fetchone()
    return _fact_dict(row) if row else None


@_locked
def search_facts(query_embedding: list[float], k: int) -> list[dict[str, Any]]:
    """Nearest live facts, closest first, each with `similarity` (cosine)."""
    conn = db.get_connection()
    rows = conn.execute(
        f"""
        SELECT {", ".join("facts." + c.strip() for c in _FACT_COLUMNS.split(","))}, facts_vec.distance
        FROM facts_vec
        JOIN facts ON facts.id = facts_vec.rowid
        WHERE facts_vec.embedding MATCH ? AND k = ? AND facts.superseded_by IS NULL
        ORDER BY facts_vec.distance
        """,
        (_prep(query_embedding), k),
    ).fetchall()
    return [{**_fact_dict(r), "similarity": similarity(r[9])} for r in rows]


@_locked
def touch_fact(fact_id: int, *, bump: float = 0.05) -> None:
    """The same fact was stated again: refresh it and nudge confidence up."""
    now = time.time()
    conn = db.get_connection()
    conn.execute(
        "UPDATE facts SET last_seen_at = ?, decayed_at = ?, confidence = MIN(1.0, confidence + ?) WHERE id = ?",
        (now, now, bump, fact_id),
    )
    db.commit(conn)


@_locked
def set_pinned(fact_id: int, pinned: bool) -> None:
    conn = db.get_connection()
    conn.execute("UPDATE facts SET pinned = ? WHERE id = ?", (int(pinned), fact_id))
    db.commit(conn)


def _retire_facts(fact_ids: list[int], marker: int) -> None:
    conn = db.get_connection()
    marks = ",".join("?" for _ in fact_ids)
    conn.execute(f"UPDATE facts SET superseded_by = ? WHERE id IN ({marks}) AND superseded_by IS NULL", [marker, *fact_ids])
    conn.execute(f"DELETE FROM facts_vec WHERE rowid IN ({marks})", fact_ids)
    db.commit(conn)


@_locked
def supersede_fact(old_id: int, new_id: int) -> None:
    _retire_facts([old_id], new_id)


@_locked
def forget_facts(fact_ids: list[int]) -> None:
    """The user asked for these to be forgotten. Rows stay (reversible with
    `restore_fact`); recall and search stop seeing them. No-op on an empty list."""
    if fact_ids:
        _retire_facts(list(fact_ids), FORGOTTEN)


@_locked
def restore_fact(fact_id: int) -> bool:
    """Undo a supersede or forget. The vector comes back via the writer's backfill."""
    conn = db.get_connection()
    cursor = conn.execute("UPDATE facts SET superseded_by = NULL WHERE id = ? AND superseded_by IS NOT NULL", (fact_id,))
    db.commit(conn)
    return cursor.rowcount > 0


@_locked
def set_fact_confidence(fact_id: int, confidence: float, decayed_at: float) -> None:
    conn = db.get_connection()
    conn.execute("UPDATE facts SET confidence = ?, decayed_at = ? WHERE id = ?", (confidence, decayed_at, fact_id))
    db.commit(conn)


# ---------------------------------------------------------------------------
# episodes
# ---------------------------------------------------------------------------

_EPISODE_COLUMNS = "id, summary, salience, created_ts, last_used_at, decayed_at, superseded_by"


def _episode_dict(row) -> dict[str, Any]:
    return {
        "id": row[0],
        "summary": row[1],
        "salience": row[2],
        "created_ts": row[3],
        "last_used_at": row[4],
        "decayed_at": row[5],
        "superseded_by": row[6],
    }


@_locked
def add_episode(
    summary: str,
    embedding: list[float] | None,
    *,
    salience: float = 0.5,
    session_id: str | None = None,
    turns: int | None = None,
    tokens: int | None = None,
) -> int:
    """Inserts an episode and (if there is one) its vector under the same rowid,
    in one transaction. `embedding=None` keeps the text but leaves it out of
    semantic search; only the Luna importer uses that."""
    summary = summary.strip()
    now = time.time()
    conn = db.get_connection()
    if embedding is not None:
        _check_dimension(conn, embedding)
    cursor = conn.execute(
        "INSERT INTO episodes (summary, salience, created_ts, decayed_at, session_id, turns, tokens) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (summary, salience, now, now, session_id, turns, tokens),
    )
    episode_id = cursor.lastrowid
    if embedding is not None:
        conn.execute(
            "INSERT INTO episode_vectors (rowid, embedding) VALUES (?, ?)", (episode_id, _prep(embedding))
        )
    db.commit(conn)
    return episode_id


@_locked
def search_episodes_scored(query_embedding: list[float], k: int) -> list[dict[str, Any]]:
    """Nearest live episodes with their similarity, salience and timestamps."""
    rows = db.get_connection().execute(
        f"""
        SELECT {", ".join("episodes." + c.strip() for c in _EPISODE_COLUMNS.split(","))}, episode_vectors.distance
        FROM episode_vectors
        JOIN episodes ON episodes.id = episode_vectors.rowid
        WHERE episode_vectors.embedding MATCH ? AND k = ? AND episodes.superseded_by IS NULL
        ORDER BY episode_vectors.distance
        """,
        (_prep(query_embedding), k),
    ).fetchall()
    return [{**_episode_dict(r), "similarity": similarity(r[7])} for r in rows]


@_locked
def search_episodes(query_embedding: list[float], top_k: int) -> list[str]:
    """Summaries only, nearest first. Empty on a fresh install, which is the
    normal starting state rather than an error. (sqlite-vec KNN needs the
    parameterised `k = ?` form; a bound `LIMIT ?` is rejected.)"""
    return [e["summary"] for e in search_episodes_scored(query_embedding, top_k)]


@_locked
def touch_episodes(episode_ids: list[int]) -> None:
    if not episode_ids:
        return
    conn = db.get_connection()
    marks = ",".join("?" for _ in episode_ids)
    conn.execute(f"UPDATE episodes SET last_used_at = ? WHERE id IN ({marks})", [time.time(), *episode_ids])
    db.commit(conn)


@_locked
def get_episode_rows(*, include_retired: bool = False) -> list[dict[str, Any]]:
    query = f"SELECT {_EPISODE_COLUMNS} FROM episodes"
    if not include_retired:
        query += " WHERE superseded_by IS NULL"
    query += " ORDER BY id"
    return [_episode_dict(r) for r in db.get_connection().execute(query).fetchall()]


@_locked
def set_episode_salience(episode_id: int, salience: float, decayed_at: float) -> None:
    conn = db.get_connection()
    conn.execute("UPDATE episodes SET salience = ?, decayed_at = ? WHERE id = ?", (salience, decayed_at, episode_id))
    db.commit(conn)


@_locked
def retire_episodes(episode_ids: list[int], marker: int = DECAYED) -> None:
    """Drops episodes from search (marker = DECAYED, or the id of the summary
    that replaced them). The rows stay."""
    if not episode_ids:
        return
    conn = db.get_connection()
    marks = ",".join("?" for _ in episode_ids)
    conn.execute(
        f"UPDATE episodes SET superseded_by = ? WHERE id IN ({marks}) AND superseded_by IS NULL",
        [marker, *episode_ids],
    )
    conn.execute(f"DELETE FROM episode_vectors WHERE rowid IN ({marks})", episode_ids)
    db.commit(conn)


# ---------------------------------------------------------------------------
# transcript buffer
# ---------------------------------------------------------------------------


@_locked
def add_transcript_turn(user_text: str, assistant_text: str, *, temp: bool = False) -> None:
    """One row per turn. It is both the user's reviewable log and the durable
    buffer the writer pass reads. `temp=True` (Temp Chat) still shows in the log
    but is never read by the writer, so a temporary chat leaves no memory. A turn
    with nothing on either side is not worth a row; a user line with no reply
    (cut off before a word was spoken) still is."""
    user_text = user_text.strip()
    assistant_text = assistant_text.strip()
    if not user_text and not assistant_text:
        return
    conn = db.get_connection()
    conn.execute(
        "INSERT INTO transcript_log (user_text, assistant_text, temp) VALUES (?, ?, ?)",
        (user_text, assistant_text, int(temp)),
    )
    db.commit(conn)


@_locked
def get_transcript_log() -> list[dict[str, str]]:
    """Oldest first, like a chat transcript. Includes temp turns: it is the
    user's own record of what happened on screen."""
    rows = db.get_connection().execute(
        "SELECT user_text, assistant_text, created_at FROM transcript_log ORDER BY created_at ASC, id ASC"
    ).fetchall()
    return [{"user": r[0], "assistant": r[1], "ts": r[2]} for r in rows]


@_locked
def get_transcript_after(after_id: int, limit: int | None = None) -> list[dict[str, Any]]:
    """Rows newer than `after_id`, oldest first, temp rows included and flagged
    so the writer can skip them while still moving its checkpoint past them."""
    query = (
        "SELECT id, user_text, assistant_text, created_at, temp FROM transcript_log WHERE id > ? ORDER BY id"
    )
    params: list[Any] = [after_id]
    if limit is not None:
        query += " LIMIT ?"
        params.append(limit)
    rows = db.get_connection().execute(query, params).fetchall()
    return [{"id": r[0], "user": r[1], "assistant": r[2], "ts": r[3], "temp": bool(r[4])} for r in rows]


@_locked
def clear_transcript_log() -> None:
    conn = db.get_connection()
    conn.execute("DELETE FROM transcript_log")
    db.commit(conn)
