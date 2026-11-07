"""
Time-based fading, so memory stays small and recent without ever silently
deleting anything.

- Episode salience halves every `half_life_days` (default 30). Recall multiplies
  by it, so old chatter stops surfacing before it stops existing.
- An episode below `drop_below` (0.05) and unused for `drop_after_days` (90) is
  dropped from search. The row is kept and the drop is logged.
- A batch of low-salience episodes (below `compress_below`, `compress_batch` at a
  time) is merged into one short summary episode by the writer role.
- Fact confidence decays on a slower clock (`fact_half_life_days`, default 180) to
  a floor of 0.1. Facts never disappear and pinned facts do not decay at all.

Decay is applied lazily: the stored value is "as of `decayed_at`", and
`effective_*` projects it to now. `sweep()` writes the projection back.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import router
from config import CONFIG

from . import db, store
from .embeddings import embed

DAY = 86400.0
CONFIDENCE_FLOOR = 0.1
RECALL_REINFORCE = 0.1


def _setting(key: str, default: Any) -> Any:
    return CONFIG.memory.decay.get(key, default)


def enabled() -> bool:
    return bool(_setting("enabled", True))


def project(value: float, since: float | None, now: float, half_life_days: float) -> float:
    """`value` after (now - since) days of exponential decay."""
    if since is None or half_life_days <= 0 or now <= since:
        return value
    return value * 0.5 ** ((now - since) / DAY / half_life_days)


def effective_salience(row: dict[str, Any], now: float | None = None) -> float:
    return project(row["salience"], row.get("decayed_at"), now or time.time(), float(_setting("half_life_days", 30)))


def reinforce(rows: list[dict[str, Any]], now: float | None = None) -> None:
    """Recalling an episode keeps it alive: bump its salience and last-used time."""
    now = now or time.time()
    for row in rows:
        bumped = min(1.0, effective_salience(row, now) + RECALL_REINFORCE)
        store.set_episode_salience(row["id"], bumped, now)
    store.touch_episodes([r["id"] for r in rows])


def sweep(now: float | None = None) -> dict[str, int]:
    """Writes decay back to the rows and drops long-dead episodes from search.
    One transaction, so it either all applies or not at all."""
    counts = {"episodes_decayed": 0, "episodes_dropped": 0, "facts_decayed": 0}
    if not enabled():
        return counts
    now = now or time.time()
    drop_below = float(_setting("drop_below", 0.05))
    drop_after = float(_setting("drop_after_days", 90)) * DAY
    fact_half = float(_setting("fact_half_life_days", 180))
    with db.transaction():
        dropped: list[int] = []
        for ep in store.get_episode_rows():
            sal = effective_salience(ep, now)
            idle_since = max(ep["last_used_at"] or 0.0, ep["created_ts"] or 0.0)
            if sal <= drop_below and now - idle_since >= drop_after:
                dropped.append(ep["id"])
                continue
            if abs(sal - ep["salience"]) > 1e-6:
                store.set_episode_salience(ep["id"], sal, now)
                counts["episodes_decayed"] += 1
        if dropped:
            store.retire_episodes(dropped)
            store.log_event("decay_drop", "decay", {"episode_ids": dropped})
            counts["episodes_dropped"] = len(dropped)
        for fact in store.get_fact_rows():
            if fact["pinned"]:
                continue
            conf = max(CONFIDENCE_FLOOR, project(fact["confidence"], _fact_decayed_at(fact), now, fact_half))
            if abs(conf - fact["confidence"]) > 1e-6:
                store.set_fact_confidence(fact["id"], conf, now)
                counts["facts_decayed"] += 1
    return counts


def _fact_decayed_at(fact: dict[str, Any]) -> float | None:
    # last_seen_at is refreshed whenever the fact is restated, so decay runs from
    # the last time the user actually told us this (or from creation).
    return fact.get("last_seen_at") or fact.get("created_ts")


def compression_batch(now: float | None = None) -> list[dict[str, Any]]:
    """The oldest `compress_batch` low-salience episodes, or [] if there are not
    enough yet to be worth a model call."""
    now = now or time.time()
    below = float(_setting("compress_below", 0.15))
    size = int(_setting("compress_batch", 10))
    low = [e for e in store.get_episode_rows() if effective_salience(e, now) < below]
    low.sort(key=lambda e: e["created_ts"] or 0.0)
    return low[:size] if len(low) >= size else []


_COMPRESS_PROMPT = (
    "Merge these old, low-importance memories into ONE short paragraph (at most 4 sentences). Keep names, "
    "dates, decisions and anything the person said they wanted; drop small talk. Plain text, no preamble."
)


async def compress_low_salience(now: float | None = None) -> int | None:
    """Merges one batch into a single new episode and retires the originals.
    Returns the new episode id, or None if there was nothing to do or the model
    or embedder was unavailable (the batch simply waits for the next pass)."""
    batch = await asyncio.to_thread(compression_batch, now)
    if not batch:
        return None
    body = "\n".join(f"- {e['summary']}" for e in batch)
    messages = [
        {"role": "system", "content": _COMPRESS_PROMPT},
        {"role": "user", "content": body},
    ]
    try:
        summary = "".join([d async for d in router.stream_text("memory_writer", messages)]).strip()
        vector = await embed(summary) if summary else None
    except Exception:  # noqa: BLE001 -- model or embedder down; the batch waits for the next pass
        return None
    if not summary or vector is None:
        return None
    ids = [e["id"] for e in batch]

    def _apply() -> int:
        with db.transaction():
            new_id = store.add_episode(summary, vector, salience=0.2)
            store.retire_episodes(ids, new_id)
            store.log_event("compress", "decay", {"from": ids, "to": new_id})
            return new_id

    return await asyncio.to_thread(_apply)
