"""
The profile card: a short, always-on summary of who the user is, derived from the
live facts. It is a cache, not a source of truth: it can be rebuilt from `facts`
at any time, and when it is wrong the fix is to correct the fact it came from.

It is rebuilt (not patched) after a writer pass changed facts. If the model is
unavailable a deterministic card (the strongest facts, joined) is written
instead, and the card stays marked stale so a later pass replaces it.
"""

from __future__ import annotations

import asyncio
import sys

import router
from config import CONFIG

from . import store

_MAX_FACTS = 60

_PROMPT = (
    "Write a compact third-person profile of the user from the facts below, as plain sentences in one "
    "paragraph, at most {limit} characters. Use ONLY what the facts say: no guesses, no advice, no "
    "bullets, no preamble."
)


def mark_dirty() -> None:
    store.set_meta("profile_dirty", "1")


def _limit() -> int:
    return int(CONFIG.memory.profile_card.get("max_chars", 800))


def _cut(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    cut = text[:limit]
    boundary = max(cut.rfind(". "), cut.rfind("; "))
    return (cut[: boundary + 1] if boundary > limit // 2 else cut.rsplit(" ", 1)[0]).strip()


def _fallback(facts: list[dict]) -> str:
    return _cut("; ".join(f["content"] for f in facts) + ".", _limit())


def _ranked_facts() -> list[dict]:
    rows = store.get_fact_rows(limit=_MAX_FACTS * 2)
    rows.sort(key=lambda f: (not f["pinned"], -f["confidence"], -f["id"]))
    return rows[:_MAX_FACTS]


async def rebuild_if_dirty() -> bool:
    """Returns True if a card was written this call."""
    if await asyncio.to_thread(store.get_meta, "profile_dirty", "0") != "1":
        return False
    facts = await asyncio.to_thread(_ranked_facts)
    if not facts:
        await asyncio.to_thread(store.set_profile_card, "")
        await asyncio.to_thread(store.set_meta, "profile_dirty", "0")
        return True

    limit = _limit()
    messages = [
        {"role": "system", "content": _PROMPT.format(limit=limit)},
        {"role": "user", "content": "\n".join(f"- {f['content']}" for f in facts)},
    ]
    card = ""
    try:
        card = "".join([d async for d in router.stream_text("memory_writer", messages)]).strip()
    except Exception as exc:  # noqa: BLE001 -- the deterministic card below covers it
        print(f"[yuri] profile card: model unavailable, using a plain card for now: {exc}", file=sys.stderr)

    used_model = bool(card) and len(card) <= limit * 2
    final = _cut(card, limit) if used_model else _fallback(facts)

    def _save() -> None:
        store.set_profile_card(final)
        store.log_event("profile_card", "writer", {"model": used_model, "chars": len(final)})
        if used_model:
            store.set_meta("profile_dirty", "0")

    await asyncio.to_thread(_save)
    return True
