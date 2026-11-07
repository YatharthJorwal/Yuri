"""
Turns the current user message into a memory block app.py injects into the LLM
call for that turn, as a separate ephemeral system message that is never written
into the session `history` (so it cannot be fed back into the writer as if
someone had said it).

Recall is budgeted, never wholesale. Each consumer gets a token budget and a
count of facts/episodes (config `memory.recall`): the cheap reflex path gets a
small one, the Mind a larger one. What goes in, in priority order:

1. the profile card (always, trimmed to half the budget so facts can still fit);
2. facts nearest to the message by embedding, above a similarity floor;
3. episodes ranked by similarity * decayed salience * recency, above a floor.

When the embedder is down it degrades instead of failing: facts are matched by
shared words and episodes are skipped. The injected contents (not the fixed
guidance paragraph) are hard-capped at the budget, and every recall writes an
audit event listing exactly which ids were used.
"""

from __future__ import annotations

import asyncio
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Any

import context_budget

from . import decay, settings, store
from .contradiction import _stem
from .db import MemoryUnavailableError
from .embeddings import EmbeddingUnreachableError, embed

DEFAULT_TOP_K = 5
RECENCY_HALF_LIFE_DAYS = 180.0
OVERFETCH = 3  # candidates fetched per slot before filtering and re-ranking
KEYWORD_SCAN_LIMIT = 500

_STOPWORDS = frozenset(
    "a about again all also am an and any are as at be been but by can could did do does for from had has have "
    "he her him his how i if in into is it its just me my no not of on or our she so some than that the their "
    "them then there these they this to too was we were what when where which who why will with would you your".split()
)

PREAMBLE = (
    "Background memory from earlier sessions -- some of it may be "
    "relevant right now, most of it probably is not. Only bring "
    "something up if it is genuinely relevant to what the user just "
    "said; do not force a reference in just because it is listed "
    "below, and do not bring up more than one thing per reply. A "
    "fact being available here does not mean this turn is about it "
    "-- if the user is talking about something else entirely, the "
    "right move is to just not mention it, the same way a person "
    "who happened to remember something wouldn't force it into an "
    "unrelated conversation. Reaching for a listed fact as a jab or "
    "example when the user brings up something unrelated (a "
    "different app, a different topic) is exactly the wrong move, "
    "even if it is a real fact -- it will read as fixating on "
    "something the user mentioned once, not as actually knowing "
    "them. Never "
    "invent a specific incident, date, or detail that is not written "
    "here -- a fact like \"likes pizza\" means exactly that and "
    "nothing more; do not turn it into a story, an event, or a "
    "callback to something that supposedly happened (no made-up "
    "\"last Tuesday\"s). Never read this list aloud, and never call "
    "it \"notes\" or \"memory\" out loud -- if it comes up, say it "
    "the way someone would who simply already knows it. "
)


@dataclass
class Selection:
    card: str | None = None
    facts: list[dict[str, Any]] = field(default_factory=list)
    episodes: list[dict[str, Any]] = field(default_factory=list)
    path: str = "vector"  # vector | keyword
    tokens: int = 0

    @property
    def empty(self) -> bool:
        return not (self.card or self.facts or self.episodes)


def _words(text: str) -> set[str]:
    return {_stem(w) for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) >= 3 and w not in _STOPWORDS}


def _subject_allows(fact: dict[str, Any], query_words: set[str]) -> bool:
    """Facts tagged with a free-form subject (a person, a project) only apply
    when the message mentions it; user/machine/untagged facts always can."""
    subject = fact.get("subject")
    if subject in (None, "", "user", "machine"):
        return True
    return bool(_words(subject) & query_words)


def _keyword_ranked(facts: list[dict[str, Any]], query_words: set[str]) -> list[dict[str, Any]]:
    scored = []
    for fact in facts:
        shared = len(_words(fact["content"]) & query_words)
        if shared and _subject_allows(fact, query_words):
            scored.append(({**fact, "similarity": min(1.0, 0.4 + 0.1 * shared)}, shared))
    scored.sort(key=lambda pair: (-pair[1], -pair[0]["id"]))
    return [fact for fact, _ in scored]


def _recency(episode: dict[str, Any], now: float) -> float:
    age_days = max(0.0, (now - (episode["created_ts"] or now)) / decay.DAY)
    return 0.5 ** (age_days / RECENCY_HALF_LIFE_DAYS)


def _trim(text: str, max_tokens: int) -> str:
    """Cuts at a line or sentence boundary so the card never ends mid-word."""
    if context_budget.estimate_tokens(text) <= max_tokens:
        return text
    kept = ""
    for piece in re.split(r"(?<=[.!?\n])\s+", text):
        candidate = (kept + " " + piece).strip()
        if context_budget.estimate_tokens(candidate) > max_tokens:
            break
        kept = candidate
    return kept


def fit(selection: Selection, budget_tokens: int) -> Selection:
    """Greedy fit in priority order. Guarantees the contents stay within budget."""
    fitted = Selection(path=selection.path)
    used = 0
    if selection.card:
        card = _trim(selection.card, budget_tokens // 2)
        if card:
            fitted.card = card
            used += context_budget.estimate_tokens(card)
    for key in ("facts", "episodes"):
        text_key = "content" if key == "facts" else "summary"
        for item in getattr(selection, key):
            cost = context_budget.estimate_tokens(item[text_key] + "; ")
            if used + cost <= budget_tokens:
                getattr(fitted, key).append(item)
                used += cost
    fitted.tokens = used
    return fitted


async def select(user_text: str, top_k: int = DEFAULT_TOP_K, *, budget: str = "reflex") -> Selection:
    cfg = settings.section("recall")
    limits = cfg[budget]
    max_facts, max_eps = int(limits["facts"]), min(int(limits["episodes"]), top_k)
    now = time.time()
    query_words = _words(user_text)
    selection = Selection()

    try:
        selection.card = await asyncio.to_thread(store.get_profile_card)
    except MemoryUnavailableError as exc:
        print(f"[yuri] memory recall: unavailable, skipping: {exc}", file=sys.stderr)
        return selection

    try:
        query_embedding: list[float] | None = await embed(user_text)
    except EmbeddingUnreachableError as exc:
        print(f"[yuri] memory recall: embedding unavailable, using keyword facts only: {exc}", file=sys.stderr)
        query_embedding = None

    try:
        if query_embedding is None:
            selection.path = "keyword"
            live = await asyncio.to_thread(store.get_fact_rows, limit=KEYWORD_SCAN_LIMIT)
            selection.facts = _keyword_ranked(live, query_words)[:max_facts]
            return selection

        floor = float(cfg["fact_similarity_floor"])
        hits = await asyncio.to_thread(store.search_facts, query_embedding, max_facts * OVERFETCH)
        facts = [f for f in hits if f["similarity"] >= floor and _subject_allows(f, query_words)]
        # Facts the embedder never reached (written while it was down, or imported)
        # still deserve a chance, matched by words.
        pending_ids = await asyncio.to_thread(store.facts_missing_vectors, KEYWORD_SCAN_LIMIT)
        if pending_ids:
            ids = {fid for fid, _ in pending_ids}
            rows = await asyncio.to_thread(store.get_fact_rows, limit=KEYWORD_SCAN_LIMIT)
            facts += _keyword_ranked([r for r in rows if r["id"] in ids], query_words)
        facts.sort(key=lambda f: -f["similarity"])
        selection.facts = facts[:max_facts]

        ep_floor = float(cfg["episode_similarity_floor"])
        candidates = await asyncio.to_thread(store.search_episodes_scored, query_embedding, max_eps * OVERFETCH)
        ranked = [
            (e["similarity"] * decay.effective_salience(e, now) * _recency(e, now), e)
            for e in candidates
            if e["similarity"] >= ep_floor
        ]
        ranked.sort(key=lambda pair: -pair[0])
        selection.episodes = [e for _, e in ranked[:max_eps]]
    except MemoryUnavailableError as exc:
        print(f"[yuri] memory recall: search unavailable, skipping: {exc}", file=sys.stderr)
    return selection


async def build_recall_context(user_text: str, top_k: int = DEFAULT_TOP_K, *, budget: str = "reflex") -> str | None:
    cfg = settings.section("recall")
    chosen = await select(user_text, top_k, budget=budget)
    fitted = fit(chosen, int(cfg[budget]["tokens"]))
    if fitted.empty:
        return None

    parts: list[str] = []
    if fitted.card:
        parts.append("About the user: " + fitted.card)
    if fitted.facts:
        parts.append("Relevant facts: " + "; ".join(f["content"] for f in fitted.facts) + ".")
    if fitted.episodes:
        parts.append("Relevant past context: " + "; ".join(e["summary"] for e in fitted.episodes) + ".")

    await _record(fitted, budget)
    return PREAMBLE + " ".join(parts)


async def _record(fitted: Selection, budget: str) -> None:
    """Audit what was injected and keep used episodes alive. Best effort: a
    failure here must never cost the turn its recall."""

    def _write() -> None:
        store.log_event(
            "recall",
            "recall",
            {
                "budget": budget,
                "path": fitted.path,
                "tokens": fitted.tokens,
                "fact_ids": [f["id"] for f in fitted.facts],
                "episode_ids": [e["id"] for e in fitted.episodes],
                "card": bool(fitted.card),
            },
        )
        decay.reinforce(fitted.episodes)

    try:
        await asyncio.to_thread(_write)
    except MemoryUnavailableError as exc:
        print(f"[yuri] memory recall: couldn't record usage: {exc}", file=sys.stderr)
