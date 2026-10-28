"""
Turns the current user message into a memory context block app.py can
inject into the LLM call for that turn -- "recall injected into the
system prompt each turn" per docs/ROADMAP.md's Phase 3 line.

Injected as a *separate*, ephemeral system message built fresh per call,
never written into the persisted session `history` list app.py keeps --
history stays exactly the real conversation (system persona + user/
assistant turns), so it doesn't grow a stale memory block on every single
turn or get fed back into consolidation.py's session-summarization pass
as if it were something the user or Luna actually said.

Degrades in two independent stages rather than all-or-nothing:
- Facts are plain SQLite reads with no embedding step, so a fact-only
  block still comes back even if the embedding server is down.
- Episode recall needs an embedding of the user's message; if that fails
  (Ollama down, model not pulled), it's skipped and logged, but doesn't
  block facts from still being included.
Returns None only if there is truly nothing to inject (fresh DB, or every
attempted lookup came up empty) -- app.py skips adding a memory message
at all in that case rather than sending an empty/pointless one.
"""

from __future__ import annotations

import asyncio
import sys

from . import store
from .db import MemoryUnavailableError
from .embeddings import EmbeddingUnreachableError, embed

# Bounds how many facts get read back into a prompt on any given turn --
# not a cap on how many facts exist (see store.py). Keeps the memory block
# from growing unbounded as facts accumulate over many sessions; most
# recent facts win on the (reasonable) assumption that a newer fact is
# more likely to reflect the user's current situation than an old one.
MAX_FACTS_IN_RECALL = 30

# How many past-episode summaries get pulled in per turn, nearest-first by
# embedding distance. Small on purpose -- these are meant to be a few
# genuinely relevant callbacks ("that bug we fixed last week"), not a full
# session-history dump; config.yaml's memory.recall_top_k is the same
# value, kept here as the hardcoded fallback if that key is ever missing.
DEFAULT_TOP_K = 5


async def build_recall_context(user_text: str, top_k: int = DEFAULT_TOP_K) -> str | None:
    facts = await _get_facts_safely()
    episodes = await _get_relevant_episodes_safely(user_text, top_k)

    if not facts and not episodes:
        return None

    parts: list[str] = []
    if facts:
        parts.append("Known facts about the user: " + "; ".join(facts) + ".")
    if episodes:
        parts.append("Relevant past context: " + "; ".join(episodes) + ".")
    return (
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
        "the way someone would who simply already knows it. " + " ".join(parts)
    )


async def _get_facts_safely() -> list[str]:
    try:
        return await asyncio.to_thread(store.get_all_facts, MAX_FACTS_IN_RECALL)
    except MemoryUnavailableError as exc:
        print(f"[luna] memory recall: facts unavailable, skipping: {exc}", file=sys.stderr)
        return []


async def _get_relevant_episodes_safely(user_text: str, top_k: int) -> list[str]:
    try:
        query_embedding = await embed(user_text)
    except EmbeddingUnreachableError as exc:
        print(
            f"[luna] memory recall: embedding unavailable, skipping episode "
            f"recall this turn (facts still included if any): {exc}",
            file=sys.stderr,
        )
        return []

    try:
        return await asyncio.to_thread(store.search_episodes, query_embedding, top_k)
    except MemoryUnavailableError as exc:
        print(f"[luna] memory recall: episode search unavailable, skipping: {exc}", file=sys.stderr)
        return []
