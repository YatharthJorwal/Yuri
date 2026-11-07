"""
The writer pass: the one place conversation becomes durable memory.

Every finished turn is already in `transcript_log` (a durable buffer; a crash
loses nothing). A pass reads the rows after `writer_checkpoint_id`, skips Temp
Chat rows (they never become memory, but the checkpoint still moves past them),
asks the model for facts + an episode summary, decides what each fact does to
what is already stored (contradiction.py), and applies the lot, together with
the new checkpoint, in ONE transaction. A crash before commit reprocesses the
same turns and writes nothing twice; a crash after commit has nothing to redo.

When it runs: on a timer (`memory.writer.interval_s`), at session end, and on
shutdown. The timer pass never starts while Yuri is listening, thinking or
speaking and is cancelled if a turn begins (`preempt`), so memory work never
competes with a reply for the model. Session-end and shutdown passes always run.

Failure behaviour, by design:
- model unreachable: nothing is written, the checkpoint stays, the next pass retries;
- embedder unreachable: facts are still written (without vectors; a later pass
  backfills them) and this chunk's episode is skipped;
- anything raising inside the transaction: rolled back whole.
"""

from __future__ import annotations

import asyncio
import re
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import context_budget

from . import consolidation, contradiction, decay, profile, settings, store
from . import db
from .db import MemoryUnavailableError
from .embeddings import EmbeddingUnreachableError, embed

MAX_CHUNKS_PER_RUN = 6
MAX_TURN_CHARS = 1500
BACKFILL_BATCH = 50
SPEAKER = "Yuri"

BASE_SALIENCE = 0.5
CORRECTION_SALIENCE = 0.8
REMEMBER_SALIENCE = 1.0

# "remember that ..." is the user marking something important. "don't remember" is
# the opposite (forget.py handles it), so a negated form does not count.
_REMEMBER = re.compile(r"(?<!don't )(?<!don\u2019t )(?<!do not )\b(remember (that|this)|keep in mind|note that)\b", re.IGNORECASE)


@dataclass
class PassResult:
    reason: str = ""
    turns: int = 0
    facts_added: int = 0
    duplicates: int = 0
    superseded: int = 0
    blocked: int = 0
    episodes: int = 0
    skipped: str | None = None  # busy | model_unreachable | memory_unavailable | nothing_new
    notes: list[str] = field(default_factory=list)


@dataclass
class _PlannedFact:
    text: str
    subject: str | None
    source: str
    confidence: float
    decision: contradiction.Decision
    embedding: list[float] | None


@dataclass
class _Plan:
    facts: list[_PlannedFact] = field(default_factory=list)
    episode_summary: str = ""
    episode_embedding: list[float] | None = None
    salience: float = BASE_SALIENCE
    turns: int = 0
    tokens: int = 0


def _format_turn(row: dict) -> str:
    lines = []
    if row["user"]:
        lines.append(f"User: {row['user'][:MAX_TURN_CHARS]}")
    if row["assistant"]:
        lines.append(f"{SPEAKER}: {row['assistant'][:MAX_TURN_CHARS]}")
    return "\n".join(lines)


def _take_window(rows: list[dict], window_tokens: int) -> list[dict]:
    """Oldest rows that fit in the window; always at least one, so one oversized
    turn can never wedge the checkpoint."""
    taken: list[dict] = []
    used = 0
    for row in rows:
        cost = context_budget.estimate_tokens(_format_turn(row))
        if taken and used + cost > window_tokens:
            break
        taken.append(row)
        used += cost
    return taken


class MemoryWriter:
    def __init__(self, is_busy: Callable[[], bool] | None = None) -> None:
        self._is_busy = is_busy or (lambda: False)
        self._lock = asyncio.Lock()
        self._loop_task: asyncio.Task | None = None
        self._timer_pass: asyncio.Future | None = None
        self._stopping = False

    def configure(self, *, is_busy: Callable[[], bool]) -> None:
        self._is_busy = is_busy

    # ---- lifecycle -------------------------------------------------------

    def start(self) -> None:
        interval = float(settings.section("writer")["interval_s"])
        if interval <= 0 or self._loop_task is not None:
            return
        self._stopping = False
        self._loop_task = asyncio.create_task(self._loop(interval), name="memory-writer")

    async def stop(self, *, final_pass: bool = True) -> None:
        task, self._loop_task = self._loop_task, None
        self._stopping = True
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        if final_pass:
            try:
                await self.run_pass("shutdown")
            except Exception as exc:  # noqa: BLE001 -- shutdown must not hang on memory
                print(f"[yuri] writer: shutdown pass failed: {exc}", file=sys.stderr)

    def preempt(self) -> None:
        """A turn is starting: abandon an in-flight *timer* pass. Safe at any
        point: the pass commits atomically or not at all."""
        pending = self._timer_pass
        if pending is not None and not pending.done():
            pending.cancel()

    async def _loop(self, interval: float) -> None:
        retry = float(settings.section("writer")["busy_retry_s"])
        delay = interval
        while True:
            await asyncio.sleep(delay)
            delay = interval
            self._timer_pass = asyncio.ensure_future(self.run_pass("timer"))
            try:
                result = await self._timer_pass
                if result.skipped == "busy":
                    delay = retry
            except asyncio.CancelledError:
                if self._stopping:  # the loop itself was stopped
                    raise
                delay = retry  # only the pass was preempted; try again shortly
            except Exception as exc:  # noqa: BLE001 -- the loop must survive anything
                print(f"[yuri] writer: pass failed: {exc}", file=sys.stderr)

    # ---- the pass --------------------------------------------------------

    async def run_pass(self, reason: str = "manual") -> PassResult:
        result = PassResult(reason=reason)
        if not self._is_enabled_for(reason):
            result.skipped = "disabled"
            return result
        async with self._lock:
            if reason == "timer" and self._is_busy():
                result.skipped = "busy"
                return result
            try:
                await self._backfill_vectors()
                for _ in range(MAX_CHUNKS_PER_RUN):
                    if not await self._one_chunk(result):
                        break
                if result.turns:
                    await asyncio.to_thread(
                        store.log_event,
                        "writer_pass",
                        "writer",
                        {
                            "reason": reason,
                            "turns": result.turns,
                            "added": result.facts_added,
                            "duplicates": result.duplicates,
                            "superseded": result.superseded,
                            "blocked": result.blocked,
                            "episodes": result.episodes,
                        },
                    )
                await self._maintenance(result)
            except MemoryUnavailableError as exc:
                print(f"[yuri] writer: memory unavailable, will retry: {exc}", file=sys.stderr)
                result.skipped = "memory_unavailable"
        return result

    @staticmethod
    def _is_enabled_for(reason: str) -> bool:
        if reason in ("session_end", "shutdown"):
            return bool(settings.section("writer")["on_session_end"])
        return True

    async def _one_chunk(self, result: PassResult) -> bool:
        """Processes one window of transcript. Returns True if there may be more."""
        cfg = settings.section("writer")
        checkpoint = int(await asyncio.to_thread(store.get_meta, "writer_checkpoint_id", "0") or 0)
        rows = await asyncio.to_thread(store.get_transcript_after, checkpoint, 400)
        if not rows:
            result.skipped = result.skipped or ("nothing_new" if not result.turns else None)
            return False

        window = _take_window(rows, int(cfg["window_tokens"]))
        last_id = window[-1]["id"]
        live = [r for r in window if not r["temp"]]
        more = len(window) < len(rows)

        if not live:  # all Temp Chat: skip them, but never look at them again
            await asyncio.to_thread(self._advance, last_id)
            return more

        transcript = "\n".join(_format_turn(r) for r in live)
        distilled = await consolidation.distill(transcript)
        if distilled is None:
            result.skipped = "model_unreachable"
            return False

        plan = await self._plan(distilled, live, transcript)
        counts = await asyncio.to_thread(self._apply, plan, last_id)
        result.turns += len(live)
        result.facts_added += counts["added"]
        result.duplicates += counts["duplicates"]
        result.superseded += counts["superseded"]
        result.blocked += counts["blocked"]
        result.episodes += counts["episodes"]
        return more

    async def _plan(self, distilled: consolidation.Distilled, live: list[dict], transcript: str) -> _Plan:
        cfg = settings.section("contradiction")
        plan = _Plan(episode_summary=distilled.episode_summary, turns=len(live))
        plan.tokens = context_budget.estimate_tokens(transcript)
        embedder_down = False
        seen: set[str] = set()

        for cand in distilled.facts:
            key = contradiction.normalize(cand.text)
            if not key or key in seen:
                continue
            seen.add(key)
            embedding = None
            if not embedder_down:
                try:
                    embedding = await embed(cand.text)
                except EmbeddingUnreachableError as exc:
                    embedder_down = True
                    print(f"[yuri] writer: embedding unavailable, facts saved without vectors: {exc}", file=sys.stderr)
            candidates = await asyncio.to_thread(self._candidates, cand.text, embedding, int(cfg["candidates"]))
            source, confidence = ("correction", 1.0) if cand.correction else ("writer_pass", 0.7)
            decision = await contradiction.decide(
                cand.text,
                candidates,
                new_confidence=confidence,
                explicit_correction=cand.correction,
                model_similarity_floor=float(cfg["model_similarity_floor"]),
            )
            subject = cand.subject if cand.subject not in (None, "") else None
            plan.facts.append(_PlannedFact(cand.text, subject, source, confidence, decision, embedding))

        if any(f.source == "correction" for f in plan.facts):
            plan.salience = CORRECTION_SALIENCE
        if any(_REMEMBER.search(r["user"]) for r in live):
            plan.salience = REMEMBER_SALIENCE

        if plan.episode_summary:
            if embedder_down:
                plan.episode_summary = ""
            else:
                try:
                    plan.episode_embedding = await embed(plan.episode_summary)
                except EmbeddingUnreachableError as exc:
                    print(f"[yuri] writer: embedding unavailable, episode not stored (facts still are): {exc}", file=sys.stderr)
                    plan.episode_summary = ""
        return plan

    @staticmethod
    def _candidates(text: str, embedding: list[float] | None, k: int) -> list[dict]:
        """Live facts that might clash with `text`: nearest by vector when we
        have an embedding, plus word-overlap matches (the only way to find a fact
        that has no vector yet)."""
        found: dict[int, dict] = {}
        if embedding is not None:
            for fact in store.search_facts(embedding, k):
                found[fact["id"]] = fact
            pending = {fid for fid, _ in store.facts_missing_vectors(500)}
            pool = [f for f in store.get_fact_rows(limit=500) if f["id"] in pending]
        else:
            pool = store.get_fact_rows(limit=500)
        for fact in pool:
            overlap = contradiction.keyword_overlap(text, fact["content"])
            if overlap > 0:
                found.setdefault(fact["id"], {**fact, "similarity": overlap})
        ranked = sorted(found.values(), key=lambda f: -f["similarity"])
        return ranked[:k]

    @staticmethod
    def _advance(last_id: int) -> None:
        with db.transaction():
            store.set_meta("writer_checkpoint_id", str(last_id))
            store.set_meta("last_writer_at", str(time.time()))

    @staticmethod
    def _apply(plan: _Plan, last_id: int) -> dict[str, int]:
        """Everything this chunk decided, plus the checkpoint, in one transaction."""
        counts = {"added": 0, "duplicates": 0, "superseded": 0, "blocked": 0, "episodes": 0}
        with db.transaction():
            for pf in plan.facts:
                d = pf.decision
                if d.action == contradiction.DUPLICATE and d.target_id is not None:
                    store.touch_fact(d.target_id)
                    store.log_event("fact_duplicate", "writer", {"fact_id": d.target_id, "via": d.via})
                    counts["duplicates"] += 1
                    continue
                new_id = store.add_fact(pf.text, source=pf.source, confidence=pf.confidence, subject=pf.subject)
                if new_id is None:
                    continue
                if pf.embedding is not None:
                    store.set_fact_vector(new_id, pf.embedding)
                counts["added"] += 1
                if d.action == contradiction.SUPERSEDE and d.target_id is not None:
                    store.supersede_fact(d.target_id, new_id)
                    store.log_event("supersede", "writer", {"old": d.target_id, "new": new_id, "via": d.via})
                    counts["superseded"] += 1
                elif d.blocked:
                    store.log_event(
                        "supersede_blocked",
                        "writer",
                        {"proposed_old": d.proposed_target_id, "new": new_id, "reason": d.blocked},
                    )
                    counts["blocked"] += 1
                else:
                    store.log_event("fact_add", "writer", {"fact_id": new_id, "source": pf.source})
            if plan.episode_summary and plan.episode_embedding is not None:
                store.add_episode(
                    plan.episode_summary,
                    plan.episode_embedding,
                    salience=plan.salience,
                    turns=plan.turns,
                    tokens=plan.tokens,
                )
                counts["episodes"] = 1
            if counts["added"] or counts["superseded"]:
                profile.mark_dirty()
            store.set_meta("writer_checkpoint_id", str(last_id))
            store.set_meta("last_writer_at", str(time.time()))
        return counts

    async def _backfill_vectors(self) -> None:
        """Gives vectors to live facts that lack one (written while the embedder
        was down, imported from Luna, or restored). Stops at the first failure."""
        pending = await asyncio.to_thread(store.facts_missing_vectors, BACKFILL_BATCH)
        for fact_id, content in pending:
            try:
                vector = await embed(content)
            except EmbeddingUnreachableError:
                return
            await asyncio.to_thread(store.set_fact_vector, fact_id, vector)

    async def _maintenance(self, result: PassResult) -> None:
        """Decay, compression and the profile card. Each is independent and
        failure of one never blocks the others."""
        try:
            await asyncio.to_thread(decay.sweep)
            await decay.compress_low_salience()
            await profile.rebuild_if_dirty()
        except MemoryUnavailableError:
            raise
        except Exception as exc:  # noqa: BLE001
            print(f"[yuri] writer: maintenance step failed: {exc}", file=sys.stderr)


WRITER = MemoryWriter()
