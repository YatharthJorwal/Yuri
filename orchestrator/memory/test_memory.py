"""
Sandbox verification for Phase 3 memory. Unlike the LLM/TTS/STT backends
(which need a real GPU/Windows/model server to fully verify -- see every
"not verified in this sandbox" note in docs/DECISIONS.md), sqlite-vec is a
pure local library with no GPU/network/OS dependency, so the DB layer
(db.py/store.py) is fully, honestly testable here, not just stub-shaped.
This is the first committed test file in the repo -- worth noting in
docs/DECISIONS.md since every prior phase's "verified in sandbox" work was
done ad hoc and not committed; this one's different because it doesn't
carry the same "can't actually verify this without real hardware" caveat,
so it's worth keeping around to catch a real regression later.

What IS stubbed, and why: the embedding HTTP call (no Ollama in this
sandbox) and the LLM streaming call consolidation.py makes (same reason).
Both are monkeypatched at the function-call boundary, same pattern
previous phases used for llm.py/tts.py stub servers.

Run with: python -m pytest memory/test_memory.py -v
(from orchestrator/, with requirements.txt installed)
"""

from __future__ import annotations

import struct

import pytest

from . import consolidation, db, embeddings, recall, store, forget


def _fake_vector(seed: float, dim: int = 4) -> list[float]:
    """Small, deterministic vectors for test purposes -- dim=4 throughout
    (not the real 768 nomic-embed-text uses) so tests run instantly and
    the nearest-neighbor math is easy to reason about by hand."""
    return [seed, 1.0 - seed, 0.0, 0.0]


@pytest.fixture
def memory_db(tmp_path, monkeypatch):
    """Real sqlite-vec connection against a throwaway temp file, dim=4.
    Monkeypatches db.get_connection so store.py (which only ever calls
    that, never _open_connection directly) transparently uses it --
    every other memory module goes through store.py, so this one fixture
    covers all of them."""
    conn = db._open_connection(path=tmp_path / "test_memory.db", dimension=4)
    monkeypatch.setattr(db, "_conn", conn)
    monkeypatch.setattr(db, "get_connection", lambda: conn)
    yield conn
    conn.close()


# ---------------------------------------------------------------------------
# db.py / store.py -- real sqlite-vec, no stubs
# ---------------------------------------------------------------------------


def test_facts_roundtrip(memory_db):
    store.add_fact("likes tabs over spaces")
    store.add_fact("working on Luna, a desktop companion app")
    facts = store.get_all_facts()
    assert facts == [
        "working on Luna, a desktop companion app",
        "likes tabs over spaces",
    ]  # most recent first


def test_add_fact_skips_blank(memory_db):
    store.add_fact("   ")
    assert store.get_all_facts() == []


def test_facts_respects_limit(memory_db):
    for i in range(5):
        store.add_fact(f"fact {i}")
    assert len(store.get_all_facts(limit=2)) == 2


def test_episode_search_returns_nearest_first(memory_db):
    store.add_episode("fixed the lipsync bug", _fake_vector(1.0))  # far from query
    store.add_episode("talked about the Flappy Bird project", _fake_vector(0.05))  # near
    store.add_episode("discussed GPU VRAM budgeting", _fake_vector(0.5))  # mid

    results = store.search_episodes(_fake_vector(0.0), top_k=3)
    assert results == [
        "talked about the Flappy Bird project",
        "discussed GPU VRAM budgeting",
        "fixed the lipsync bug",
    ]


def test_episode_search_respects_top_k(memory_db):
    for i in range(10):
        store.add_episode(f"episode {i}", _fake_vector(i / 10))
    assert len(store.search_episodes(_fake_vector(0.0), top_k=3)) == 3


def test_episode_search_empty_db_returns_empty_list(memory_db):
    assert store.search_episodes(_fake_vector(0.0), top_k=5) == []


def test_pack_matches_struct_format():
    packed = store._pack([1.0, 2.0, 3.0])
    assert packed == struct.pack("3f", 1.0, 2.0, 3.0)


# ---------------------------------------------------------------------------
# db.py -- dimension mismatch check
# ---------------------------------------------------------------------------


class _FakeEmbeddingCfg:
    def __init__(self, dimension: int):
        self.dimension = dimension


class _FakeMemoryCfg:
    def __init__(self, dimension: int):
        self.embedding = _FakeEmbeddingCfg(dimension)


class _FakeConfig:
    def __init__(self, dimension: int):
        self.memory = _FakeMemoryCfg(dimension)


def test_dimension_mismatch_warns(tmp_path, monkeypatch, capsys):
    path = tmp_path / "mismatch.db"
    conn = db._open_connection(path=path, dimension=4)
    monkeypatch.setattr(db, "_conn", conn)
    monkeypatch.setattr(db, "get_connection", lambda: conn)
    # CONFIG is a frozen dataclass -- can't mutate an attribute on the real
    # instance, so replace db.py's whole module-level `CONFIG` name
    # instead (that's the only name check_embedding_dimension_matches()
    # actually reads).
    monkeypatch.setattr(db, "CONFIG", _FakeConfig(dimension=999))  # deliberately doesn't match dim=4

    db.check_embedding_dimension_matches()
    captured = capsys.readouterr()
    assert "WARNING" in captured.err
    assert "999" in captured.err
    conn.close()


def test_dimension_match_is_silent(tmp_path, monkeypatch, capsys):
    path = tmp_path / "match.db"
    conn = db._open_connection(path=path, dimension=4)
    monkeypatch.setattr(db, "_conn", conn)
    monkeypatch.setattr(db, "get_connection", lambda: conn)
    monkeypatch.setattr(db, "CONFIG", _FakeConfig(dimension=4))  # matches

    db.check_embedding_dimension_matches()
    captured = capsys.readouterr()
    assert captured.err == ""
    conn.close()


# ---------------------------------------------------------------------------
# recall.py -- store is real (via memory_db fixture), embed() is stubbed
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_recall_combines_facts_and_episodes(memory_db, monkeypatch):
    store.add_fact("uses an RTX 3060")
    store.add_episode("debugged the STT CUDA DLL issue together", _fake_vector(0.0))

    async def fake_embed(text: str) -> list[float]:
        return _fake_vector(0.0)

    monkeypatch.setattr(recall, "embed", fake_embed)
    block = await recall.build_recall_context("what GPU do I have again?", top_k=3)
    assert block is not None
    assert "RTX 3060" in block
    assert "STT CUDA DLL" in block


@pytest.mark.asyncio
async def test_recall_returns_none_when_empty(memory_db, monkeypatch):
    async def fake_embed(text: str) -> list[float]:
        return _fake_vector(0.0)

    monkeypatch.setattr(recall, "embed", fake_embed)
    block = await recall.build_recall_context("anything", top_k=3)
    assert block is None


@pytest.mark.asyncio
async def test_recall_degrades_when_embedding_unreachable(memory_db, monkeypatch):
    """Facts alone should still come back even if the embed() call fails --
    this is the actual graceful-degradation behavior this module exists
    for, not just a nice-to-have edge case."""
    store.add_fact("prefers concise answers")

    async def failing_embed(text: str) -> list[float]:
        raise embeddings.EmbeddingUnreachableError("Ollama not running")

    monkeypatch.setattr(recall, "embed", failing_embed)
    block = await recall.build_recall_context("hello", top_k=3)
    assert block is not None
    assert "concise answers" in block


@pytest.mark.asyncio
async def test_recall_degrades_when_db_unavailable(monkeypatch):
    """Simulates the DB itself being unavailable (not just embeddings) --
    e.g. a locked file. Both facts and episode lookups should fail closed
    (return None), not raise out of build_recall_context()."""

    def raise_unavailable(*_a, **_kw):
        raise db.MemoryUnavailableError("simulated: disk full")

    monkeypatch.setattr(store, "get_all_facts", raise_unavailable)
    monkeypatch.setattr(store, "search_episodes", raise_unavailable)

    async def fake_embed(text: str) -> list[float]:
        return _fake_vector(0.0)

    monkeypatch.setattr(recall, "embed", fake_embed)
    block = await recall.build_recall_context("hello", top_k=3)
    assert block is None


# ---------------------------------------------------------------------------
# consolidation.py -- llm.stream_reply and embed() are stubbed
# ---------------------------------------------------------------------------


def _stub_llm_module(monkeypatch, output_text: str):
    async def fake_stream_reply(messages):
        for ch in output_text:
            yield ch

    monkeypatch.setattr(consolidation.llm, "stream_reply", fake_stream_reply)


@pytest.mark.asyncio
async def test_consolidation_parses_clean_json(memory_db, monkeypatch):
    _stub_llm_module(
        monkeypatch,
        '{"facts": ["likes dark mode", "building a Tauri app"], '
        '"episode_summary": "Helped debug a lipsync issue."}',
    )

    async def fake_embed(text: str) -> list[float]:
        return _fake_vector(0.2)

    monkeypatch.setattr(consolidation, "embed", fake_embed)

    history = [
        {"role": "system", "content": "persona"},
        {"role": "user", "content": "why is the mouth not moving"},
        {"role": "assistant", "content": "found it, fixed"},
    ]
    await consolidation.consolidate_session(history)

    facts = store.get_all_facts()
    assert "likes dark mode" in facts
    assert "building a Tauri app" in facts
    episodes = store.search_episodes(_fake_vector(0.2), top_k=1)
    assert episodes == ["Helped debug a lipsync issue."]


@pytest.mark.asyncio
async def test_consolidation_parses_json_wrapped_in_prose(memory_db, monkeypatch):
    """A small model padding its output with commentary/markdown fences
    around the JSON -- the realistic failure mode this parser exists for,
    per this module's own docstring."""
    _stub_llm_module(
        monkeypatch,
        "Sure, here's the summary:\n```json\n"
        '{"facts": ["mentioned a cat named Miso"], "episode_summary": "Chatted about pets."}'
        "\n```\nHope that helps!",
    )

    async def fake_embed(text: str) -> list[float]:
        return _fake_vector(0.3)

    monkeypatch.setattr(consolidation, "embed", fake_embed)

    history = [{"role": "system", "content": "persona"}, {"role": "user", "content": "I have a cat, Miso"}]
    await consolidation.consolidate_session(history)

    assert "mentioned a cat named Miso" in store.get_all_facts()


@pytest.mark.asyncio
async def test_consolidation_falls_back_on_unparseable_output(memory_db, monkeypatch, capsys):
    _stub_llm_module(monkeypatch, "I don't think I can do that in JSON, sorry!")

    async def fake_embed(text: str) -> list[float]:
        return _fake_vector(0.4)

    monkeypatch.setattr(consolidation, "embed", fake_embed)

    history = [{"role": "system", "content": "persona"}, {"role": "user", "content": "hi"}]
    await consolidation.consolidate_session(history)

    # No facts extracted, but the raw text still got stored as a fallback
    # episode summary rather than the whole session being silently lost.
    assert store.get_all_facts() == []
    episodes = store.search_episodes(_fake_vector(0.4), top_k=1)
    assert episodes and "I don't think I can do that" in episodes[0]
    assert "didn't parse as the expected" in capsys.readouterr().err


@pytest.mark.asyncio
async def test_consolidation_skips_empty_session(memory_db, monkeypatch):
    """system-prompt-only history (nothing actually happened) shouldn't
    even make the LLM call."""
    called = False

    async def fake_stream_reply(messages):
        nonlocal called
        called = True
        return
        yield  # pragma: no cover -- unreachable, keeps this an async generator

    monkeypatch.setattr(consolidation.llm, "stream_reply", fake_stream_reply)
    await consolidation.consolidate_session([{"role": "system", "content": "persona"}])
    assert called is False
    assert store.get_all_facts() == []


@pytest.mark.asyncio
async def test_consolidation_survives_llm_unreachable(memory_db, monkeypatch):
    async def failing_stream_reply(messages):
        raise consolidation.llm.LLMUnreachableError("no server")
        yield  # pragma: no cover -- unreachable, keeps this an async generator

    monkeypatch.setattr(consolidation.llm, "stream_reply", failing_stream_reply)
    history = [{"role": "system", "content": "persona"}, {"role": "user", "content": "hi"}]
    # Should not raise.
    await consolidation.consolidate_session(history)
    assert store.get_all_facts() == []


@pytest.mark.asyncio
async def test_consolidation_keeps_facts_when_embedding_fails(memory_db, monkeypatch):
    """Facts should still get written even if the episode's embedding call
    fails -- only the episode summary is lost, per this module's docstring."""
    _stub_llm_module(
        monkeypatch,
        '{"facts": ["prefers terse commit messages"], "episode_summary": "Talked about git."}',
    )

    async def failing_embed(text: str) -> list[float]:
        raise embeddings.EmbeddingUnreachableError("Ollama not running")

    monkeypatch.setattr(consolidation, "embed", failing_embed)

    history = [{"role": "system", "content": "persona"}, {"role": "user", "content": "hi"}]
    await consolidation.consolidate_session(history)

    assert "prefers terse commit messages" in store.get_all_facts()


# ---------------------------------------------------------------------------
# store.py -- get_all_facts_with_ids / delete_facts (forget.py's building blocks)
# ---------------------------------------------------------------------------


def test_get_all_facts_with_ids_and_delete(memory_db):
    store.add_fact("likes pizza")
    store.add_fact("likes ice cream")
    with_ids = store.get_all_facts_with_ids()
    assert [content for _id, content in with_ids] == ["likes ice cream", "likes pizza"]

    pizza_id = next(fact_id for fact_id, content in with_ids if content == "likes pizza")
    store.delete_facts([pizza_id])
    assert store.get_all_facts() == ["likes ice cream"]


def test_delete_facts_empty_list_is_noop(memory_db):
    store.add_fact("likes pizza")
    store.delete_facts([])
    assert store.get_all_facts() == ["likes pizza"]


# ---------------------------------------------------------------------------
# Phase 9 -- transcript_log (the persistent conversation-log panel). Fully
# real against the same memory_db fixture, no stubbing needed -- this is a
# plain SQLite table with no embedding/LLM dependency at all.
# ---------------------------------------------------------------------------


def test_transcript_log_roundtrip_oldest_first(memory_db):
    store.add_transcript_turn("hello", "hi here to help...")
    store.add_transcript_turn("how are you", "tch, fine I guess")
    turns = store.get_transcript_log()
    assert [t["user"] for t in turns] == ["hello", "how are you"]  # oldest first,
    # the opposite ordering from get_all_facts() -- see store.py's docstring
    assert turns[0]["assistant"] == "hi here to help..."
    assert turns[1]["assistant"] == "tch, fine I guess"
    assert all("ts" in t for t in turns)


def test_add_transcript_turn_skips_when_both_blank(memory_db):
    store.add_transcript_turn("   ", "")
    assert store.get_transcript_log() == []


def test_add_transcript_turn_keeps_real_user_text_with_blank_reply(memory_db):
    # A turn stopped before she said anything -- still an honest record of
    # what actually happened, not something to silently drop (see
    # add_transcript_turn's own docstring).
    store.add_transcript_turn("wait stop", "")
    turns = store.get_transcript_log()
    assert len(turns) == 1
    assert turns[0]["user"] == "wait stop"
    assert turns[0]["assistant"] == ""


def test_clear_transcript_log(memory_db):
    store.add_transcript_turn("hello", "hi")
    store.add_transcript_turn("bye", "bye bye")
    store.clear_transcript_log()
    assert store.get_transcript_log() == []


def test_clear_transcript_log_empty_db_is_noop(memory_db):
    store.clear_transcript_log()
    assert store.get_transcript_log() == []


# ---------------------------------------------------------------------------
# forget.py -- store is real (via memory_db fixture), llm.stream_reply stubbed
# ---------------------------------------------------------------------------


def _stub_forget_llm(monkeypatch, output_text: str):
    async def fake_stream_reply(messages):
        for ch in output_text:
            yield ch

    monkeypatch.setattr(forget.llm, "stream_reply", fake_stream_reply)


@pytest.mark.asyncio
async def test_forget_skips_llm_call_with_no_trigger_word(memory_db, monkeypatch):
    """The cheap regex gate should mean a completely unrelated message
    never even reaches the LLM -- if it did, this stub would raise and
    fail the test, since it's never supposed to be called."""
    store.add_fact("likes pizza")

    async def should_not_be_called(messages):
        raise AssertionError("LLM should never be called without a forget-intent trigger word")
        yield  # pragma: no cover -- unreachable, keeps this an async generator

    monkeypatch.setattr(forget.llm, "stream_reply", should_not_be_called)
    result = await forget.maybe_forget("what's the weather like today")
    assert result is None
    assert store.get_all_facts() == ["likes pizza"]


@pytest.mark.asyncio
async def test_forget_removes_matched_fact(memory_db, monkeypatch):
    store.add_fact("likes pizza")
    store.add_fact("uses an RTX 3060")
    _stub_forget_llm(monkeypatch, '{"remove_indices": [2]}')  # facts listed newest-first: [1]=RTX 3060, [2]=pizza

    result = await forget.maybe_forget("forget that I like pizza")
    assert result is not None
    assert "likes pizza" in result
    assert store.get_all_facts() == ["uses an RTX 3060"]


@pytest.mark.asyncio
async def test_forget_no_facts_referenced_removes_nothing(memory_db, monkeypatch):
    store.add_fact("likes pizza")
    _stub_forget_llm(monkeypatch, '{"remove_indices": []}')

    result = await forget.maybe_forget("forget what I said about my ex")
    assert result is None
    assert store.get_all_facts() == ["likes pizza"]


@pytest.mark.asyncio
async def test_forget_ignores_out_of_range_indices(memory_db, monkeypatch):
    """A hallucinated index the small model made up (e.g. it saw 1 fact
    but said to remove index 5) should be dropped, not crash or delete the
    wrong row."""
    store.add_fact("likes pizza")
    _stub_forget_llm(monkeypatch, '{"remove_indices": [5]}')

    result = await forget.maybe_forget("forget that")
    assert result is None
    assert store.get_all_facts() == ["likes pizza"]


@pytest.mark.asyncio
async def test_forget_parses_json_wrapped_in_prose(memory_db, monkeypatch):
    store.add_fact("likes pizza")
    _stub_forget_llm(monkeypatch, 'Sure!\n```json\n{"remove_indices": [1]}\n```\nDone.')

    result = await forget.maybe_forget("forget that I like pizza")
    assert result is not None
    assert store.get_all_facts() == []


@pytest.mark.asyncio
async def test_forget_falls_back_to_nothing_on_unparseable_output(memory_db, monkeypatch):
    store.add_fact("likes pizza")
    _stub_forget_llm(monkeypatch, "sorry, I can't do that in JSON")

    result = await forget.maybe_forget("forget that I like pizza")
    assert result is None
    assert store.get_all_facts() == ["likes pizza"]


@pytest.mark.asyncio
async def test_forget_survives_llm_unreachable(memory_db, monkeypatch):
    store.add_fact("likes pizza")

    async def failing_stream_reply(messages):
        raise forget.llm.LLMUnreachableError("no server")
        yield  # pragma: no cover -- unreachable, keeps this an async generator

    monkeypatch.setattr(forget.llm, "stream_reply", failing_stream_reply)
    result = await forget.maybe_forget("forget that I like pizza")
    assert result is None
    assert store.get_all_facts() == ["likes pizza"]


@pytest.mark.asyncio
async def test_forget_with_no_facts_at_all_skips_llm_call(memory_db):
    """Trigger word present but the DB has zero facts -- nothing to
    possibly remove, so this should short-circuit before ever calling the
    LLM (no stub installed here at all; a real call would raise since
    there's no Ollama in this sandbox)."""
    result = await forget.maybe_forget("forget everything about me")
    assert result is None
