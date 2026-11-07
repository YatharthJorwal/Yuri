"""Recall v2: budgeted, relevance-floored, audited."""

from __future__ import annotations

import json

import pytest

import context_budget
from memory import recall, store
from memory_helpers import UNRELATED, fake_vector, stub_embed, stub_embed_down

QUERY = fake_vector(0.0)  # what the stubbed embedder returns for the user's message


def fact(text, vector=None, **kw):
    fid = store.add_fact(text, **kw)
    store.set_fact_vector(fid, vector if vector is not None else fake_vector(0.0))
    return fid


@pytest.fixture
def embedder(monkeypatch):
    stub_embed(monkeypatch, lambda text: QUERY)


@pytest.mark.asyncio
async def test_reflex_recall_never_exceeds_its_fact_and_episode_counts(memory_db, embedder):
    for i in range(20):
        fact(f"relevant fact number {i}")
        store.add_episode(f"relevant episode number {i}", fake_vector(0.0))
    chosen = await recall.select("anything", 5, budget="reflex")
    assert len(chosen.facts) == 3 and len(chosen.episodes) == 2


@pytest.mark.asyncio
async def test_mind_recall_gets_a_bigger_budget(memory_db, embedder):
    for i in range(20):
        fact(f"relevant fact number {i}")
        store.add_episode(f"relevant episode number {i}", fake_vector(0.0))
    chosen = await recall.select("anything", 10, budget="mind")
    assert len(chosen.facts) == 8 and len(chosen.episodes) == 5


@pytest.mark.asyncio
async def test_contents_are_hard_capped_at_the_token_budget(memory_db, embedder):
    store.set_profile_card("A long profile card sentence. " * 80)
    for i in range(3):
        fact("a quite long fact about the user, " * 12 + str(i))
    for i in range(3):
        store.add_episode("a long episode summary that goes on, " * 12 + str(i), fake_vector(0.0))
    fitted = recall.fit(await recall.select("anything", 5), 300)
    assert fitted.tokens <= 300
    contents = (fitted.card or "") + "".join(f["content"] + "; " for f in fitted.facts) + "".join(e["summary"] + "; " for e in fitted.episodes)
    assert context_budget.estimate_tokens(contents) <= 300 + 4  # estimator rounds per piece


@pytest.mark.asyncio
async def test_profile_card_is_always_there_but_leaves_room_for_facts(memory_db, embedder):
    store.set_profile_card("Sentence one is here. " * 100)
    fact("relevant short fact")
    block = await recall.build_recall_context("hello", 5)
    assert "About the user: Sentence one" in block and "relevant short fact" in block
    fitted = recall.fit(await recall.select("hello", 5), 300)
    assert context_budget.estimate_tokens(fitted.card) <= 150


@pytest.mark.asyncio
async def test_facts_below_the_similarity_floor_are_not_injected(memory_db, embedder):
    fact("on topic", fake_vector(0.0))
    fact("totally unrelated", UNRELATED)
    chosen = await recall.select("anything", 5)
    assert [f["content"] for f in chosen.facts] == ["on topic"]


@pytest.mark.asyncio
async def test_nothing_relevant_means_no_block_at_all(memory_db, embedder):
    fact("totally unrelated", UNRELATED)
    assert await recall.build_recall_context("anything", 5) is None


@pytest.mark.asyncio
async def test_episodes_are_ranked_by_similarity_times_salience(memory_db, embedder):
    store.add_episode("important but slightly off-topic", fake_vector(0.1), salience=1.0)
    store.add_episode("exact match but forgettable", fake_vector(0.0), salience=0.05)
    chosen = await recall.select("anything", 5)
    assert chosen.episodes[0]["summary"] == "important but slightly off-topic"


@pytest.mark.asyncio
async def test_superseded_and_forgotten_facts_never_come_back(memory_db, embedder):
    keep = fact("keep me")
    gone = fact("forgotten fact")
    store.forget_facts([gone])
    chosen = await recall.select("anything", 5)
    assert [f["id"] for f in chosen.facts] == [keep]


@pytest.mark.asyncio
async def test_facts_about_other_subjects_need_the_subject_mentioned(memory_db, embedder):
    fact("is allergic to nuts", subject="Priya")
    fact("likes tea", subject="user")
    assert [f["content"] for f in (await recall.select("what should I cook", 5)).facts] == ["likes tea"]
    assert len((await recall.select("what should I cook for Priya", 5)).facts) == 2


@pytest.mark.asyncio
async def test_without_an_embedder_facts_match_by_words_and_episodes_are_skipped(memory_db, monkeypatch):
    stub_embed_down(monkeypatch)
    store.add_fact("prefers concise answers")
    store.add_fact("collects vintage keyboards")
    store.add_episode("an episode", fake_vector(0.0))
    chosen = await recall.select("please keep answers concise", 5)
    assert chosen.path == "keyword"
    assert [f["content"] for f in chosen.facts] == ["prefers concise answers"]
    assert chosen.episodes == []


@pytest.mark.asyncio
async def test_facts_the_embedder_never_reached_are_still_found_by_words(memory_db, embedder):
    store.add_fact("uses an RTX 3060 GPU")  # no vector yet
    chosen = await recall.select("what GPU do I have", 5)
    assert [f["content"] for f in chosen.facts] == ["uses an RTX 3060 GPU"]


@pytest.mark.asyncio
async def test_every_recall_is_audited_with_the_ids_it_used(memory_db, embedder):
    fid = fact("on topic")
    eid = store.add_episode("on topic episode", fake_vector(0.0))
    await recall.build_recall_context("anything", 5)
    detail = json.loads(store.get_events("recall")[0]["detail"])
    assert detail["fact_ids"] == [fid] and detail["episode_ids"] == [eid] and detail["budget"] == "reflex"
    assert detail["tokens"] > 0


@pytest.mark.asyncio
async def test_recalling_an_episode_keeps_it_alive(memory_db, embedder):
    eid = store.add_episode("on topic episode", fake_vector(0.0), salience=0.3)
    await recall.build_recall_context("anything", 5)
    row = store.get_episode_rows()[0]
    assert row["salience"] > 0.3 and row["last_used_at"] is not None and row["id"] == eid
