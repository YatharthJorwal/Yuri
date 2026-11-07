"""Decay, compression and the derived profile card."""

from __future__ import annotations

import time

import pytest

from memory import decay, profile, store
from memory_helpers import fake_vector, stub_embed, stub_model, stub_model_down, system_prompt

DAY = 86400.0


def age_episode(conn, episode_id, days, salience=None):
    then = time.time() - days * DAY
    conn.execute(
        "UPDATE episodes SET created_ts = ?, decayed_at = ?, last_used_at = NULL"
        + (", salience = ?" if salience is not None else "")
        + " WHERE id = ?",
        (then, then, *([salience] if salience is not None else []), episode_id),
    )
    conn.commit()


def test_salience_halves_every_half_life():
    now = time.time()
    assert decay.project(0.8, now - 30 * DAY, now, 30) == pytest.approx(0.4)
    assert decay.project(0.8, now - 60 * DAY, now, 30) == pytest.approx(0.2)
    assert decay.project(0.8, now, now, 30) == 0.8


def test_sweep_writes_decay_back_and_keeps_young_episodes(memory_db):
    eid = store.add_episode("recent", fake_vector(0.1), salience=0.8)
    age_episode(memory_db, eid, 30)
    counts = decay.sweep()
    assert counts["episodes_decayed"] == 1
    assert store.get_episode_rows()[0]["salience"] == pytest.approx(0.4, abs=0.02)


def test_long_dead_episodes_leave_search_but_the_row_stays(memory_db):
    eid = store.add_episode("ancient chatter", fake_vector(0.1), salience=0.04)
    age_episode(memory_db, eid, 120, salience=0.04)
    counts = decay.sweep()
    assert counts["episodes_dropped"] == 1
    assert store.search_episodes(fake_vector(0.1), 5) == []  # gone from search
    assert store.get_episode_rows() == []
    retired = store.get_episode_rows(include_retired=True)
    assert retired[0]["summary"] == "ancient chatter" and retired[0]["superseded_by"] == -2
    assert store.get_events("decay_drop")


def test_low_salience_but_recently_used_episode_is_not_dropped(memory_db):
    eid = store.add_episode("still useful", fake_vector(0.1), salience=0.04)
    age_episode(memory_db, eid, 120, salience=0.04)
    memory_db.execute("UPDATE episodes SET last_used_at = ? WHERE id = ?", (time.time() - 5 * DAY, eid))
    memory_db.commit()
    assert decay.sweep()["episodes_dropped"] == 0


def test_facts_lose_confidence_to_a_floor_but_never_disappear_and_pins_never_decay(memory_db):
    a = store.add_fact("old unpinned", confidence=0.7)
    b = store.add_fact("old pinned", confidence=0.7, pinned=True)
    long_ago = time.time() - 3650 * DAY
    memory_db.execute("UPDATE facts SET last_seen_at = ?, created_ts = ?", (long_ago, long_ago))
    memory_db.commit()
    decay.sweep()
    assert store.get_fact(a)["confidence"] == pytest.approx(0.1)
    assert store.get_fact(a)["superseded_by"] is None
    assert store.get_fact(b)["confidence"] == 0.7


def test_restating_a_fact_resets_its_decay_clock(memory_db):
    fid = store.add_fact("likes tea", confidence=0.7)
    long_ago = time.time() - 365 * DAY
    memory_db.execute("UPDATE facts SET last_seen_at = ?, created_ts = ?", (long_ago, long_ago))
    memory_db.commit()
    store.touch_fact(fid)
    decay.sweep()
    assert store.get_fact(fid)["confidence"] >= 0.7


def test_compression_needs_a_full_batch_of_low_salience_episodes(memory_db):
    for i in range(9):
        age_episode(memory_db, store.add_episode(f"old {i}", fake_vector(0.1), salience=0.05), 10, 0.05)
    assert decay.compression_batch() == []
    age_episode(memory_db, store.add_episode("old 9", fake_vector(0.1), salience=0.05), 10, 0.05)
    assert len(decay.compression_batch()) == 10


@pytest.mark.asyncio
async def test_compression_merges_a_batch_into_one_and_keeps_the_originals(memory_db, monkeypatch):
    ids = [store.add_episode(f"old {i}", fake_vector(0.1), salience=0.05) for i in range(10)]
    for i in ids:
        age_episode(memory_db, i, 10, 0.05)
    calls = stub_model(monkeypatch, "Ten old chats about assorted things.")
    stub_embed(monkeypatch)

    new_id = await decay.compress_low_salience()

    assert new_id is not None and system_prompt(calls[0]).startswith("Merge these")
    live = store.get_episode_rows()
    assert [e["summary"] for e in live] == ["Ten old chats about assorted things."]
    assert all(e["superseded_by"] == new_id for e in store.get_episode_rows(include_retired=True) if e["id"] in ids)
    assert store.get_events("compress")


@pytest.mark.asyncio
async def test_compression_leaves_the_batch_alone_when_the_model_is_down(memory_db, monkeypatch):
    for i in range(10):
        age_episode(memory_db, store.add_episode(f"old {i}", fake_vector(0.1), salience=0.05), 10, 0.05)
    stub_model_down(monkeypatch)
    stub_embed(monkeypatch)
    assert await decay.compress_low_salience() is None
    assert len(store.get_episode_rows()) == 10


# ---- profile card ------------------------------------------------------------


@pytest.mark.asyncio
async def test_card_is_only_rebuilt_when_marked_dirty(memory_db, monkeypatch):
    calls = stub_model(monkeypatch, "Whatever.")
    store.add_fact("likes tea")
    assert await profile.rebuild_if_dirty() is False and calls == []


@pytest.mark.asyncio
async def test_card_is_built_from_facts_and_clears_the_dirty_flag(memory_db, monkeypatch):
    store.add_fact("likes tea")
    store.add_fact("builds desktop apps", pinned=True)
    calls = stub_model(monkeypatch, "Likes tea and builds desktop apps.")
    profile.mark_dirty()
    assert await profile.rebuild_if_dirty() is True
    assert store.get_profile_card() == "Likes tea and builds desktop apps."
    assert store.get_meta("profile_dirty") == "0"
    body = calls[0][1]["content"]
    assert body.index("builds desktop apps") < body.index("likes tea")  # pinned first


@pytest.mark.asyncio
async def test_card_falls_back_to_the_facts_themselves_and_stays_dirty_when_the_model_is_down(memory_db, monkeypatch):
    store.add_fact("likes tea")
    stub_model_down(monkeypatch)
    profile.mark_dirty()
    await profile.rebuild_if_dirty()
    assert "likes tea" in store.get_profile_card()
    assert store.get_meta("profile_dirty") == "1"  # a later pass will replace it


@pytest.mark.asyncio
async def test_card_is_capped_at_the_configured_length(memory_db, monkeypatch):
    store.add_fact("likes tea")
    stub_model(monkeypatch, "A sentence that repeats. " * 100)
    profile.mark_dirty()
    await profile.rebuild_if_dirty()
    assert len(store.get_profile_card()) <= 800


@pytest.mark.asyncio
async def test_card_empties_when_every_fact_is_gone(memory_db, monkeypatch):
    fid = store.add_fact("likes tea")
    stub_model(monkeypatch, "Likes tea.")
    profile.mark_dirty()
    await profile.rebuild_if_dirty()
    store.forget_facts([fid])
    profile.mark_dirty()
    await profile.rebuild_if_dirty()
    assert not store.get_profile_card()
