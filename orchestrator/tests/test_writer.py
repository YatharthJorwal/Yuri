"""The writer pass end to end: real SQLite, stubbed chat model and embedder.
Includes the cases that used to cover consolidate_session (clean JSON, JSON in
prose, unparseable output, model down, embedder down)."""

from __future__ import annotations

import asyncio
import json

import pytest

from memory import settings, store, writer
from memory_helpers import fake_vector, stub_embed, stub_embed_down, stub_model, stub_model_down, system_prompt


def distilled(facts=(), summary="", correction=()):
    items = [{"text": f, "subject": "user", "correction": f in correction} for f in facts]
    return json.dumps({"facts": items, "episode_summary": summary})


def log(user, assistant="ok", temp=False):
    store.add_transcript_turn(user, assistant, temp=temp)


def checkpoint():
    return int(store.get_meta("writer_checkpoint_id", "0"))


def distiller_calls(calls):
    return [m for m in calls if system_prompt(m).startswith("You distill")]


@pytest.fixture
def w():
    return writer.MemoryWriter()


# ---- what a pass writes ------------------------------------------------------


@pytest.mark.asyncio
async def test_pass_writes_facts_and_episode_and_moves_the_checkpoint(memory_db, monkeypatch, w):
    stub_model(monkeypatch, distilled(["likes dark mode", "building a Tauri app"], "Helped debug a lipsync issue."))
    stub_embed(monkeypatch)
    log("why is the mouth not moving", "found it, fixed")

    result = await w.run_pass("session_end")

    assert (result.turns, result.facts_added, result.episodes) == (1, 2, 1)
    assert sorted(store.get_all_facts()) == ["building a Tauri app", "likes dark mode"]
    assert store.search_episodes(fake_vector(0.2), 1) == ["Helped debug a lipsync issue."]
    assert checkpoint() == 1
    assert store.get_meta("last_writer_at")
    assert {"fact_add", "writer_pass"} <= {e["kind"] for e in store.get_events()}


@pytest.mark.asyncio
async def test_json_wrapped_in_prose_and_fences_still_parses(memory_db, monkeypatch, w):
    stub_model(
        monkeypatch,
        "Sure, here's the summary:\n```json\n"
        + distilled(["has a cat named Miso"], "Chatted about pets.")
        + "\n```\nHope that helps!",
    )
    stub_embed(monkeypatch)
    log("I have a cat, Miso")
    await w.run_pass("session_end")
    assert "has a cat named Miso" in store.get_all_facts()


@pytest.mark.asyncio
async def test_plain_string_facts_are_still_accepted(memory_db, monkeypatch, w):
    stub_model(monkeypatch, '{"facts": ["likes tea"], "episode_summary": "Tea talk."}')
    stub_embed(monkeypatch)
    log("I like tea")
    await w.run_pass("session_end")
    assert store.get_all_facts() == ["likes tea"]


@pytest.mark.asyncio
async def test_unparseable_output_is_kept_as_a_raw_episode(memory_db, monkeypatch, capsys, w):
    stub_model(monkeypatch, "I don't think I can do that in JSON, sorry!")
    stub_embed(monkeypatch)
    log("hi")
    await w.run_pass("session_end")
    assert store.get_all_facts() == []
    assert "I don't think I can do that" in store.search_episodes(fake_vector(0.2), 1)[0]
    assert "didn't parse as the expected" in capsys.readouterr().err


@pytest.mark.asyncio
async def test_nothing_new_means_no_model_call(memory_db, monkeypatch, w):
    calls = stub_model(monkeypatch, distilled())
    result = await w.run_pass("session_end")
    assert calls == [] and result.skipped == "nothing_new" and result.turns == 0


@pytest.mark.asyncio
async def test_only_turns_after_the_checkpoint_are_read(memory_db, monkeypatch, w):
    calls = stub_model(monkeypatch, distilled(["a fact"], "Summary."))
    stub_embed(monkeypatch)
    log("first thing said")
    await w.run_pass("session_end")
    log("second thing said")
    await w.run_pass("session_end")
    sent = [m[1]["content"] for m in distiller_calls(calls)]
    assert "first thing said" in sent[0] and "second thing said" not in sent[0]
    assert "second thing said" in sent[1] and "first thing said" not in sent[1]
    assert checkpoint() == 2


# ---- Temp Chat ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_temp_turns_never_reach_the_model_but_the_checkpoint_moves_past_them(memory_db, monkeypatch, w):
    calls = stub_model(monkeypatch, distilled(["ordinary fact"], "Ordinary chat."))
    stub_embed(monkeypatch)
    log("my secret temp chat", temp=True)
    log("an ordinary message")
    log("another temp thing", temp=True)

    await w.run_pass("session_end")

    prompt = distiller_calls(calls)[0][1]["content"]
    assert "an ordinary message" in prompt
    assert "secret temp" not in prompt and "another temp" not in prompt
    assert checkpoint() == 3


@pytest.mark.asyncio
async def test_a_session_of_only_temp_turns_makes_no_model_call(memory_db, monkeypatch, w):
    calls = stub_model(monkeypatch, distilled(["should never exist"], "nope"))
    log("temp only", temp=True)
    await w.run_pass("session_end")
    assert calls == [] and store.get_all_facts() == [] and checkpoint() == 1


def test_temp_flag_is_stored_and_the_user_log_still_shows_everything(memory_db):
    log("shown but temporary", temp=True)
    assert store.get_transcript_log()[0]["user"] == "shown but temporary"
    assert store.get_transcript_after(0)[0]["temp"] is True


# ---- failure behaviour -------------------------------------------------------


@pytest.mark.asyncio
async def test_model_down_writes_nothing_and_retries_later(memory_db, monkeypatch, w):
    stub_embed(monkeypatch)
    stub_model_down(monkeypatch)
    log("remember this session")
    result = await w.run_pass("session_end")
    assert result.skipped == "model_unreachable" and checkpoint() == 0 and store.get_all_facts() == []

    stub_model(monkeypatch, distilled(["a late fact"], "Late summary."))
    await w.run_pass("session_end")
    assert store.get_all_facts() == ["a late fact"] and checkpoint() == 1


@pytest.mark.asyncio
async def test_embedder_down_keeps_facts_skips_episode_and_backfills_later(memory_db, monkeypatch, w):
    stub_model(monkeypatch, distilled(["prefers terse commit messages"], "Talked about git."))
    stub_embed_down(monkeypatch)
    log("hi")
    await w.run_pass("session_end")
    assert store.get_all_facts() == ["prefers terse commit messages"]
    assert store.get_episode_rows() == []
    assert len(store.facts_missing_vectors()) == 1

    stub_embed(monkeypatch)
    log("nothing durable")  # a later pass also backfills the vector
    stub_model(monkeypatch, distilled())
    await w.run_pass("timer")
    assert store.facts_missing_vectors() == []


@pytest.mark.asyncio
async def test_a_failure_while_applying_rolls_back_the_whole_pass(memory_db, monkeypatch, w):
    stub_model(monkeypatch, distilled(["fact one", "fact two"], "Summary."))
    stub_embed(monkeypatch)
    log("something")

    def boom(*a, **k):
        raise RuntimeError("disk exploded")

    real_add_episode = store.add_episode
    monkeypatch.setattr(store, "add_episode", boom)
    with pytest.raises(RuntimeError):
        await w.run_pass("session_end")

    assert store.get_all_facts() == []  # facts written before the failure were rolled back
    assert checkpoint() == 0  # so the same turns are retried

    monkeypatch.setattr(store, "add_episode", real_add_episode)
    await w.run_pass("session_end")
    assert sorted(store.get_all_facts()) == ["fact one", "fact two"]  # once, not twice


# ---- contradictions through a real pass ---------------------------------------


@pytest.mark.asyncio
async def test_a_contradicting_fact_supersedes_instead_of_piling_up(memory_db, monkeypatch, w):
    old_id = store.add_fact("likes pizza")
    store.set_fact_vector(old_id, fake_vector(0.2))
    stub_model(monkeypatch, distilled(["doesn't like pizza"], "Talked food."))
    stub_embed(monkeypatch)
    log("actually I don't like pizza anymore")

    result = await w.run_pass("session_end")

    assert result.superseded == 1
    assert store.get_all_facts() == ["doesn't like pizza"]
    old = store.get_fact(old_id)
    assert old["superseded_by"] not in (None, -1)  # points at the replacement, row kept
    assert store.get_events("supersede")


@pytest.mark.asyncio
async def test_restating_a_fact_refreshes_it_rather_than_duplicating(memory_db, monkeypatch, w):
    fid = store.add_fact("likes pizza", confidence=0.7)
    store.set_fact_vector(fid, fake_vector(0.2))
    stub_model(monkeypatch, distilled(["Likes pizza."], "Food again."))
    stub_embed(monkeypatch)
    log("I really do like pizza")
    result = await w.run_pass("session_end")
    assert result.duplicates == 1 and len(store.get_all_facts()) == 1
    assert store.get_fact(fid)["confidence"] > 0.7


@pytest.mark.asyncio
async def test_a_pinned_fact_is_not_replaced_without_an_explicit_correction(memory_db, monkeypatch, w):
    fid = store.add_fact("likes pizza", pinned=True)
    store.set_fact_vector(fid, fake_vector(0.2))
    stub_model(monkeypatch, distilled(["doesn't like pizza"], "Food."))
    stub_embed(monkeypatch)
    log("hmm pizza is meh")
    result = await w.run_pass("session_end")
    assert result.blocked == 1 and store.get_fact(fid)["superseded_by"] is None
    assert store.get_events("supersede_blocked")


@pytest.mark.asyncio
async def test_an_explicit_correction_replaces_a_pinned_fact(memory_db, monkeypatch, w):
    fid = store.add_fact("likes pizza", pinned=True)
    store.set_fact_vector(fid, fake_vector(0.2))
    stub_model(monkeypatch, distilled(["doesn't like pizza"], "Food.", correction=["doesn't like pizza"]))
    stub_embed(monkeypatch)
    log("no, I stopped liking pizza")
    await w.run_pass("session_end")
    assert store.get_all_facts() == ["doesn't like pizza"]
    new = store.get_fact_rows()[0]
    assert new["source"] == "correction" and new["confidence"] == 1.0


@pytest.mark.asyncio
async def test_duplicates_inside_one_pass_are_written_once(memory_db, monkeypatch, w):
    stub_model(monkeypatch, distilled(["likes tea", "Likes tea."], "Tea."))
    stub_embed(monkeypatch)
    log("tea tea tea")
    await w.run_pass("session_end")
    assert store.get_all_facts() == ["likes tea"]


# ---- salience ----------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "user,correction,expected",
    [
        ("we talked about weather", (), 0.5),
        ("actually it's Linux now", ("uses Linux",), 0.8),
        ("remember that my sister's birthday is in May", (), 1.0),
        ("don't remember that", (), 0.5),
    ],
)
async def test_episode_salience_reflects_how_much_it_mattered(memory_db, monkeypatch, w, user, correction, expected):
    stub_model(monkeypatch, distilled(["uses Linux"] if correction else [], "Something.", correction=correction))
    stub_embed(monkeypatch)
    log(user)
    await w.run_pass("session_end")
    assert store.get_episode_rows()[0]["salience"] == expected


# ---- windows, scheduling -------------------------------------------------------


def test_window_always_takes_at_least_one_row():
    rows = [{"user": "x" * 5000, "assistant": ""}, {"user": "y", "assistant": ""}]
    assert len(writer._take_window(rows, 10)) == 1


@pytest.mark.asyncio
async def test_a_backlog_larger_than_the_window_is_worked_through_in_chunks(memory_db, monkeypatch, w):
    real = settings.section
    monkeypatch.setattr(settings, "section", lambda n: {**real(n), "window_tokens": 30} if n == "writer" else real(n))
    calls = stub_model(monkeypatch, distilled([], "Chunk."))
    stub_embed(monkeypatch)
    for i in range(6):
        log(f"message number {i} " + "word " * 20)
    result = await w.run_pass("session_end")
    assert len(distiller_calls(calls)) > 1 and result.turns == 6 and checkpoint() == 6


@pytest.mark.asyncio
async def test_timer_pass_waits_while_busy_but_session_end_does_not(memory_db, monkeypatch):
    busy = writer.MemoryWriter(is_busy=lambda: True)
    calls = stub_model(monkeypatch, distilled(["a fact"], "S."))
    stub_embed(monkeypatch)
    log("hello")
    assert (await busy.run_pass("timer")).skipped == "busy" and calls == []
    assert (await busy.run_pass("session_end")).facts_added == 1


@pytest.mark.asyncio
async def test_session_end_pass_respects_the_config_switch(memory_db, monkeypatch, w):
    real = settings.section
    monkeypatch.setattr(settings, "section", lambda n: {**real(n), "on_session_end": False} if n == "writer" else real(n))
    calls = stub_model(monkeypatch, distilled(["a fact"], "S."))
    log("hello")
    assert (await w.run_pass("session_end")).skipped == "disabled" and calls == []


@pytest.mark.asyncio
async def test_preempting_a_timer_pass_leaves_nothing_half_written(memory_db, monkeypatch, w):
    stub_embed(monkeypatch)

    async def slow_stream(messages):
        await asyncio.sleep(5)
        yield distilled(["never lands"], "never")

    import llm

    monkeypatch.setattr(llm, "stream_reply", slow_stream)
    log("hello")
    w._timer_pass = asyncio.ensure_future(w.run_pass("timer"))
    await asyncio.sleep(0.05)
    w.preempt()
    with pytest.raises(asyncio.CancelledError):
        await w._timer_pass
    assert store.get_all_facts() == [] and checkpoint() == 0
    # and the lock was released, so the next pass can run
    stub_model(monkeypatch, distilled(["lands later"], "ok"))
    await w.run_pass("session_end")
    assert store.get_all_facts() == ["lands later"]


@pytest.mark.asyncio
async def test_the_profile_card_is_rebuilt_after_facts_change(memory_db, monkeypatch, w):
    def responder(messages):
        if system_prompt(messages).startswith("Write a compact"):
            return "Likes tea and builds desktop apps."
        return distilled(["likes tea"], "Tea.")

    stub_model(monkeypatch, responder)
    stub_embed(monkeypatch)
    log("I like tea")
    await w.run_pass("session_end")
    assert store.get_profile_card() == "Likes tea and builds desktop apps."
    assert store.get_meta("profile_dirty") == "0"
