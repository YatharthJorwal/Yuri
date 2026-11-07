"""
Phase 3 gate, the parts that can be proven without the user's hardware
(docs/phases/PHASE-3-MEMORY.md):

- a hard kill loses nothing that was committed and nothing half-written
  survives: tested by really killing a child process, mid-model-call and
  mid-transaction;
- 10 seeded facts, 5 corrections, 3 forgets and 20 recalls, all correct,
  against a real SQLite file;
- recall stays inside 300 / 1200 tokens.

The 50-turn persona-consistency run needs the real model and is not here.
"""

from __future__ import annotations

import asyncio
import json
import pathlib
import subprocess
import sys
import textwrap

import pytest

import context_budget
from memory import db, forget, recall, store, writer
from memory_helpers import stub_embed, stub_model

ORCH = str(pathlib.Path(__file__).resolve().parent.parent)

# ---------------------------------------------------------------------------
# kill -9
# ---------------------------------------------------------------------------

_CHILD = textwrap.dedent(
    """
    import asyncio, sys, time
    sys.path.insert(0, {orch!r})
    import llm
    from memory import db, store, writer

    mode, path = sys.argv[1], sys.argv[2]
    db._conn = db._open_connection(path=path, dimension=4)

    async def fake_embed(text):
        return [0.2, 0.8, 0.0, 0.0]
    writer.embed = fake_embed

    def reply(messages):
        return '{{"facts": [{{"text": "likes tea", "subject": "user", "correction": false}}], "episode_summary": "Tea chat."}}'

    async def good_stream(messages):
        yield reply(messages)
    llm.stream_reply = good_stream

    async def main():
        for i in range(3):
            store.add_transcript_turn(f"turn {{i}}", "ok")
        await writer.MemoryWriter().run_pass("session_end")
        print("COMMITTED", flush=True)

        store.add_transcript_turn("turn after the first pass", "ok")
        if mode == "mid_model":
            async def hang(messages):
                print("WAITING", flush=True)
                await asyncio.sleep(3600)
                yield ""
            llm.stream_reply = hang
        else:  # mid_transaction: die while the pass holds its write transaction
            real = store.add_episode
            def stall(*a, **k):
                real(*a, **k)
                print("IN_TX", flush=True)
                time.sleep(3600)
            store.add_episode = stall
            async def other_stream(messages):
                yield '{{"facts": [{{"text": "likes coffee", "subject": "user", "correction": false}}], "episode_summary": "Coffee."}}'
            llm.stream_reply = other_stream
        await writer.MemoryWriter().run_pass("session_end")

    asyncio.run(main())
    """
)


def _run_and_kill(tmp_path, mode: str) -> pathlib.Path:
    path = tmp_path / "killed.db"
    proc = subprocess.Popen(
        [sys.executable, "-c", _CHILD.format(orch=ORCH), mode, str(path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    seen = []
    marker = "WAITING" if mode == "mid_model" else "IN_TX"
    for line in proc.stdout:
        seen.append(line.strip())
        if line.strip() == marker:
            break
    else:
        proc.kill()
        pytest.fail(f"child never reached {marker}: {seen}\n{proc.stderr.read()}")
    proc.kill()  # SIGKILL: no cleanup handlers, no finally blocks
    proc.wait(timeout=10)
    assert "COMMITTED" in seen
    return path


@pytest.fixture
def reopened(monkeypatch):
    opened = []

    def _open(path):
        conn = db._open_connection(path=path, dimension=4)
        monkeypatch.setattr(db, "_conn", conn)
        monkeypatch.setattr(db, "get_connection", lambda: conn)
        opened.append(conn)
        return conn

    yield _open
    for conn in opened:
        conn.close()


@pytest.mark.asyncio
async def test_kill_during_a_model_call_loses_nothing_that_was_written(tmp_path, monkeypatch, reopened):
    path = _run_and_kill(tmp_path, "mid_model")
    reopened(path)

    assert store.get_all_facts() == ["likes tea"]  # the committed pass survived
    assert int(store.get_meta("writer_checkpoint_id")) == 3
    # the turn that was in flight is still in the durable buffer, unprocessed
    pending = store.get_transcript_after(3)
    assert [r["user"] for r in pending] == ["turn after the first pass"]

    # ...and the next pass picks it up
    stub_model(monkeypatch, '{"facts": [{"text": "likes coffee", "subject": "user"}], "episode_summary": "Coffee."}')
    stub_embed(monkeypatch)
    await writer.MemoryWriter().run_pass("startup")
    assert sorted(store.get_all_facts()) == ["likes coffee", "likes tea"]
    assert int(store.get_meta("writer_checkpoint_id")) == 4


@pytest.mark.asyncio
async def test_kill_in_the_middle_of_the_write_transaction_leaves_no_partial_pass(tmp_path, monkeypatch, reopened):
    path = _run_and_kill(tmp_path, "mid_transaction")
    reopened(path)

    # the second pass had already inserted a fact and an episode when it was killed;
    # none of it may be visible, and the checkpoint must not have moved
    assert store.get_all_facts() == ["likes tea"]
    assert len(store.get_episode_rows()) == 1
    assert int(store.get_meta("writer_checkpoint_id")) == 3

    stub_model(monkeypatch, '{"facts": [{"text": "likes coffee", "subject": "user"}], "episode_summary": "Coffee."}')
    stub_embed(monkeypatch)
    await writer.MemoryWriter().run_pass("startup")
    assert sorted(store.get_all_facts()) == ["likes coffee", "likes tea"]  # once each


# ---------------------------------------------------------------------------
# 10 seeded facts, 5 corrections, 3 forgets, 20 recalls
# ---------------------------------------------------------------------------

TOPICS = ["pizza", "gpu", "editor", "pet", "city", "music", "coffee", "language", "game", "sleep"]
DIM = 16


def vec(text: str) -> list[float]:
    """One axis per topic keyword, so similarity is 1.0 within a topic and 0 across."""
    out = [0.0] * DIM
    for i, topic in enumerate(TOPICS):
        if topic in text.lower():
            out[i] = 1.0
    if not any(out):
        out[DIM - 1] = 1.0
    return out


SEED = {
    "pizza": "loves pepperoni pizza",
    "gpu": "owns an RTX 3060 gpu",
    "editor": "uses neovim as their editor",
    "pet": "has a cat named Miso as a pet",
    "city": "lives in Pune, their city",
    "music": "listens to lofi music",
    "coffee": "drinks coffee every morning",
    "language": "is learning Japanese, their second language",
    "game": "plays Hades as their main game",
    "sleep": "goes to sleep around midnight",
}
CORRECTIONS = {  # topic -> the corrected fact (5)
    "pizza": "hates pizza now",
    "editor": "uses helix as their editor",
    "city": "lives in Delhi, their city",
    "coffee": "no longer drinks coffee",
    "game": "plays Celeste as their main game",
}
FORGOTTEN = ["music", "sleep", "pet"]  # (3)


@pytest.fixture
def real_file_db(tmp_path, monkeypatch):
    conn = db._open_connection(path=tmp_path / "gate.db", dimension=DIM)
    monkeypatch.setattr(db, "_conn", conn)
    monkeypatch.setattr(db, "get_connection", lambda: conn)
    yield conn
    conn.close()


async def _pass_stating(monkeypatch, texts, *, corrections=()):
    """One writer pass over a transcript in which the user stated `texts`. The stub
    model plays an ideal contradiction judge (picks the on-topic fact it is shown);
    how well the real 9B does at that is NOT proven here."""
    stub_embed(monkeypatch, vec)
    items = [{"text": t, "subject": "user", "correction": t in corrections} for t in texts]
    distilled = json.dumps({"facts": items, "episode_summary": ""})

    def responder(messages):
        if not messages[0]["content"].startswith("You compare"):
            return distilled
        body = messages[1]["content"]
        new_fact = body.rsplit("New fact: ", 1)[1].strip()
        for line in body.split("Facts on file:\n", 1)[1].split("\n\n", 1)[0].splitlines():
            fact_id, _, content = line.partition(": ")
            if vec(content) == vec(new_fact):
                return json.dumps({"action": "supersede", "target_id": int(fact_id)})
        return json.dumps({"action": "add", "target_id": None})

    stub_model(monkeypatch, responder)
    store.add_transcript_turn("saying things", "ok")
    await writer.MemoryWriter().run_pass("session_end")


@pytest.mark.asyncio
async def test_seeded_facts_corrections_forgets_and_recalls_are_all_correct(real_file_db, monkeypatch):
    await _pass_stating(monkeypatch, list(SEED.values()))
    assert len(store.get_all_facts()) == 10

    await _pass_stating(monkeypatch, list(CORRECTIONS.values()), corrections=set(CORRECTIONS.values()))
    live = store.get_all_facts()
    assert len(live) == 10  # five replaced, none piled up
    for topic, text in CORRECTIONS.items():
        assert text in live and SEED[topic] not in live

    for topic in FORGOTTEN:
        stub_model(monkeypatch, '{"remove_indices": [%d]}' % (
            [c for _i, c in store.get_all_facts_with_ids()].index(SEED[topic]) + 1
        ))
        assert await forget.maybe_forget(f"forget that about {topic}")
    assert len(store.get_all_facts()) == 7

    stub_embed(monkeypatch, vec)
    expected = {t: CORRECTIONS.get(t, SEED[t]) for t in TOPICS if t not in FORGOTTEN}
    queries = [(t, f"tell me about my {t}") for t in TOPICS for _ in range(2)]  # 20 recalls
    assert len(queries) == 20
    for topic, query in queries:
        block = await recall.build_recall_context(query, 5)
        if topic in FORGOTTEN:
            assert block is None or SEED[topic] not in block
            continue
        assert block is not None and expected[topic] in block, (topic, block)
        # nothing superseded or forgotten ever leaks back in
        leaked = [text for text in list(SEED.values()) if text not in expected.values() and text in (block or "")]
        assert leaked == [], (topic, leaked)

    # every supersede and forget is on the audit trail and reversible
    assert len(store.get_events("supersede")) == 5
    assert len(store.get_events("forget")) == 3
    for row in store.get_fact_rows(include_superseded=True):
        assert row["content"]  # nothing was deleted: 10 seeded + 5 corrections
    assert len(store.get_fact_rows(include_superseded=True)) == 15


@pytest.mark.asyncio
async def test_recall_stays_inside_the_documented_token_budgets(real_file_db, monkeypatch):
    stub_embed(monkeypatch, lambda t: vec("pizza"))
    for i in range(30):
        fid = store.add_fact(f"pizza fact {i}: " + "detail " * 25)
        store.set_fact_vector(fid, vec("pizza"))
        store.add_episode(f"pizza episode {i}: " + "detail " * 40, vec("pizza"))
    store.set_profile_card("Profile sentence. " * 60)

    for budget, limit in (("reflex", 300), ("mind", 1200)):
        fitted = recall.fit(await recall.select("pizza", 10, budget=budget), limit)
        assert fitted.tokens <= limit
        assert fitted.facts or fitted.episodes
        contents = (fitted.card or "") + "".join(f["content"] + "; " for f in fitted.facts)
        contents += "".join(e["summary"] + "; " for e in fitted.episodes)
        assert context_budget.estimate_tokens(contents) <= limit + 6  # estimator rounds each piece up
