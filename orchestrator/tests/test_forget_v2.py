"""forget.py on the v2 store: reversible, audited, and not triggered by 'don't forget'."""

from __future__ import annotations

import json

import pytest

from memory import forget, store
from memory_helpers import stub_model


@pytest.mark.parametrize(
    "text",
    ["forget that", "please forget my address", "Forget it", "don't remember that", "delete that", "stop remembering my birthday"],
)
def test_real_forget_requests_trigger(text):
    assert forget._FORGET_TRIGGER.search(text)


@pytest.mark.parametrize(
    "text",
    ["don't forget to buy milk", "Don\u2019t forget the meeting", "do not forget my keys"],
)
def test_dont_forget_is_the_opposite_of_forget(text):
    assert not forget._FORGET_TRIGGER.search(text)


@pytest.mark.asyncio
async def test_forgetting_is_soft_audited_and_reversible(memory_db, monkeypatch):
    pizza = store.add_fact("likes pizza")
    store.add_fact("likes tea")
    stub_model(monkeypatch, '{"remove_indices": [2]}')  # newest-first: 1 = tea, 2 = pizza

    note = await forget.maybe_forget("forget that I like pizza")

    assert note and "likes pizza" in note
    assert store.get_all_facts() == ["likes tea"]
    row = store.get_fact(pizza)
    assert row is not None and row["superseded_by"] == -1
    event = store.get_events("forget")[0]
    assert event["actor"] == "user" and json.loads(event["detail"])["fact_ids"] == [pizza]
    assert store.restore_fact(pizza)
    assert "likes pizza" in store.get_all_facts()


@pytest.mark.asyncio
async def test_dont_forget_makes_no_model_call(memory_db, monkeypatch):
    store.add_fact("likes pizza")
    calls = stub_model(monkeypatch, '{"remove_indices": [1]}')
    assert await forget.maybe_forget("don't forget to remind me about pizza night") is None
    assert calls == [] and store.get_all_facts() == ["likes pizza"]
