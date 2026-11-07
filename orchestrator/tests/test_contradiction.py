"""Contradiction handling: the deterministic pass, the model pass, and the vetoes
that stop either from replacing a fact it should not."""

from __future__ import annotations

import pytest

from memory import contradiction as c
from memory_helpers import stub_model, stub_model_down, system_prompt


def cand(id_, content, *, confidence=0.7, pinned=False, similarity=0.9):
    return {"id": id_, "content": content, "confidence": confidence, "pinned": pinned, "similarity": similarity}


@pytest.mark.parametrize(
    "new,old",
    [("likes pizza", "Likes pizza."), ("uses an RTX 3060", "uses an RTX 3060"), ("Likes  Pizza!", "likes pizza")],
)
def test_same_statement_is_a_duplicate(new, old):
    d = c.deterministic(new, [cand(1, old)])
    assert d and d.action == c.DUPLICATE and d.target_id == 1


@pytest.mark.parametrize(
    "new,old",
    [
        ("doesn't like pizza", "likes pizza"),
        ("likes pizza", "does not like pizza"),
        ("no longer uses VRoid", "uses VRoid"),
        ("never eats meat", "eats meat"),
    ],
)
def test_opposite_polarity_of_the_same_claim_supersedes(new, old):
    d = c.deterministic(new, [cand(7, old)])
    assert d and d.action == c.SUPERSEDE and d.target_id == 7


def test_unrelated_facts_are_left_to_the_model_pass():
    assert c.deterministic("likes pizza", [cand(1, "uses an RTX 3060")]) is None
    assert c.deterministic("likes pizza", [cand(1, "likes ice cream")]) is None


@pytest.mark.asyncio
async def test_no_candidates_means_add_and_no_model_call(monkeypatch):
    calls = stub_model(monkeypatch, "{}")
    d = await c.decide("likes pizza", [], new_confidence=0.7)
    assert d.action == c.ADD and calls == []


@pytest.mark.asyncio
async def test_deterministic_decisions_cost_no_model_call(monkeypatch):
    calls = stub_model(monkeypatch, "{}")
    d = await c.decide("doesn't like pizza", [cand(3, "likes pizza")], new_confidence=0.7)
    assert d.action == c.SUPERSEDE and d.via == "deterministic" and calls == []


@pytest.mark.asyncio
async def test_model_decides_when_a_close_neighbour_is_not_obviously_the_same(monkeypatch):
    calls = stub_model(monkeypatch, '{"action": "supersede", "target_id": 4}')
    d = await c.decide("switched to Linux", [cand(4, "uses Windows 11", similarity=0.7)], new_confidence=0.7)
    assert (d.action, d.target_id, d.via) == (c.SUPERSEDE, 4, "model")
    assert "uses Windows 11" in calls[0][1]["content"] and system_prompt(calls[0]).startswith("You compare")


@pytest.mark.asyncio
async def test_model_not_asked_about_distant_neighbours(monkeypatch):
    calls = stub_model(monkeypatch, '{"action": "supersede", "target_id": 4}')
    d = await c.decide("switched to Linux", [cand(4, "uses Windows 11", similarity=0.2)], new_confidence=0.7)
    assert d.action == c.ADD and calls == []


@pytest.mark.asyncio
async def test_model_naming_an_id_it_was_not_shown_is_ignored(monkeypatch):
    stub_model(monkeypatch, '{"action": "supersede", "target_id": 999}')
    d = await c.decide("switched to Linux", [cand(4, "uses Windows 11")], new_confidence=0.7)
    assert d.action == c.ADD


@pytest.mark.asyncio
@pytest.mark.parametrize("garbage", ["I think they conflict!", '{"action": "delete_everything", "target_id": 4}', "{not json}"])
async def test_unparseable_or_unknown_model_answers_become_add(monkeypatch, garbage):
    stub_model(monkeypatch, garbage)
    d = await c.decide("switched to Linux", [cand(4, "uses Windows 11")], new_confidence=0.7)
    assert d.action == c.ADD


@pytest.mark.asyncio
async def test_model_down_means_add(monkeypatch):
    stub_model_down(monkeypatch)
    d = await c.decide("switched to Linux", [cand(4, "uses Windows 11")], new_confidence=0.7)
    assert d.action == c.ADD


@pytest.mark.asyncio
async def test_pinned_fact_is_immune_to_a_normal_supersede():
    d = await c.decide("doesn't like pizza", [cand(1, "likes pizza", pinned=True)], new_confidence=0.7)
    assert d.action == c.ADD and d.blocked == "pinned" and d.proposed_target_id == 1


@pytest.mark.asyncio
async def test_explicit_correction_does_replace_a_pinned_fact():
    d = await c.decide("doesn't like pizza", [cand(1, "likes pizza", pinned=True)], new_confidence=1.0, explicit_correction=True)
    assert d.action == c.SUPERSEDE and d.target_id == 1


@pytest.mark.asyncio
async def test_lower_confidence_cannot_replace_higher_confidence():
    d = await c.decide("doesn't like pizza", [cand(1, "likes pizza", confidence=1.0)], new_confidence=0.7)
    assert d.action == c.ADD and d.blocked == "lower_confidence"


@pytest.mark.asyncio
async def test_equal_confidence_may_replace():
    d = await c.decide("doesn't like pizza", [cand(1, "likes pizza", confidence=0.7)], new_confidence=0.7)
    assert d.action == c.SUPERSEDE


def test_keyword_overlap_ignores_negation_and_filler():
    assert c.keyword_overlap("likes pizza", "doesn't like pizza") == 1.0
    assert c.keyword_overlap("likes pizza", "uses Windows") == 0.0
