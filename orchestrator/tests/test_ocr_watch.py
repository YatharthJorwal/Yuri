"""
Tests for ocr_watch.py -- same shape as test_task_guide.py's own tests,
since ocr_watch.py deliberately mirrors task_guide.py's architecture
(module-level state, forgiving JSON parse, a one-shot VLM call). What's
NOT covered here, same caveat as task_guide's own tests: whether
qwen3.5:9b actually produces the requested JSON reliably against a real
server, and whether the "is this worth a comment" judgment reads as
well-calibrated (not naggy, not silent) in actual use.
"""

from __future__ import annotations

import time

import pytest

import ocr_watch


@pytest.fixture(autouse=True)
def _reset_state():
    """ocr_watch's WatchState is module-level (see its own docstring on
    why) -- reset it before and after every test so tests don't leak
    state into each other."""
    ocr_watch.set_active(False)
    yield
    ocr_watch.set_active(False)


# ---------------------------------------------------------------------------
# set_active -- the state machine itself
# ---------------------------------------------------------------------------


def test_set_active_true_starts_watching():
    ocr_watch.set_active(True)
    state = ocr_watch.get_state()
    assert state.active is True
    assert state.last_seen_summary == ""


def test_set_active_false_stops_watching():
    ocr_watch.set_active(True)
    ocr_watch.set_active(False)
    assert ocr_watch.get_state().active is False


def test_starting_resets_check_clock_and_summary():
    ocr_watch.set_active(True)
    state = ocr_watch.get_state()
    assert abs(time.time() - state.last_check_at) < 1.0
    assert state.last_seen_summary == ""


def test_restarting_clears_stale_summary():
    ocr_watch.set_active(True)
    ocr_watch.mark_checked("an old summary from before")
    ocr_watch.set_active(False)
    ocr_watch.set_active(True)
    # A fresh start shouldn't carry over a summary from a previous
    # watching session -- the next check has nothing to compare against.
    assert ocr_watch.get_state().last_seen_summary == ""


# ---------------------------------------------------------------------------
# mark_checked
# ---------------------------------------------------------------------------


def test_mark_checked_noop_when_not_active():
    before = ocr_watch.get_state()
    ocr_watch.mark_checked("something")
    after = ocr_watch.get_state()
    assert before == after


def test_mark_checked_updates_summary_when_given():
    ocr_watch.set_active(True)
    ocr_watch.mark_checked("a code editor open")
    assert ocr_watch.get_state().last_seen_summary == "a code editor open"


def test_mark_checked_keeps_summary_when_omitted():
    ocr_watch.set_active(True)
    ocr_watch.mark_checked("a code editor open")
    ocr_watch.mark_checked()  # no new summary given
    assert ocr_watch.get_state().last_seen_summary == "a code editor open"


# ---------------------------------------------------------------------------
# due_for_check
# ---------------------------------------------------------------------------


def test_due_for_check_false_when_not_active():
    state = ocr_watch.WatchState(active=False, last_check_at=0.0)
    assert ocr_watch.due_for_check(state, interval_seconds=10) is False


def test_due_for_check_true_past_interval():
    state = ocr_watch.WatchState(active=True, last_check_at=time.time() - 100)
    assert ocr_watch.due_for_check(state, interval_seconds=10) is True


def test_due_for_check_false_within_interval():
    state = ocr_watch.WatchState(active=True, last_check_at=time.time())
    assert ocr_watch.due_for_check(state, interval_seconds=10) is False


# ---------------------------------------------------------------------------
# check_for_comment -- llm.describe_image stubbed, forgiving JSON parse
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_check_for_comment_worthy(monkeypatch):
    async def fake_describe_image(prompt, image_b64, **_kw):
        assert "nothing yet" in prompt
        return '{"comment_worthy": true, "summary": "a browser open to a game store page", "note": "looks like they are eyeing a new game"}'

    monkeypatch.setattr(ocr_watch.llm, "describe_image", fake_describe_image)
    result = await ocr_watch.check_for_comment("", "fake_b64")
    assert result == {
        "comment_worthy": True,
        "summary": "a browser open to a game store page",
        "note": "looks like they are eyeing a new game",
    }


@pytest.mark.asyncio
async def test_check_for_comment_not_worthy(monkeypatch):
    async def fake_describe_image(prompt, image_b64, **_kw):
        assert "a code editor open" in prompt
        return '{"comment_worthy": false, "summary": "a code editor, same as before", "note": ""}'

    monkeypatch.setattr(ocr_watch.llm, "describe_image", fake_describe_image)
    result = await ocr_watch.check_for_comment("a code editor open", "fake_b64")
    assert result == {"comment_worthy": False, "summary": "a code editor, same as before", "note": ""}


@pytest.mark.asyncio
async def test_check_for_comment_llm_unreachable_returns_none(monkeypatch):
    async def failing_describe_image(prompt, image_b64, **_kw):
        raise ocr_watch.llm.LLMUnreachableError("no server")

    monkeypatch.setattr(ocr_watch.llm, "describe_image", failing_describe_image)
    result = await ocr_watch.check_for_comment("", "fake_b64")
    assert result is None


@pytest.mark.asyncio
async def test_check_for_comment_unparseable_returns_none(monkeypatch):
    async def fake_describe_image(prompt, image_b64, **_kw):
        return "sure, nothing much going on I guess"

    monkeypatch.setattr(ocr_watch.llm, "describe_image", fake_describe_image)
    result = await ocr_watch.check_for_comment("", "fake_b64")
    assert result is None


def test_parse_watch_result_handles_markdown_fenced_json():
    raw = '```json\n{"comment_worthy": true, "summary": "x", "note": "y"}\n```'
    result = ocr_watch._parse_watch_result(raw)
    assert result == {"comment_worthy": True, "summary": "x", "note": "y"}


def test_parse_watch_result_missing_comment_worthy_key_returns_none():
    result = ocr_watch._parse_watch_result('{"summary": "x", "note": "y"}')
    assert result is None


def test_parse_watch_result_non_bool_comment_worthy_returns_none():
    result = ocr_watch._parse_watch_result('{"comment_worthy": "yes", "summary": "x", "note": "y"}')
    assert result is None


def test_parse_watch_result_missing_summary_and_note_default_empty():
    result = ocr_watch._parse_watch_result('{"comment_worthy": false}')
    assert result == {"comment_worthy": False, "summary": "", "note": ""}


# ---------------------------------------------------------------------------
# Deterministic quiet rules -- the model said "worth a comment" on 12 of 15
# checks in a real session, five times in a row about the identical summary.
# ---------------------------------------------------------------------------

SAME_SCREEN_PAIRS = [
    # the literal consecutive summaries from the real log
    ("debugging a Python application with an anime avatar", "Debugging a Python app with an anime avatar"),
    ("Minecraft gameplay with anime avatar overlay", "Minecraft gameplay with anime avatar overlay"),
    ("player mining wood in Minecraft with new recipe unlocked", "player mining wood in Minecraft with new recipe unlocked"),
]
DIFFERENT_SCREEN_PAIRS = [
    ("player mining wood in Minecraft with new recipe unlocked", "player exploring a lush forest biome in Minecraft"),
    ("Minecraft gameplay with anime avatar overlay", "Canva design interface with laptop mockup"),
    ("Canva design interface with laptop mockup", "File explorer showing game-related folders and Unity files"),
    ("Minecraft launcher settings with a cute anime avatar and luxury car wallpaper",
     "player mining wood in Minecraft with new recipe unlocked"),
]


@pytest.mark.parametrize("a,b", SAME_SCREEN_PAIRS)
def test_summaries_match_real_repeats(a, b):
    assert ocr_watch.summaries_match(a, b, 0.6) is True


@pytest.mark.parametrize("a,b", DIFFERENT_SCREEN_PAIRS)
def test_summaries_match_real_changes_are_not_matches(a, b):
    assert ocr_watch.summaries_match(a, b, 0.6) is False


def test_summaries_match_empty_side_is_never_a_match():
    assert ocr_watch.summaries_match("", "anything on screen", 0.6) is False
    assert ocr_watch.summaries_match("anything on screen", "", 0.6) is False


def _worthy(summary="a new thing", note="something funny"):
    return {"comment_worthy": True, "summary": summary, "note": note}


def test_decide_comment_speaks_for_a_genuinely_new_screen():
    speak, why = ocr_watch.decide_comment(_worthy("Yahtzee game in progress"), "Canva design", 0.0, 120, 0.6)
    assert speak is True and why == ""


def test_decide_comment_model_said_no():
    result = {"comment_worthy": False, "summary": "x", "note": ""}
    assert ocr_watch.decide_comment(result, "", 0.0, 120, 0.6)[0] is False


def test_decide_comment_needs_a_note_to_react_to():
    assert ocr_watch.decide_comment(_worthy(note=""), "", 0.0, 120, 0.6)[0] is False


def test_decide_comment_suppresses_an_unchanged_screen_even_if_model_says_yes():
    speak, why = ocr_watch.decide_comment(
        _worthy("Debugging a Python app with an anime avatar"),
        "debugging a Python application with an anime avatar",
        0.0, 120, 0.6,
    )
    assert speak is False and "same screen" in why


def test_decide_comment_enforces_the_minimum_gap():
    now = 10_000.0
    speak, why = ocr_watch.decide_comment(_worthy("brand new"), "old", now - 30, 120, 0.6, now=now)
    assert speak is False and "minimum gap" in why
    speak, _ = ocr_watch.decide_comment(_worthy("brand new"), "old", now - 121, 120, 0.6, now=now)
    assert speak is True


def test_decide_comment_first_comment_of_a_watch_ignores_the_gap():
    # last_comment_at == 0 means "never spoke this watch"
    assert ocr_watch.decide_comment(_worthy("brand new"), "old", 0.0, 120, 0.6, now=5.0)[0] is True


def test_mark_commented_records_time_only_while_active():
    ocr_watch.mark_commented()
    assert ocr_watch.get_state().last_comment_at == 0.0
    ocr_watch.set_active(True)
    ocr_watch.mark_commented()
    assert ocr_watch.get_state().last_comment_at > 0.0


def test_defer_check_restarts_the_clock_but_keeps_summary_and_last_comment():
    ocr_watch.set_active(True)
    ocr_watch.mark_checked("old screen")
    ocr_watch.mark_commented()
    commented = ocr_watch.get_state().last_comment_at
    ocr_watch._state = ocr_watch.replace(ocr_watch._state, last_check_at=time.time() - 9999)  # noqa: SLF001
    ocr_watch.defer_check()
    state = ocr_watch.get_state()
    assert ocr_watch.due_for_check(state, interval_seconds=60) is False
    assert state.last_seen_summary == "old screen" and state.last_comment_at == commented


def test_defer_check_is_a_noop_when_inactive():
    ocr_watch.defer_check()
    assert ocr_watch.get_state().active is False


@pytest.mark.asyncio
async def test_watch_check_sends_the_schema(monkeypatch):
    seen = {}

    async def fake_describe_image(prompt, image_b64, json_schema=None):
        seen["schema"] = json_schema
        return '{"comment_worthy": true, "summary": "a game", "note": "funny"}'

    monkeypatch.setattr(ocr_watch.llm, "describe_image", fake_describe_image)
    result = await ocr_watch.check_for_comment("", "b64")
    assert seen["schema"] == ocr_watch.WATCH_SCHEMA and result["comment_worthy"] is True
    assert set(ocr_watch.WATCH_SCHEMA["required"]) == {"comment_worthy", "summary", "note"}


@pytest.mark.asyncio
async def test_watch_unparseable_output_is_logged_raw(monkeypatch, capsys):
    async def prose(prompt, image_b64, json_schema=None):
        return "Nothing much is going on."

    monkeypatch.setattr(ocr_watch.llm, "describe_image", prose)
    assert await ocr_watch.check_for_comment("", "b64") is None
    assert "unparseable check output: 'Nothing much is going on.'" in capsys.readouterr().err
