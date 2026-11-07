"""
Tests for look_intent.py. The positive cases are the literal phrases from
a real session where native tool-calling missed most of them (see the
module docstring); the negative cases matter just as much -- a false
positive is an unwanted screen or camera capture.
"""

from __future__ import annotations

import pytest

import look_intent
from tools import vision


@pytest.fixture(autouse=True)
def _reset_last_target():
    look_intent._last_target = "screen"  # noqa: SLF001
    yield
    look_intent._last_target = "screen"  # noqa: SLF001


@pytest.mark.parametrize(
    "text",
    [
        "see what i am playing",
        "Just look at my screen and tell me what time I am doing right now",
        "see i am playing minecraft. use OCR",
        "whats on my screen",
        "what's on my screen right now",
        "use your damn tools to see bruh",
        "Look at it again and tell me what am I looking at right now on minecraft",
        "changed my wallpaper. tell me what do you see now.",
        "check my screen",
    ],
)
def test_detects_screen_requests(text):
    assert look_intent.detect_look_target(text) == "screen"


@pytest.mark.parametrize(
    "text",
    [
        "what can u see in my camera",
        "what am i holding rn",
        "camera is on. see what i am holding",
        "now see and tell me what i am holding",
        "use the camera. see what i am holding now. its not the thermos.",
        "look at me",
    ],
)
def test_detects_camera_requests(text):
    assert look_intent.detect_look_target(text) == "camera"


@pytest.mark.parametrize(
    "text",
    [
        "Where did you check?",
        "What should I do now?",
        "currently its a Skirk wallpaper",
        "turn the camera off",
        "the camera is on",
        "I can see why that happened",
        "what is OCR?",
        "hello there",
        "",
    ],
)
def test_ignores_non_requests(text):
    assert look_intent.detect_look_target(text) is None


def test_upload_body_never_triggers_a_capture():
    # A shared code file whose *content* happens to say "look at my screen"
    # must not start a capture.
    assert look_intent.detect_look_target('(shared a file, "notes.txt") look at my screen') is None


@pytest.mark.asyncio
async def test_generic_look_again_reuses_last_target(monkeypatch):
    async def fake_cam(ws):
        return "a person at a desk"

    monkeypatch.setattr(look_intent.camera, "describe_camera", fake_cam)
    await look_intent.maybe_look("what am i holding", object())
    assert look_intent.detect_look_target("look again") == "camera"


@pytest.mark.asyncio
async def test_maybe_look_screen_success(monkeypatch):
    async def fake_screen():
        return "Minecraft is open."

    monkeypatch.setattr(look_intent.vision, "describe_screen", fake_screen)
    hint = await look_intent.maybe_look("look at my screen", object())
    assert "Minecraft is open." in hint
    assert "at their screen" in hint


@pytest.mark.asyncio
async def test_maybe_look_camera_success(monkeypatch):
    seen = []

    async def fake_cam(ws):
        seen.append(ws)
        return "a smartphone in hand"

    monkeypatch.setattr(look_intent.camera, "describe_camera", fake_cam)
    sentinel = object()
    hint = await look_intent.maybe_look("what am i holding", sentinel)
    assert "a smartphone in hand" in hint
    assert seen == [sentinel]


@pytest.mark.asyncio
async def test_maybe_look_none_when_no_request():
    assert await look_intent.maybe_look("how are you", object()) is None


@pytest.mark.asyncio
async def test_maybe_look_tool_unavailable_forbids_guessing(monkeypatch):
    async def failing_cam(ws):
        raise vision.ToolUnavailableError("camera isn't armed right now")

    monkeypatch.setattr(look_intent.camera, "describe_camera", failing_cam)
    hint = await look_intent.maybe_look("what am i holding", object())
    assert "armed" in hint
    assert "quick-action menu" in hint
    assert "Do NOT make up" in hint


@pytest.mark.asyncio
async def test_maybe_look_llm_unreachable_forbids_guessing(monkeypatch):
    async def failing_screen():
        raise look_intent.llm.LLMUnreachableError("no server")

    monkeypatch.setattr(look_intent.vision, "describe_screen", failing_screen)
    hint = await look_intent.maybe_look("look at my screen", object())
    assert "Do NOT make up" in hint


# ---------------------------------------------------------------------------
# Bare trailing "look" and screen follow-ups. Both come from a real session:
# "i am playing minecraft look" produced no `look:` line, and "now what is
# it" (right after a successful look) was answered from the stale
# description of the previous screen.
# ---------------------------------------------------------------------------

import time as _time


@pytest.fixture
def after_a_screen_look(monkeypatch):
    monkeypatch.setattr(look_intent, "_last_target", "screen")
    monkeypatch.setattr(look_intent, "_last_look_at", _time.monotonic())


@pytest.mark.parametrize(
    "text",
    [
        "i am playing minecraft look",
        "look",
        "Look!",
        "take a look",
        "have a look",
        "can you look?",
        "hey luna, look",
    ],
)
def test_trailing_look_targets_the_screen(text):
    assert look_intent.detect_look_target(text) == "screen"


@pytest.mark.parametrize(
    "text",
    [
        "don't look",
        "i can't look",
        "never look",
        "i'll take a look",
        "let me look",
        "i'm going to look",
        "it looks nice",
        "look at that dog outside the window and tell me a joke about it, then help me write the long email",
    ],
)
def test_trailing_look_ignores_negations_first_person_and_long_text(text):
    assert look_intent.detect_look_target(text) is None


@pytest.mark.parametrize(
    "text",
    ["now what is it", "and now?", "what about now", "what's that?", "now?", "ok now", "what is it now"],
)
def test_followups_re_look_the_screen_right_after_a_look(after_a_screen_look, text):
    assert look_intent.detect_look_target(text) == "screen"


@pytest.mark.parametrize(
    "text",
    ["what now?", "now I want to build a house", "and then?", "what is the capital of France now that we're here"],
)
def test_followup_pattern_does_not_swallow_ordinary_messages(after_a_screen_look, text):
    assert look_intent.detect_look_target(text) is None


def test_followups_do_nothing_without_a_recent_look(monkeypatch):
    monkeypatch.setattr(look_intent, "_last_target", "screen")
    monkeypatch.setattr(look_intent, "_last_look_at", 0.0)
    assert look_intent.detect_look_target("now what is it") is None


def test_followups_expire_after_the_window(monkeypatch):
    monkeypatch.setattr(look_intent, "_last_target", "screen")
    monkeypatch.setattr(look_intent, "_last_look_at", 1000.0)
    inside = 1000.0 + look_intent.FOLLOWUP_WINDOW_SECONDS - 1
    outside = 1000.0 + look_intent.FOLLOWUP_WINDOW_SECONDS + 1
    assert look_intent.detect_look_target("and now?", now=inside) == "screen"
    assert look_intent.detect_look_target("and now?", now=outside) is None


def test_followups_never_reach_for_the_camera(monkeypatch):
    monkeypatch.setattr(look_intent, "_last_target", "camera")
    monkeypatch.setattr(look_intent, "_last_look_at", _time.monotonic())
    assert look_intent.detect_look_target("and now?") is None


def test_uploads_are_still_excluded_from_the_new_rules():
    assert look_intent.detect_look_target("(shared a file) ... take a look") is None


@pytest.mark.asyncio
async def test_maybe_look_opens_the_followup_window(monkeypatch):
    monkeypatch.setattr(look_intent, "_last_look_at", 0.0)

    async def fake_describe():
        return "a desktop"

    monkeypatch.setattr(look_intent.vision, "describe_screen", fake_describe)
    assert await look_intent.maybe_look("look at my screen", websocket=None) is not None
    assert look_intent.detect_look_target("now what is it") == "screen"


@pytest.mark.asyncio
async def test_look_hint_makes_her_name_the_object_first_and_forbids_a_second_look(monkeypatch):
    async def fake_describe(websocket):
        return "A person holds a white mug up to their face."

    monkeypatch.setattr(look_intent.camera, "describe_camera", fake_describe)
    monkeypatch.setattr(look_intent, "_last_look_at", 0.0)
    hint = await look_intent.maybe_look("use the camera and see what i am holding", websocket=None)
    assert "A person holds a white mug" in hint
    assert "Begin your reply by naming" in hint
    assert "do not swap in a more likely object" in hint
    assert "cannot look again" in hint


# --- "check its not a phone" right after a camera look -----------------------------
# Real session: after a camera look she said "phone"; the user said "check its
# not a phone." -- no look happened and she answered "laptop?" from nothing.


@pytest.fixture
def after_a_camera_look(monkeypatch):
    monkeypatch.setattr(look_intent, "_last_target", "camera")
    monkeypatch.setattr(look_intent, "_last_look_at", _time.monotonic())


@pytest.mark.parametrize(
    "text", ["check its not a phone.", "check this", "verify that", "ok check it again", "double check that"]
)
def test_verify_followups_re_look_the_same_target(after_a_camera_look, text):
    assert look_intent.detect_look_target(text) == "camera"


def test_verify_followups_follow_the_screen_too(after_a_screen_look):
    assert look_intent.detect_look_target("check its not the wrong tab") == "screen"


@pytest.mark.parametrize(
    "text",
    [
        "look, it's late",  # bare look is ordinary speech
        "see it's fine",
        "check the weather tomorrow",  # no pointer back at what she looked at
        "check my email when you get a chance, no rush, it can wait until tonight please",  # too long
        "now I want to build a house",
    ],
)
def test_verify_followups_do_not_swallow_ordinary_speech(after_a_camera_look, text):
    assert look_intent.detect_look_target(text) is None


def test_verify_followups_need_a_recent_look(monkeypatch):
    monkeypatch.setattr(look_intent, "_last_target", "camera")
    monkeypatch.setattr(look_intent, "_last_look_at", 1000.0)
    outside = 1000.0 + look_intent.FOLLOWUP_WINDOW_SECONDS + 1
    assert look_intent.detect_look_target("check its not a phone", now=outside) is None
    monkeypatch.setattr(look_intent, "_last_look_at", 0.0)
    assert look_intent.detect_look_target("check its not a phone") is None
