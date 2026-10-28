"""
Speech-safe text tests.

The corpus is model-shaped text (the kind a 9B tsundere actually produces),
not invented punctuation. Each case pins one rule; the invariants at the
bottom are the ones that matter if a rule is ever reordered.
"""

import dataclasses

import pytest

import config
import speech


CFG = config.CONFIG.speech


def prep(text: str) -> speech.PreparedSpeech:
    return speech.prepare(text, CFG)


# --- individual rules --------------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("Wait\u2014what?", "Wait, what?"),                 # em dash
        ("Wait\u2013what?", "Wait, what?"),                 # en dash
        ("Wait \u2014 what?", "Wait, what?"),               # spaced em dash
        ("It\u2019s fine", "It\u2019s fine"),               # apostrophe survives
        ("She said \u201chello\u201d to me.", "She said hello to me."),
        ("She said \"hello\" to me.", "She said hello to me."),
        ("I don't know\u2026", "I don't know,"),            # ellipsis -> pause
        ("**Bold** text", "Bold text"),                     # markdown emphasis
        ("run `git status` now", "run git status now"),     # inline code
        ("see https://example.com/x for that", "see a link for that"),
        ("Nice one \U0001F600", "Nice one"),                # emoji
        ("Really!!!", "Really!"),                           # repeated punctuation
        ("well-being matters", "well being matters"),       # compound hyphen
        ("first; second", "first. second"),                 # semicolon -> stop
    ],
)
def test_rules(raw: str, expected: str):
    assert prep(raw).spoken == expected


def test_code_fence_is_never_read_aloud():
    raw = "Here:\n```python\nprint('secret')\n```\nThat is it."
    result = prep(raw)
    assert "code block" in result.spoken
    assert "print" not in result.spoken
    assert any(entry.startswith("code_block:") for entry in result.dropped)


def test_bullets_become_sentences():
    raw = "- buy milk\n- fix the bug\n- sleep"
    spoken = prep(raw).spoken
    assert "-" not in spoken
    assert "buy milk" in spoken and "fix the bug" in spoken


def test_short_parenthetical_is_dropped():
    assert prep("A thing (small note) happened.").spoken == "A thing happened."


def test_caps_are_lowered_but_acronyms_survive():
    spoken = prep("THIS IS FINE, check the VRAM tab").spoken
    assert "this is fine" in spoken
    assert "VRAM" in spoken


def test_emotion_tag_is_not_speech():
    # extract_emotion_tag runs before this in the pipeline; if it did not,
    # brackets would still be stripped rather than spoken.
    assert "[" not in prep("Sure thing. [teasing]").spoken


# --- segmentation ------------------------------------------------------------


def test_segments_split_on_sentences():
    segments = prep("One. Two! Three?").segments
    assert segments == ("One.", "Two!", "Three?")


def test_no_segment_exceeds_the_limit():
    cfg = dataclasses.replace(CFG, max_segment_chars=40, min_segment_chars=8)
    text = " ".join(["this is a fairly long clause"] * 6)
    for segment in speech.prepare(text, cfg).segments:
        assert len(segment) <= 40


def test_no_segment_splits_a_word():
    cfg = dataclasses.replace(CFG, max_segment_chars=30, min_segment_chars=5)
    result = speech.prepare("supercalifragilisticexpialidocious word", cfg)
    for segment in result.segments:
        assert " " in segment or len(segment) <= 30


def test_runt_fragments_are_merged():
    # A fragment with no terminal punctuation belongs to the sentence before it.
    segments = prep("This one is long enough to stand alone. yes").segments
    assert segments == ("This one is long enough to stand alone. yes",)


def test_short_complete_sentences_stay_separate():
    # "No." is a real clause; merging it would only make streaming coarser.
    assert prep("No. This one is long enough to stand alone.").segments == (
        "No.",
        "This one is long enough to stand alone.",
    )


def test_empty_input():
    result = prep("   ")
    assert result.spoken == ""
    assert result.segments == ()


# --- invariants --------------------------------------------------------------


GOLDEN = [
    "Ugh, fine\u2014I'll look at it. [teasing]",
    "**Look** at line 42; it's broken\u2026 again.",
    "I said \u201copen the terminal\u201d, not \u201cdelete the repo\u201d.",
    "Here:\n```js\nconsole.log('x')\n```\nNow run it.",
    "Seriously?! You named it `test_final_v2`?",
    "See https://example.com/docs \U0001F644",
    "- first step\n- second step\n- done",
    "It's 3-4 minutes, give or take.",
    "No. Just no.",
]


@pytest.mark.parametrize("raw", GOLDEN)
def test_banned_characters_never_survive(raw):
    assert speech.banned_chars_in(prep(raw).spoken) == set()


@pytest.mark.parametrize("raw", GOLDEN)
def test_normalization_is_idempotent(raw):
    once = prep(raw).spoken
    twice = speech.prepare(once, CFG).spoken
    assert twice == once


@pytest.mark.parametrize("raw", GOLDEN)
def test_display_keeps_the_original_wording(raw):
    result = prep(raw)
    assert result.display == raw.strip()


def test_disabled_returns_text_unchanged():
    cfg = dataclasses.replace(CFG, enabled=False)
    raw = "Wait\u2014what?"
    result = speech.prepare(raw, cfg)
    assert result.spoken == raw
    assert result.enabled is False
