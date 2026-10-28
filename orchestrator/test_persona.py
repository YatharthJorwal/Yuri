"""
Tests for persona.py's apply_persona_pass() -- both safety nets are pure
regex logic with no network/LLM/event-loop involved, so (unlike most of
this project) these are genuinely, fully testable rather than "logically
checked, not confirmed." No test file existed for the original dash
safety net before this one; added here as real regression coverage
alongside the new you're/You're one, rather than leaving both to ad hoc
sandbox checks only.
"""

from persona import SYSTEM_PROMPT, apply_persona_pass


def test_dash_replaced_with_comma():
    assert apply_persona_pass("wait - actually never mind") == "wait, actually never mind"


def test_compound_word_dash_replaced():
    assert apply_persona_pass("your well-being matters") == "your well, being matters"


def test_em_dash_and_en_dash_both_replaced():
    assert apply_persona_pass("fine\u2014whatever") == "fine, whatever"
    assert apply_persona_pass("fine\u2013whatever") == "fine, whatever"


def test_no_dash_is_unchanged():
    assert apply_persona_pass("nothing to change here.") == "nothing to change here."


def test_youre_lowercase_replaced():
    assert apply_persona_pass("you're being annoying") == "you are being annoying"


def test_youre_capitalized_replaced_keeping_capitalization():
    assert apply_persona_pass("You're being annoying") == "You are being annoying"


def test_youre_does_not_touch_other_contractions():
    # Only "you're" was flagged as mispronounced -- everything else
    # should pass through untouched (see persona.py's own comment on
    # _YOURE_PATTERN for why this one word specifically).
    text = "I can't believe you're not done, don't worry though"
    assert apply_persona_pass(text) == "I can't believe you are not done, don't worry though"


def test_youre_and_dash_both_apply_in_one_pass():
    assert apply_persona_pass("you're right - I checked") == "you are right, I checked"


# ---------------------------------------------------------------------------
# Canary: the "you can only observe and advise" framing that predated
# Phase 4's tools was found, live on the user's own machine, to actively
# suppress capture_screen/read_clipboard tool use -- an isolated raw
# Ollama call with the real tool schemas and no system prompt called
# capture_screen correctly; the same call through Luna's real system
# prompt never called any tool across several real attempts. Fixed by
# telling her plainly she has these two real tools and should use them,
# while keeping the "no mouse/keyboard/code/files" boundary for
# everything else. This test exists so a future prompt edit can't
# silently reintroduce the same blanket "you can only observe and
# advise" line without a test failing -- see docs/DECISIONS.md for the
# full diagnosis.
# ---------------------------------------------------------------------------


def test_system_prompt_tells_her_to_actually_use_the_tools():
    lowered = SYSTEM_PROMPT.lower()
    assert "screenshot" in lowered
    assert "clipboard" in lowered


def test_system_prompt_no_longer_says_observe_and_advise_only():
    assert "you can only observe and advise" not in SYSTEM_PROMPT.lower()


# --- the prompt must match the tools she actually has --------------------------
# It used to promise a task tool (now withheld: the classifier owns tasks) and
# to say "call the tool again right then" on every look request, which a
# tool-happy model obeyed in addition to the look the app had already done.


def test_prompt_no_longer_promises_a_task_tool():
    lowered = SYSTEM_PROMPT.lower()
    assert "three real tools" in lowered and "four real tools" not in lowered
    assert "call the task tool" not in lowered
    assert "one starts or stops tracking a task" not in lowered


def test_prompt_says_a_look_is_usually_already_done_for_her():
    lowered = SYSTEM_PROMPT.lower()
    assert "already been done for you" in lowered
    assert "do not call a tool to look again" in lowered
    assert "call the tool again right then" not in lowered
