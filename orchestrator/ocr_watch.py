"""
Quick-action menu's Continuous OCR toggle -- ambient screen-watching,
independent of Task Guide Mode's goal-directed drift checking
(task_guide.py). Where that requires an active tracked task and checks
specifically for "has the user drifted off it," this is a general "glance
at the screen every so often and remark on anything genuinely notable"
mode -- no task, no goal, just ambient commentary. Both can run at once;
neither depends on the other.

Explicitly a lower-confidence feature by the user's own framing ("low
stakes... cuz 9b's limit") -- built lean on purpose, expecting it may need
real tuning after actual use, same as everything else vision-related in
this project. The single biggest risk with an ambient-commentary feature
is being naggy/creepy rather than charming, so every design choice here
leans toward under-commenting: a missed genuinely-interesting moment costs
nothing; an unwanted comment about someone's screen costs trust.

Single-slot, module-level state -- same reasoning as task_guide.py's
single-active-task design: this is a single-driver-connection app, so
there's never a genuine need for more than one watch state at a time.
"""

from __future__ import annotations

import json
import re
import sys
import time
from dataclasses import dataclass, replace

import llm

# Polled far less often than Task Guide Mode's own loop (15s) -- ambient
# commentary should be rare by design, not prompt-timely the way noticing
# task drift is. The actual real-world cadence is controlled by
# CONFIG.ocr_watch.comment_interval_seconds (app.py reads it), this is
# just how granular the "is a check due yet" polling is.
WATCH_POLL_SECONDS = 30


@dataclass(frozen=True)
class WatchState:
    active: bool = False
    last_check_at: float = 0.0
    # A brief, one-line gist of what was on screen at the last check --
    # not a transcript, just enough for the next check's prompt to reason
    # about "has this changed since last time" rather than judging each
    # screenshot in isolation, which would make repeat comments about an
    # unchanging screen much more likely.
    last_seen_summary: str = ""
    # When she last actually SPOKE an ambient remark (0 = never this watch).
    # Drives the minimum gap in decide_comment().
    last_comment_at: float = 0.0


_state = WatchState()


def get_state() -> WatchState:
    """Frozen snapshot -- the caller can't accidentally mutate module
    state through the object it gets back."""
    return _state


def set_active(active: bool) -> None:
    """The set_ocr_watch message's actual implementation (see app.py).
    Starting fresh always clears last_seen_summary and resets the check
    clock -- a freshly enabled watch shouldn't immediately fire using a
    stale summary from a previous session, and shouldn't be treated as
    already overdue for a check the instant it's turned on."""
    global _state
    if active:
        _state = WatchState(active=True, last_check_at=time.time(), last_seen_summary="")
    else:
        _state = WatchState(active=False)


def defer_check() -> None:
    """Restarts the check clock without touching the summary or the
    last-comment time. Called every poll while a Task Guide task is active
    (ocr_watch.pause_during_task) so that, once the task ends, the first
    ambient check is a full interval away instead of firing instantly with
    a stale summary."""
    global _state
    if _state.active:
        _state = replace(_state, last_check_at=time.time())


def mark_commented() -> None:
    """Called once an ambient remark has actually been spoken."""
    global _state
    if _state.active:
        _state = replace(_state, last_comment_at=time.time())


def due_for_check(state: WatchState, interval_seconds: float) -> bool:
    if not state.active:
        return False
    return (time.time() - state.last_check_at) >= interval_seconds


def mark_checked(summary: str = "") -> None:
    """Called right before/after a check actually runs (whether or not it
    finds something comment-worthy) -- due_for_check() measures from "last
    time we actually looked," not "last time we found something to say."
    summary, when given, becomes the next check's last_seen_summary --
    left unchanged (not cleared) if omitted, e.g. on a failed check where
    there's nothing new to record."""
    global _state
    if _state.active:
        _state = replace(_state, last_check_at=time.time(), last_seen_summary=summary or _state.last_seen_summary)


# Words that carry no information about WHAT is on screen; dropped before
# comparing two summaries.
_SUMMARY_STOPWORDS = frozenset(
    "a an the with in on of and at to is are for by its it this that as from "
    "showing shows displaying displays screen".split()
)


def _summary_tokens(text: str) -> frozenset[str]:
    return frozenset(
        t for t in re.findall(r"[a-z0-9']+", text.lower()) if len(t) > 1 and t not in _SUMMARY_STOPWORDS
    )


def summaries_match(previous: str, current: str, threshold: float) -> bool:
    """True when two screen summaries describe essentially the same thing:
    Jaccard overlap of their content words >= threshold. Deterministic on
    purpose -- the 9b was told "do not comment unless the screen changed"
    and said "worth a comment" five checks in a row about the identical
    summary ("debugging a Python app with an anime avatar"). Empty on
    either side = can't tell = not a match."""
    a, b = _summary_tokens(previous), _summary_tokens(current)
    if not a or not b:
        return False
    return len(a & b) / len(a | b) >= threshold


def decide_comment(
    result: dict,
    previous_summary: str,
    last_comment_at: float,
    min_gap_seconds: float,
    similarity_threshold: float,
    now: float | None = None,
) -> tuple[bool, str]:
    """Final say on whether a check turns into a spoken remark: (speak,
    reason-if-not). The model's comment_worthy is necessary but not
    sufficient -- code enforces the quiet rules the prompt could not.
    Order matters only for which reason gets logged."""
    if not result.get("comment_worthy"):
        return False, "model found nothing worth saying"
    if not result.get("note"):
        return False, "no observation to react to"
    if summaries_match(previous_summary, result.get("summary", ""), similarity_threshold):
        return False, "same screen as the last check"
    now = time.time() if now is None else now
    since = now - last_comment_at
    if last_comment_at and since < min_gap_seconds:
        return False, f"spoke {int(since)}s ago (minimum gap {int(min_gap_seconds)}s)"
    return True, ""


_WATCH_SYSTEM_PROMPT = """\
You are glancing at someone's screen every so often, just to notice what \
they're up to -- not tracking a specific task, not checking for \
anything in particular, just an occasional look. Respond with ONLY a \
JSON object, no markdown fences, no commentary before or after it:

{"comment_worthy": true, "summary": "one short phrase describing what is on screen right now", "note": "one short observation worth remarking on"}

or, when there is nothing worth saying:

{"comment_worthy": false, "summary": "one short phrase describing what is on screen right now", "note": ""}

"summary" is always filled in either way -- it is compared against what \
was on screen last time you looked, given below. Only set \
"comment_worthy" to true if something is genuinely interesting, has \
changed meaningfully since last time, or is funny/notable enough that a \
person glancing over someone's shoulder would actually say something out \
loud -- a new application, something that looks like trouble, a clear \
milestone or achievement, something genuinely funny. Do NOT set it true \
just because the screen has content, because enough time has passed, or \
because the summary is merely different in wording from last time with \
nothing actually new happening. Most glances should find nothing worth \
commenting on -- when in doubt, set it false. Never remark on the same \
thing you already commented on before.

What was on screen last time you looked (empty if this is the first \
check): __PREVIOUS_SUMMARY__\
"""


# Structured-output schema for the check (see task_guide.CHECK_SCHEMA).
WATCH_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "comment_worthy": {"type": "boolean"},
        "summary": {"type": "string"},
        "note": {"type": "string"},
    },
    "required": ["comment_worthy", "summary", "note"],
}


async def check_for_comment(previous_summary: str, image_b64: str) -> dict | None:
    """One-shot VLM call judging whether the current screen is worth an
    unprompted remark, given a brief note on what was there last time.
    Returns {"comment_worthy": bool, "summary": str, "note": str} or None
    if the call failed or came back unparseable -- both treated the same
    by the caller (app.py's _run_ocr_watch_check): skip this check, try
    again next interval, never crash the loop or comment on a guess.
    """
    # Plain string replacement, not str.format() -- the template above
    # contains literal JSON examples with their own {curly braces},
    # which .format() would try to interpret as format fields too (and
    # did, the first time this was written -- caught by this function's
    # own tests, not by inspection).
    prompt = _WATCH_SYSTEM_PROMPT.replace("__PREVIOUS_SUMMARY__", repr(previous_summary or "(nothing yet)"))
    try:
        raw_output = await llm.describe_image(prompt, image_b64, json_schema=WATCH_SCHEMA)
    except llm.LLMUnreachableError as exc:
        print(f"[luna] ocr watch: vision call failed ({type(exc).__name__}): {exc}", file=sys.stderr)
        return None
    result = _parse_watch_result(raw_output)
    if result is None:
        print(f"[luna] ocr watch: unparseable check output: {raw_output[:300]!r}", file=sys.stderr)
    return result


def _parse_watch_result(raw_output: str) -> dict | None:
    """Forgiving parse, same shape as task_guide.py's _parse_check_result
    and forget.py's _parse_remove_indices -- a small model's output
    shouldn't be trusted to be clean JSON and nothing else. Returns None
    on anything that doesn't parse into the expected shape -- the safe
    direction to fail in is "skip this check," not "assume it's worth a
    comment."""
    candidates = [raw_output.strip()]
    match = re.search(r"\{.*\}", raw_output, re.DOTALL)
    if match:
        candidates.append(match.group(0))

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if not isinstance(parsed, dict):
            continue
        comment_worthy = parsed.get("comment_worthy")
        if not isinstance(comment_worthy, bool):
            continue
        summary = parsed.get("summary")
        note = parsed.get("note")
        return {
            "comment_worthy": comment_worthy,
            "summary": summary.strip() if isinstance(summary, str) else "",
            "note": note.strip() if isinstance(note, str) else "",
        }
    return None
