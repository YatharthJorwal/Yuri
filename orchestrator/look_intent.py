"""
Explicit "look at X" requests, handled deterministically instead of leaving
them to the model's own native tool-calling.

Real usage (a pasted orchestrator.log) showed capture_screen firing for
only 2 of ~9 requests that were plainly asking her to look -- "see what i
am playing", "look at my screen", "use OCR", "use your damn tools to see",
"whats on my screen", "tell me what do you see now". Every miss was a
`replied directly` with no tool_calls key at all, and each one meant she
answered from stale context or made something up (an invented clock
reading, a description of a scene the user had already changed). Same
failure class as set_active_task (task_guide.py) and the camera
re-invocation gap: qwen3.5:9b's autonomous decision to call a tool is not
reliable enough to depend on.

So, same approach as task_guide.maybe_update_task: take the decision away
from the model for the cases where the user's intent is unambiguous. A
regex gate (no extra LLM call, so no per-turn latency cost on ordinary
turns) detects an explicit look-request, this module does the capture
itself in Python via the same describe_screen()/describe_camera() the
tools already wrap, and app.py's _run_turn splices the result in as
ephemeral context -- exactly how forget/task/recall hints already work.
Phrasings the patterns don't catch still fall through to the model's native
tool-calling, which does work some of the time; this only adds a reliable
path for the common explicit ones, it doesn't remove the fallback.

The patterns are deliberately conservative -- a false positive means an
unwanted screen capture (or, worse, an unwanted *camera* capture), which
costs more than a missed phrase costs. In particular the camera only
triggers on look-verb + camera phrasings or "what am I holding/wearing",
never on the bare word "camera", so "turn the camera off" can't capture a
frame.
"""

from __future__ import annotations

import re
import sys
import time

import camera
import llm
from tools import vision

# Explicit screen requests. Objects limited to screen/monitor/display/
# desktop/wallpaper (not "window"/"tab") -- those are common enough in
# ordinary sentences ("I see a window") to cause false positives.
_SCREEN_RE = re.compile(
    r"""
      \b(?:look|see|check|read|show|watch)\b[^.?!]{0,30}\b(?:screen|monitor|display|desktop|wallpaper)\b
    | \bwhat(?:['’]?s|\ is|\ am\ i|\ are\ we)\b[^.?!]{0,20}\b(?:on\ (?:my\ |the\ )?(?:screen|monitor|display)|playing|watching)\b
    | \bsee\b[^.?!]{0,10}\bi(?:['’]?m|\ am)\b[^.?!]{0,6}\b(?:playing|doing|watching|working)\b
    | \buse\b[^.?!]{0,15}\b(?:ocr|tools?|eyes|vision)\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Camera requests. Never the bare word "camera" -- see module docstring.
_CAMERA_RE = re.compile(
    r"""
      \b(?:look|see|check|use|through)\b[^.?!]{0,15}\b(?:camera|webcam)\b
    | \b(?:look\ at|see|watch)\ me\b
    | \b(?:what|see|tell\ me)\b[^.?!]{0,25}\b(?:holding|wearing)\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

# "Look again" style requests that don't name a target -- resolved to
# whatever was looked at last (default: the screen).
_GENERIC_RE = re.compile(
    r"""
      \b(?:what\ do\ you\ see|what\ can\ you\ see|what\ u\ see|tell\ me\ what\ you\ see
        |look\ again|check\ again|look\ at\ it\ again|see\ it\ again|take\ another\ look
        |what\ am\ i\ looking\ at)\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

# A bare trailing "look" ("i am playing minecraft look") is a request to
# look at the screen, but "look" is common speech, so it only counts at the
# very end of a short message and never after a negation or a first-person
# "I'll/let me" (the user looking, not asking her to).
_TRAILING_LOOK_RE = re.compile(r"\b(?:take\s+a\s+|have\s+a\s+)?look\W*$", re.IGNORECASE)
_TRAILING_LOOK_GUARD_RE = re.compile(
    r"(?:n['’]?t|not|never|\bno|\bdont|\bcant|\bwont|i['’]?ll|\bill|i\s+will|let\s+me|lemme"
    r"|we['’]?ll|i['’]?m\s+(?:gonna|going\s+to)|gonna|going\s+to)\s*$",
    re.IGNORECASE,
)
_TRAILING_LOOK_MAX_CHARS = 100

# Bare follow-ups that only make sense right after she looked ("now what is
# it", "and now?", "what about now"). Matched against the WHOLE message and
# only inside the follow-up window, and only when the last look was the
# screen -- a stray "now?" must never reach for the camera.
_FOLLOWUP_RE = re.compile(
    r"""
    ^\W*(?:(?:and|ok|okay|so|well|hm+|hey)\W+)?
    (?:
        now(?:\W+what(?:['’]?s|\ is)\ (?:it|that|this))?
      | what(?:['’]?s|\ is)\ (?:it|that|this)(?:\ now)?
      | (?:what|how)\ about\ now
    )\W*$
    """,
    re.IGNORECASE | re.VERBOSE,
)
FOLLOWUP_WINDOW_SECONDS = 240

# "check its not a phone", "verify that", "check this" right after she looked:
# an explicit verification verb aimed at what was just looked at, so it
# re-looks the SAME target, camera included (a bare "now?" still never reaches
# for the camera -- only an explicit verb does). Deliberately not bare "look"
# / "see": "look, it's late" and "see it's fine" are ordinary speech.
_VERIFY_FOLLOWUP_RE = re.compile(
    r"""
    ^\W*(?:(?:ok(?:ay)?|no|so|wait|hey)\W+)?
    (?:double[-\ ]check|check|verify|confirm)\b
    [^.?!]{0,40}?\b(?:it['’]?s?|that|this|again|now)\b
    """,
    re.IGNORECASE | re.VERBOSE,
)
_VERIFY_FOLLOWUP_MAX_CHARS = 70

_last_target = "screen"
_last_look_at = 0.0  # time.monotonic() of the last look she acted on


def detect_look_target(user_text: str, now: float | None = None) -> str | None:
    """"screen", "camera", or None. Pure and synchronous -- no side effects
    (the last-target memory is only updated by maybe_look, once it actually
    acts on a detection). `now` (monotonic seconds) is injectable for tests."""
    # Uploaded files/images arrive as a synthetic "(shared a ...)" user_text
    # whose body can be anything -- a code file that happens to contain
    # "look at the screen" must not trigger a capture.
    if user_text.lstrip().startswith("(shared a"):
        return None
    if _CAMERA_RE.search(user_text):
        return "camera"
    if _SCREEN_RE.search(user_text):
        return "screen"
    if _GENERIC_RE.search(user_text):
        return _last_target
    if len(user_text) <= _TRAILING_LOOK_MAX_CHARS:
        trailing = _TRAILING_LOOK_RE.search(user_text.strip())
        if trailing and not _TRAILING_LOOK_GUARD_RE.search(user_text.strip()[: trailing.start()]):
            return "screen"
    now = time.monotonic() if now is None else now
    recent = bool(_last_look_at) and now - _last_look_at <= FOLLOWUP_WINDOW_SECONDS
    if recent and _last_target == "screen" and _FOLLOWUP_RE.match(user_text.strip()):
        return "screen"
    if (
        recent
        and len(user_text) <= _VERIFY_FOLLOWUP_MAX_CHARS
        and _VERIFY_FOLLOWUP_RE.match(user_text.strip())
    ):
        return _last_target
    return None


async def maybe_look(user_text: str, websocket) -> str | None:
    """If user_text is an explicit look-request, does the capture and
    returns an instruction fragment for _run_turn's ephemeral context
    block; None if nothing was asked for. Never raises for an expected
    failure (capture failed, camera not armed, vision model unreachable) --
    those become a hint telling her to say so plainly instead of guessing,
    which is the whole failure this exists to prevent."""
    global _last_target, _last_look_at
    target = detect_look_target(user_text)
    if target is None:
        return None
    _last_target = target
    _last_look_at = time.monotonic()

    where = "through their webcam" if target == "camera" else "at their screen"
    try:
        if target == "camera":
            description = await camera.describe_camera(websocket)
        else:
            description = await vision.describe_screen()
    except vision.ToolUnavailableError as exc:
        print(f"[luna] look: {target} capture failed: {exc}", file=sys.stderr)
        extra = (
            " If it was the camera, they probably need to turn it on from "
            "their quick-action menu."
            if target == "camera"
            else ""
        )
        return (
            f"The user just asked you to look {where}, but it failed: "
            f"{exc}. Tell them plainly, in character, that you couldn't "
            f"see it.{extra} Do NOT make up or guess what you would have seen."
        )
    except llm.LLMUnreachableError as exc:
        print(f"[luna] look: vision model unreachable: {exc}", file=sys.stderr)
        return (
            f"The user just asked you to look {where}, but your vision "
            "isn't working right now. Tell them plainly, in character, "
            "that you couldn't see it. Do NOT make up or guess what you "
            "would have seen."
        )

    # 300 chars, not 80: the short version made it impossible to tell
    # whether a wrong-sounding reply was the vision model's fault or hers.
    print(f"[luna] look: {target} -> {description[:300]!r}", file=sys.stderr)
    return (
        f"You just looked {where} right now, because they asked. What you "
        f"saw: {description} Begin your reply by naming, accurately and in "
        "your own words, the main thing it shows (what they are holding, or "
        "what is on screen), then react in character. Answer using only this "
        "-- do not add details it doesn't mention, do not swap in a more "
        "likely object, do not invent readings like a clock time it doesn't "
        "show, and do not say you looked at any other moment. You have "
        "already looked this turn and cannot look again until they ask."
    )
