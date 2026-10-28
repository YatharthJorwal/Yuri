"""
Phase 4 Round 2 -- Task Guide Mode's other half (docs/ARCHITECTURE.md's
"Task Guide Mode" section): tracking whether a task is currently active,
and checking a screenshot against it. The actual scheduling (waking up
on an interval, deciding when to run a check, speaking the chide) lives
in app.py's `_task_guide_loop`/`_run_task_guide_check` -- this module is
just the state and the one LLM call that turns a screenshot + task
description into an on-task/off-task verdict, kept separate so both
halves stay independently testable (same split as vision.py's
capture_screen/describe_screen).

Single active task at a time, tracked as plain module-level state --
same pattern as app.py's `_driver`/`_scene_state` globals. This is a
single-user, single-driver-connection app (see app.py's driver/observer
comment), so there's never a real need for more than one "what am I
tracking right now" slot; a second concurrent task genuinely doesn't
make sense for one person at one screen.
"""

from __future__ import annotations

import json
import re
import sys
import time
from dataclasses import dataclass, replace

import llm


@dataclass(frozen=True)
class TaskState:
    active: bool = False
    description: str = ""
    # When set_active_task() last actually changed something -- not used
    # for idle detection (that's last_interaction_at below), just useful
    # for debugging/logging.
    updated_at: float = 0.0
    # Bumped by mark_interaction() on every real user turn (see app.py's
    # _run_turn), regardless of whether that turn was about the tracked
    # task at all -- any real interaction means the user is still there.
    # is_idle() compares against this, not updated_at, so a task that's
    # been tracked a long time but the user keeps actively talking never
    # times out just because the *step itself* hasn't changed.
    last_interaction_at: float = 0.0
    # Bumped by mark_checked() every time a screen check actually runs
    # (whether or not it found drift) -- due_for_check() compares against
    # this so the interval is measured from the last real check, not from
    # when the task was first set.
    last_check_at: float = 0.0


_state = TaskState()


def get_state() -> TaskState:
    """Frozen snapshot -- the caller can't accidentally mutate module
    state through the object it gets back."""
    return _state


def set_active_task(active: bool, description: str = "") -> str:
    """The set_active_task tool's actual implementation (see
    tools/__init__.py for the schema/handler wiring). Returns a short
    plain-string confirmation for the tool result the model sees --
    never raises; there's no real failure mode here beyond "the
    description was empty," handled by just falling back to a generic
    label rather than refusing.

    Starting a new task (active=True) always resets last_interaction_at
    and last_check_at to now -- a freshly stated task shouldn't be
    treated as already overdue for a check the instant it's set, and the
    user just talked about it, so the idle clock should start fresh too.
    """
    global _state
    now = time.time()
    if active:
        clean_description = description.strip() or "an unspecified task"
        _state = TaskState(
            active=True,
            description=clean_description,
            updated_at=now,
            last_interaction_at=now,
            last_check_at=now,
        )
        return f"Task tracking started: now watching for progress on '{clean_description}'."
    _state = TaskState(active=False, description="", updated_at=now)
    return "Task tracking stopped."


def clear_active_task() -> None:
    """Same effect as set_active_task(False), but for app.py's own
    internal idle-timeout path (_task_guide_loop) rather than a model
    tool call -- no confirmation string needed there, nothing reads it."""
    global _state
    _state = TaskState(active=False, description="", updated_at=time.time())


def mark_interaction() -> None:
    """Called from app.py's _run_turn on every real user turn. A no-op
    if no task is active -- nothing to reset the idle clock on."""
    global _state
    if _state.active:
        _state = replace(_state, last_interaction_at=time.time())


def mark_checked() -> None:
    """Called right before/after a screen check actually runs (whether
    or not it finds drift) -- due_for_check() measures the interval from
    this, not from when the task was set."""
    global _state
    if _state.active:
        _state = replace(_state, last_check_at=time.time())


def is_idle(state: TaskState, idle_timeout_seconds: float) -> bool:
    if not state.active:
        return False
    return (time.time() - state.last_interaction_at) >= idle_timeout_seconds


def due_for_check(state: TaskState, capture_interval_seconds: float) -> bool:
    if not state.active:
        return False
    return (time.time() - state.last_check_at) >= capture_interval_seconds


_CHECK_SYSTEM_PROMPT = """\
You are checking whether a computer screen matches what someone is \
supposed to be working on. You will be given their current task/step \
and a screenshot. Respond with ONLY a JSON object, no markdown fences, \
no commentary before or after it:

{"on_task": true, "note": "one short sentence on what you actually see"}

Be conservative: if the screen plausibly relates to the task, or you \
genuinely cannot tell what is on screen, set on_task to true. Only set \
it to false when the screen is clearly something unrelated -- a \
different app or website that has nothing to do with the stated task \
(e.g. a game, unrelated social media, an unrelated video). A blank \
screen, a lock screen, or an unreadable capture should also be treated \
as on_task true -- there is no real evidence of drift there, just \
missing information.\
"""


# Sent as Ollama's structured-output `format` so the check always comes back as
# parseable JSON (llm.describe_image falls back to prompt-only if the server
# rejects it). 15 of 16 checks on the abliterated GGUF "didn't parse" without it.
CHECK_SCHEMA: dict = {
    "type": "object",
    "properties": {"on_task": {"type": "boolean"}, "note": {"type": "string"}},
    "required": ["on_task", "note"],
}


async def check_task_progress(task_description: str, image_b64: str) -> dict | None:
    """One-shot VLM call comparing a screenshot against the tracked
    task, via llm.describe_image (same entry point vision.py's
    describe_screen uses) with a different, comparison-focused prompt.
    Returns {"on_task": bool, "note": str} or None if the call failed or
    came back unparseable -- both treated the same by the caller
    (app.py's _run_task_guide_check): skip this check, try again next
    interval, never crash the loop or chide on a guess.

    **Not verified against a real server** -- same caveat as every other
    describe_image/stream_reply_with_tools call in this project (see
    llm.py's own docstrings): this exact prompt shape has never actually
    been sent to qwen3.5:9b.
    """
    prompt = (
        f"{_CHECK_SYSTEM_PROMPT}\n\nCurrent task/step: {task_description!r}"
    )
    try:
        raw_output = await llm.describe_image(prompt, image_b64, json_schema=CHECK_SCHEMA)
    except llm.LLMUnreachableError as exc:
        print(f"[luna] task guide: vision call failed ({type(exc).__name__}): {exc}", file=sys.stderr)
        return None
    result = _parse_check_result(raw_output)
    if result is None:
        # The raw text is the only way to tell "the model wrote prose" from
        # "the model wrote JSON with the wrong shape" -- it used to be dropped.
        print(f"[luna] task guide: unparseable check output: {raw_output[:300]!r}", file=sys.stderr)
    return result


def _parse_check_result(raw_output: str) -> dict | None:
    """Forgiving, same reasoning/shape as forget.py's
    _parse_remove_indices and consolidation.py's own JSON parsing -- a
    small model's output shouldn't be trusted to be clean JSON and
    nothing else. Returns None (not a guessed default) on anything that
    doesn't parse into the expected shape -- the safe direction to fail
    in is "skip this check," not "assume drift" or "assume on task"."""
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
        on_task = parsed.get("on_task")
        if not isinstance(on_task, bool):
            continue
        note = parsed.get("note")
        if not isinstance(note, str):
            note = ""
        return {"on_task": on_task, "note": note.strip()}
    return None


# ---------------------------------------------------------------------------
# Programmatic start/stop detection -- bypasses Ollama's native tool-calling
# ---------------------------------------------------------------------------
#
# Originally, starting/stopping task tracking was left entirely to the
# model's own tool-calling (set_active_task, see tools/__init__.py) --
# the same mechanism capture_screen/read_clipboard already use. Real
# testing showed this doesn't work reliably for *this* tool specifically:
# even on an exact, explicit trigger phrase ("I'm going to work on X,
# keep an eye on me"), qwen3.5:9b never emitted a tool_calls field at
# all across every test turn (confirmed via llm.py's own diagnostic
# logging). Initially suspected as an Ollama-side bug affecting Qwen
# 3.5's tool-calling format specifically (a real, documented bug --
# github.com/ollama/ollama#14493 and related issues -- fixed upstream as
# of Ollama v0.17.6), but the user confirmed running a *newer* Ollama
# (0.34.2) and *also* saw the same 9B model fail to call a tool in a
# completely different agent framework on a direct, explicit command
# ("open a browser"). That combination -- fixed-Ollama-version plus
# cross-framework failure on an explicit ask -- points at a genuine
# capability ceiling for this model size on agentic tool-use, not
# something more prompting can fix.
#
# So: don't fight it. `set_active_task` (the tool) is left in place as a
# redundant path in case the model does call it sometimes, but the
# *reliable* path is this module's own narrow classification call --
# structurally identical to forget.py's maybe_forget() and
# consolidation.py's session distillation, both of which already work
# fine with this same model, because neither depends on Ollama's
# tool-calling machinery at all: just a system prompt asking for a
# small JSON object, and a forgiving parse. Unlike forget.py, there's no
# cheap keyword pre-filter here -- task-starting phrasing is too varied
# for a narrow regex to catch reliably (unlike the word "forget"), and
# given Task Guide Mode is the product's flagship behavior, one extra
# short classification call per turn is a reasonable reliability trade,
# not a wasteful one.

_TASK_DETECT_SYSTEM_PROMPT = """\
You detect whether a message means the user is starting, continuing, or \
stopping a task they want their AI companion to keep an eye on -- \
distinct from just chatting, asking a question, or mentioning something \
in passing. Respond with ONLY a JSON object, no markdown fences, no \
commentary before or after it, in exactly one of these three shapes:

{"action": "start", "description": "short concrete description of the task or step"}
{"action": "stop"}
{"action": "none"}

Use "start" only when the message says the USER themselves is about to \
do, or is currently doing, some piece of work or a project -- e.g. "I'm \
going to debug this function", "working on a poster for a bit", "keep an \
eye on me while I do X". "description" should be a short, concrete \
phrase about THEIR work, not a full sentence. A request or command \
directed at the AI itself is NOT a task, no matter how it is phrased: \
"use OCR", "look at my screen", "check what's on screen", "see what I'm \
playing", "what am I holding", "what time is it", "read this", or \
telling it to use a tool are all "none". Merely mentioning what they are \
playing or watching ("I'm playing Minecraft", "I'm watching a video") is \
also "none" -- that is a hobby or leisure, not something they asked to \
be kept on track for. Use "stop" only when the message says the task is \
finished, asks to stop being watched, or clearly drops/abandons it -- \
e.g. "I'm done", "stop watching me", "never mind that". A message about \
doing something else for a bit (checking a video, taking a break) is NOT \
a stop -- that is a normal distraction, not the task ending, so use \
"none" for it. Use "none" for everything else, including ordinary \
conversation and questions. When genuinely unsure, use "none" -- a \
missed task is a smaller problem than falsely claiming to track one, or \
falsely ending a real one that's still in progress.\
"""


async def maybe_update_task(user_text: str) -> str | None:
    """Runs every turn (see this section's own comment above for why
    there's no cheap keyword gate first, unlike forget.maybe_forget).
    Returns an instruction fragment for app.py to inject into the same
    ephemeral memory-block message forget.py's hint goes into, or None
    if nothing changed this turn -- covers "classified as none,"
    "classified as stop but nothing was actually active," and any
    failure (LLM unreachable, unparseable output), all treated the same:
    no state change, turn proceeds completely normally either way."""
    messages = [
        {"role": "system", "content": _TASK_DETECT_SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
    ]
    raw_output = ""
    try:
        async for delta in llm.stream_reply(messages):
            raw_output += delta
    except llm.LLMUnreachableError as exc:
        print(f"[luna] task guide: detection LLM unreachable, skipping: {exc}", file=sys.stderr)
        return None

    parsed = _parse_task_detection(raw_output)
    if parsed is None or parsed["action"] == "none":
        return None

    if parsed["action"] == "start":
        description = parsed["description"] or "an unspecified task"
        set_active_task(True, description)
        print(f"[luna] task guide: detected task start -> '{description}'", file=sys.stderr)
        return (
            f"The user just described a task, and it is now being "
            f"tracked: {description!r}. You do not need to explicitly "
            "announce that you are tracking it -- just react naturally "
            "to what they said, in character."
        )

    # action == "stop"
    if not get_state().active:
        return None
    set_active_task(False)
    print("[luna] task guide: detected task stop", file=sys.stderr)
    return (
        "The user just indicated the task you were tracking is done, "
        "dropped, or paused, and tracking has been stopped. React "
        "naturally in character -- do not explicitly narrate that "
        "tracking stopped."
    )


def _parse_task_detection(raw_output: str) -> dict | None:
    """Forgiving, same shape as _parse_check_result above and forget.py's
    _parse_remove_indices."""
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
        action = parsed.get("action")
        if action not in ("start", "stop", "none"):
            continue
        description = parsed.get("description")
        if not isinstance(description, str):
            description = ""
        return {"action": action, "description": description.strip()}
    return None
