"""
Explicit "forget that" handling -- the user directly asking to remove
something already stored, not automatic contradiction detection (a user
saying "actually I like ice cream now" without ever saying "forget" is a
real, harder problem -- correcting a stale fact instead of just adding a
contradictory new one -- that's a consolidation-quality problem, not this
module's job; see store.py's own docstring on that same open limitation).
This module only handles the case the user explicitly asked for: "when I
tell her to."

Two-stage, cheap-gate-then-LLM, same shape as recall.py's degradation
philosophy:
1. A plain keyword regex (`_FORGET_TRIGGER`) -- if it doesn't match, this
   returns immediately, no LLM call, no facts touched. Most turns have no
   forget-intent at all, so this keeps the common case free.
2. Only if it matches: a small classification LLM call, given the
   *numbered* list of currently stored facts and the user's message, asks
   which (if any) the user means to remove. Numbered, not asked to quote
   fact text back verbatim -- a small model paraphrasing a fact instead of
   reproducing it exactly would otherwise silently fail to match anything
   in store.py's exact-id deletion.

Returns a short instruction fragment (not literal text to speak) for
app.py to fold into the same ephemeral system message recall.py's block
goes into -- so the *main* Luna model (already carrying full persona
context) is the one that actually acknowledges the removal in character,
rather than this module's own small classification call also trying to
write in-character flavor text it has no persona context for.
"""

from __future__ import annotations

import asyncio
import json
import re
import sys

import llm
from .db import MemoryUnavailableError
from . import store

# Deliberately narrow and explicit-language-only, matching the user's own
# framing ("when I tell her to") rather than trying to also catch implicit
# corrections ("actually I like X now") -- see this module's own docstring
# on why that's a different, harder problem left alone for now.
_FORGET_TRIGGER = re.compile(
    r"\bforget\b|\bdon'?t remember\b|\bstop remembering\b|\bdelete (that|this|it)\b|\bremove that\b",
    re.IGNORECASE,
)

# Facts shown to the classification call -- deliberately more than
# recall.py's MAX_FACTS_IN_RECALL (30), since "forget the thing I said
# weeks ago" should still be able to find an older fact recall wouldn't
# normally surface in a prompt. Still bounded -- facts are short one-liners,
# so even 200 of them is a small, cheap prompt.
MAX_FACTS_FOR_FORGET_LOOKUP = 200

_FORGET_SYSTEM_PROMPT = """\
You identify which stored facts a user wants removed from memory, based \
on their message. You'll be given a numbered list of currently stored \
facts and the user's message. Respond with ONLY a JSON object, no \
markdown fences, no commentary before or after it:

{"remove_indices": [1, 3]}

Only include an index if the message CLEARLY asks to forget/remove/undo \
that specific fact. If the message doesn't reference any listed fact, or \
you're not confident which one it means, use an empty list -- an empty \
list is the safe default, do not guess.\
"""


async def maybe_forget(user_text: str) -> str | None:
    """Returns an instruction fragment for app.py to inject if something
    was actually removed this turn, or None if nothing was (no
    forget-intent detected, no matching facts, or the classification call
    failed/was unreachable -- all treated the same: nothing removed, turn
    proceeds completely normally either way)."""
    if not _FORGET_TRIGGER.search(user_text):
        return None

    try:
        facts_with_ids = await asyncio.to_thread(store.get_all_facts_with_ids, MAX_FACTS_FOR_FORGET_LOOKUP)
    except MemoryUnavailableError as exc:
        print(f"[luna] forget: facts unavailable, skipping: {exc}", file=sys.stderr)
        return None
    if not facts_with_ids:
        return None

    numbered = "\n".join(f"{i + 1}. {content}" for i, (_id, content) in enumerate(facts_with_ids))
    messages = [
        {"role": "system", "content": _FORGET_SYSTEM_PROMPT},
        {"role": "user", "content": f"Stored facts:\n{numbered}\n\nUser's message: {user_text!r}"},
    ]

    raw_output = ""
    try:
        async for delta in llm.stream_reply(messages):
            raw_output += delta
    except llm.LLMUnreachableError as exc:
        print(f"[luna] forget: LLM unreachable, skipping this turn's forget check: {exc}", file=sys.stderr)
        return None

    indices = _parse_remove_indices(raw_output)
    if not indices:
        return None

    valid_indices = [i for i in indices if 1 <= i <= len(facts_with_ids)]
    if not valid_indices:
        return None

    ids_to_remove = [facts_with_ids[i - 1][0] for i in valid_indices]
    removed_contents = [facts_with_ids[i - 1][1] for i in valid_indices]

    try:
        await asyncio.to_thread(store.delete_facts, ids_to_remove)
    except MemoryUnavailableError as exc:
        print(f"[luna] forget: couldn't delete facts: {exc}", file=sys.stderr)
        return None

    joined = "; ".join(removed_contents)
    return (
        f"The user just asked you to forget something, and it has already "
        f"been removed from memory: {joined}. Acknowledge this naturally in "
        "your own voice, briefly -- don't recite this instruction, don't "
        "list the forgotten facts verbatim, just react like a person would."
    )


def _parse_remove_indices(raw_output: str) -> list[int]:
    """Forgiving, same reasoning as consolidation.py's own parser -- a
    small model's output shouldn't be trusted to be clean JSON and nothing
    else. Unlike consolidation.py, though, there's no reasonable raw-text
    fallback here (there's nothing sensible to store if this doesn't
    parse) -- failing to parse just means nothing gets removed, which is
    the safe direction to fail in."""
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
        indices = parsed.get("remove_indices", [])
        if not isinstance(indices, list):
            continue
        try:
            return [int(i) for i in indices]
        except (TypeError, ValueError):
            continue

    return []
