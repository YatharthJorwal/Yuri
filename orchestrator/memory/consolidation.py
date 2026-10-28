"""
Runs once per session, at disconnect (app.py's `finally`) -- distills the
session's real conversation turns into durable memory: candidate facts,
plus one episode summary. Per docs/ARCHITECTURE.md's memory design:
"never dump raw message logs into long-term memory -- distill, don't
transcribe."

One extra LLM call, asked to respond with *only* a JSON object:

    {"facts": ["...", "..."], "episode_summary": "..."}

`qwen3.5:9b` is a small model without much reliable structured-output
discipline (same caveat docs/DECISIONS.md already names for why the
Phase 8 emotion system plans to hardcode some triggers rather than trust
a small model's self-reported structured state) -- so parsing here is
deliberately forgiving rather than a strict `json.loads` that throws the
whole session's memory away over one stray sentence before or after the
JSON block. See `_parse_consolidation_output()`.

Degrades independently per piece, same philosophy as recall.py:
- LLM unreachable at session end -> nothing gets distilled, session's
  memory is lost (there's no raw transcript kept to retry from later --
  an accepted limitation, not a bug, per the "distill don't transcribe"
  rule above). Logged, not raised -- a session ending because the user
  closed the app shouldn't ever be blocked by this.
- Facts still get written even if the episode's embedding call fails
  (embedding server down) -- only the episode summary is lost in that
  case, not the facts.
"""

from __future__ import annotations

import json
import re
import sys

import llm
from .db import MemoryUnavailableError
from .embeddings import EmbeddingUnreachableError, embed
from . import store

_CONSOLIDATION_SYSTEM_PROMPT = """\
You distill a conversation transcript between a user and their AI \
companion into durable memory. Respond with ONLY a single JSON object, \
no markdown fences, no commentary before or after it, in exactly this \
shape:

{"facts": ["short standalone fact about the user", ...], "episode_summary": "one or two sentence summary of what happened this session"}

Facts are durable things worth remembering long-term and bringing up in \
*future, unrelated* conversations: stated preferences, identity details, \
and projects or tools the user is actually working with on an ongoing \
basis. The test for each one: would it still make sense to casually \
mention this back to the user weeks from now, in a conversation about \
something else entirely? A single passing mention of a tool, app, \
game, or website -- named once, in the context of that session, never \
described as something the user actually uses or works on regularly -- \
fails that test and should be skipped, even though it might look like \
"a tool mentioned." For example: the user saying "let me pull up VRoid \
real quick" during an unrelated session is not durable; the user saying \
"I've been using VRoid for my avatar project" is. When genuinely unsure \
whether something clears this bar, leave it out -- an incomplete fact \
list is a much smaller problem than a stale one-off mention resurfacing \
in unrelated conversations for weeks. If nothing durable came up, use \
an empty facts list. The episode_summary should read like a memory of \
what happened, not a transcript -- brief enough to recall at a glance \
later.\
"""


async def consolidate_session(history: list[dict[str, str]]) -> None:
    """`history` matches the shape app.py keeps per connection -- index 0
    is the persona system prompt (skipped here, it's not part of what
    happened), the rest are the real user/assistant turns. May be a
    filtered copy rather than the literal same list object (app.py
    excludes Temp Chat turns before calling this) -- doesn't matter here,
    this function only ever reads it."""
    turns = history[1:]
    if not turns:
        return  # nothing actually happened this session -- nothing to distill

    transcript = _format_transcript(turns)
    consolidation_messages = [
        {"role": "system", "content": _CONSOLIDATION_SYSTEM_PROMPT},
        {"role": "user", "content": transcript},
    ]

    raw_output = ""
    try:
        async for delta in llm.stream_reply(consolidation_messages):
            raw_output += delta
    except llm.LLMUnreachableError as exc:
        print(f"[luna] consolidation: LLM unreachable, session's memory is lost: {exc}", file=sys.stderr)
        return

    facts, episode_summary = _parse_consolidation_output(raw_output)

    try:
        for fact in facts:
            store.add_fact(fact)
    except MemoryUnavailableError as exc:
        print(f"[luna] consolidation: couldn't write facts: {exc}", file=sys.stderr)

    if not episode_summary:
        return

    try:
        embedding = await embed(episode_summary)
    except EmbeddingUnreachableError as exc:
        print(
            f"[luna] consolidation: embedding unavailable, episode summary "
            f"not stored (facts above were still written if any): {exc}",
            file=sys.stderr,
        )
        return

    try:
        store.add_episode(episode_summary, embedding)
    except MemoryUnavailableError as exc:
        print(f"[luna] consolidation: couldn't write episode: {exc}", file=sys.stderr)


def _format_transcript(turns: list[dict[str, str]]) -> str:
    lines = []
    for turn in turns:
        speaker = "User" if turn["role"] == "user" else "Luna"
        lines.append(f"{speaker}: {turn['content']}")
    return "\n".join(lines)


def _parse_consolidation_output(raw_output: str) -> tuple[list[str], str]:
    """Deliberately forgiving, not a strict json.loads() -- see this
    module's docstring on why a small model's output shouldn't be trusted
    to be clean JSON and nothing else. Falls back, in order:
    1. Parse the whole trimmed output as JSON.
    2. Extract the first {...} block (in case the model wrapped it in
       markdown fences or added a stray sentence) and parse that.
    3. Give up on structure entirely -- store the raw output itself
       (truncated) as the episode summary with no facts, so the session
       isn't a total loss just because the model didn't follow the format.
    """
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
        facts = [str(f).strip() for f in parsed.get("facts", []) if str(f).strip()]
        summary = str(parsed.get("episode_summary", "")).strip()
        return facts, summary

    print(
        f"[luna] consolidation: model output didn't parse as the expected "
        f"JSON shape, falling back to storing it as a raw episode summary: "
        f"{raw_output[:300]!r}",
        file=sys.stderr,
    )
    return [], raw_output.strip()[:500]
