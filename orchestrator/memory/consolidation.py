"""
The distillation half of the writer pass: turns a chunk of transcript into
candidate facts plus one episode summary, per the memory design in
docs/SPEC-v0.1.md: "never dump raw message logs into long-term memory --
distill, don't transcribe."

One LLM call, asked for only a JSON object:

    {"facts": [{"text": "...", "subject": "user", "correction": false}],
     "episode_summary": "..."}

The configured chat model is small and not reliable at structured output, so
parsing is forgiving rather than a strict `json.loads` that discards a whole
chunk over one stray sentence. A bare string in `facts` is accepted as a fact
with no subject, and if nothing parses the raw text becomes the episode summary
so the chunk is not a total loss.

Writing the results (dedupe, contradictions, checkpoint) is writer.py's job.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field

import router

_SYSTEM_PROMPT = """\
You distill a conversation transcript between a user and their AI \
companion into durable memory. Respond with ONLY a single JSON object, \
no markdown fences, no commentary before or after it, in exactly this \
shape:

{"facts": [{"text": "short standalone fact about the user", "subject": "user", "correction": false}], "episode_summary": "one or two sentence summary of what happened"}

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
in unrelated conversations for weeks.

"subject" is "user" for facts about the person, "machine" for their \
computer or setup, or the name of a specific other person or project the \
fact is about. "correction" is true ONLY when the user explicitly changed \
or retracted something they said before ("actually I...", "I no longer...", \
"that's wrong, it's..."); otherwise false. If nothing durable came up, use \
an empty facts list. The episode_summary should read like a memory of \
what happened, not a transcript -- brief enough to recall at a glance \
later.\
"""


@dataclass
class FactCandidate:
    text: str
    subject: str | None = None
    correction: bool = False


@dataclass
class Distilled:
    facts: list[FactCandidate] = field(default_factory=list)
    episode_summary: str = ""


async def distill(transcript: str) -> Distilled | None:
    """Returns the distilled chunk, or None if the model could not be reached
    (the caller must then leave its checkpoint where it is and retry later)."""
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": transcript},
    ]
    raw = ""
    try:
        async for delta in router.stream_text("memory_writer", messages):
            raw += delta
    except Exception as exc:  # noqa: BLE001 -- a background pass must never take the app down
        print(f"[yuri] writer: model unreachable, will retry next pass: {exc}", file=sys.stderr)
        return None
    return parse_output(raw)


def _fact_from(item) -> FactCandidate | None:
    if isinstance(item, dict):
        text = str(item.get("text", "")).strip()
        subject = str(item.get("subject") or "").strip() or None
        return FactCandidate(text, subject, item.get("correction") is True) if text else None
    text = str(item).strip()
    return FactCandidate(text) if text else None


def parse_output(raw_output: str) -> Distilled:
    """Falls back, in order: the whole output as JSON; the first {...} block
    (markdown fences, a stray sentence); then the raw text, truncated, as the
    episode summary with no facts."""
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
        raw_facts = parsed.get("facts", [])
        facts = [f for f in (_fact_from(i) for i in (raw_facts if isinstance(raw_facts, list) else [])) if f]
        return Distilled(facts, str(parsed.get("episode_summary", "")).strip())

    print(
        f"[yuri] writer: model output didn't parse as the expected JSON shape, falling back to "
        f"storing it as a raw episode summary: {raw_output[:300]!r}",
        file=sys.stderr,
    )
    return Distilled([], raw_output.strip()[:500])
