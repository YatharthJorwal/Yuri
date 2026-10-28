"""
Keeps every LLM call inside the model's context window, and keeps unprompted
comments from flooding history.

Why this exists (docs/DECISIONS.md, "Reply cutoffs were the context window"):
replies were ending after 24-132 characters with done_reason='length' while
llm.max_tokens (512) was nowhere near reached. The window was full. A turn's
prompt is the persona system prompt (~9 KB) + tool schemas (~2.7 KB) + the
recall/hint block + up to 40 history messages, and Continuous OCR and Task
Guide were each adding one assistant message per check. `_trim_history` only
counts messages, never size, so nothing noticed the window filling.

Two independent guards live here, both pure functions so they are testable
without a server:

- fit_messages(): at call time, drops the OLDEST droppable history messages
  from the copy being sent until the estimated prompt plus a reserved reply
  fits num_ctx. System messages (the persona prompt and the per-turn
  memory/hint block) and the final message are never dropped. `history`
  itself is untouched, so end-of-session consolidation still sees the whole
  conversation.
- cap_unprompted(): permanently removes all but the newest few unprompted
  assistant turns (Task Guide chides, OCR remarks) from `history`. They add
  little, and stacked back-to-back with no user turn between them they made
  the model reply to a user who never spoke.

Token counts are an estimate (chars / 3.5, deliberately a bit pessimistic for
English) because nothing here may call the model just to count. Ollama's real
prompt_eval_count is logged per turn by llm.py, which is how to check it.
"""

from __future__ import annotations

import json
import math
from typing import Any

CHARS_PER_TOKEN = 3.5
MESSAGE_OVERHEAD_TOKENS = 6
# Slack for the chat template's own markup and for estimate error.
SAFETY_TOKENS = 400


# Self-calibration. chars/3.5 over-counts for this model (a real session
# measured ~4.1-4.4 chars per token on her replies), which made the fit drop
# history while the real prompt was only ~66% of the window. Ollama reports
# the true prompt size on every reply, so llm.py feeds (estimate, actual)
# pairs to observe() and the estimate is scaled by a smoothed ratio. Clamped
# so one odd sample (a big image, a huge tool result) can never make the
# estimate wildly optimistic; it can only ever shave off the over-count.
MIN_RATIO = 0.7
MAX_RATIO = 1.05
_RATIO_SMOOTHING = 0.3
_ratio = 1.0
_have_sample = False


def calibration_ratio() -> float:
    return _ratio


def reset_calibration() -> None:
    global _ratio, _have_sample
    _ratio, _have_sample = 1.0, False


def observe(estimated_tokens: int, actual_tokens: int | None) -> None:
    """Records one real prompt size against the raw (uncalibrated) estimate
    for the same messages. Ignores nonsense (no actual count, tiny prompts)."""
    global _ratio, _have_sample
    if not actual_tokens or estimated_tokens < 200:
        return
    sample = min(MAX_RATIO, max(MIN_RATIO, actual_tokens / estimated_tokens))
    if not _have_sample:
        _ratio, _have_sample = sample, True
    else:
        _ratio = (1 - _RATIO_SMOOTHING) * _ratio + _RATIO_SMOOTHING * sample


def estimate_tokens(text: str) -> int:
    """Raw chars/3.5 estimate -- NOT calibrated (calibration is applied once,
    to the totals in fit_messages, so observe() compares like with like)."""
    return math.ceil(len(text) / CHARS_PER_TOKEN)


def estimate_prompt(messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> int:
    """Raw, uncalibrated estimate for a whole request: messages + tools."""
    return sum(message_tokens(m) for m in messages) + tools_tokens(tools)


def message_tokens(message: dict[str, Any]) -> int:
    tokens = MESSAGE_OVERHEAD_TOKENS + estimate_tokens(str(message.get("content") or ""))
    calls = message.get("tool_calls")
    if calls:
        tokens += estimate_tokens(json.dumps(calls, ensure_ascii=False, default=str))
    return tokens


def tools_tokens(tools: list[dict[str, Any]] | None) -> int:
    if not tools:
        return 0
    return estimate_tokens(json.dumps(tools, ensure_ascii=False))


def fit_messages(
    messages: list[dict[str, Any]],
    *,
    num_ctx: int | None,
    reply_reserve: int,
    tools: list[dict[str, Any]] | None = None,
    extra: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], int, int, int]:
    """Returns (fitted, dropped_count, estimated_prompt_tokens, budget).

    `extra` is appended to the request but never dropped (a tool round's
    call/result messages); it only reduces the room for history. `budget` is
    what the prompt may use: num_ctx minus the reply reserve, the safety
    margin, the tool schemas and `extra`. If the fixed parts alone exceed it
    nothing more can be dropped and the over-budget list is returned as is --
    the caller logs that, since it means the persona/memory block itself is
    too big for num_ctx. num_ctx None (server default, unknown) = no-op.
    """
    if not num_ctx:
        return list(messages), 0, math.ceil(sum(message_tokens(m) for m in messages) * _ratio), 0

    ratio = _ratio
    budget = (
        num_ctx
        - reply_reserve
        - SAFETY_TOKENS
        - math.ceil(tools_tokens(tools) * ratio)
        - math.ceil(sum(message_tokens(m) for m in (extra or [])) * ratio)
    )
    sizes = [math.ceil(message_tokens(m) * ratio) for m in messages]
    total = sum(sizes)
    last = len(messages) - 1
    droppable = [i for i, m in enumerate(messages) if m.get("role") != "system" and i != last]

    dropped: set[int] = set()
    for i in droppable:
        if total <= budget:
            break
        dropped.add(i)
        total -= sizes[i]

    fitted = [m for i, m in enumerate(messages) if i not in dropped]
    return fitted, len(dropped), total, budget


def cap_unprompted(
    history: list[dict[str, Any]],
    temp_turn_flags: list[bool] | None,
    unprompted: list[dict[str, Any]],
    cap: int,
) -> int:
    """Drops all but the newest `cap` unprompted messages from `history`
    (index 0 is the system prompt and is never touched) and removes the
    matching entry from `temp_turn_flags`, which is kept in lockstep with
    history[1:]. `unprompted` is the connection's list of the message dicts
    that were appended as unprompted comments; it is matched by identity and
    pruned in place, including of messages _trim_history already dropped.
    Returns how many messages were removed from history."""
    live = [m for m in unprompted if any(m is h for h in history[1:])]
    unprompted[:] = live

    removed = 0
    while len(unprompted) > max(cap, 0):
        oldest = unprompted.pop(0)
        for i in range(1, len(history)):
            if history[i] is oldest:
                del history[i]
                if temp_turn_flags is not None and 0 <= i - 1 < len(temp_turn_flags):
                    del temp_turn_flags[i - 1]
                removed += 1
                break
    return removed
