"""
Splits a growing text buffer into sentence-ish chunks as soon as they're
ready, so app.py can start TTS on the first sentence instead of waiting for
the whole LLM reply -- this is the "streamed text" half of
docs/ARCHITECTURE.md's "streamed text -> TTS -> lip-sync" pipeline (token-
level audio streaming isn't practical with pyttsx3, so chunking at
sentence granularity is the level that's actually achievable in Phase 2).

Not real NLP sentence segmentation -- it'll misfire on things like
abbreviations ("Mr. Smith"). Good enough for spoken-companion chat; revisit
only if it's audibly wrong often enough to matter.
"""

from __future__ import annotations

import re

# Sentences shorter than this get folded into the next one before being
# treated as ready -- pyttsx3 re-inits its engine per call (see tts.py), so
# firing it for every three-word fragment is wasteful and sounds choppy.
# Doesn't apply to the final trailing chunk, which always flushes regardless
# of length once the stream ends.
MIN_CHUNK_CHARS = 40

_SENTENCE_BOUNDARY = re.compile(r"[.!?]+(?=\s|$)")


def extract_ready_chunks(buffer: str) -> tuple[list[str], str]:
    """Given the buffer so far (previous remainder + newly arrived text),
    returns (chunks ready to speak, new remainder to keep buffering)."""
    chunks: list[str] = []
    start = 0
    for match in _SENTENCE_BOUNDARY.finditer(buffer):
        end = match.end()
        candidate = buffer[start:end].strip()
        if len(candidate) >= MIN_CHUNK_CHARS:
            chunks.append(candidate)
            start = end
    return chunks, buffer[start:]


def flush(buffer: str) -> list[str]:
    """Call once the LLM stream ends: whatever's left, however short or
    unpunctuated, becomes one final chunk."""
    tail = buffer.strip()
    return [tail] if tail else []
