"""
Speech-safe text.

The persona prompt asks the model not to use em dashes, quotes, markdown or
emoji. This module is the part that *guarantees* it. A 9B model does not hold
a typographic rule for fifty turns; a pure function with a golden corpus does.

`prepare()` is the only entry point. It is side-effect free (no model, no disk,
no clock), which is what makes the golden corpus meaningful.

Design notes:
- Text is normalized for the *voice* only. `display` keeps the model's own
  wording (minus the emotion tag, which `persona.extract_emotion_tag` removes
  earlier) so the chat panel can still read naturally.
- Apostrophes inside words are preserved (`don't` stays `don't`); standalone
  single quotes are removed. Removing apostrophes is the classic way to make a
  voice say "dont".
- Hyphens inside compound words become a space (`well-being` -> `well being`);
  a hyphen used as a dash becomes a comma. Both sound better than "minus".
- Nothing here decides *what* is said; it only makes what was said speakable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Sequence

from config import CONFIG, SpeechConfig

_CODE_FENCE = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE = re.compile(r"`([^`]+)`")
_MD_LINK = re.compile(r"!?\[([^\]]*)\]\([^)]*\)")
_URL = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_HEADING = re.compile(r"(?m)^\s{0,3}#{1,6}\s+")
_BLOCKQUOTE = re.compile(r"(?m)^\s*>\s?")
_BULLET = re.compile(r"(?m)^\s*[-*+]\s+")
_NUMBERED = re.compile(r"(?m)^\s*\d+[.)]\s+")
_EMPHASIS = re.compile(r"\*\*|__|\*|_")
# The dash family + the minus sign. \u2010-\u2015 covers hyphen..horizontal bar.
_UNICODE_DASH = re.compile(r"\s*[\u2010\u2011\u2012\u2013\u2014\u2015\u2212]\s*")
_SPACED_HYPHEN = re.compile(r"\s+-\s+")
_COMPOUND_HYPHEN = re.compile(r"(?<=\w)-(?=\w)")
_QUOTES = re.compile(r"[\"\u201c\u201d\u00ab\u00bb\u201e\u201f]")
_SINGLE_QUOTES = re.compile(r"(?<![A-Za-z])['\u2018\u2019]|['\u2018\u2019](?![A-Za-z])")
_ELLIPSIS = re.compile(r"\u2026|\.{3,}")
_SEMICOLON = re.compile(r"\s*;\s*")
_PARENS = re.compile(r"\(([^)]*)\)")
_EMOJI = re.compile(
    "["
    "\U0001F000-\U0001FAFF"
    "\U00002600-\U000027BF"
    "\U0001F1E6-\U0001F1FF"
    "\u2B00-\u2BFF\u3030\u303D\u3297\u3299"
    "\uFE0F\u200D"
    "]+"
)
_EMOTICON = re.compile(r"[:;]-?[)D(Pp3]")
# A stray [emotion] style tag. persona.extract_emotion_tag removes the real one
# before this runs; this is the backstop, so a hallucinated tag is never read
# aloud verbatim.
_BRACKET_TAG = re.compile(r"\s*\[[^\]\n]{0,24}\]")
_REPEATED_PUNCT = re.compile(r"([!?])\1+")
_NEWLINES = re.compile(r"\s*\n+\s*")
_WHITESPACE = re.compile(r"[ \t]{2,}")
_LEADING_JUNK = re.compile(r"^[\s,.;:!?]+")
_ACRONYM = re.compile(r"\b[A-Z]{2,}\b")

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_COMMA_SPLIT = re.compile(r"(?<=,)\s+")


@dataclass(frozen=True)
class PreparedSpeech:
    """What the caller needs: the text for the panel, the text for the voice,
    the clause chunks to synthesize, and a short audit of what was removed."""

    display: str
    spoken: str
    segments: tuple[str, ...] = ()
    dropped: tuple[str, ...] = ()
    enabled: bool = True

    def __bool__(self) -> bool:
        return bool(self.spoken) or bool(self.segments)


def _strip_code(text: str, marker: str, dropped: list[str]) -> str:
    for block in _CODE_FENCE.findall(text):
        dropped.append(f"code_block:{len(block)}")
    text = _CODE_FENCE.sub(f", {marker}, " if marker else ", ", text)
    return _INLINE_CODE.sub(r"\1", text)


def _strip_markdown(text: str) -> str:
    text = _MD_LINK.sub(r"\1", text)
    text = _URL.sub("a link", text)
    text = _HEADING.sub("", text)
    text = _BLOCKQUOTE.sub("", text)
    text = _BULLET.sub("", text)
    text = _NUMBERED.sub("", text)
    return _EMPHASIS.sub("", text)


def _strip_parentheticals(text: str, max_inline: int = 40) -> str:
    def repl(match: re.Match) -> str:
        inner = match.group(1).strip()
        if not inner:
            return " "
        if len(inner) <= max_inline:
            return " "
        return f", {inner}, "

    return _PARENS.sub(repl, text)


def _strip_quotes(text: str) -> str:
    text = _QUOTES.sub("", text)
    return _SINGLE_QUOTES.sub("", text)


def _normalize_dashes(text: str) -> str:
    text = _UNICODE_DASH.sub(", ", text)
    text = _SPACED_HYPHEN.sub(", ", text)
    return _COMPOUND_HYPHEN.sub(" ", text)


def normalize_spoken(text: str, cfg: SpeechConfig | None = None) -> tuple[str, list[str]]:
    """Apply every rule and return (spoken_text, dropped). Exposed separately
    so tests can check one rule at a time without building segments."""
    cfg = cfg or CONFIG.speech
    dropped: list[str] = []
    marker = "code block" if cfg.code_blocks == "say_marker" else ""

    out = text
    out = _strip_code(out, marker, dropped)
    out = _strip_markdown(out)
    out = _BRACKET_TAG.sub(" ", out)
    out = _normalize_dashes(out)
    out = _strip_quotes(out)
    out = _ELLIPSIS.sub(", ", out)
    out = _SEMICOLON.sub(". ", out)
    out = _strip_parentheticals(out)
    out = _EMOJI.sub(" ", out)
    out = _EMOTICON.sub("", out)
    out = _REPEATED_PUNCT.sub(r"\1", out)
    out = _NEWLINES.sub(". ", out)
    out = _WHITESPACE.sub(" ", out)

    # ALL CAPS reads like shouting. Acronyms (VRAM, GPU, ...) are exempt.
    if cfg.acronym_allowlist:
        allowed = set(cfg.acronym_allowlist)
        out = _ACRONYM.sub(lambda m: m.group(0) if m.group(0) in allowed else m.group(0).lower(), out)

    out = _LEADING_JUNK.sub("", out)
    out = re.sub(r"\s+([,.;:!?])", r"\1", out)
    return out.strip(), dropped


def split_segments(
    text: str,
    max_chars: int = 160,
    min_chars: int = 12,
) -> tuple[str, ...]:
    """Clause chunks for TTS. Sentence boundaries first, commas only when a
    sentence is too long, word boundaries as the last resort, never empty."""
    if not text:
        return ()

    pieces: list[str] = []
    for sentence in _SENTENCE_SPLIT.split(text.strip()):
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(sentence) <= max_chars:
            pieces.append(sentence)
            continue

        # Too long: prefer comma boundaries, then pack without exceeding the
        # limit, and fall back to word packing for a single huge clause.
        clauses = [c.strip() for c in _COMMA_SPLIT.split(sentence) if c.strip()]
        buffer = ""
        for clause in clauses:
            if len(clause) > max_chars:
                if buffer:
                    pieces.append(buffer)
                    buffer = ""
                pieces.extend(_pack_words(clause, max_chars))
                continue
            candidate = f"{buffer} {clause}".strip()
            if buffer and len(candidate) > max_chars:
                pieces.append(buffer)
                buffer = clause
            else:
                buffer = candidate
        if buffer:
            pieces.append(buffer)

    # Merge a non-terminal runt into its predecessor so TTS never speaks a
    # one-word fragment with a full pause after it. Short complete sentences
    # ("No." "Fine.") are intentionally left alone: they are real clauses and
    # merging them would make streaming coarser for no benefit.
    merged: list[str] = []
    for piece in pieces:
        complete = piece.endswith((".", "!", "?"))
        if merged and len(piece) < min_chars and not complete:
            merged[-1] = f"{merged[-1]} {piece}"
        else:
            merged.append(piece)
    return tuple(merged)


def _pack_words(text: str, max_chars: int) -> list[str]:
    out: list[str] = []
    buffer = ""
    for word in text.split():
        candidate = f"{buffer} {word}".strip()
        if buffer and len(candidate) > max_chars:
            out.append(buffer)
            buffer = word
        else:
            buffer = candidate
    if buffer:
        out.append(buffer)
    return out


def prepare(text: str, cfg: SpeechConfig | None = None) -> PreparedSpeech:
    """The single entry point: model text in, speakable text and clauses out."""
    cfg = cfg or CONFIG.speech
    display = (text or "").strip()
    if not cfg.enabled:
        segments = (display,) if display else ()
        return PreparedSpeech(display=display, spoken=display, segments=segments, enabled=False)

    spoken, dropped = normalize_spoken(display, cfg)
    segments = split_segments(spoken, cfg.max_segment_chars, cfg.min_segment_chars)
    return PreparedSpeech(
        display=display,
        spoken=spoken,
        segments=segments,
        dropped=tuple(dropped),
        enabled=True,
    )


# Characters that must never reach TTS. Apostrophes are deliberately absent:
# contractions keep theirs ("don't" read as "dont" is worse than a curly
# apostrophe). Tests assert against this set; adding a rule means adding the
# character here.
BANNED_CHARS: frozenset[str] = frozenset(
    "\u2010\u2011\u2012\u2013\u2014\u2015\u2212"      # dash family + minus
    "\u201c\u201d\u00ab\u00bb\u201e\u201f\""           # quotation marks
    "\u2026"                                            # ellipsis
    "*_`"                                               # markdown emphasis/code
)


def banned_chars_in(text: str) -> set[str]:
    return BANNED_CHARS & set(text)
