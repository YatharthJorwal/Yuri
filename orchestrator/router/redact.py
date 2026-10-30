"""Redaction for anything leaving the machine.

Rules run before serialization, in this order: secrets, file paths, emails, long
digit runs. Counts go to the audit row; content never does.

This is deliberately blunt. A redaction pass that is clever is one that can be
argued with; a regex that always removes a Windows path may occasionally eat a
sentence, and that is the correct direction of failure for a cloud egress gate.
"""

from __future__ import annotations

import re
from typing import Mapping

_SECRET = re.compile(
    r"(?:sk|pk|rk)-[A-Za-z0-9_\-]{16,}"
    r"|gh[pousr]_[A-Za-z0-9]{16,}"
    r"|(?i:bearer)\s+[A-Za-z0-9._\-]{16,}"
    r"|(?i:api[_-]?key)\s*[:=]\s*\S{8,}",
)
_WINDOWS_PATH = re.compile(r"[A-Za-z]:\\[^\s\"'<>|]+")
_UNC_PATH = re.compile(r"\\\\[^\s\"'<>|]+")
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_DIGITS = re.compile(r"\b\d{12,}\b")

_REPLACEMENTS = (
    ("secret", _SECRET),
    ("path", _WINDOWS_PATH),
    ("path", _UNC_PATH),
    ("email", _EMAIL),
    ("digits", _DIGITS),
)


def redact(text: str, settings: Mapping[str, object] | None = None) -> tuple[str, dict[str, int]]:
    """Return (redacted_text, counts_by_kind). A setting of False disables that
    rule; anything else enables it (default-on, which is the safe default)."""
    settings = settings or {}
    enabled = {
        "secret": settings.get("secrets", True),
        "path": settings.get("file_paths", True),
        "email": settings.get("emails", True),
        "digits": settings.get("long_digit_runs", True),
    }
    counts: dict[str, int] = {}
    out = text
    for kind, pattern in _REPLACEMENTS:
        if not enabled.get(kind, True):
            continue
        out, n = pattern.subn(f"[{kind}]", out)
        if n:
            counts[kind] = counts.get(kind, 0) + n
    return out, counts


def redact_messages(
    messages: list[dict], settings: Mapping[str, object] | None = None
) -> tuple[list[dict], dict[str, int]]:
    """Redact every string content in a message list, without mutating the input.
    Image payloads (`images`, base64) are never sent to a cloud provider at all,
    so they are left untouched here and blocked by the privacy class instead."""
    counts: dict[str, int] = {}
    out: list[dict] = []
    for message in messages:
        copy = dict(message)
        content = copy.get("content")
        if isinstance(content, str):
            copy["content"], found = redact(content, settings)
            for kind, n in found.items():
                counts[kind] = counts.get(kind, 0) + n
        out.append(copy)
    return out, counts


def cap(text: str, limit: int) -> str:
    """Truncate to `limit` chars on a word boundary, marking the cut."""
    if limit <= 0 or len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0].rstrip()
    return cut + "..."
