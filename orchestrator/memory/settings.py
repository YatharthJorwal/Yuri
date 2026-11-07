"""Memory config sections with defaults filled in, so a config.yaml that sets
only some keys of a section (writer, recall, ...) still works."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from config import CONFIG, MemoryConfig


def _merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def section(name: str) -> dict[str, Any]:
    return _merge(getattr(MemoryConfig(), name), getattr(CONFIG.memory, name))
