"""Router data types.

Small, frozen, provider-agnostic. The provider interface converts its own wire
format into `ModelChunk`, and nothing above the provider ever sees Ollama's
NDJSON or an OpenAI-style SSE frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Privacy classes, weakest to strongest. The call's class is the max over its
# messages (ROUTING.md §5).
PUBLIC = "public"
INTERNAL = "internal"
SECRET = "secret"
_PRIVACY_ORDER = {PUBLIC: 0, INTERNAL: 1, SECRET: 2}

TRUSTED = "trusted"
UNTRUSTED = "untrusted"


def max_privacy(*classes: str) -> str:
    return max(classes or (INTERNAL,), key=lambda c: _PRIVACY_ORDER.get(c, 1))


@dataclass(frozen=True)
class Usage:
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    estimated: bool = False


@dataclass(frozen=True)
class ModelRequest:
    role: str
    messages: tuple[dict[str, Any], ...] = ()
    tools: tuple[dict[str, Any], ...] = ()
    stream: bool = True
    privacy_class: str = INTERNAL
    trust: str = TRUSTED
    explicit_escalation: bool = False
    background: bool = False
    job_id: str | None = None
    max_tokens: int | None = None
    temperature: float | None = None
    schema: dict[str, Any] | None = None

    def with_privacy(self, privacy_class: str) -> "ModelRequest":
        return ModelRequest(
            role=self.role,
            messages=self.messages,
            tools=self.tools,
            stream=self.stream,
            privacy_class=privacy_class,
            trust=self.trust,
            explicit_escalation=self.explicit_escalation,
            background=self.background,
            job_id=self.job_id,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            schema=self.schema,
        )


@dataclass(frozen=True)
class ModelChunk:
    kind: str                                   # content | tool_calls | usage | done
    text: str = ""
    calls: tuple[dict[str, Any], ...] = ()
    usage: Usage | None = None

    @staticmethod
    def content(text: str) -> "ModelChunk":
        return ModelChunk(kind="content", text=text)

    @staticmethod
    def tool_calls(calls: list[dict[str, Any]]) -> "ModelChunk":
        return ModelChunk(kind="tool_calls", calls=tuple(calls))

    @staticmethod
    def done(usage: Usage | None = None) -> "ModelChunk":
        return ModelChunk(kind="done", usage=usage)


@dataclass(frozen=True)
class RouteDecision:
    role: str
    reason: str
    candidates: tuple[str, ...]
    chosen: str
    shadow: bool = False
    est_prompt_tokens: int = 0
    est_cost_usd: float = 0.0
    executed: str | None = None      # filled in by the router after the call

    def described(self) -> str:
        return f"{self.role}: {self.reason} -> {self.chosen}" + (" (shadow)" if self.shadow else "")


@dataclass
class RouterState:
    """Everything `decide()` needs that is not in the config. Mutable, owned by
    the router, serialized to SQLite only for the spend counters."""

    cloud_available: bool = False
    offline_until: float = 0.0
    spend_hour_usd: float = 0.0
    spend_day_usd: float = 0.0
    spend_month_usd: float = 0.0
    est_cost_usd: float = 0.0
    est_prompt_tokens: int = 0
    steps_estimate: int = 1
    tool_count: int = 0
    prices: dict[str, float] = field(default_factory=dict)   # "provider:model" -> usd/1M in
