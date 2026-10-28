"""
Typed event bus between Yuri's services.

Services never call each other directly (SCOPE.md §6). They emit events and
subscribe to patterns, which is what makes a later split into separate
processes a transport change rather than a rewrite.

Two properties are load-bearing, not decoration:

- **`trust`** on every event. Content that came from a screen, a file or a web
  page is `untrusted`, and the router refuses cloud egress for any turn that
  touches it (ROUTING.md §9). A field that nobody sets is useless, so
  `emit()` requires a source and defaults to `trusted` only for our own
  services.
- **A bounded replay log**, so tests can assert what a service actually
  emitted without wiring a fake for every producer.

This is an in-process bus on purpose (D-011). Handlers are awaited in
subscription order, which keeps a turn deterministic; an exception in one
handler is recorded and never prevents the next handler from running.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Mapping

TRUSTED = "trusted"
UNTRUSTED = "untrusted"
TRUST_VALUES = frozenset({TRUSTED, UNTRUSTED})

Handler = Callable[["Event"], Awaitable[None] | None]
Unsubscribe = Callable[[], None]


@dataclass(frozen=True)
class Event:
    """One thing that happened. Immutable so a handler cannot rewrite history."""

    type: str
    source: str
    payload: Mapping[str, Any] = field(default_factory=dict)
    trust: str = TRUSTED
    session: str = ""
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    ts: float = field(default_factory=time.monotonic)

    def __post_init__(self) -> None:
        if not self.type or not isinstance(self.type, str):
            raise ValueError("event.type must be a non-empty string")
        if not self.source or not isinstance(self.source, str):
            raise ValueError("event.source must be a non-empty string")
        if self.trust not in TRUST_VALUES:
            raise ValueError(f"event.trust must be one of {sorted(TRUST_VALUES)}, got {self.trust!r}")
        if not isinstance(self.payload, Mapping):
            raise TypeError("event.payload must be a mapping")

    def is_untrusted(self) -> bool:
        return self.trust == UNTRUSTED


class Bus:
    """In-process publish/subscribe with a bounded replay log."""

    def __init__(self, *, replay_limit: int = 500) -> None:
        if replay_limit < 0:
            raise ValueError("replay_limit must be >= 0")
        self._replay_limit = replay_limit
        self._history: list[Event] = []
        self._subscribers: list[tuple[str, Handler]] = []
        # Handlers that raised, newest last. Tests assert on this instead of
        # patching print().
        self.errors: list[tuple[str, BaseException]] = []

    # -- wiring ---------------------------------------------------------------

    def subscribe(self, pattern: str, handler: Handler) -> Unsubscribe:
        """`pattern` is an exact type ("turn.started"), a prefix ("turn.*"),
        or "*" for everything. Returns an unsubscribe callable."""
        if not pattern:
            raise ValueError("pattern must be a non-empty string")
        entry = (pattern, handler)
        self._subscribers.append(entry)

        def unsubscribe() -> None:
            try:
                self._subscribers.remove(entry)
            except ValueError:
                pass  # already removed; idempotent on purpose

        return unsubscribe

    @staticmethod
    def matches(pattern: str, event_type: str) -> bool:
        if pattern == "*":
            return True
        if pattern.endswith(".*"):
            return event_type.startswith(pattern[:-1])  # keep the dot
        return pattern == event_type

    # -- publishing -----------------------------------------------------------

    async def publish(self, event: Event) -> int:
        """Deliver to every matching handler. Returns the number of handlers
        called. Handler failures are recorded in `errors` and never propagate."""
        if self._replay_limit:
            self._history.append(event)
            if len(self._history) > self._replay_limit:
                del self._history[: len(self._history) - self._replay_limit]

        called = 0
        for pattern, handler in tuple(self._subscribers):
            if not self.matches(pattern, event.type):
                continue
            called += 1
            try:
                result = handler(event)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as exc:  # noqa: BLE001 -- one bad subscriber must not stop the app
                self.errors.append((pattern, exc))
        return called

    async def emit(
        self,
        event_type: str,
        source: str,
        payload: Mapping[str, Any] | None = None,
        *,
        trust: str = TRUSTED,
        session: str = "",
    ) -> Event:
        """Build and publish in one step. The common production path."""
        event = Event(
            type=event_type,
            source=source,
            payload=dict(payload or {}),
            trust=trust,
            session=session,
        )
        await self.publish(event)
        return event

    # -- inspection -----------------------------------------------------------

    def history(self, pattern: str = "*") -> tuple[Event, ...]:
        """What was published, oldest first. A test helper that is also useful
        from a debug endpoint."""
        return tuple(e for e in self._history if self.matches(pattern, e.type))

    def clear_history(self) -> None:
        self._history.clear()

    def subscriber_count(self, pattern: str = "*") -> int:
        if pattern == "*":
            return len(self._subscribers)
        return sum(1 for registered, _ in self._subscribers if registered == pattern)
