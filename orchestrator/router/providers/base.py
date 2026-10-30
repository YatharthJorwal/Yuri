"""Provider interface.

A provider turns a `ModelRequest` into `ModelChunk`s. It does not decide
anything, does not retry, and does not write audit rows -- the router owns all
three so behavior stays identical across providers.
"""

from __future__ import annotations

from typing import AsyncIterator, Protocol

from router.types import ModelChunk, ModelRequest


class Provider(Protocol):
    name: str

    def complete(self, request: ModelRequest, *, model: str) -> AsyncIterator[ModelChunk]: ...

    async def health(self) -> bool: ...
