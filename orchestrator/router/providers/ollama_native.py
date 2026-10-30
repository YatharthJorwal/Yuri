"""The local provider.

This is a thin adapter, not a second implementation: it calls `llm.py` by module
attribute so the existing, tested Ollama code remains the only place the local
wire format is understood. That also keeps the ported tests honest -- they
monkeypatch `module.llm.describe_image`, and that patch still takes effect here.
"""

from __future__ import annotations

from typing import AsyncIterator

import llm
from router.types import ModelChunk, ModelRequest


class OllamaNativeProvider:
    name = "local_ollama"

    async def health(self) -> bool:
        caps = await llm.model_capabilities(llm.CONFIG.llm.model)
        return caps is not None

    async def complete(self, request: ModelRequest, *, model: str) -> AsyncIterator[ModelChunk]:
        messages = [dict(message) for message in request.messages]
        if request.tools:
            async for event in llm.stream_reply_with_tools(messages, list(request.tools)):
                if event["type"] == "content":
                    yield ModelChunk.content(event["text"])
                elif event["type"] == "tool_calls":
                    yield ModelChunk.tool_calls(event["calls"])
        else:
            async for delta in llm.stream_reply(messages):
                yield ModelChunk.content(delta)
        yield ModelChunk.done()

    async def describe_image(
        self, prompt: str, image_b64: str, *, model: str, json_schema: dict | None = None
    ) -> str:
        # Only pass the keyword when there is one: several tests (and any future
        # alternate vision backend) define `describe_image(prompt, image)` and
        # would reject an explicit None.
        if json_schema is None:
            return await llm.describe_image(prompt, image_b64)
        return await llm.describe_image(prompt, image_b64, json_schema=json_schema)
