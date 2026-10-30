"""AI Router provider (OpenAI-compatible SSE).

Everything here is testable against `httpx.MockTransport`, which is how the
suite exercises 200/429/500, tool-call deltas, missing usage, and malformed
frames without a key or a network.

No retries for network errors: the router's candidate chain owns fallback, and
two retry layers fighting each other is how a 429 becomes a four-second pause.
The single exception is a 400 caused by `response_format`, retried once without
it, because some providers reject the field rather than ignoring it.
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator

import httpx

from config import ProviderConfig
from router.errors import ProviderError
from router.types import ModelChunk, ModelRequest, Usage


class AirouterProvider:
    name = "airouter"

    def __init__(
        self,
        cfg: ProviderConfig,
        api_key: str | None,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        timeout_s: float | None = None,
    ) -> None:
        self.cfg = cfg
        self.api_key = api_key
        self._transport = transport
        self._timeout = httpx.Timeout(timeout_s or cfg.request_timeout_s, connect=5.0)

    # -- requests ------------------------------------------------------------

    def _payload(self, request: ModelRequest, model: str) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model,
            "messages": [dict(message) for message in request.messages],
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        temperature = request.temperature
        if temperature is None:
            temperature = 0.6
        payload["temperature"] = temperature
        if request.max_tokens:
            payload["max_tokens"] = request.max_tokens
        if request.tools:
            payload["tools"] = list(request.tools)
            payload["tool_choice"] = "auto"
        if request.schema:
            payload["response_format"] = {"type": "json_object"}
        return payload

    async def health(self) -> bool:
        try:
            async with httpx.AsyncClient(transport=self._transport, timeout=10.0) as client:
                response = await client.get(f"{self.cfg.base_url.rstrip('/')}/models")
            return response.status_code < 400
        except httpx.RequestError:
            return False

    # -- streaming -----------------------------------------------------------

    async def complete(self, request: ModelRequest, *, model: str) -> AsyncIterator[ModelChunk]:
        payload = self._payload(request, model)
        url = f"{self.cfg.base_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key or ''}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }

        for attempt in range(2):
            try:
                async with httpx.AsyncClient(transport=self._transport, timeout=self._timeout) as client:
                    async with client.stream("POST", url, json=payload, headers=headers) as response:
                        if response.status_code >= 400:
                            body = (await response.aread()).decode("utf-8", "replace")
                            if (
                                attempt == 0
                                and response.status_code == 400
                                and "response_format" in payload
                            ):
                                payload.pop("response_format", None)
                                continue
                            raise ProviderError(
                                f"{response.status_code} from {url}: {body[:300]!r}"
                            )
                        async for chunk in self._frames(response):
                            yield chunk
                        return
            except httpx.TimeoutException as exc:
                raise ProviderError(f"timed out waiting for {url}: {exc}") from exc
            except httpx.RequestError as exc:
                raise ProviderError(f"couldn't reach {url}: {exc}") from exc

    async def _frames(self, response: httpx.Response) -> AsyncIterator[ModelChunk]:
        usage: Usage | None = None
        tool_parts: dict[int, dict[str, str]] = {}

        async for raw in response.aiter_lines():
            line = raw.strip()
            if not line or not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            if data == "[DONE]":
                break
            try:
                event = json.loads(data)
            except ValueError:
                continue
            if not isinstance(event, dict):
                continue

            reported = event.get("usage")
            if isinstance(reported, dict):
                usage = Usage(
                    prompt_tokens=_int(reported.get("prompt_tokens")),
                    completion_tokens=_int(reported.get("completion_tokens")),
                )

            choices = event.get("choices")
            if not isinstance(choices, list) or not choices:
                continue
            choice = choices[0] if isinstance(choices[0], dict) else {}
            delta = choice.get("delta") if isinstance(choice.get("delta"), dict) else {}

            text = delta.get("content")
            if isinstance(text, str) and text:
                yield ModelChunk.content(text)

            for raw_call in delta.get("tool_calls") or []:
                if not isinstance(raw_call, dict):
                    continue
                index = raw_call.get("index", 0)
                function = raw_call.get("function") if isinstance(raw_call.get("function"), dict) else {}
                part = tool_parts.setdefault(int(index), {"name": "", "arguments": ""})
                if isinstance(function.get("name"), str) and function["name"]:
                    part["name"] = function["name"]
                if isinstance(function.get("arguments"), str):
                    part["arguments"] += function["arguments"]

        calls = [call for call in _finish_tool_parts(tool_parts) if call["name"]]
        if calls:
            yield ModelChunk.tool_calls(calls)
        yield ModelChunk.done(usage)


def _finish_tool_parts(parts: dict[int, dict[str, str]]) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for index in sorted(parts):
        part = parts[index]
        raw = part.get("arguments") or ""
        try:
            arguments = json.loads(raw) if raw.strip() else {}
        except ValueError:
            arguments = {}
        if not isinstance(arguments, dict):
            arguments = {}
        calls.append({"name": part.get("name", ""), "arguments": arguments})
    return calls


def _int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None
