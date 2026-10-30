"""AI Router provider against a fake SSE server. No key, no network."""

import json

import httpx
import pytest

from config import ProviderConfig
from router.errors import ProviderError
from router.providers.airouter import AirouterProvider
from router.types import ModelRequest

CFG = ProviderConfig(
    name="airouter",
    base_url="https://api.example/v1",
    api_style="openai",
)


def sse(*events: dict) -> bytes:
    lines = [f"data: {json.dumps(event)}\n\n" for event in events]
    lines.append("data: [DONE]\n\n")
    return "".join(lines).encode("utf-8")


def chunk(delta: dict, finish: str | None = None) -> dict:
    return {"choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}


def provider(handler, *, key="test-key"):
    return AirouterProvider(CFG, key, transport=httpx.MockTransport(handler))


async def collect(p):
    return [
        c
        async for c in p.complete(ModelRequest(role="mind", messages=({"role": "user", "content": "hi"},)), model="m")
    ]


@pytest.mark.asyncio
async def test_streams_content_chunks():
    body = sse(chunk({"content": "Hel"}), chunk({"content": "lo"}, "stop"))
    events = await collect(provider(lambda request: httpx.Response(200, content=body)))
    assert "".join(c.text for c in events if c.kind == "content") == "Hello"
    assert events[-1].kind == "done"


@pytest.mark.asyncio
async def test_tool_call_fragments_are_assembled():
    body = sse(
        chunk(
            {
                "tool_calls": [
                    {"index": 0, "function": {"name": "capture_screen", "arguments": '{"reg'}}
                ]
            }
        ),
        chunk({"tool_calls": [{"index": 0, "function": {"arguments": 'ion": "full"}'}}]}, "tool_calls"),
    )
    events = await collect(provider(lambda request: httpx.Response(200, content=body)))
    calls = [c for e in events if e.kind == "tool_calls" for c in e.calls]
    assert calls == [{"name": "capture_screen", "arguments": {"region": "full"}}]


@pytest.mark.asyncio
async def test_usage_is_reported():
    body = sse(chunk({"content": "x"}), {"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 3}})
    events = await collect(provider(lambda request: httpx.Response(200, content=body)))
    done = events[-1]
    assert done.usage is not None
    assert done.usage.prompt_tokens == 10
    assert done.usage.completion_tokens == 3


@pytest.mark.asyncio
async def test_missing_usage_is_none():
    body = sse(chunk({"content": "x"}, "stop"))
    events = await collect(provider(lambda request: httpx.Response(200, content=body)))
    assert events[-1].usage is None


@pytest.mark.asyncio
async def test_rate_limit_raises_provider_error():
    p = provider(lambda request: httpx.Response(429, json={"error": "slow down"}))
    with pytest.raises(ProviderError) as err:
        await collect(p)
    assert "429" in str(err.value)


@pytest.mark.asyncio
async def test_server_error_raises_provider_error():
    p = provider(lambda request: httpx.Response(500, text="boom"))
    with pytest.raises(ProviderError):
        await collect(p)


@pytest.mark.asyncio
async def test_malformed_frames_are_ignored():
    body = b'data: {"choices": [{"delta": {"content": "a"}}]}\n\ndata: not json\n\ndata: [DONE]\n\n'
    events = await collect(provider(lambda request: httpx.Response(200, content=body)))
    assert "".join(c.text for c in events if c.kind == "content") == "a"


@pytest.mark.asyncio
async def test_json_schema_retried_without_response_format():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        seen.append("response_format" in payload)
        if "response_format" in payload:
            return httpx.Response(400, json={"error": "unsupported response_format"})
        return httpx.Response(200, content=sse(chunk({"content": '{"ok":true}'}, "stop")))

    p = provider(handler)
    events = [
        c
        async for c in p.complete(
            ModelRequest(
                role="mind",
                messages=({"role": "user", "content": "json"},),
                schema={"type": "object"},
            ),
            model="m",
        )
    ]
    assert seen == [True, False]
    assert "".join(c.text for c in events if c.kind == "content") == '{"ok":true}'


@pytest.mark.asyncio
async def test_authorization_header_is_sent():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["auth"] = request.headers.get("authorization")
        return httpx.Response(200, content=sse(chunk({"content": "x"}, "stop")))

    await collect(provider(handler, key="secret-key"))
    assert captured["auth"] == "Bearer secret-key"
