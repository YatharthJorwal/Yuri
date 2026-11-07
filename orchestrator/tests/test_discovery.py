"""Catalog discovery: parse, cache, staleness, fail-closed."""

import json

import httpx
import pytest

from router.discovery import ModelCatalog

PAYLOAD = {
    "object": "list",
    "data": [
        {
            "id": "deepseek/deepseek-v4-flash",
            "type": "language",
            "context_length": 1_000_000,
            "max_tokens": 384_000,
            "pricing": {"prompt": "0.13", "completion": "0.26"},
            "tags": ["tool-use", "reasoning", "structured-output"],
            "zdr": True,
        },
        {
            "id": "alibaba/qwen-3-235b",
            "type": "language",
            "context_length": 262_144,
            "max_tokens": 16_384,
            "pricing": {"prompt": "0.22", "completion": "0.88"},
            "tags": ["tool-use"],
            "zdr": False,
        },
    ],
}


def _client(payload=None, status=200):
    def handler(request: httpx.Request) -> httpx.Response:
        if status >= 400:
            return httpx.Response(status, json={"error": "boom"})
        return httpx.Response(status, json=payload if payload is not None else PAYLOAD)

    return httpx.MockTransport(handler)


def _catalog(tmp_path, transport, clock=lambda: 1000.0, ttl=24):
    cache = tmp_path / "models.json"
    cache.write_text(json.dumps({"fetched_at": 0, "models": {}}), encoding="utf-8")
    return ModelCatalog("https://api.example/v1", cache, ttl_hours=ttl, transport=transport, clock=clock)


@pytest.mark.asyncio
async def test_refresh_parses_catalog(tmp_path):
    catalog = _catalog(tmp_path, _client())
    assert await catalog.refresh() is True
    assert catalog.available
    assert catalog.model_ids() == ("alibaba/qwen-3-235b", "deepseek/deepseek-v4-flash")
    info = catalog.info("deepseek/deepseek-v4-flash")
    assert info.context_length == 1_000_000
    assert info.price_in == pytest.approx(0.13)
    assert info.price_out == pytest.approx(0.26)
    assert info.zdr is True
    assert info.supports("tool-use")


@pytest.mark.asyncio
async def test_refresh_writes_cache_and_reloads(tmp_path):
    catalog = _catalog(tmp_path, _client())
    await catalog.refresh()
    cache_text = catalog.cache_path.read_text(encoding="utf-8")
    assert "deepseek/deepseek-v4-flash" in cache_text

    fresh = ModelCatalog("https://api.example/v1", catalog.cache_path, transport=_client(status=500))
    assert fresh.load_cache() is True
    assert fresh.info("deepseek/deepseek-v4-flash").price_in == pytest.approx(0.13)


@pytest.mark.asyncio
async def test_refresh_is_skipped_when_cache_fresh(tmp_path):
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(200, json=PAYLOAD)

    catalog = _catalog(tmp_path, httpx.MockTransport(handler), clock=lambda: 1000.0)
    await catalog.refresh()
    await catalog.refresh()
    assert calls["n"] == 1          # second call hit the cache
    await catalog.refresh(force=True)
    assert calls["n"] == 2


@pytest.mark.asyncio
async def test_stale_cache_is_refreshed(tmp_path):
    now = {"t": 1000.0}
    catalog = _catalog(tmp_path, _client(), clock=lambda: now["t"], ttl=1)
    await catalog.refresh()
    now["t"] += 2 * 3600
    assert catalog.is_stale()
    assert await catalog.refresh() is True


@pytest.mark.asyncio
async def test_endpoint_down_with_cache_stays_available(tmp_path):
    catalog = _catalog(tmp_path, _client())
    await catalog.refresh()
    broken = ModelCatalog(
        "https://api.example/v1", catalog.cache_path, transport=_client(status=503)
    )
    assert broken.load_cache() is True
    assert await broken.refresh(force=True) is True
    assert broken.available
    assert broken.last_error is not None


@pytest.mark.asyncio
async def test_endpoint_down_without_cache_is_unavailable(tmp_path):
    catalog = ModelCatalog(
        "https://api.example/v1", tmp_path / "none.json", transport=_client(status=503)
    )
    assert await catalog.refresh() is False
    assert catalog.available is False
    assert "503" in (catalog.last_error or "")


@pytest.mark.asyncio
async def test_empty_catalog_is_unavailable(tmp_path):
    catalog = _catalog(tmp_path, _client(payload={"object": "list", "data": []}))
    assert await catalog.refresh() is False
    assert catalog.available is False


@pytest.mark.asyncio
async def test_malformed_payload_is_unavailable(tmp_path):
    catalog = _catalog(tmp_path, _client(payload={"nope": True}))
    assert await catalog.refresh() is False


@pytest.mark.asyncio
async def test_closest_offers_near_misses(tmp_path):
    catalog = _catalog(tmp_path, _client())
    await catalog.refresh()
    assert "deepseek/deepseek-v4-flash" in catalog.closest("deepseek/deepseek-v4-flsh")


def test_price_of_unknown_model_is_zero(tmp_path):
    catalog = ModelCatalog("https://api.example/v1", tmp_path / "none.json")
    assert catalog.price("nope") == (0.0, 0.0)
    assert catalog.context_length("nope") == 0
