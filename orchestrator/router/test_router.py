"""Router end-to-end with fake providers: shadow, enforce, fallback, privacy."""

import dataclasses
import json

import pytest

import config
from router import Router
from router.audit import AuditWriter
from router.budget import SpendLedger
from router.errors import ProviderError
from router.types import ModelChunk, ModelRequest, SECRET, UNTRUSTED, Usage


class FakeCatalog:
    available = True
    last_error = None

    def model_ids(self):
        return ("deepseek/deepseek-v4-flash",)

    def load_cache(self):
        return True

    def is_stale(self):
        return False

    async def refresh(self, *, force=False):
        return True

    def price(self, model):
        return (0.5, 1.5)


class FakeProvider:
    def __init__(self, name, *, error=None):
        self.name = name
        self.calls = []
        self.error = error

    async def health(self):
        return True

    async def complete(self, request, *, model):
        self.calls.append((request, model))
        if self.error:
            raise self.error
        yield ModelChunk.content("hello")
        yield ModelChunk.done(Usage(prompt_tokens=10, completion_tokens=2))

    async def describe_image(self, prompt, image_b64, *, model, json_schema=None):
        self.calls.append((prompt, model))
        return "a described image"


def make_config(tmp_path, *, mode="shadow"):
    providers = {
        **config.CONFIG.providers,
        "airouter": dataclasses.replace(
            config.CONFIG.providers["airouter"], enabled=True, models_cache=str(tmp_path / "models.json")
        ),
    }
    return dataclasses.replace(
        config.CONFIG,
        providers=providers,
        routing=dataclasses.replace(config.CONFIG.routing, mode=mode, enforce_roles=("mind",)),
        audit=dataclasses.replace(config.CONFIG.audit, path=str(tmp_path / "audit.jsonl")),
    )


def make_router(tmp_path, *, mode="shadow", cloud_error=None):
    local = FakeProvider("local_ollama")
    cloud = FakeProvider("airouter", error=cloud_error)
    audit = AuditWriter(tmp_path / "audit.jsonl")
    ledger = SpendLedger(tmp_path / "spend.db")
    router = Router(
        make_config(tmp_path, mode=mode),
        catalog=FakeCatalog(),
        local_provider=local,
        cloud_provider=cloud,
        audit=audit,
        ledger=ledger,
    )
    return router, local, cloud


def audit_rows(tmp_path):
    path = tmp_path / "audit.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


async def drain(router, request):
    return [chunk async for chunk in router.complete(request)]


MIND = ModelRequest(role="mind", messages=({"role": "user", "content": "plan this"},), background=True)


@pytest.mark.asyncio
async def test_shadow_computes_but_executes_local(tmp_path):
    router, local, cloud = make_router(tmp_path, mode="shadow")
    chunks = await drain(router, MIND)

    assert "".join(c.text for c in chunks if c.kind == "content") == "hello"
    assert len(local.calls) == 1
    assert cloud.calls == []

    started = audit_rows(tmp_path)[0]
    assert started["shadow"] is True
    assert started["chosen"].startswith("airouter:")
    assert started["mode"] == "shadow"


@pytest.mark.asyncio
async def test_enforce_calls_the_cloud(tmp_path):
    router, local, cloud = make_router(tmp_path, mode="enforce")
    await drain(router, MIND)

    assert len(cloud.calls) == 1
    assert local.calls == []
    assert cloud.calls[0][1] == "deepseek/deepseek-v4-flash"


@pytest.mark.asyncio
async def test_cloud_failure_falls_back_to_local(tmp_path):
    router, local, cloud = make_router(tmp_path, mode="enforce", cloud_error=ProviderError("429 nope"))
    await drain(router, MIND)

    assert len(cloud.calls) == 1
    assert len(local.calls) == 1
    finished = audit_rows(tmp_path)[-1]
    assert finished["event"] == "model.call.finished"
    assert finished["fallback_used"] is True
    assert "429" in (finished["error"] or "")


@pytest.mark.asyncio
async def test_cloud_failure_falls_through_to_a_failing_local_fallback(tmp_path):
    # A cloud-only chain is impossible by construction: the local model is always
    # the last candidate. So the real terminal failure is both providers down.
    router, local, _ = make_router(tmp_path, mode="enforce", cloud_error=ProviderError("429 nope"))
    local.error = ProviderError("ollama down")
    with pytest.raises(ProviderError):
        await drain(router, MIND)


@pytest.mark.asyncio
async def test_decision_candidates_always_include_local(tmp_path):
    router, _, _ = make_router(tmp_path, mode="enforce")
    await router.startup()
    decision = router.route(MIND)
    assert decision.chosen.startswith("airouter:")
    assert decision.candidates[-1].startswith("local_ollama:")


@pytest.mark.asyncio
async def test_untrusted_context_never_reaches_cloud(tmp_path):
    router, local, cloud = make_router(tmp_path, mode="enforce")
    request = dataclasses.replace(MIND, trust=UNTRUSTED)
    await drain(router, request)

    assert cloud.calls == []
    assert len(local.calls) == 1
    assert audit_rows(tmp_path)[0]["reason"] == "UNTRUSTED_CONTEXT"


@pytest.mark.asyncio
async def test_secret_content_never_reaches_cloud(tmp_path):
    router, local, cloud = make_router(tmp_path, mode="enforce")
    await drain(router, dataclasses.replace(MIND, privacy_class=SECRET))

    assert cloud.calls == []
    assert audit_rows(tmp_path)[0]["reason"] == "PRIVACY_LOCAL_ONLY"


@pytest.mark.asyncio
async def test_budget_zero_never_spends(tmp_path):
    router, local, cloud = make_router(tmp_path, mode="enforce")
    router.config = dataclasses.replace(
        router.config,
        budget=dataclasses.replace(router.config.budget, hourly_usd=0),
    )
    router.state = dataclasses.replace(router.state)
    await drain(router, MIND)

    assert cloud.calls == []
    assert audit_rows(tmp_path)[0]["reason"] == "BUDGET_EXHAUSTED"


@pytest.mark.asyncio
async def test_local_only_role_is_never_escalated(tmp_path):
    router, local, cloud = make_router(tmp_path, mode="enforce")
    await drain(router, ModelRequest(role="reflex", messages=({"role": "user", "content": "hi"},)))

    assert cloud.calls == []
    assert len(local.calls) == 1
    assert audit_rows(tmp_path)[0]["reason"] == "ROLE_PINNED_LOCAL"


@pytest.mark.asyncio
async def test_cloud_payload_is_redacted(tmp_path):
    router, _, cloud = make_router(tmp_path, mode="enforce")
    request = dataclasses.replace(
        MIND,
        messages=(
            {
                "role": "user",
                "content": r"key sk-abcdefghijklmnop1234 path C:\Users\User\secret.txt",
            },
        ),
    )
    await drain(router, request)

    sent = cloud.calls[0][0].messages[0]["content"]
    assert "sk-" not in sent
    assert "C:\\Users" not in sent
    started = audit_rows(tmp_path)[0]
    assert started["redactions"] == {"secret": 1, "path": 1}


@pytest.mark.asyncio
async def test_images_are_never_sent_to_cloud(tmp_path):
    router, local, cloud = make_router(tmp_path, mode="enforce")
    text = await router.describe_image("vision", "what is this", "aGVsbG8=")

    assert text == "a described image"
    assert cloud.calls == []
    assert local.calls[0][0] == "what is this"
    # vision is local-only by role (rule 1), which is even stronger than the
    # privacy class; both keep pixels here.
    assert audit_rows(tmp_path)[0]["reason"] == "ROLE_PINNED_LOCAL"


@pytest.mark.asyncio
async def test_image_payload_makes_even_mind_local(tmp_path):
    router, local, cloud = make_router(tmp_path, mode="enforce")
    text = await router.describe_image("mind", "what is this", "aGVsbG8=")

    assert text == "a described image"
    assert cloud.calls == []
    assert audit_rows(tmp_path)[0]["reason"] == "PRIVACY_LOCAL_ONLY"


@pytest.mark.asyncio
async def test_spend_is_recorded_after_a_cloud_call(tmp_path):
    router, _, cloud = make_router(tmp_path, mode="enforce")
    await drain(router, MIND)

    hour, day, month = router.ledger.totals()
    assert day > 0
    # 10 prompt @ $0.5/1M + 2 completion @ $1.5/1M
    assert day == pytest.approx((10 * 0.5 + 2 * 1.5) / 1_000_000)


@pytest.mark.asyncio
async def test_stream_text_matches_the_old_shape(tmp_path):
    router, _, _ = make_router(tmp_path)
    pieces = [text async for text in router.stream_text("reflex", [{"role": "user", "content": "hi"}])]
    assert pieces == ["hello"]


# -- the runtime toggle (tray / quick-action "Use API model") -----------------


@pytest.mark.asyncio
async def test_toggle_on_overrides_a_shadow_config(tmp_path):
    router, local, cloud = make_router(tmp_path, mode="shadow")
    router.set_use_api_model(True)
    await router.startup()
    await drain(router, MIND)

    assert len(cloud.calls) == 1
    assert local.calls == []
    assert audit_rows(tmp_path)[0]["shadow"] is False


@pytest.mark.asyncio
async def test_toggle_off_overrides_an_enforce_config(tmp_path):
    router, local, cloud = make_router(tmp_path, mode="enforce")
    router.set_use_api_model(False)
    await router.startup()
    await drain(router, MIND)

    assert cloud.calls == []
    assert len(local.calls) == 1
    assert audit_rows(tmp_path)[0]["shadow"] is True


@pytest.mark.asyncio
async def test_toggle_on_still_refuses_untrusted_context(tmp_path):
    router, _, cloud = make_router(tmp_path, mode="shadow")
    router.set_use_api_model(True)
    await router.startup()
    await drain(router, dataclasses.replace(MIND, trust=UNTRUSTED))

    assert cloud.calls == []


@pytest.mark.asyncio
async def test_toggle_on_still_refuses_when_budget_is_zero(tmp_path):
    router, _, cloud = make_router(tmp_path, mode="shadow")
    router.config = dataclasses.replace(
        router.config, budget=dataclasses.replace(router.config.budget, hourly_usd=0)
    )
    router.set_use_api_model(True)
    await router.startup()
    await drain(router, MIND)

    assert cloud.calls == []
    assert audit_rows(tmp_path)[0]["reason"] == "BUDGET_EXHAUSTED"


@pytest.mark.asyncio
async def test_toggle_off_keeps_local_only_roles_local(tmp_path):
    router, local, cloud = make_router(tmp_path, mode="enforce")
    router.set_use_api_model(True)
    await router.startup()
    await drain(router, ModelRequest(role="reflex", messages=({"role": "user", "content": "hi"},)))

    assert cloud.calls == []
    assert len(local.calls) == 1


@pytest.mark.asyncio
async def test_ensure_cloud_picks_up_a_key_pasted_after_launch(tmp_path, monkeypatch):
    # Isolated from the real secrets.local.yaml/.env on this machine: the point
    # is the transition, not whether the developer has a key today.
    monkeypatch.setattr(config, "resolve_api_key", lambda *args, **kwargs: None)
    router = Router(
        make_config(tmp_path),
        catalog=FakeCatalog(),
        local_provider=FakeProvider("local_ollama"),
        cloud_provider=None,
        audit=AuditWriter(tmp_path / "audit.jsonl"),
        ledger=SpendLedger(tmp_path / "spend.db"),
    )
    assert await router.ensure_cloud() is False

    monkeypatch.setattr(config, "resolve_api_key", lambda *args, **kwargs: "pasted-just-now")
    assert await router.ensure_cloud() is True
    assert router.cloud_provider is not None
    assert router.state.cloud_available is True
