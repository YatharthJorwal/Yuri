"""The router: the only way Yuri reaches a model.

Callers ask for a role, never a model. The router decides (pure function), writes
the audit row, checks the budget, and executes -- local unless the decision says
otherwise and the mode allows it.

In `shadow` mode the decision is still computed and logged, but execution is
forced local. That is the default, and it is how the decision rules get exercised
against real traffic before any of them are allowed to cost money.
"""

from __future__ import annotations

import asyncio
import dataclasses
import time
from typing import AsyncIterator

import context_budget
from config import CONFIG, Config, ProviderConfig, RoleConfig
from router import decision as decide_mod
from router.audit import AuditWriter
from router.budget import SpendLedger, actual_cost, estimate_cost
from router.decision import decide
from router.discovery import ModelCatalog, repo_path
from router.errors import ProviderError, RouterError
from router.providers import AirouterProvider, OllamaNativeProvider
from router.redact import redact_messages
from router.types import (
    INTERNAL,
    SECRET,
    TRUSTED,
    ModelChunk,
    ModelRequest,
    RouteDecision,
    RouterState,
    Usage,
)

__all__ = [
    "Router",
    "ROUTER",
    "ModelChunk",
    "ModelRequest",
    "RouteDecision",
    "RouterState",
    "complete",
    "describe_image",
    "stream_text",
    "stream_with_tools",
    "summarize",
]

_MODEL_FALLBACKS = {"reflex": "qwen3.5:9b"}


def _default_role(role: str) -> RoleConfig:
    # Unknown role -> local only. Fail safe, not fail open.
    return RoleConfig(chain=("local_ollama",), local_only=True)


class Router:
    def __init__(
        self,
        config: Config = CONFIG,
        *,
        catalog: ModelCatalog | None = None,
        local_provider=None,
        cloud_provider=None,
        audit: AuditWriter | None = None,
        ledger: SpendLedger | None = None,
        clock=time.time,
    ) -> None:
        self.config = config
        self.clock = clock
        self._started = False
        self._lock = asyncio.Lock()

        local_cfg = config.providers.get("local_ollama")
        cloud_cfg = config.providers.get("airouter")
        self.local_provider = local_provider or OllamaNativeProvider()
        self.cloud_provider = cloud_provider
        self._api_key_present = cloud_provider is not None
        if cloud_provider is None and cloud_cfg is not None:
            from config import resolve_api_key

            key = resolve_api_key(cloud_cfg.api_key_env)
            self._api_key_present = bool(key)
            if key:
                self.cloud_provider = AirouterProvider(cloud_cfg, key)

        self.catalog = catalog or ModelCatalog(
            cloud_cfg.base_url if cloud_cfg else "https://api.airouter.in/v1",
            (cloud_cfg.models_cache if cloud_cfg else "data/provider_models.json"),
            ttl_hours=(cloud_cfg.models_refresh_hours if cloud_cfg else 24),
            models_url=(cloud_cfg.models_url if cloud_cfg else None),
        )
        self.audit = audit or AuditWriter(repo_path(config.audit.path))
        self.ledger = ledger or SpendLedger(repo_path(config.budget.ledger_path))
        self.state = RouterState(cloud_available=False)
        self._cloud_cfg: ProviderConfig | None = cloud_cfg
        # None = follow routing.mode/enforce_roles. True/False = the user's live
        # choice from the tray/quick-action toggle, which outranks the config
        # until they change it again (see settings.py).
        self.use_api_model_override: bool | None = None

    def set_use_api_model(self, enabled: bool | None) -> None:
        self.use_api_model_override = enabled

    def _enforce(self, role: str) -> bool:
        """May this role execute a cloud decision right now?"""
        role_cfg = self.config.roles.get(role) or _default_role(role)
        if role_cfg.local_only:
            return False
        if self.use_api_model_override is False:
            return False
        if self.use_api_model_override is True:
            return True
        return (
            self.config.routing.mode == "enforce"
            and role in self.config.routing.enforce_roles
        )

    # -- lifecycle -----------------------------------------------------------

    async def ensure_cloud(self) -> bool:
        """Make cloud usable right now if it can be, without a restart.

        A key pasted into `.env` after launch is picked up here, the provider is
        built, and a stale/absent catalog is fetched. Returns whether cloud is
        actually available afterwards. Never raises: a toggle that cannot reach
        the network must still respond with "no key" instead of an exception.
        """
        cloud_cfg = self._cloud_cfg
        if cloud_cfg is None or not cloud_cfg.enabled:
            self.state.cloud_available = False
            return False
        if self.cloud_provider is None:
            from config import resolve_api_key

            key = resolve_api_key(cloud_cfg.api_key_env)
            if key:
                self.cloud_provider = AirouterProvider(cloud_cfg, key)
                self._api_key_present = True
        if self.cloud_provider is None:
            self.state.cloud_available = False
            return False
        try:
            loaded = self.catalog.load_cache()
            if not loaded or self.catalog.is_stale():
                await self.catalog.refresh(force=not loaded)
        except Exception as exc:  # noqa: BLE001 -- a toggle must not crash the app
            print(f"[yuri] router: catalog refresh failed: {exc}", flush=True)
        self.state.cloud_available = bool(self.catalog.available and self._api_key_present)
        return self.state.cloud_available

    async def _ensure_started(self) -> None:
        if self._started:
            return
        async with self._lock:
            if self._started:
                return
            await self.startup()
            self._started = True

    async def startup(self) -> bool:
        """Load the catalog and the spend ledger. Returns cloud availability."""
        cloud_cfg = self._cloud_cfg
        cloud_ok = bool(
            cloud_cfg is not None
            and cloud_cfg.enabled
            and self.cloud_provider is not None
            and self._api_key_present
        )
        if cloud_ok:
            loaded = self.catalog.load_cache()
            if not loaded or self.catalog.is_stale():
                await self.catalog.refresh(force=not loaded)
            cloud_ok = self.catalog.available
            if not cloud_ok:
                print(
                    "[yuri] router: cloud catalog unavailable -- Mind stays local"
                    + (f" ({self.catalog.last_error})" if self.catalog.last_error else ""),
                    flush=True,
                )
        self.state.cloud_available = cloud_ok
        now = self.clock()
        hour, day, month = self.ledger.totals(now)
        self.state.spend_hour_usd, self.state.spend_day_usd, self.state.spend_month_usd = hour, day, month
        return cloud_ok

    # -- decisions -----------------------------------------------------------

    def _estimate(self, request: ModelRequest) -> int:
        if not request.messages:
            return max(1, sum(len(message.get("content") or "") for message in request.messages) // 4)
        try:
            return context_budget.estimate_prompt(
                [dict(message) for message in request.messages], list(request.tools)
            )
        except Exception:  # noqa: BLE001 -- estimation must never block a turn
            return max(1, sum(len(str(m.get("content") or "")) for m in request.messages) // 4)

    def _anticipated_cloud_cost(self, role: str, prompt_tokens: int, max_tokens: int) -> float:
        worst = 0.0
        role_cfg = self.config.roles.get(role) or _default_role(role)
        for provider in role_cfg.chain:
            if provider == "local_ollama":
                continue
            model = decide_mod.model_for(provider, role, self.config.models)
            price_in, price_out = self.catalog.price(model)
            worst = max(worst, estimate_cost(prompt_tokens, max_tokens, price_in, price_out))
        return worst

    def _effective_privacy(self, request: ModelRequest) -> str:
        for message in request.messages:
            if message.get("images"):
                return SECRET
        return request.privacy_class

    def route(self, request: ModelRequest) -> RouteDecision:
        """Compute (but do not execute) the decision. Public so `--check` and the
        tests can look at it without a provider."""
        role_cfg = self.config.roles.get(request.role) or _default_role(request.role)
        prompt_tokens = self._estimate(request)
        max_tokens = request.max_tokens or self.config.llm.max_tokens
        state = dataclasses.replace(
            self.state,
            est_prompt_tokens=prompt_tokens,
            est_cost_usd=self._anticipated_cloud_cost(request.role, prompt_tokens, max_tokens),
            tool_count=len(request.tools),
        )
        decision = decide(
            role=request.role,
            role_cfg=role_cfg,
            request=dataclasses.replace(request, privacy_class=self._effective_privacy(request)),
            routing=self.config.routing,
            budget=self.config.budget,
            state=state,
            models=self.config.models,
            providers=self.config.providers,
            now=self.clock(),
        )
        if not self._enforce(request.role):
            decision = dataclasses.replace(decision, shadow=True)
        return decision

    def _execution_chain(self, decision: RouteDecision) -> list[str]:
        if decision.shadow or not self._enforce(decision.role):
            # Shadow, or a role we have not opted into: run local, keep the
            # decision as the record of what would have happened.
            local = [c for c in decision.candidates if c.startswith("local_ollama:")]
            return local or [decision.chosen]
        chain = [decision.chosen] + [c for c in decision.candidates if c != decision.chosen]
        return chain

    def _provider(self, name: str):
        if name == "local_ollama":
            return self.local_provider
        if name == "airouter":
            if self.cloud_provider is None:
                raise RouterError("airouter selected but no provider is configured")
            return self.cloud_provider
        raise RouterError(f"unknown provider {name!r}")

    # -- execution -----------------------------------------------------------

    async def complete(self, request: ModelRequest) -> AsyncIterator[ModelChunk]:
        await self._ensure_started()
        decision = self.route(request)
        chain = self._execution_chain(decision)
        first = chain[0]
        executed_decision = dataclasses.replace(decision, executed=first)

        redactions: dict[str, int] = {}
        if first.startswith("airouter:"):
            # Counts for the audit row; the actual redacted copy is rebuilt per
            # target below, so a fallback to local still sends the original.
            _, redactions = self._redact_for_cloud(request)

        trace_id = self.audit.start(
            decision=executed_decision,
            privacy_class=self._effective_privacy(request),
            trust=request.trust,
            redactions=redactions,
            mode=self.config.routing.mode,
            job_id=request.job_id,
        )
        started = time.monotonic()
        usage: Usage | None = None
        fallback_used = False
        error: str | None = None
        cost: float | None = None
        try:
            for index, target in enumerate(chain):
                provider_name, _, model = target.partition(":")
                if index > 0:
                    fallback_used = True
                try:
                    provider = self._provider(provider_name)
                    outbound = request
                    if provider_name == "airouter":
                        outbound, _ = self._redact_for_cloud(request)
                    async for chunk in provider.complete(outbound, model=model):
                        if chunk.usage is not None:
                            usage = chunk.usage
                        yield chunk
                    if provider_name == "airouter":
                        price_in, price_out = self.catalog.price(model)
                        cost = actual_cost(
                            usage.prompt_tokens if usage else None,
                            usage.completion_tokens if usage else None,
                            price_in,
                            price_out,
                        )
                    break
                except ProviderError as exc:
                    error = str(exc)
                    if provider_name == "airouter":
                        self.state.offline_until = self.clock() + 60.0
                    if index + 1 < len(chain):
                        continue
                    raise
        finally:
            latency_ms = (time.monotonic() - started) * 1000.0
            if cost:
                try:
                    self.ledger.record(
                        first.split(":", 1)[0],
                        first.partition(":")[2],
                        cost,
                        prompt_tokens=usage.prompt_tokens if usage else None,
                        completion_tokens=usage.completion_tokens if usage else None,
                        estimated=bool(usage and usage.estimated),
                    )
                    hour, day, month = self.ledger.totals(self.clock())
                    self.state.spend_hour_usd, self.state.spend_day_usd, self.state.spend_month_usd = hour, day, month
                except Exception as exc:  # noqa: BLE001 -- bookkeeping must not kill a reply
                    print(f"[yuri] router: spend ledger write failed: {exc}", flush=True)
            self.audit.finish(
                trace_id,
                role=request.role,
                latency_ms=latency_ms,
                prompt_tokens=usage.prompt_tokens if usage else None,
                completion_tokens=usage.completion_tokens if usage else None,
                cost_usd=cost,
                estimated=bool(usage and usage.estimated),
                fallback_used=fallback_used,
                error=error,
            )

    def _redact_for_cloud(self, request: ModelRequest) -> tuple[ModelRequest, dict[str, int]]:
        if any(message.get("images") for message in request.messages):
            raise RouterError("refusing to send images to a cloud provider")
        settings = dict(self.config.privacy.redaction)
        messages, counts = redact_messages([dict(m) for m in request.messages], settings)
        return dataclasses.replace(request, messages=tuple(messages)), counts

    # -- convenience, shaped like llm.py so callers barely change ------------

    async def stream_text(
        self,
        role: str,
        messages: list[dict],
        *,
        privacy_class: str = INTERNAL,
        trust: str = TRUSTED,
        background: bool = False,
        explicit_escalation: bool = False,
        job_id: str | None = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        request = ModelRequest(
            role=role,
            messages=tuple(messages),
            stream=True,
            privacy_class=privacy_class,
            trust=trust,
            background=background,
            explicit_escalation=explicit_escalation,
            job_id=job_id,
            **kwargs,
        )
        async for chunk in self.complete(request):
            if chunk.kind == "content":
                yield chunk.text

    async def stream_with_tools(
        self,
        role: str,
        messages: list[dict],
        tools: list[dict],
        *,
        privacy_class: str = INTERNAL,
        trust: str = TRUSTED,
        **kwargs,
    ) -> AsyncIterator[dict]:
        request = ModelRequest(
            role=role,
            messages=tuple(messages),
            tools=tuple(tools),
            stream=True,
            privacy_class=privacy_class,
            trust=trust,
            **kwargs,
        )
        async for chunk in self.complete(request):
            if chunk.kind == "content":
                yield {"type": "content", "text": chunk.text}
            elif chunk.kind == "tool_calls":
                yield {"type": "tool_calls", "calls": list(chunk.calls)}

    async def describe_image(
        self,
        role: str,
        prompt: str,
        image_b64: str,
        json_schema: dict | None = None,
    ) -> str:
        """Vision always runs local: pixels are `secret` and rule 2 fires."""
        request = ModelRequest(
            role=role,
            messages=({"role": "user", "content": prompt, "images": [image_b64]},),
            privacy_class=SECRET,
            stream=False,
        )
        await self._ensure_started()
        decision = self.route(request)
        model = decide_mod.model_for("local_ollama", role, self.config.models)
        trace_id = self.audit.start(
            decision=dataclasses.replace(decision, executed="local_ollama:" + model),
            privacy_class=SECRET,
            trust=TRUSTED,
            mode=self.config.routing.mode,
        )
        started = time.monotonic()
        error = None
        try:
            return await self.local_provider.describe_image(
                prompt, image_b64, model=model, json_schema=json_schema
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            self.audit.finish(
                trace_id,
                role=role,
                latency_ms=(time.monotonic() - started) * 1000.0,
                error=error,
            )


# -- module-level singleton + functions ---------------------------------------
# Importing this module must not hit the network; Router.startup() is called on
# the first complete()/describe_image(), i.e. from the app's own lifespan.

_router: Router | None = None


def get_router() -> Router:
    global _router
    if _router is None:
        _router = Router()
    return _router


async def complete(request: ModelRequest) -> AsyncIterator[ModelChunk]:
    async for chunk in get_router().complete(request):
        yield chunk


async def stream_text(role: str, messages: list[dict], **kwargs) -> AsyncIterator[str]:
    async for text in get_router().stream_text(role, messages, **kwargs):
        yield text


async def stream_with_tools(role: str, messages: list[dict], tools: list[dict], **kwargs) -> AsyncIterator[dict]:
    async for event in get_router().stream_with_tools(role, messages, tools, **kwargs):
        yield event


async def describe_image(role: str, prompt: str, image_b64: str, json_schema: dict | None = None) -> str:
    return await get_router().describe_image(role, prompt, image_b64, json_schema)


def summarize(router: Router | None = None) -> str:
    """Human-readable status, for `check.py` and the startup log."""
    r = router or get_router()
    lines = [f"mode: {r.config.routing.mode} (enforce roles: {', '.join(r.config.routing.enforce_roles) or '-'})"]
    lines.append(
        f"cloud: {'available' if r.state.cloud_available else 'unavailable'}"
        f"  catalog: {len(r.catalog.model_ids())} models"
        + (f"  last error: {r.catalog.last_error}" if r.catalog.last_error else "")
    )
    hour, day, month = r.ledger.totals()
    lines.append(
        f"spend: hour ${hour:.4f}/{r.config.budget.hourly_usd}  "
        f"day ${day:.4f}/{r.config.budget.daily_usd}  "
        f"month ${month:.4f}/{r.config.budget.monthly_usd}"
    )
    for name, role in r.config.roles.items():
        model = decide_mod.model_for("local_ollama", name, r.config.models)
        print_chain = " -> ".join(
            decide_mod.model_for(provider, name, r.config.models) if provider == "local_ollama"
            else f"{provider}:{decide_mod.model_for(provider, name, r.config.models)}"
            for provider in role.chain
        )
        lines.append(f"  {name:<14} {'local-only' if role.local_only else 'escalatable':<11} {print_chain}")
    return "\n".join(lines)
