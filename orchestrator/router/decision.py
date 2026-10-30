"""The routing decision.

Pure: no I/O, no clock, no asyncio, no provider. Everything it needs is passed
in, which is what makes the ten rules in ROUTING.md §4 testable one at a time,
including their precedence, which is the part that actually goes wrong.

Rule order is the contract. Do not reorder without editing ROUTING.md.
"""

from __future__ import annotations

from typing import Mapping

from config import (
    BudgetConfig,
    ModelsConfig,
    ProviderConfig,
    RoleConfig,
    RoutingConfig,
)
from router.types import ModelRequest, RouteDecision, RouterState, SECRET, UNTRUSTED

# Reason codes. These strings are the audit log's vocabulary.
ROLE_PINNED_LOCAL = "ROLE_PINNED_LOCAL"
PRIVACY_LOCAL_ONLY = "PRIVACY_LOCAL_ONLY"
UNTRUSTED_CONTEXT = "UNTRUSTED_CONTEXT"
CLOUD_DISABLED = "CLOUD_DISABLED"
OFFLINE = "OFFLINE"
BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"
EXPLICIT_ESCALATION = "EXPLICIT_ESCALATION"
BACKGROUND_JOB = "BACKGROUND_JOB"
COMPLEXITY_THRESHOLD = "COMPLEXITY_THRESHOLD"
LOCAL_DEFAULT = "LOCAL_DEFAULT"

LOCAL_PROVIDER = "local_ollama"


def model_for(provider: str, role: str, models: ModelsConfig) -> str:
    """Which model a (provider, role) pair uses. A null local vision model means
    'the reflex model', not 'no model'."""
    if provider == LOCAL_PROVIDER:
        name = models.local.get(role)
        if name is None:
            name = models.local.get("reflex")
        return name or "qwen3.5:9b"
    return models.cloud.get(role) or models.cloud.get("mind") or ""


def _render(provider: str, role: str, models: ModelsConfig) -> str:
    return f"{provider}:{model_for(provider, role, models)}"


def _local_chain(role: str, models: ModelsConfig) -> list[str]:
    return [_render(LOCAL_PROVIDER, role, models)]


def _cloud_chain(
    role: str,
    role_cfg: RoleConfig,
    models: ModelsConfig,
    providers: Mapping[str, ProviderConfig],
    available: bool,
) -> list[str]:
    if not available:
        return []
    out: list[str] = []
    for provider in role_cfg.chain:
        if provider == LOCAL_PROVIDER:
            continue
        cfg = providers.get(provider)
        if cfg is None or not cfg.enabled:
            continue
        rendered = _render(provider, role, models)
        if rendered.rsplit(":", 1)[-1]:
            out.append(rendered)
    return out


def _budget_exhausted(budget: BudgetConfig, state: RouterState) -> bool:
    """Caps are per window. A cap of 0 means 'never spend' (legal and useful
    before a key is trusted); a negative cap means unlimited."""
    if not budget.enabled:
        return False
    windows = (
        (state.spend_hour_usd, budget.hourly_usd),
        (state.spend_day_usd, budget.daily_usd),
        (state.spend_month_usd, budget.monthly_usd),
    )
    for spent, cap in windows:
        if cap < 0:
            continue
        if cap == 0 or spent + state.est_cost_usd > cap:
            return True
    return False


def _escalation_reason(request: ModelRequest, routing: RoutingConfig, state: RouterState) -> str | None:
    when = routing.escalate_when
    if request.explicit_escalation and when.get("explicit_think_hard", True):
        return EXPLICIT_ESCALATION
    if request.background and when.get("background_job", True):
        return BACKGROUND_JOB
    if state.steps_estimate >= int(when.get("task_steps_gte", 4)):
        return COMPLEXITY_THRESHOLD
    if state.tool_count >= int(when.get("tool_calls_gte", 2)):
        return COMPLEXITY_THRESHOLD
    if state.est_prompt_tokens >= int(when.get("prompt_tokens_gte", 6000)):
        return COMPLEXITY_THRESHOLD
    return None


def decide(
    *,
    role: str,
    role_cfg: RoleConfig,
    request: ModelRequest,
    routing: RoutingConfig,
    budget: BudgetConfig,
    state: RouterState,
    models: ModelsConfig,
    providers: Mapping[str, ProviderConfig],
    now: float = 0.0,
) -> RouteDecision:
    local = _local_chain(role, models)
    cloud = _cloud_chain(role, role_cfg, models, providers, state.cloud_available)

    def local_only(reason: str) -> RouteDecision:
        return RouteDecision(
            role=role,
            reason=reason,
            candidates=tuple(local),
            chosen=local[0],
            est_prompt_tokens=state.est_prompt_tokens,
            est_cost_usd=state.est_cost_usd,
        )

    def escalate(reason: str) -> RouteDecision:
        return RouteDecision(
            role=role,
            reason=reason,
            candidates=tuple(cloud + local),
            chosen=cloud[0],
            est_prompt_tokens=state.est_prompt_tokens,
            est_cost_usd=state.est_cost_usd,
        )

    # 1. The role itself may not leave the machine (reflex, vision, classifier,
    #    embed, memory_writer). This is the strongest rule: it is a startup
    #    invariant, re-checked here so a config edit at runtime cannot bypass it.
    if role_cfg.local_only:
        return local_only(ROLE_PINNED_LOCAL)

    # 2. Secret content never leaves, regardless of role.
    if request.privacy_class == SECRET:
        return local_only(PRIVACY_LOCAL_ONLY)

    # 3. Injected / screen-sourced content in the turn blocks egress.
    if request.trust == UNTRUSTED and routing.untrusted_context_blocks_cloud:
        return local_only(UNTRUSTED_CONTEXT)

    # 4. No cloud provider at all: disabled, no key, or no catalog.
    if not cloud:
        return local_only(CLOUD_DISABLED)

    # 5. We already failed to reach it recently.
    if state.offline_until and now < state.offline_until:
        return local_only(OFFLINE)

    # 6. Spending caps.
    if _budget_exhausted(budget, state):
        return local_only(BUDGET_EXHAUSTED)

    # 7-9. The reasons to actually pay for it, most specific first.
    reason = _escalation_reason(request, routing, state)
    if reason:
        return escalate(reason)

    # 10. Nothing argued for the cloud.
    return local_only(LOCAL_DEFAULT)
