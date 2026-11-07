"""Decision rules: one test per rule, then the precedence between them."""

import dataclasses

import pytest

import config
from config import BudgetConfig, RoleConfig, RoutingConfig
from router import decision as d
from router.types import ModelRequest, RouterState, SECRET, UNTRUSTED

MODELS = config.CONFIG.models
PROVIDERS = {
    **config.CONFIG.providers,
    "airouter": dataclasses.replace(config.CONFIG.providers["airouter"], enabled=True),
}
MIND = config.CONFIG.roles["mind"]
REFLEX = config.CONFIG.roles["reflex"]
ROUTING = config.CONFIG.routing
BUDGET = BudgetConfig(enabled=True, hourly_usd=1.0, daily_usd=10.0, monthly_usd=100.0)


def state(**overrides):
    base = RouterState(
        cloud_available=True,
        est_prompt_tokens=200,
        est_cost_usd=0.001,
        steps_estimate=1,
        tool_count=0,
    )
    return dataclasses.replace(base, **overrides)


def run(
    *,
    role="mind",
    role_cfg=MIND,
    request=None,
    routing=ROUTING,
    budget=BUDGET,
    st=None,
    now=1000.0,
):
    request = request or ModelRequest(role=role)
    return d.decide(
        role=role,
        role_cfg=role_cfg,
        request=request,
        routing=routing,
        budget=budget,
        state=st or state(),
        models=MODELS,
        providers=PROVIDERS,
        now=now,
    )


def test_local_only_role_is_pinned():
    result = run(role="reflex", role_cfg=REFLEX)
    assert result.reason == d.ROLE_PINNED_LOCAL
    assert result.chosen.startswith("local_ollama:")


def test_secret_content_never_escalates():
    result = run(request=ModelRequest(role="mind", privacy_class=SECRET, explicit_escalation=True))
    assert result.reason == d.PRIVACY_LOCAL_ONLY


def test_untrusted_context_blocks_cloud():
    result = run(request=ModelRequest(role="mind", trust=UNTRUSTED, background=True))
    assert result.reason == d.UNTRUSTED_CONTEXT


def test_untrusted_context_can_be_configured_open():
    routing = dataclasses.replace(ROUTING, untrusted_context_blocks_cloud=False)
    result = run(request=ModelRequest(role="mind", trust=UNTRUSTED, background=True), routing=routing)
    assert result.reason == d.BACKGROUND_JOB


def test_cloud_disabled_when_unavailable():
    result = run(st=state(cloud_available=False))
    assert result.reason == d.CLOUD_DISABLED
    assert all(c.startswith("local_ollama:") for c in result.candidates)


def test_offline_cooldown():
    result = run(st=state(offline_until=2000.0), now=1500.0)
    assert result.reason == d.OFFLINE


def test_offline_expired_is_not_blocking():
    result = run(st=state(offline_until=2000.0), now=2500.0, request=ModelRequest(role="mind", background=True))
    assert result.reason == d.BACKGROUND_JOB


def test_budget_exhausted_hour_cap():
    st = state(spend_hour_usd=1.0)
    result = run(st=st)
    assert result.reason == d.BUDGET_EXHAUSTED


def test_budget_zero_means_never_spend():
    budget = dataclasses.replace(BUDGET, hourly_usd=0)
    assert run(budget=budget).reason == d.BUDGET_EXHAUSTED


def test_budget_negative_means_unlimited():
    budget = dataclasses.replace(BUDGET, hourly_usd=-1, daily_usd=-1, monthly_usd=-1)
    assert run(budget=budget).reason == d.LOCAL_DEFAULT


def test_explicit_escalation():
    result = run(request=ModelRequest(role="mind", explicit_escalation=True))
    assert result.reason == d.EXPLICIT_ESCALATION
    assert result.chosen.startswith("airouter:")


def test_background_job_escalates():
    assert run(request=ModelRequest(role="mind", background=True)).reason == d.BACKGROUND_JOB


def test_step_estimate_escalates():
    assert run(st=state(steps_estimate=4)).reason == d.COMPLEXITY_THRESHOLD


def test_tool_count_escalates():
    assert run(st=state(tool_count=2)).reason == d.COMPLEXITY_THRESHOLD


def test_prompt_size_escalates():
    assert run(st=state(est_prompt_tokens=6000)).reason == d.COMPLEXITY_THRESHOLD


def test_nothing_escalates():
    result = run()
    assert result.reason == d.LOCAL_DEFAULT
    assert result.chosen.startswith("local_ollama:")


# -- precedence: the order is the contract -----------------------------------


def test_pinned_role_beats_secret():
    result = run(role="reflex", role_cfg=REFLEX, request=ModelRequest(role="reflex", privacy_class=SECRET))
    assert result.reason == d.ROLE_PINNED_LOCAL


def test_secret_beats_untrusted():
    result = run(request=ModelRequest(role="mind", privacy_class=SECRET, trust=UNTRUSTED))
    assert result.reason == d.PRIVACY_LOCAL_ONLY


def test_untrusted_beats_offline():
    result = run(request=ModelRequest(role="mind", trust=UNTRUSTED), st=state(offline_until=2000.0))
    assert result.reason == d.UNTRUSTED_CONTEXT


def test_offline_beats_explicit():
    result = run(
        request=ModelRequest(role="mind", explicit_escalation=True),
        st=state(offline_until=2000.0),
    )
    assert result.reason == d.OFFLINE


def test_budget_beats_explicit():
    result = run(
        request=ModelRequest(role="mind", explicit_escalation=True),
        st=state(spend_day_usd=10.0),
    )
    assert result.reason == d.BUDGET_EXHAUSTED


def test_explicit_beats_background():
    result = run(request=ModelRequest(role="mind", explicit_escalation=True, background=True))
    assert result.reason == d.EXPLICIT_ESCALATION


def test_escalated_candidates_include_local_fallback():
    result = run(request=ModelRequest(role="mind", background=True))
    assert result.candidates[0].startswith("airouter:")
    assert any(c.startswith("local_ollama:") for c in result.candidates)


@pytest.mark.parametrize("role_cfg", [REFLEX, MIND])
def test_decision_is_reproducible(role_cfg):
    a = run(role_cfg=role_cfg)
    b = run(role_cfg=role_cfg)
    assert (a.reason, a.chosen, a.candidates) == (b.reason, b.chosen, b.candidates)
