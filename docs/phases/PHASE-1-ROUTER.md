# Phase 1 — Foundation + router

The phase that makes "local by default, cloud when it earns it" real, without
changing a single thing about how she currently behaves.

Read with `ROUTING.md` (the policy) and `config/config.example.yaml` (the
surface). P0 already landed the bus, the config v2 loader, and the tests.

---

## 0. Goal

Every model call in Yuri goes through one router. In `shadow` mode the router
computes a decision, writes the audit row, and executes the local path anyway.
In `enforce` mode it does what it decided, per role. The conversation must be
byte-identical to today in shadow mode; that is the whole point of the phase.

Dependencies: P0 (done). Nothing here needs the network to be tested.

---

## 1. Module layout

```
orchestrator/router/
  __init__.py        Router facade: complete(role, request) -> AsyncIterator[Chunk]
  types.py           ModelRequest, ModelChunk, RouteDecision, Usage, ProviderError
  decision.py        decide(...) -- pure, no I/O, no asyncio
  discovery.py       /v1/models fetch, cache, capability lookup
  budget.py          cost estimation, caps, spend ledger
  audit.py           JSONL writer (before + after)
  redact.py          secrets/paths/emails/digits, excerpt caps
  providers/
    base.py          Provider protocol + health
    ollama_native.py NDJSON streaming, tools, images, format
    airouter.py      SSE streaming, Bearer auth, tool calls, usage
```

`app.py`, `memory/consolidation.py`, `memory/forget.py`, `task_guide.py`,
`ocr_watch.py` and `tools/vision.py` stop importing `llm.py` directly and call
`router.complete(...)` with an explicit role. `llm.py` keeps its low-level
parsing helpers (they are tested) and becomes the provider's implementation
detail rather than the public entry point.

---

## 2. Types

```python
@dataclass(frozen=True)
class ModelRequest:
    role: str                       # reflex | vision | classifier | embed | memory_writer | mind
    messages: list[dict]            # OpenAI-ish, provider adapts
    tools: list[dict] | None = None
    stream: bool = True
    privacy_class: str = "internal" # public | internal | secret (max over messages)
    trust: str = "trusted"          # untrusted blocks cloud egress
    explicit_escalation: bool = False
    background: bool = False
    job_id: str | None = None
    max_tokens: int | None = None
    temperature: float | None = None
    schema: dict | None = None      # structured output

@dataclass(frozen=True)
class ModelChunk:
    kind: str                       # content | tool_calls | usage | error
    text: str = ""
    calls: tuple[dict, ...] = ()
    usage: Usage | None = None
    error: str | None = None

@dataclass(frozen=True)
class RouteDecision:
    role: str
    reason: str                     # one of ROUTING.md §4's codes
    candidates: tuple[str, ...]     # ["airouter:deepseek/...", "local:qwen3.5:9b"]
    chosen: str
    shadow: bool
    est_prompt_tokens: int
    est_cost_usd: float
```

`decide()` takes `(role, request, config, state)` where `state` carries budget
spent, offline flag, catalog freshness. It returns a decision and never raises
for policy reasons; a bad request is a `ValueError` before it gets there.

---

## 3. Provider contract

```python
class Provider(Protocol):
    name: str
    async def complete(self, req: ModelRequest, *, model: str) -> AsyncIterator[ModelChunk]: ...
    async def health(self) -> bool: ...
    def caps(self, model: str) -> ModelCaps | None: ...
```

- `ollama_native` reuses `llm.py`'s framing: `fold_system_messages`, one
  `num_ctx`, `think`, `keep_alive`, streaming tool calls, `format` schemas,
  inline `images` for vision. No behavior change is allowed here; the existing
  `test_llm.py` suite must pass against the moved code.
- `airouter` is new code: `POST {base}/chat/completions`, `Bearer` from
  `resolve_api_key`, SSE `data:` lines, `[DONE]`, `tool_calls` deltas, and the
  `usage` block when present. Timeouts: connect 5 s, read 120 s, no retry inside
  the provider (the chain owns retries).
- Both map their wire format into `ModelChunk`; nothing above the provider sees
  provider-specific shapes.

---

## 4. Discovery

- `GET {base}/models` (no key needed today), parse `id`, `context_length`,
  `max_tokens`, `pricing`, `tags`, `zdr`, `type`.
- Cache `config/providers/models_cache.json` (path configurable) with
  `fetched_at`; refresh if older than `models_refresh_hours` (24) at startup.
- If the fetch fails: use the cache. If there is no cache: mark the provider
  unavailable, reason `CLOUD_DISABLED`, print one warning. Never guess a price.
- Model selection for a role: explicit id from config; if missing from the
  catalog, log once with the closest three ids by edit distance (a real help
  when a provider renames something) and fall back to the chain's next entry.

---

## 5. Budget and audit

- `budget.py` holds a spend ledger in memory and appends to SQLite
  (`spend(ts, provider, model, prompt_tokens, completion_tokens, cost_usd,
  estimated)`), so caps survive restarts.
- Before a call: estimate prompt tokens (chars/4, calibrated by
  `context_budget`'s ratio), estimate cost with the worst-case completion
  (`max_tokens`), refuse if it would cross hour/day/month.
- After: reconcile with the provider's `usage`; store both. The ledger is the
  source of truth, not the estimate.
- `audit.py` writes the row in `ROUTING.md` §7 before the request, and a second
  short `model.call.finished` line after. A crash mid-call leaves a row with
  `error: "incomplete"` rather than silence.
- Redaction runs in `redact.py` before the audit row is built, so counts are
  available and content is not.

---

## 6. Wiring, in order

1. `providers/ollama_native.py` + tests, behavior preserved by `test_llm.py`.
2. `decision.py` + tests (pure, no providers).
3. `budget.py`, `audit.py`, `redact.py` + tests.
4. `providers/airouter.py` against a fake SSE server (no real key).
5. `discovery.py` with a fixture catalog + a live smoke test behind `--live`.
6. `Router` facade; `app.py` switched role by role. In shadow mode the facade
   ignores the chosen cloud candidate and calls local.
7. `--check` CLI: validate config, print resolved chains, print catalog status.

Every switch is a separate commit so a regression can be bisected to a role.

---

## 7. Tests

- `test_route_rules.py` — one per rule, precedence order, boundary values.
- `test_route_privacy.py` — a `secret` message never reaches a cloud fake;
  untrusted context forces local; memory writer never cloud.
- `test_route_budget.py` — exact cap, over cap, estimate vs actual, persistence
  across a restart (reload the SQLite ledger).
- `test_route_fallback.py` — 429, 500, timeout, malformed SSE, no `usage`,
  local failure; exactly one retry.
- `test_route_shadow.py` — cloud fake's `call_count == 0`; decision logged.
- `test_discovery.py` — fresh cache, stale refresh, endpoint down with/without
  cache, unknown model id.
- `test_providers_ollama.py` — the existing `test_llm.py`, moved and kept.
- `test_providers_airouter.py` — SSE chunking, tool calls, usage, abort.

Gate: all of the above plus `ROUTING.md` §12; 50 replayed turns in shadow with
zero cloud calls and one audit row each; Luna's suite green.

---

## 8. Risks

1. **SSE differences.** Airouter may omit `usage`, split tool-call deltas
   differently, or send keep-alives. Mitigation: fake-server tests with those
   exact shapes, and tolerate missing usage (`cost_estimated: true`).
2. **Catalog drift.** A model id can vanish. Mitigation: discovery + fallback
   chain + a nearest-id log line.
3. **Shadow drift.** A refactor that changes output is invisible until enforce.
   Mitigation: a replay test that hashes the reflex output for fixed inputs
   (temperature 0) before and after each wiring commit.
4. **Ledger corruption.** A half-written SQLite row under kill -9. Mitigation:
   WAL + one row per call in a transaction, plus a rebuild-from-audit path.
5. **Accidental enforce.** Mitigation: `mode: shadow` default, `enforce_roles`
   empty by default, and a startup banner naming the mode.

---

## 9. After this phase

`mode: enforce` for `mind` only, after a week of shadow logs are reviewed. Then
P2, where the voice path must stay local regardless of what the router says.
