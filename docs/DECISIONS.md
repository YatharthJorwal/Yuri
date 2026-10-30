# Decisions

Lean log. One entry per decision that would otherwise be re-argued. Format:
decision, why, what was rejected, what it costs. Append-only; superseding an
entry gets a new number that says so.

---

## D-001 — Port Luna's shell, restructure its orchestrator

**Decision.** Copy the Tauri shell nearly verbatim. Copy the orchestrator file by
file into service-shaped modules, preserving logic. Rewrite only `config.py`,
`stt.py`, `tts.py` and the transport split.

**Why.** Luna at `2015462b` contains about a year of fixes specific to this model,
this GPU, and this Windows install: the `num_ctx` reload trap, the system-message
folding for GGUF chat templates, the no-tools retry, the context calibration
ratio, the regex look gate, the Job Object orphan fix. A clean rewrite throws all
of it away and re-earns each bug.

**Rejected.** Greenfield orchestrator (the spec's lean, 12 months of new bugs);
forking Luna in place (inherits a 1,307-line `app.py` as the architecture).

**Cost.** Two repos during the transition; the port itself has to be verified by
Luna's own test suite (314) or it is not a port.

---

## D-002 — AI Router is the only cloud provider

**Decision.** `https://api.airouter.in/v1`, OpenAI-compatible, key in
`AIROUTER_API_KEY`. Model metadata, context length and pricing are fetched from
its public `/v1/models`.

**Why.** It is the key the user has. It exposes the catalog over HTTP, including
a `zdr` flag per model, so the router can make privacy and cost choices from live
data instead of a hand-maintained table.

**Rejected.** Groq direct (a second key and a second catalog to maintain, and it
was the earlier answer, superseded by the user's); OpenRouter (functionally the
same, not the key on hand); OmniRoute as the gateway (see D-003).

**Cost.** One vendor in the middle of the cloud path; if it is down, Mind runs
local. Acceptable because Mind is not on the interactive path.

---

## D-003 — The router is ours; Hermes and OmniRoute are references, not dependencies

**Decision.** Implement `router/` inside Yuri, shaped after Hermes's config
vocabulary (`default`, `fallback`, `custom_providers`).

**Why.** The Ollama-native features Yuri depends on (`num_ctx`, `think`,
inline images, `format` schemas, streaming tool calls) are not uniformly exposed
through an OpenAI-compatible gateway. A gateway in the voice path is a second
process and a second failure mode for a single-user desktop app. The decision
logic also needs to be unit-testable, which means it belongs in our process.

**Rejected.** Calling Hermes as the orchestrator (an agent harness, wrong
abstraction, and it would own the loop); routing through OmniRoute (the 160+
provider catalog is not the problem we have; one provider and a privacy gate is).

**Cost.** We own provider code. It is two providers and one interface.

---

## D-004 — Reflex is `qwen3.5:9b` at Q4_K_M

**Decision.** `qwen3.5:9b` (6.55 GB), Ollama-native, `num_ctx` 8192, `keep_alive`
set, `think: false`.

**Why.** It reports `vision` and `tools`, so one model covers chat and eyes. At
Q4_K_M it leaves ~3 GB headroom next to the desktop. Q6_K_L (9.37 GB) is what
Luna is configured with today and it does not leave room for KV cache plus a game.

**Rejected.** Q6_K_L (over the line); `qwen3.5:4b` as the reflex model (weaker
persona, and it is doing classifier duty); the abliterated GGUF (Luna's handoff
records it ignoring structured-output instructions).

**Cost.** Some quality versus Q6; measurable in the P2 persona check. If it
regresses, Q5_K_M is the next rung, not Q6.

---

## D-005 — No resident 14B; Mind is cloud

**Decision.** Drop the spec's 14B/MoE "deep mind". Mind is a cloud role with a
local fallback to the same 9B.

**Why.** A 14B dense at Q4 is ~9 GB, which does not coexist with the reflex model
on 12 GB; the spec already said so. With a cloud Mind, the 14B is a slower, worse
version of a capability we already have. It also solves the model-swap problem in
v0.1 §3: nothing has to be swapped for Mind.

**Rejected.** Two local models with swap orchestration (the original design, high
complexity, GPU contention); MoE with CPU offload (low-teens tok/s, still worse
than the cloud path and still costs RAM).

**Cost.** Mind needs the network; offline deep work degrades to the 9B. Accepted:
offline is the fallback case, not the default case.

---

## D-006 — The entire voice stack is local

**Decision.** STT, TTS, VAD and embeddings never leave the machine, and no cloud
provider is cloud-eligible in those roles.

**Why.** Audio identifies a person; it is the most sensitive stream in the
system. It is also latency-critical, and a 429 in the voice path would be a
multi-second silence. It is enforced at config load, not by discipline.

**Rejected.** Cloud STT as an accuracy upgrade (whisper-small is adequate for
short utterances); cloud TTS quality (it would break the cloned voice).

**Cost.** Voice quality is capped by local models. Fine.

---

## D-007 — GPT-SoVITS: managed launcher, no reinstall, zero-shot for v1

**Decision.** Use the existing install at
`D:\AI\Project Luna\GPT-SoVITS-v3lora-20250228` (bundled Python 3.9.13,
`api_v2.py`, `tts_infer.yaml`). Start it from the supervisor inside the Job
Object. Voice comes from the 8.52 s reference clip
`C:\Users\User\Downloads\character_files_main_sample.wav`, zero-shot.

**Why.** It is already installed and configured, and Luna already speaks through
its API. The `GPT_weights*` and `SoVITS_weights*` directories are empty, so there
is no fine-tuned model to load; zero-shot is the available path.

**Rejected.** Reinstalling a fresh SoVITS (no benefit, new dependency risk);
fine-tuning first (a day of GPU time before a single sentence is spoken).

**Cost.** Zero-shot quality is reference-limited. A fine-tune is a Phase 2.5
option, not a prerequisite.

---

## D-008 — Speech normalization is enforced in code

**Decision.** `speech.py` rewrites model output before TTS: no em/en dashes,
quotes, ellipses, asterisks, parentheses, semicolons, lists, emoji, URLs, or code
blocks. The persona prompt also states the rule.

**Why.** The user's requirement is that TTS never struggles with typography. On
this model, prompt rules decay over a session (see D-010's evidence). A pure
function with a golden corpus is testable; a prompt is not.

**Rejected.** Prompt-only (the complaint would return); post-hoc regex at the TTS
call site (untestable, duplicated, and it loses segmentation).

**Cost.** Normalization can make a reply sound terse. The chat panel still shows
the model's original text, so nothing is lost visually.

---

## D-009 — One TTS request per clause

**Decision.** Split the reply into clauses client-side and request one WAV per
clause, prefetch depth 2.

**Why.** Complete WAVs make gapless ordering, lip-sync and abort straightforward;
a failed clause retries alone.

**Rejected.** Server-side `streaming_mode: true` with `cut5` (opaque stream,
harder abort, untested splitter).

**Cost.** Per-request overhead. If the §2.5 benchmark shows it dominating, the
fallback is streaming per sentence group, recorded as D-009a.

---

## D-010 — Routing is decided in Python

**Decision.** A pure function over `(role, request, config, state)` chooses the
provider. The model is never asked whether a task is hard.

**Why.** Luna's handoff documents four separate cases of this model failing to
self-assess mid-reply: calling `set_active_task` on a bare "hi" from remembered
context, double camera calls, ignoring "don't repeat yourself" over five
identical OCR frames, and needing a code-level similarity check to stop repeating
comments. The pattern is consistent enough to design around.

**Rejected.** A "needs planning?" model call (adds latency and a failure mode);
trusting tool-call behavior to signal complexity.

**Cost.** The complexity estimate is a heuristic and will sometimes be wrong.
Shadow mode (D-014) exists to measure how wrong before it costs anything.

---

## D-011 — One supervisor process, logical services

**Decision.** Services are modules with typed bus interfaces, in one asyncio
process. Ollama and SoVITS remain separate processes because they already are.

**Why.** Eight Python processes on Windows is eight startup paths and eight logs
for a single-user app, and nothing here is CPU-bound enough to need isolation.
The bus is the seam: a process split later touches the transport, not the
callers.

**Rejected.** The spec's literal service split (operational cost with no current
benefit).

**Cost.** One crash takes everything down. The Job Object and a supervisor
restart guard cover the actual failure mode (orphans), which is what Luna
actually suffered from.

---

## D-012 — Content carries a trust flag, and untrusted blocks cloud egress

**Decision.** Every bus event has `trust: trusted|untrusted`. Untrusted content
in a turn forces local Mind for that turn and raises the minimum tool tier.

**Why.** v0.1 §4 names prompt injection as the failure that makes autonomous
agents go wrong. Cloud egress is a new way for injected text to exfiltrate: the
attacker does not need a tool if the model will paste context into a remote
request. Tagging is the only place to stop it reliably.

**Rejected.** Prompt-based "ignore injected instructions" (the exact failure
mode); redaction-only (does not stop the model from choosing to send).

**Cost.** Some Mind calls that could have used the cloud run local. Fine.

---

## D-013 — Budget from provider metadata, caps checked before the call, fail closed

**Decision.** Prices come from `/v1/models`; caps are hourly/daily/monthly in
config; a call that would cross a cap is not made; no cached catalog means cloud
is disabled, not guessed.

**Why.** A hardcoded price table drifts silently, and a router that guesses prices
cannot be trusted with a cap. Failing closed on missing metadata means the
failure mode is "Mind stayed local", never "unexpected bill".

**Rejected.** Per-call user confirmation (kills the autonomy the app is for);
unlimited spend with logs only.

**Cost.** A stale catalog can disable cloud until refreshed. Acceptable.

---

## D-014 — Shadow mode before enforce

**Decision.** `routing.mode: shadow` by default. Decisions are computed and
logged; execution stays local. `enforce` is per-role.

**Why.** It answers "how often would we have escalated, and were the reasons
right" on real traffic before any money is spent or any behavior changes.

**Rejected.** Straight to enforce (untestable against real usage patterns).

**Cost.** A week of logs before Mind is actually used. Cheap.

---

## D-015 — Memory writes never go to the cloud

**Decision.** `memory_writer` is not cloud-eligible, even though a cloud model
would write better summaries.

**Why.** Memory is the densest personal data in the system, and it is written
continuously and silently. One redaction mistake there is worse than a hundred
bad Mind calls.

**Rejected.** Cloud writer with redaction (the asymmetry of harm does not justify
it).

**Cost.** Slightly worse memory curation; `qwen3.5:4b` with structured output is
enough for summaries.

---

## D-016 — Barge-in ships only if it measures clean

**Decision.** Full duplex with barge-in is gated on <= 1/20 false triggers; else
half-duplex behind a config flag.

**Why.** Echo from her own speakers is the likely failure, and a companion that
stops talking at random is worse than one that cannot be interrupted.

**Rejected.** Ship it and tune later (a bad first impression of the core feature).

**Cost.** Possibly no barge-in in v1. The spec listed it as required; this
supersedes that until the measurement says otherwise.

---

## D-017 — Owner ping deferred to P6; ntfy recommended

**Decision.** Keep the channel as a single Tier 3 entry point to one recipient,
off by default, and do not choose the transport now.

**Why.** It is the last phase and it depends on tiers (P4). Choosing a transport
today would be a decision with no consumer. ntfy is the recommendation because it
is one URL, one recipient, no bot identity, and no third-party contact graph.

**Rejected.** Telegram/Discord (bot identity and a contact surface the app does
not need).

**Cost.** None yet.

---

## D-018 — The router wraps `llm.py` instead of replacing it

**Decision.** `router/providers/ollama_native.py` calls `llm.stream_reply`,
`llm.stream_reply_with_tools` and `llm.describe_image` by module attribute. It is
an adapter, not a reimplementation. Cloud errors subclass
`llm.LLMUnreachableError`, so every existing `except` and every existing test
stub keeps working.

**Why.** `llm.py` contains the accumulated Ollama knowledge: one `num_ctx` for
every call, `think` handling, the no-tools model memory, the `format` retry, the
context calibration feed, the 120 s timeouts. Rewriting it to fit a new interface
would have thrown all of that away and re-earned each bug. The ported suite (314
tests) also stubs `llm.<function>` directly; attribute access keeps those stubs
effective, which is why the whole suite stayed green through the wiring.

**Rejected.** A clean provider implementation replacing `llm.py` (equal behavior
for a week of work and a new round of model bugs); wrapping at the `app.py` level
only (every caller would keep knowing provider details).

**Cost.** The dependency arrow points "up" from router to llm, which is unusual.
It is deliberate and bounded: when a second local backend appears (llama.cpp
server), the adapter becomes two adapters and `llm.py` becomes one of them.

---

## D-019 — Cloud is enabled by default in config, disabled by a missing key

**Decision.** `providers.airouter.enabled: true` ships on. With no resolvable
`AIROUTER_API_KEY`, startup marks the provider unavailable, prints one line, and
`mind` stays local. `routing.mode` remains `shadow` until the user opts in.

**Why.** A key in `.env` should be the only step needed to make cloud work; a
config edit as a second step is a support load and a footgun. The dangerous
switch is not "is the provider configured" but "do we execute cloud decisions",
and that one stays off.

**Rejected.** `enabled: false` by default (people leave it off and wonder why
costs stay at zero); auto-enforcing `mind` when a key exists (spending money as a
side effect of a config file).

**Cost.** A fresh clone prints one warning about the missing key. Acceptable.

---

## Superseded / historical

- **D-002 (first version).** "Groq is the cloud provider" was drafted before the
  user supplied the AI Router endpoint. Superseded by D-002 as written above.
