# Yuri — Scope v0.2

Supersedes nothing; amends `SPEC-v0.1.md` in place. Every place this document
disagrees with v0.1, this document wins, and the reason is written down in
§5 or `DECISIONS.md`.

Drafted 29 Oct 2026, against the machine as measured on 28 Oct 2026.

---

## 1. What Yuri is

A companion that runs a loop on your PC, keeps its own goals, and talks. Chat
is one interface to it. It is Luna's body and Luna's instincts with a different
spine: services instead of a monolith, a router instead of one hardcoded model,
and a permission system that is enforced in code instead of described in prose.

It is **not** a general agent framework, **not** a coding assistant, and **not**
a chat wrapper. It is one character, on one machine, with a long memory.

---

## 2. Where it starts from

Luna is not a paper design; it is a working app. Yuri starts by porting it, not
by rewriting it. Snapshot used for this scope:

| Fact | Value |
| --- | --- |
| Local checkout | `D:\AI\Project Luna\luna-main\luna` |
| Commit | `2015462b` (clean `main`, level with `origin/main`) |
| Done and confirmed on the machine | Luna phases 0-4, 7, 8, 9, 9.5; 13 in progress |
| Not started | Agent Mode (Luna 11), Live Voice Chat (Luna 12), game-context (Luna 5) |
| Tests | 314 pytest in `orchestrator/`, 13 frontend tests, `tsc`, `vite build` |
| Voice today | whole-reply WAV from GPT-SoVITS, no streaming, no VAD, no barge-in |
| Memory today | SQLite + sqlite-vec: facts, episodes, consolidation at session end |

The important consequence: **"port Luna" is cheap and "rewrite Luna" throws away
about a year of model-specific fixes.** The orchestrator is restructured, not
rewritten. The shell is copied almost verbatim.

---

## 3. Principles (v0.2)

v0.1 §1 stands, with one sentence changed and one added.

1. **Agent, not chatbot.** Unchanged.
2. **Local by default.** Inference, voice, memory, and screen data stay on the
   machine. *Revised:* the single exception is text-only Mind-tier work sent to
   AI Router, behind a budget, an audit log, and a redaction gate. Pixels,
   audio, and raw transcripts never leave. See `ROUTING.md`.
3. **Two clocks** *revised to three:* sub-second reflex (local 9B), seconds
   cloud mind (AI Router), minutes background jobs (local or cloud).
4. **Permissioned autonomy.** Unchanged in intent; tiers are now executable
   metadata on every tool, not a table in a document.
5. **Config over code.** Unchanged. Stronger: every key in `config.example.yaml`
   has a code default, because a missing key used to be a startup `KeyError`.
6. **Keep from Luna:** VRM avatar, persona/reasoning separation, camera
   permission path, Tauri shell, Python orchestrator, Windows Job Object
   kill-on-close, the `num_ctx` discipline, the model-capability startup check.

Added:

7. **Spoken text is normalized in code, not requested in a prompt.** A 9B model
   will keep emitting em dashes and quotes no matter what the system prompt
   says. The prompt asks; the normalizer guarantees. See `PHASE-2-FAST-PATH.md`.
8. **The router decides in Python.** Never ask the small model to judge its own
   difficulty. Luna proved this four separate times (see `DECISIONS.md`
   D-010 for the list).

---

## 4. Spec traceability

Every section of v0.1 resolves to a decision and an owning phase. Nothing is
left unanswered.

| v0.1 | Decision | Phase | Status |
| --- | --- | --- | --- |
| §1 Principles | Amended: cloud mind tier, speech normalization, router-in-Python | — | revised |
| §2 Hardware / 14B verdict | No resident 14B and no second vision model. Reflex is `qwen3.5:9b` Q4_K_M (6.6 GB measured). Depth comes from AI Router. Budget rewritten with measured numbers. | P2 | revised |
| §3 Architecture | Same service *names*, one supervisor process. Typed asyncio bus with a trust field. Physical process split deferred behind the bus interface. | P1 | revised |
| §4 Agent loop + autonomy | Tiers kept. Tool manifests carry tier + JSON schema + audit line. Tier 2 confirmation is a shell surface, not a prompt. Cloud egress is a second outbound channel with its own consent. Untrusted content is tagged and blocks cloud egress. | P4 | kept |
| §5 Memory | Keep SQLite + sqlite-vec (Luna has it working). Add the async writer pass on a timer, contradiction detection, decay, and a profile card. | P3 | kept + extended |
| §6 Voice | Rewritten in detail: streaming SoVITS, VAD, barge-in, latency budget. All local. See `PHASE-2-FAST-PATH.md`. | P2 | kept, detailed |
| §7 Affect + avatar | Keep the state vector; drive Luna's existing idle/eyes/expression modules from it; feed TTS style. | P5 | kept |
| §8 Proactivity | Kept. Shared speech arbiter with voice, so an ambient remark can never talk over the user. | P6 | kept |
| §9 Runtime | Tauri shell kept, orchestrator restructured on one supervisor, Windows Job Object kept for every child, one YAML + gitignored secrets, per-service logs. | P1 | kept |
| §10 Evals | Kept, plus router evals (no unexpected cloud call), a speech-normalizer golden corpus, and a latency harness. | P7 (+ per-phase) | extended |
| §11 Phases | Renumbered with a Phase 0 bootstrap. Router lands in P1, fast path in P2. | all | revised |
| §12 Open questions | All five answered below. | — | resolved |

### §12 answers

1. **Which 9B quant is resident?** `qwen3.5:9b` Q4_K_M, 6.6 GB, Ollama-native.
   Q6_K_L (9.4 GB) is dropped: it leaves no headroom for a game plus KV cache,
   and the quality delta does not buy back the swap risk.
2. **14B dense or MoE for Mind?** Neither. Mind is cloud. The local fallback is
   the same resident 9B, not a second model. `phi4:14b` stays installed but out
   of scope.
3. **Which outbound channel for the owner ping?** Deferred to P6; recommendation
   is ntfy (one topic, one recipient, no bot identity). Independent of the
   router's egress, which is a different consent surface.
4. **Does Work Mode ship in v1?** No. Tool actions ship as Tier 2 with a hard
   confirmation in P4. Reversing "observe and advise only" is a later decision
   and needs the injection-resistant confirmation design Luna's phase 11
   already scoped.
5. **Port Luna or restart the orchestrator?** Port the shell verbatim. Restructure
   the orchestrator file-by-file; rewrite only `config.py`, `stt.py`, `tts.py`
   and the transport split. See §11.

---

## 5. Spec revisions (numbered, with reasons)

**R1. The cloud exception is written down.** v0.1 says nothing leaves the
machine except an owner ping. v0.2 adds AI Router for Mind. Consequence: two
distinct egress paths (model calls, owner pings) with two distinct consent
switches. Neither implies the other.

**R2. No resident 14B.** v0.1 §2 recommends a 14B or MoE "deep mind" on demand.
With a cloud Mind, that model is a worse version of what the router already
gets, paid for in VRAM that the reflex model needs. It survives only as an
offline fallback in the future, and the fallback today is the 9B.

**R3. One supervisor process, logical services.** v0.1 §3 implies separate
services. Physically splitting eight Python processes on Windows buys isolation
and costs eight startup paths, eight logs, and IPC debugging for a single-user
desktop app. Services are modules with typed interfaces on one bus; the bus is
the seam that makes a process split possible later without touching callers.

**R4. Vision and embeddings stay on the reflex model.** `qwen3.5:9b` reports
`vision`, so there is no second VLM. That saves roughly 2-3 GB versus Luna's
separate vision model, which is most of the reason Q4_K_M fits comfortably.

**R5. Speech normalization is code, not a prompt rule.** The prompt carries a
hard no (as requested). `speech.py` enforces it deterministically, because the
prompt alone has never held up over 50 turns on this model.

**R6. Barge-in is a measured feature, not a promised one.** Full duplex with
barge-in ships only if false triggers measure under 1/20 turns with browser AEC.
Otherwise it degrades to half-duplex behind a config flag. v0.1 treats
"no push-to-talk required" as settled; it is not settled until it is measured.

**R7. Three clocks, but only two are user-visible.** The cloud mind is fast
enough to stop feeling like a "slow path". Latency targets are per path, in
`PHASE-2-FAST-PATH.md` §7.

**R8. Evals gain a router dimension.** "Did she behave" is not enough; the suite
must also assert "did we spend money we did not mean to", and "did anything
leave the machine that should not have".

**R9. Phase numbering gains P0.** v0.1 phase 1 (foundation) presumes a repo.
P0 is bootstrap: port, baseline tests, config v2. Otherwise phase 1 has nowhere
to live.

---

## 6. Architecture

```
Tauri shell (Rust core + WebView2)
  window / tray / hotkeys / VRM / lipsync / audio playback / mic capture
        |  one localhost WebSocket, typed envelopes
        v
Supervisor (one Python process, asyncio)
  bus ............ typed events, trust field, replayable for tests
  turn ........... perceive -> decide -> act -> observe -> reflect
  router ......... role -> candidate chain -> provider call  (ROUTING.md)
  voice .......... VAD, STT, streaming TTS, arbiter            (PHASE-2)
  mind ........... background job runner, step limits, tools
  memory ......... facts, episodes, writer pass, recall, decay
  perception ..... screen OCR / camera on demand, window + idle signals
  affect ......... mood, energy, affection, irritation -> prompt, VRM, TTS style
  scheduler ...... timers, triggers, proactivity budget
  tools .......... typed, sandboxed, tiered, audited
        |                 |                       |
        v                 v                       v
  Ollama (local)    GPT-SoVITS (local)      SQLite + sqlite-vec (local)
        |
        +--> AI Router (network, mind role only, budgeted, audited)
```

The shell is dumb on purpose: it renders state and sends input. Every decision
lives in Python, where it can be tested without a VRM model.

### Bus envelope

```python
Event(
    id: str,            # uuid, for tracing
    type: str,          # "turn.started", "route.decided", "tool.requested" ...
    ts: float,          # monotonic seconds
    source: str,        # "voice", "shell", "mind", "memory", "scheduler"
    trust: str,         # "trusted" | "untrusted"  <- from screen/file/web content
    session: str,
    payload: dict,      # per-type schema, validated on emit
)
```

`trust` is load-bearing, not decorative: `untrusted` blocks cloud egress and
raises the tier requirement on any action it influences.

### Event types (v1 vocabulary)

`input.utterance`, `speech.started`, `speech.ended`, `turn.started`,
`turn.delta`, `turn.finished`, `route.decided`, `model.call.started`,
`model.call.finished`, `tts.segment`, `audio.chunk`, `playback.started`,
`playback.ended`, `barge_in.detected`, `mind.job.requested`,
`mind.job.finished`, `tool.requested`, `tool.result`, `tool.denied`,
`ambient.proposed`, `affect.updated`, `scheduler.fired`, `task_guide.tick`.

---

## 7. Model and memory split

| Role | Where | Model | Why |
| --- | --- | --- | --- |
| `reflex` | local | `qwen3.5:9b` Q4_K_M, `num_ctx` 8192 | voice latency, persona, tool *requests* |
| `vision` | local | same 9B (it reports `vision`) | pixels never leave; no second model |
| `embed` | local | `nomic-embed-text` | 274 MB, SQLite vector width fixed at 768 |
| `classifier` | local | `qwen3.5:4b` | intent / task / JSON, tiny prompts |
| `memory_writer` | local | `qwen3.5:4b` | most sensitive content on the box |
| `mind` | airouter | shortlist benchmarked in P1 | planning, multi-step tools, reflection |

Mind is the only role that can call out. Its chain is
`airouter -> local reflex` (see `ROUTING.md` §4). A cloud failure degrades to the
9B and says so in the log; it never surfaces as "I can't reach my own brain".

Memory stays in one SQLite file, one store, one embedding width, as v0.1 says.
The only structural addition is a writer pass that can run on a timer as well as
at session end, so a hard kill costs minutes rather than the session.

---

## 8. Hardware budget (measured, not estimated)

Target machine, as of 28 Oct 2026: RTX 3060 12 GB (12,288 MiB), i5-14400F,
32 GB DDR5-4800, Windows 11.

| Item | VRAM | Source |
| --- | --- | --- |
| Desktop + WebView2 + VRM render | ~0.8-1.2 GB | v0.1 estimate, consistent with observed idle 1.7 GB with a browser open |
| `qwen3.5:9b` Q4_K_M weights | 6.55 GB | `ollama list` |
| KV cache at 8,192 ctx | ~0.8-1.0 GB | estimate, verify with `nvidia-smi` during P2 |
| `nomic-embed-text` | 0.27 GB | `ollama list` |
| GPT-SoVITS | 0 GPU | runs on CPU in its own runtime Python 3.9 |
| faster-whisper small int8 | 0 GPU | CPU, as in Luna |
| **Total resident** | **~8.4-9.0 GB** | leaves ~3 GB headroom for a game or a GPU TTS experiment |

Why Q4_K_M and not the Q6_K_L currently configured in Luna: 9.37 GB of weights
plus KV plus desktop is over the line, and the model swaps under load. The
quality difference at this size is small; the swap is not.

The 14B-in-RAM plan from v0.1 §2 is dropped (R2). If a local Mind is ever wanted
again, the MoE route (few active parameters) is the one to benchmark, not dense.

---

## 9. More stuff (failure modes this scope has to answer)

These are the things that actually go wrong, in rough order of how expensive
they are to discover late.

1. **Her own voice re-triggering the mic.** The single likeliest reason barge-in
   fails is echo, not VAD. Mitigation: browser AEC in the shell
   (`echoCancellation: true`), plus an energy gate tied to current playback
   level, plus the half-duplex fallback (R6). Gate: false triggers under 1/20.
2. **Model output that TTS cannot say.** Em dashes, quotes, ellipses, asterisks,
   emoji, URLs, code fences, `1.` lists, `:)`. Mitigation: `speech.py` with a
   golden corpus, applied to every spoken string; display text kept separate.
3. **Prompt-only rules decaying.** Over a long session the persona prompt's
   "never use dashes" line stops being obeyed. Mitigation: the prompt asks, the
   normalizer enforces, and a test asserts the invariant, not the compliance.
4. **First-audio latency hiding in the wrong stage.** Without per-stage
   timestamps, "she feels slow" is unactionable. Mitigation: seven timestamps
   per turn, JSONL log, p50/p95 gate (P2 §7).
5. **Cold model load.** A 6.6 GB load takes 30-60 s, long enough to look like a
   hang. Mitigation: Ollama `keep_alive`, warmup at startup, an in-character
   "one second" line if a load is detected mid-turn.
6. **SoVITS server not running.** Luna already falls back to pyttsx3 and logs
   why. Yuri keeps that, and adds a managed launcher with a health check so the
   fallback is rare rather than routine.
7. **WebView2 autoplay policy.** Luna already hit a silent first remark because
   the `AudioContext` was suspended. Mitigation: resume on first user gesture and
   on window focus, and never treat "played" as "heard".
8. **The 9B deciding mid-reply whether to act.** Known failure. Mitigation:
   tool/look/task decisions in Python (regex gate, classifier, similarity), the
   prompt as fallback only. This is the single most transferable lesson in the
   Luna handoff.
9. **Prompt injection reaching the cloud.** Injected text on screen could ask
   Mind to send something out. Mitigation: `trust: untrusted` blocks egress, and
   cloud calls are only made for tasks that originated from a trusted user turn.
10. **Spending money silently.** Mitigation: per-hour/day/month caps, price data
    from `/v1/models`, an audit line per call, and `shadow` mode before `enforce`.
11. **Memory writes going out.** The writer pass is `local_only` even though
    cloud would be better at it. It is the most personal data in the system.
12. **Mic privacy.** Indicator in the tray while listening, a mute hotkey, no
    audio written to disk, no transcript persisted beyond the session log the
    user can clear.
13. **Two speakers, one mouth.** Task Guide chides, ambient OCR remarks, and
    voice replies can collide. Mitigation: one arbiter owns "who may speak";
    ambient remarks queue and drop, never interrupt the user.
14. **Persona drift over 50 turns.** Kept from v0.1 §10 as a real eval, not a
    vibe check.
15. **Windows path and quoting traps.** Config strings with backslashes need
    single quotes (Luna hit this). The example config uses single quotes for
    every Windows path and the loader documents it.
16. **Two clocks becoming one slow one.** If cloud were ever allowed on the
    voice path, a 429 would be a four-second pause. It is not allowed, and the
    router refuses the combination at config-load time rather than at runtime.

---

## 10. Not building (v1)

- Separate OS processes per service (R3).
- Cloud vision, cloud STT, or cloud TTS. Any of the three would be a different
  privacy conversation.
- A resident 14B or MoE Mind (R2).
- Work Mode / computer control. Tier 2 confirmed actions only, in P4.
- Third-party messaging. The owner ping is one channel to one recipient.
- Luna's sandbox apartment (`sandbox.ts`, `src/apartment/`, `camera-modes.ts`,
  `postfx.ts` stay frozen and unported unless asked for).
- Fine-tuning the voice. Zero-shot from the existing reference clip, then decide.
- Multi-user anything. One machine, one person.

---

## 11. Reuse map (Luna @ 2015462b)

| Luna file | LOC | Yuri disposition |
| --- | --- | --- |
| `orchestrator/app.py` | 1307 | Split: turn pipeline, WS transport, ambient/task-guide loops into services. Logic preserved. |
| `orchestrator/llm.py` | 648 | Split: `providers/ollama_native.py`, `providers/openai_compat.py`, `fold_system_messages`, context accounting. Gains the router in front. |
| `orchestrator/persona.py` | 225 | Port. Persona file externed, speech rules added. |
| `orchestrator/context_budget.py` | 144 | Port. Calibration ratio kept, it is real data. |
| `orchestrator/task_guide.py` | 309 | Port, becomes one trigger-driven behavior (v0.1 §8). |
| `orchestrator/ocr_watch.py` | 218 | Port, gated by the shared arbiter. |
| `orchestrator/look_intent.py` | 194 | Port. This is the Python-decides pattern in practice. |
| `orchestrator/memory/*` | 688 | Port wholesale (store, db, consolidation, forget, recall, embeddings). |
| `orchestrator/tools/*` | 310 | Port + tier metadata + audit fields. |
| `orchestrator/camera.py`, `chunking.py` | 130 | Port; `chunking.py` extended for TTS clauses. |
| `orchestrator/stt.py` | 133 | Rewrite: streaming/VAD-aware, same config surface. |
| `orchestrator/tts.py` | 113 | Rewrite: streaming, abort, managed launcher, fallback kept. |
| `orchestrator/config.py` | 205 | Rewrite for schema v2, with defaults for every key. |
| `orchestrator/test_*.py` | ~1900 | Port as the P0 baseline gate; extend per phase. |
| `src-tauri/` | — | Copy nearly verbatim (Job Object, tray, hotkeys, windowing). |
| `src/*.ts` (main, lipsync, mic, eyes, idle-motion, expression-life) | — | Copy; lipsync gains chunk-aware updates, mic gains frame streaming. |
| `src/sandbox.ts`, `src/apartment/*` | — | Frozen, not ported. |

---

## 12. Decisions still owed

Nothing blocking P0-P2. Open items, each with an owning phase:

| Item | Phase | Default if not decided |
| --- | --- | --- |
| Mind model shortlist winner | P1 | `deepseek/deepseek-v4-flash` (cheapest with 1M ctx and tool use) |
| Owner ping channel | P6 | ntfy |
| Voice fine-tune vs zero-shot | P2.5 | zero-shot, 8.5 s reference clip |
| Barge-in on or half-duplex | P2 gate | half-duplex if false triggers exceed 1/20 |
| Local Mind, ever | P7 | no |

---

## 13. Done criteria for this scope

- Every v0.1 section has a decision and a phase (§4).
- Every v0.1 open question is answered or explicitly deferred (§12).
- `ROUTING.md` and `config.example.yaml` agree key for key.
- `PHASE-2-FAST-PATH.md` has a measurable gate per component and no open design
  decisions.
- Luna is untouched; no Yuri code exists yet.
