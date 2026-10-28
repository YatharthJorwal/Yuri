# Yuri — Starting Spec v0.1

Successor to Luna. Local-first, agent-first companion that lives on your PC. Drafted 20 Oct 2026.

## 1. Principles

- **Agent, not chatbot.** Yuri runs a loop, owns goals, and acts without being poked. Chat is one interface to it, not the whole thing.
- **Local by default.** Inference, voice, memory and screen data never leave the machine. The one exception is an explicit, narrow outbound channel to *you* (see section 5).
- **Two clocks.** A fast path for talking (sub-second feel) and a slow path for thinking (can take a minute, nobody is waiting).
- **Permissioned autonomy.** Every action has a tier. Capability grows only by you promoting it.
- **Config over code.** Models, engines, tiers and budgets live in config. Nothing hardcoded.
- **Keep from Luna:** VRM avatar, persona/reasoning separation, camera permission path, Tauri shell, Python orchestrator.

## 2. Hardware budget and the 14B verdict

Target: RTX 3060 12GB, i5-14400F, 32GB DDR5-4800 (dual channel), 50GB+ disk.

**VRAM is the constraint, not RAM.** Rough resident budget with everything running:

| Item | VRAM |
| --- | --- |
| Windows desktop + WebView2/VRM render | \~0.8–1.2 GB |
| GPT-SoVITS (TTS) | \~2–3 GB |
| Embedding model (nomic-embed-text) | \~0.3 GB |
| STT (faster-whisper small) | CPU for now (0 GB) |
| **Left for the LLM + KV cache** | **\~7–8 GB** |

**Reading the numbers (estimates, benchmark before trusting):**

- **9B at Q6** is roughly 7.5 GB of weights alone. With KV cache that is already over the line while TTS is resident. Your instinct that it is the ceiling is right, and it is a *tight* ceiling.
- **9B at Q5\_K\_M or Q4\_K\_M** (\~6.3 / \~5.5 GB) leaves real room for context and KV cache. The quality loss from Q6 to Q5 is small. I would default to **Q5\_K\_M with q8 KV cache** and keep Q6 as a config option.
- **14B can run, but not fully on GPU next to everything else.** At Q4\_K\_M it is \~9 GB of weights, so some layers spill into system RAM. DDR5-4800 dual channel (\~70 GB/s) makes that usable but slower, expect roughly low-teens tokens/sec at best, and prompt processing hurts more. Fine for the slow path, bad for the fast path.
- **Best use of your 32 GB RAM:** a larger model (14B dense, or a MoE in the \~30B-total/few-B-active class if the current lineup has a good one) with CPU offload, loaded on demand for the slow path. MoE models degrade much more gracefully under CPU offload than dense ones. Check what is actually available and benchmark.

**Verdict:** 9B (Q5/Q6) stays resident as the voice and persona model. A 14B or MoE is the on-demand "deep mind" for planning, tool use and reflection, run in the background where speed does not matter. Do not make a 14B the live voice brain on this card.

## 3. Architecture

Services talk over a typed event bus (local WebSocket or in-process asyncio bus). No service calls another directly.

| Service | Job |
| --- | --- |
| **Shell** (Tauri + three.js) | VRM avatar, UI, tray, global hotkeys. Dumb: renders state, sends input. |
| **Gateway** | Event bus, session state, auth between services. |
| **Voice** | VAD, STT, streaming TTS, barge-in. |
| **Reflex** (9B resident) | Conversation, persona, short replies, tool *requests*. |
| **Mind** (14B/MoE on demand) | Planning, multi-step tool use, reflection, memory curation. |
| **Memory** | Facts, episodes, salience, decay, retrieval. |
| **Perception** | Screen OCR/VLM on demand, camera (explicit permission), window and idle signals. |
| **Tools** | Typed, sandboxed actions (section 4). |
| **Affect** | Mood/energy/affection state (section 6). |
| **Scheduler** | Timers, triggers, proactivity budget. |

Model swap rule: only one big model owns the GPU at a time. Mind loads when Reflex is idle or when a task needs it, then unloads. TTS stays resident.

## 4. Agent loop and autonomy

Loop: **perceive → decide → act → observe → reflect**, driven by events (you spoke, timer fired, window changed, idle threshold).

- Reflex handles the live turn and can *hand off* a task to Mind ("let me check").
- Mind runs the task as a background job with a step limit, writes results back to the bus, Reflex narrates them in character.
- Every tool has a JSON schema, a tier, and an audit log line.

**Permission tiers**

| Tier | Examples | Rule |
| --- | --- | --- |
| 0 Observe | read screen, read files in allowed folders, search memory | Allowed |
| 1 Draft | prepare a message, a file edit, a shell command | Allowed, nothing executes |
| 2 Ask-first | send a message to you, open apps, write files, run commands | Per-action confirmation |
| 3 Autonomous allowlist | actions you promoted (e.g. "ping me on Telegram if I have been gone 2h") | Runs alone, logged, revocable |
| 4 Never | delete outside sandbox, spend money, message third parties, touch credentials | Hard blocked in code |

**The OpenClaw behavior you want** (agent notices you are gone, finds your contact, messages you) maps to Tier 3 with a narrow target: *reach the owner*. It does not need to roam your contacts. It needs one outbound channel, one recipient.

**Two things that need deciding, not skipping:**

1. **The "100% local" rule vs messaging.** Reaching your phone requires a network hop (Telegram bot, Discord DM, ntfy). Proposal: a single allowlisted outbound channel to you only, off by default, documented as the one exception.
2. **Prompt injection.** Anything Yuri reads off a screen, web page or file is untrusted text that can contain instructions. Perception output is tagged as data, never as instructions, and Tier 2+ actions triggered while untrusted content is in context require confirmation. This is the failure mode that makes autonomous agents go wrong.

## 5. Memory

- **Stores:** facts (key-value, with source and confidence), episodes (summaries + embeddings), and a small always-in-context profile card.
- **Write path:** a separate memory-writer pass (Mind, async) after each session *and* on a timer, so a hard kill loses minutes, not the session.
- **Retrieval:** facts are semantically filtered per turn, not injected wholesale. Episodes ranked by similarity x salience x recency.
- **Corrections:** contradiction detection. "Actually I like X now" supersedes the old fact without needing the word "forget". Explicit forget stays.
- **Decay:** low-salience episodes are compressed or dropped over time.
- **Stack:** keep SQLite + sqlite-vec. It works and is honestly tested.

## 6. Voice

- Full duplex: VAD-gated mic, barge-in (you talk over her, she stops), no push-to-talk required. F9 stays as an override.
- Streaming TTS in short chunks with a pre-buffered first clause to cut time-to-first-audio.
- STT: faster-whisper, fix CUDA later once VRAM budget is settled. CPU is fine for now.
- Constraint: every GB of VRAM TTS uses comes out of the LLM budget. Measure it first.

## 7. Affect and avatar

- Small state vector: mood, energy, affection, irritation. Updated by events, decays toward baseline.
- State feeds three places: the prompt, VRM expressions, and TTS style.
- Idle life (breathing, gaze, blinks, yawns, pouts) is driven by this state, not a random timer.
- Tsundere persona lives in one swappable persona file, separate from reasoning.

## 8. Proactivity

- Triggers: idle time, time of day, window/app change, task-guide drift, long absence.
- **Proactivity budget:** max N unprompted interactions per hour, backing off when ignored. Without this she becomes a notification cannon.
- Task Guide Mode from Luna becomes one trigger-driven behavior, not a special code path.

## 9. Runtime

- Tauri shell stays. Orchestrator splits into the services above, still Python, managed by one supervisor.
- Windows Job Object kill-on-close stays (it fixed the orphan processes).
- Config: one YAML for models, tiers, budgets, channels. Secrets in a gitignored local file.
- Logs per service under `logs/`.

## 10. Evals

- Replayable conversation fixtures run against the real resident model.
- Persona consistency check (does she still sound like Yuri after 50 turns).
- Memory tests: write, correct, forget, retrieve, against a real DB.
- Tool safety tests: each tier tried from an injected-content context, must be blocked or confirmed.
- Latency budget tracked per turn: time to first audio, tokens/sec, VRAM peak.

## 11. Phases

1. **Foundation:** event bus, service split, config, supervisor. Port Luna's shell and voice onto it.
2. **Fast path:** Reflex model, streaming voice, barge-in, latency measurement.
3. **Memory v2:** writer pass, semantic facts, contradiction handling, timer consolidation.
4. **Agent core:** tool registry, tiers, audit log, Mind job runner, model swap.
5. **Affect + avatar life.**
6. **Proactivity + the owner-ping channel.**
7. **Evals hardened, docs trimmed to one spec and one decision log.**

## 12. Open questions

- Which 9B quant is the resident default after measuring real VRAM with TTS loaded?
- Is a 14B dense or a MoE the better Mind on this machine? Benchmark both.
- Which single outbound channel for the owner ping (Telegram, Discord, ntfy)?
- Does Work Mode (computer control) ship in Yuri v1, or stay a later phase?
- Port Luna's code, or restart the orchestrator clean? I lean clean orchestrator, ported shell.
