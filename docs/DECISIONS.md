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

---

## D-020 — Speech is normalized at one chokepoint, before synthesis

**Decision.** _send_speak() is the only function that turns text into audio. It
runs speech.prepare() and sends the model's wording as the caption while
synthesizing the normalized text. _run_turn no longer calls synthesize
directly, and the pipelined stream routes every clause through the same
_send_audio payload shape.

**Why.** Before this, speech.py existed and was tested but was not on the TTS
path: turns still sent raw persona text to SoVITS. One chokepoint means ambient
remarks, Task Guide chides, fallback lines, upload descriptions and normal turns
all get the same guarantee, and a future path cannot forget it. It also keeps the
caption honest (the panel shows what she meant, the voice says what it can say).

**Rejected.** Normalizing at each call site (drifts immediately); normalizing
inside 	ts.py (loses the display/spoken split and the caption would show the
mangled text).

**Cost.** One extra regex pass per spoken line, and captions sometimes differ
from the audio in punctuation.

---

## D-021 — TTS synthesis is pipelined one clause ahead, and the launcher is port-first

**Decision.** speech_stream.SpeechStream queues clauses, starts synthesizing the
next one before awaiting delivery of the current, and aborts by dropping the
queue. 	ts.ensure_server() checks the port *before* launching, and only launches
when nothing is listening.

**Why.** The old loop awaited synthesis inside the token stream, so the model's
next sentence waited on the voice. One-clause lookahead removes that wait without
reordering anything. The port-first check matters because two launchers exist (the
Tauri shell and now the Python startup check); whichever runs second must find the
server rather than start a second copy on the same port.

**Rejected.** Concurrent synthesis with reordering by index (more machinery for
the same effect at this scale); removing the shell launcher now (it still owns
child lifecycle when the app starts).

**Cost.** A clause can be synthesized that an abort then discards.
"

## D-022 — Energy VAD, not Silero, and barge-in cancels the turn

**Decision.** Voice input streams 20 ms PCM16 frames from the shell; `vad.py`
decides start/end from frame energy above an adaptive noise floor, with pre-roll,
minimum and maximum utterance bounds. No new dependency. Barge-in is VAD speech
during an active turn: the server emits `barge_in`, cancels the turn, and the
shell stops playback.

**Why.** Silero would mean onnxruntime (and in practice torch) in the
orchestrator's Python 3.13 environment for a decision a room mic makes easy.
Energy VAD is testable with synthetic frames, which is why this landed with tests
instead of "unverified on the user's machine". Barge-in has to be
server-authoritative: the shell cannot tell the user from her own playback.

**Rejected.** Shipping MediaRecorder clips with a VAD pass over the decoded file
(too late to interrupt anything); Silero first (a dependency and an untested
model for the same behavior).

**Cost.** A loud fan or a TV can still trigger a false start; the adaptive floor
and `margin_db` are the knobs, and the half-duplex fallback from D-016 remains
available if the 1/20 gate fails on the real desk.

---

## D-023 — The transcript log is the writer's durable buffer, and Temp Chat is a column

**Decision.** The writer pass reads `transcript_log` rows after
`meta.writer_checkpoint_id`, distills them, and applies facts, episode and the new
checkpoint in one transaction (`db.transaction()`, serialised by a process-wide
lock). Temp Chat turns are logged with `temp = 1`: still visible in the log panel,
never read by the writer, and the checkpoint moves past them.

**Why.** The log already survives a crash, so no second buffer is needed, and the
single transaction makes a pass all-or-nothing (a SIGKILL test proves it, in the
model call and inside the transaction). The temp flag is required, not optional:
the log was written for every turn regardless of Temp Chat, so a writer that read
it unfiltered would have memorised temporary chats.

**Rejected.** A separate scratch table (duplicates data that is already durable);
filtering temp turns from in-memory history as the old session-end code did (that
history is gone after a crash, which is the case this phase exists for).

**Cost.** "Clear log" also discards turns the writer had not processed yet. That
matches the intent of the button, but it is now a way to lose pending memory.

---

## D-024 — Replace and forget never delete

**Decision.** `superseded_by` is NULL (live), the id of the replacing fact, `-1`
(forgotten by the user) or `-2` (episode decayed out of search). Vectors are
removed so search cannot see a retired row; the row stays and
`python -m memory --restore ID` brings it back. A supersede is vetoed (and logged
as `supersede_blocked`) when the target is pinned, or has higher confidence than
the new fact, unless the new fact is flagged as an explicit correction.

**Why.** An automatic "that contradicts" decision made by a small model will
sometimes be wrong; the damage has to be one command to undo. Provenance
(`source`, `confidence`) is what lets the code tell a correction from an offhand
remark.

**Rejected.** Hard delete (unrecoverable mistakes); trusting the model's `target_id`
blindly (it must be one of the ids it was shown).

**Cost.** The database only grows. Decay removes things from search, never from disk.

---

## D-025 — Recall is relevance-gated, which changes what the old tests assumed

**Decision.** Recall injects a profile card plus up to 3 facts and 2 episodes
(reflex) or 8 and 5 (mind), only above a similarity floor, hard-capped at 300 /
1200 tokens of content. Previously up to 30 facts were injected on every turn
regardless of relevance. With no embedder, facts match by shared words and
episodes are skipped. Two old tests asserted the always-inject behaviour and were
rewritten.

**Known gap found while building this.** The Ollama-native provider
(`router/providers/ollama_native.py`) calls `llm.stream_reply(messages)` with
neither the role's model nor `schema` / `max_tokens`. So `memory_writer`
(configured as `qwen3.5:4b`) actually runs on the main chat model, and there is
no constrained-JSON output. Fixing it means passing `model` and `format` through
`llm.py`, which cannot be verified without a live Ollama, so it was not changed.
Until then: expect writer passes to compete with chat for the 9B, which is why
timer passes wait for an idle arbiter and are cancelled when a turn starts.

**Cost.** A fact that is true but not semantically near the current message is not
injected; the profile card is the always-on channel for that.

---

## D-026 — Repository layout

**Decision.** `orchestrator/` modules are grouped by job: `audio/` (voice I/O),
`senses/` (camera, look, OCR, Task Guide), alongside the existing `router/`,
`memory/`, `tools/`. All pytest files live in `orchestrator/tests/`; per-phase
plans live in `docs/phases/`; the handoff note is `docs/HANDOFF.md`.

**Why.** Twenty-odd flat modules with tests interleaved made it hard to see what
belonged to what. Imports became `from audio import tts`; the module objects are
unchanged, so tests that patch `llm.<fn>` or `tts.<fn>` by attribute still work
(the rule in `HANDOFF.md`).

**Cost.** Any local branch or open diff from before this change will conflict on
paths. `git mv` was used so history follows the files.

---

## D-027 — Everything slow at boot starts at the same time

**Decision.** The shell starts the orchestrator immediately, in parallel with
GPT-SoVITS (it used to wait up to 60 s for GPT-SoVITS's port first). Because both
the shell and `tts.py` know how to start GPT-SoVITS, the shell now sets
`YURI_SHELL_STARTS_TTS=1` on the orchestrator; in that mode `tts.py` never launches
a server, it waits for the shell's, and for the first 90 s a spoken line waits up to
8 s for it rather than using the placeholder voice. Inside the orchestrator the
chat-model warm-up, the speech-model load (new, `stt.warm_up_on_start`), the TTS
check and the router start-up run concurrently. The frontend reconnects with a
250 ms -> 2 s backoff instead of a flat 2 s.

**Why.** Measured, not guessed: a warm `import app` is about 0.6 s, so the Python
venv was not the problem. The window sat on "connecting" for as long as GPT-SoVITS
took to load, because the orchestrator was queued behind it, and then the first
spoken turn also paid for loading Whisper.

**Rejected.** Letting the orchestrator own GPT-SoVITS (the shell's Job Object and
quit handler already manage its lifetime); dropping the port wait without the
env var (the Python side would have launched a second copy and fought for port
9880).

**Cost.** In the first ~30 s after boot the first voice line may be delayed a few
seconds, or fall back to the placeholder voice if GPT-SoVITS is slower than that.
The Rust change is syntax-checked but was not compiled in the environment it was
written in; the first `tauri dev` is the real check.

---

## D-028 — Luna leftovers

**Decision.** Deleted `src/camera-modes.ts` and `src/postfx.ts` (nothing imported
them; `SCOPE.md` already listed Luna's sandbox as not ported). Renamed the stray
Luna references in comments, strings and test fixtures. One was user-visible: the
log panel labelled Yuri's lines "Luna:". The app icon (a neon cat), the tray
camera-active icon and `favicon.png` are now the bunny portrait. The status dot
became a flat SVG spinner (connecting / connected / listening / working).
`npm run clean` and `npm run clean:all` remove regenerable build output; the Cargo
dev profile uses line-table debug info so `src-tauri/target` is far smaller.

**Left on purpose.** The Luna importer (`python -m memory --import-luna`), the
GPT-SoVITS install path (`D:\AI\Project Luna\...` is a real folder on disk), and
history in the docs.

**Next candidates, not done because they run in the turn path and have no tests:**
the driver/observer WebSocket protocol, `scene_state` and `surface` in
`app.py` / `ws-client.ts` / `main.ts`. They exist for `sandbox.ts`, which is not in
this repo, so there is only ever one driver.

---

## D-029 — Frame from the model's top, and a doctor for "it doesn't load"

**Decision.** The camera looks at the head bone and pans up only as far as needed to keep
the model's highest point (bunny ears, a hat) in frame with 5 cm to spare, capped at 20 cm
(`src/framing.ts`). Before, it looked at the head bone with a fixed offset, so anything
taller than about 34 cm above the bone was cropped. `npm run doctor` (and a silent
`predev` hook) checks the usual reasons the dev window does not load and prints the fix
for each.

**Why.** A fixed offset is right for exactly one model; Luna and any future avatar would
have needed their own number. For the doctor: "it doesn't load" has several unrelated
causes (a stale process holding port 1420, no venv, no avatar file, no backend in a plain
browser tab) and they look identical from outside.

**Cost.** Framing now reads the model's skinned bounding box once at boot (a few ms). The
doctor adds about 100 ms to `npm run dev`; it never blocks the start.

---

## Superseded / historical

- **D-002 (first version).** "Groq is the cloud provider" was drafted before the
  user supplied the AI Router endpoint. Superseded by D-002 as written above.
