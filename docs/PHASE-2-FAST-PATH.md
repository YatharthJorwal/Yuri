# Phase 2 — Fast path (voice)

The phase that decides whether Yuri feels alive or feels like a chat window with
a cartoon attached.

Everything here is local: GPT-SoVITS for TTS, faster-whisper for STT, Silero for
VAD, `qwen3.5:9b` for the reflex turn. No network call is on any path this
document defines.

Read with `PHASES.md` (Phase 2), `SCOPE.md` §6 and §9, `ROUTING.md` §3.

**Status (in progress).** Landed: `speech.py` enforced inside `_send_speak` (so
every spoken path, including fallbacks and ambient remarks, is normalized);
`speech_stream.py` pipelining (clause queue, one-clause synthesis lookahead,
abort); `latency.py` with one JSONL row per turn; and `tts.ensure_server()` with
a port-first managed launch. Verified against the real server: 227 KB WAV from
`/tts`. Since: streaming 20 ms PCM frames from the shell, an energy VAD with an adaptive
noise floor, WAV packaging for STT, one arbiter gating every unprompted line, and
barge-in (VAD speech during a turn cancels it and sends `barge_in`, which stops
playback). Remaining: the §5 latency *targets* measured over a real 20-turn
microphone run; the instrumentation exists, the gate does not.

---

## 0. Shape of a spoken turn

```
mic frames ──VAD──> speech.ended ──STT──> turn.started
                                             │
                              reflex stream ─┤ first token
                                             ├──> speech.prepare()  (clauses)
                                             │         │
                                             │         ├──> TTS clause 1 ──> playback.started
                                             │         ├──> TTS clause 2 (prefetched)
                                             │         └──> ...
                                             └──> turn.finished ──> playback.ended ──> idle
```

Text input follows the same tail (turn → clauses → TTS → playback); only the head
differs. Nothing in the TTS or playback path knows whether the turn came from the
mic.

---

## 1. `speech.py` — the speech-safe normalizer

**Why it is code and not a prompt rule:** a 9B model does not hold a typographic
rule for 50 turns. Luna's log has four separate examples of prompt rules failing
on this model. The prompt asks; this module guarantees.

### 1.1 Contract

```python
@dataclass
class PreparedSpeech:
    display: str              # what the chat panel shows (model text, lightly trimmed)
    spoken: str               # normalized, safe to speak as one string
    segments: list[str]       # clause chunks, in order, each <= max_segment_chars
    dropped: list[str]        # what was removed, for the debug log only
```

`prepare(text, config) -> PreparedSpeech`

Pure function. No I/O, no model, no clock. Fully unit-testable.

### 1.2 Rules, in application order

| # | Input | Output | Note |
| --- | --- | --- | --- |
| 1 | fenced code block ```...``` | removed, replaced with "code block" | never read code aloud |
| 2 | inline `` `code` `` | the code text without backticks | short tokens are fine to say |
| 3 | markdown emphasis `**x**`, `__x__`, `*x*`, `_x_` | `x` | |
| 4 | markdown links `[x](url)` | `x` | |
| 5 | bare URLs | "a link" | |
| 6 | headings `# x`, blockquotes `> x` | `x` | |
| 7 | bullet and numbered list markers | removed, sentences joined with periods | |
| 8 | em dash `—`, en dash `–`, horizontal bar | comma | the specific complaint |
| 9 | quotes: `" " ' ' « » „ "` and straight quotes | removed | never spoken |
| 10 | ellipsis `…` or `...` | comma | |
| 11 | semicolon | period | |
| 12 | parentheses and their contents | removed if under 40 chars, else the contents kept with commas | asides rarely sound right |
| 13 | emoji and pictographs | removed | |
| 14 | repeated punctuation `!!!`, `???` | single `!`, `?` | |
| 15 | newline runs | period + space | |
| 16 | ALL CAPS words longer than 3 chars | lowercased unless in an allowlist of acronyms (`VRAM`, `GPU`, `API`, `TTS`, `STT`, `SQL`) | shouting reads badly |
| 17 | `:)`, `:(`, `:3`, `;)` | removed | |
| 18 | whitespace runs, leading/trailing punctuation runs | collapsed | |

Then segmentation (rule 19): split on sentence terminators first, then on commas
if a sentence exceeds `max_segment_chars` (160), never inside a word, never
producing a segment under `min_segment_chars` (12) unless it is the only one.

Numbers and units are left alone in v1. "12:30" and "3.5" are acceptable in
English TTS; speaking them as words is a later, riskier normalization.

### 1.3 What the model is still told

The persona prompt carries the explicit no-list, because it improves the raw
output and costs nothing. It is not the enforcement.

```
Speech rules, no exceptions:
You are speaking out loud, so write only words a person would say.
- Never use em dashes or en dashes. Use a comma.
- Never use quotation marks of any kind.
- Never use ellipses, asterisks, parentheses, semicolons, bullet points,
  numbered lists, or code blocks.
- No emoji, no emoticons, no URLs.
- One to three short sentences per reply unless asked to explain.
- If something has to be shown rather than said, say in one sentence that
  it is on screen, and stop.
```

### 1.4 Tests (`test_speech.py`)

- 30-case golden corpus: each case gives model text and the exact expected
  `spoken` and `segments`. Cases are taken from real Luna replies, not invented.
- Invariant: for every case, `banned_chars & set(spoken) == {}` where
  `banned_chars = "—–"'""…*_`>;[]()#"` plus emoji ranges.
- Idempotence: `prepare(prepare(x).spoken).spoken == prepare(x).spoken`.
- Segmentation: no segment > 160 chars, no segment splits a word, first segment
  never empty, ordering preserved.
- Code: a reply containing a fenced block produces "code block" and none of the
  code's identifiers in `spoken`.

---

## 2. Streaming TTS

### 2.1 The decision: one request per clause, not one streaming request

GPT-SoVITS `api_v2.py` supports `streaming_mode: true` with server-side
`text_split_method: cut5`. Yuri still uses **one non-streaming request per
clause**, pipelined, for four reasons:

1. Each clause comes back as a complete WAV, which the shell can buffer, gaplessly
   order, and lip-sync against. A single opaque stream makes all three harder.
2. Abort is per clause: barge-in drops the queue instead of tearing down a stream.
3. A slow or failed clause is retried alone, not the whole reply.
4. Server-side splitting is a black box; our splitter is tested.

Tradeoff: requests carry a fixed overhead each. Measured in §2.5; if the overhead
dominates, the fallback is one `streaming_mode: true` request per sentence group
(not per whole reply), which keeps abort granularity.

### 2.2 Pipeline

- `prefetch_depth: 2`: clause N+1 is requested while clause N plays.
- First clause is buffered to `first_clause_buffer_ms` (default 800 ms of audio)
  before playback starts, so the first pause is not a stutter.
- Clauses are appended to one audio queue in the shell; playback never gaps
  silently: if the queue drains, the arbiter returns to `thinking` and the shell
  shows the thinking state.

### 2.3 Managed launcher (no install needed)

SoVITS is already on this machine:

```
D:\AI\Project Luna\GPT-SoVITS-v3lora-20250228\
  runtime\python.exe          (3.9.13, bundled)
  api_v2.py
  GPT_SoVITS\configs\tts_infer.yaml
  ffmpeg.exe / ffprobe.exe
```

Launch:

```
runtime\python.exe api_v2.py -a 127.0.0.1 -p 9880 -c GPT_SoVITS\configs\tts_infer.yaml
```

Supervisor responsibilities:

- Start it as a child of the supervisor, inside the existing Windows Job Object,
  so closing Yuri never leaves an orphan (Luna had this recur three times).
- Health check: TCP connect to 9880, then a one-character `/tts` probe. Ready
  when the probe returns audio or a valid 4xx that proves the server is alive.
- `startup_timeout_s: 120`, then mark TTS unavailable and use the pyttsx3
  fallback with one visible log line (Luna's behavior, kept).
- Restart on crash, at most twice, with backoff; after that stay down and fall
  back.
- Never block startup on it: the app opens, TTS becomes available when the probe
  passes.

### 2.4 Fallback ladder

GPT-SoVITS → pyttsx3 (kept from Luna) → text only with a tray notice. Each step
logs one line naming the exact failure, because "wrong voice came out" was a
real, silent bug in Luna.

### 2.5 Gate: the CPU reality check

Zero-shot GPT-SoVITS on CPU is the main technical risk in this phase. Measure
before promising 1.2 s:

1. Synthesize three fixed sentences (8, 20, 45 words) five times each through
   the API on this machine.
2. Record real-time factor (audio seconds produced / wall seconds) per request,
   warm and cold.
3. Decide:
   - RTF under 0.6 on the first clause: the 1.2 s target stands.
   - RTF 0.6-1.0: keep CPU, accept a first-audio target of ~2.0 s, and say so in
     the UI (thinking state, not silence).
   - RTF over 1.0: allocate ~2 GB VRAM to SoVITS (9B Q4_K_M + KV leaves room),
     or benchmark a faster local TTS. This becomes an explicit VRAM decision, not
     a silent regression.
4. Record the numbers in `DECISIONS.md` either way.

This measurement runs **before** the pipeline is built against a latency target,
so the target is set by the machine, not by hope.

---

## 3. Voice input

### 3.1 Capture

- Shell captures mono 16 kHz PCM16 in 20 ms frames, from the default input
  device, with `echoCancellation: true`, `noiseSuppression: true`,
  `autoGainControl: true` in the `getUserMedia` constraints.
- Frames go over the existing WebSocket as binary, preceded by a tiny JSON
  header (`{"type":"audio.frame","seq":n,"ts":...}`), or as a separate binary
  channel if the interleaving gets awkward. 20 ms frames are ~640 bytes; the
  bandwidth is negligible.
- Mic permission is requested once, visible in the tray, and revocable. Audio is
  never written to disk (`voice.mic.save_audio: false`).

### 3.2 VAD

- Primary: Silero VAD, ONNX runtime, CPU. Small, no torch in the orchestrator's
  Python 3.13 environment if the ONNX wheel resolves.
- Fallback: `webrtcvad` if the ONNX wheel is unavailable, then an energy gate as
  last resort (documented as worse, and only to keep the app usable).
- Parameters: `speech_threshold` 0.5, `end_silence_ms` 500, `min_utterance_ms`
  250, `max_utterance_ms` 30000, 200 ms pre-roll buffer so the first phoneme is
  not clipped.
- F9 is a manual override: hold to talk, release to send, bypassing VAD. Kept from
  the spec as the escape hatch.

### 3.3 STT

- faster-whisper `small`, `int8`, CPU, language `en` (fixed, not auto: auto-detect
  costs latency and mislabels one-word utterances).
- Warm the model lazily on first use, then keep it resident in the worker thread.
- Timeout 20 s per clip, then surface "didn't catch that" rather than hanging
  (Luna had an indefinite-hang failure here; the timeout is kept).
- Transcripts shorter than 2 words or matching a known noise pattern are dropped
  as accidental noise, not sent to the model.

---

## 4. Arbiter and barge-in

### 4.1 One arbiter

`arbiter.py` owns exactly one question: who may produce sound right now. Voice
replies, ambient remarks, and Task Guide chides all go through it. This is the
answer to v0.1 §8's collision problem, implemented in one place.

```
idle ──speech.started──> listening ──silence──> transcribing ──text──> thinking
  ↑                                                                       │
  │                                                                 first audio
  └──────── playback.ended + turn.finished ──── speaking <───────────────┘
                                                   │
                                       barge_in.detected
                                                   ↓
                                              interrupting ──> listening
```

Rules:

- Only the user can interrupt. She never preempts herself, and ambient remarks
  never preempt a voice turn; they queue behind `speaking` and drop if the user
  starts talking.
- `interrupting` stops playback, cancels queued TTS, cancels the in-flight LLM
  stream, and marks the partial reply in history as interrupted (so memory
  consolidation does not treat a half sentence as a commitment).
- Typed input during `speaking` also interrupts, and is treated the same way.

### 4.2 Barge-in detection

- Requires `barge_in.min_speech_ms` (300 ms) of speech while `speaking`, above
  the VAD threshold, and after AEC has had a chance to cancel her own output.
- A second gate uses the shell's reported playback level: if mic energy tracks
  output level within a tolerance, it is echo, not the user.
- Stop latency: on detection, send `playback.stop` immediately (do not wait for
  the current clause to finish), then cancel TTS and the stream.

### 4.3 The measured gate

Run 20 scripted turns with speakers at normal volume:

- false triggers (she stops while the user is silent) must be <= 1/20;
- her own voice must never trigger a stop (0/20 by construction if AEC works);
- a genuine interruption must stop audio within 150 ms, 20/20.

Fail: `voice.barge_in.enabled` flips to false, half-duplex ships (mic muted while
speaking, F9 always available), and barge-in moves to Phase 2.5 with the recorded
numbers. This is a scope decision, not a bug to grind on: the spec's "no
push-to-talk required" is conditional on the hardware forgiving us.

---

## 5. Latency instrumentation

Seven timestamps per spoken turn, monotonic, written as one JSONL row:

| Field | Point |
| --- | --- |
| `t0` | last voiced frame before silence |
| `t1` | VAD declared end of speech |
| `t2` | transcript ready |
| `t3` | turn started (prompt assembled) |
| `t4` | first model token |
| `t5` | first TTS request sent |
| `t6` | first audio byte received |
| `t7` | playback actually started in the shell |

Derived: `stt_ms = t2-t1`, `ttft_ms = t4-t3`, `clause_ms = t5-t4`,
`tts_ms = t6-t5`, `playback_ms = t7-t6`, `total_ms = t7-t0`.

Targets (provisional until §2.5):

| Stage | Target |
| --- | --- |
| `stt_ms` | <= 700 |
| `ttft_ms` | <= 400 |
| `clause_ms` | <= 200 |
| `tts_ms` | <= 900 (or the measured value from §2.5, whichever is worse) |
| `playback_ms` | <= 50 |
| `total_ms` p50 | <= 1.2 s |
| `total_ms` p95 | <= 1.8 s |
| inter-clause gap | < 300 ms |

Measurement protocol: 20 fixed user utterances as WAV fixtures (no mic, no VAD
flakiness), replayed through the real pipeline, real model, real TTS. Report p50,
p95, and the worst turn with its `logs/latency.jsonl` line. No warm-up turn
counted; a cold-load turn is reported separately.

---

## 6. Startup order

1. Config load + validation (fails fast, no partially-running app).
2. Bus, logger, SQLite open (schema migrate if needed).
3. Ollama reachability + capability check; warm the reflex model
   (`keep_alive` set, one empty generate).
4. SoVITS launcher (non-blocking); TTS marked pending until its probe passes.
5. STT model load (non-blocking; first speech pays it, logged).
6. Shell opens, VRM renders, tray shows state.
7. Voice loop starts in `idle`.

If Ollama is missing, the app still opens with a tray warning and typed replies
fail with the same in-character line Luna uses. Nothing crashes at startup
because a model is absent.

---

## 7. Config deltas

New blocks, all defaulted in code:

```yaml
voice:
  enabled: true
  stt: {engine: faster_whisper, model_size: small, device: cpu, compute_type: int8, language: en, timeout_s: 20}
  vad: {engine: silero_onnx, speech_threshold: 0.5, end_silence_ms: 500, min_utterance_ms: 250, max_utterance_ms: 30000, pre_roll_ms: 200}
  barge_in: {enabled: true, min_speech_ms: 300, stop_within_ms: 150, aec: browser, fallback: half_duplex}
  push_to_talk_key: F9
  mic: {save_audio: false, indicator: tray}

tts:
  engine: gpt_sovits
  stream: per_clause
  gpt_sovits:
    api_url: 'http://127.0.0.1:9880/tts'
    launch:
      enabled: true
      python: 'D:\AI\Project Luna\GPT-SoVITS-v3lora-20250228\runtime\python.exe'
      script: 'D:\AI\Project Luna\GPT-SoVITS-v3lora-20250228\api_v2.py'
      args: ['-a', '127.0.0.1', '-p', '9880', '-c', 'GPT_SoVITS/configs/tts_infer.yaml']
      cwd: 'D:\AI\Project Luna\GPT-SoVITS-v3lora-20250228'
      startup_timeout_s: 120
      restart_max: 2
    ref_audio_path: 'C:\Users\User\Downloads\character_files_main_sample.wav'
    prompt_text: "This is a sample voice for you to just get started with because it sounds kind of cute, but just make sure this doesn't have long silences."
    prompt_lang: en
    text_lang: en
    media_type: wav
    prefetch_depth: 2
    first_clause_buffer_ms: 800
  fallback_engine: pyttsx3

speech:
  enabled: true
  max_segment_chars: 160
  min_segment_chars: 12
  code_blocks: say_marker
  acronym_allowlist: [VRAM, GPU, API, TTS, STT, SQL, CPU, DLL]

latency:
  log: logs/latency.jsonl
  report_after_turns: 20
```

Note the single quotes on every Windows path: double quotes treat backslashes as
escapes in this loader (Luna hit this; the loader documents it).

---

## 8. Test plan

Unit (fast, no I/O):

- `test_speech.py` — §1.4.
- `test_segmentation.py` — clause split invariants, long-word safety, first-chunk
  size.
- `test_arbiter.py` — every transition; ambient never preempts; user always can;
  interrupt cancels the queue exactly once.
- `test_barge_in.py` — synthetic frame sequences: silence, her own voice profile
  (echo), real speech, short blips under `min_speech_ms`.
- `test_tts_stream.py` — fake server: ordering, prefetch depth, abort mid-queue,
  one clause failing and retrying, fallback to pyttsx3.

Integration (fake model, real files):

- `test_turn_pipeline.py` — text in, ordered WAV fixtures out, latency row
  written with seven timestamps.
- `test_launcher.py` — start a stub server, health probe, timeout path, restart
  cap, Job Object kill on supervisor exit.

Manual, gated on this machine:

- §2.5 TTS benchmark.
- §4.3 barge-in run (20 turns).
- §5 latency run (20 turns).
- 10-minute ambient run: no clipped words, no runaway playback, VRAM under
  9.0 GB with the model warm and SoVITS running.

---

## 9. Risks, ranked

1. **CPU SoVITS is too slow for a 1.2 s target.** Mitigated by measuring first
   (§2.5) and choosing among three explicit outcomes. This is the only risk that
   can change the phase's promise.
2. **Echo causing false barge-ins.** Mitigated by AEC + level gate + the
   half-duplex fallback. If speakers are loud, headphones make this disappear;
   the config supports both.
3. **ONNX runtime on Python 3.13.** If neither Silero ONNX nor webrtcvad installs
   cleanly, VAD degrades to an energy gate and the phase's voice-input quality
   drops. Check this on day 1 of the phase, not week 3.
4. **Zero-shot voice quality from an 8.5 s reference.** Acceptable for v1; the
   weights directories are empty, so any fine-tune is new work (Phase 2.5).
5. **WebView2 mic and autoplay policies.** Luna already hit a suspended
   `AudioContext`; resume on focus/gesture is ported, and "played" is never
   assumed to mean "heard" (a log line and a UI state change confirm it).
6. **Lip-sync drift over a long reply.** Chunk-aware updates keep amplitude tied
   to the currently playing buffer; a test plays a 30 s fake reply and asserts
   the mouth returns to rest exactly when audio ends.
7. **First-token quality dropping for speech.** Normalization removes structure;
   if replies start sounding clipped, the fix is persona/prompt, not loosening
   the normalizer.

---

## 10. Definition of done

- Spoken turn: mic → transcript → reply → audio, with `total_ms` p50 and p95
  within the §5 targets (adjusted by §2.5 if it says so), reported in
  `DECISIONS.md`.
- Interrupt mid-reply: audio stops within 150 ms, the next utterance is heard
  fully, and the interrupted reply is marked in history.
- Every spoken string passed through `speech.py`; the golden corpus and a real
  20-turn run contain zero banned characters.
- SoVITS launches and dies with the supervisor; no orphan processes after three
  open/close cycles.
- Typed chat works with the mic absent, SoVITS down, and the network off.
- `pytest` green, including the new voice tests.
- VRAM at or under 9.0 GB with everything warm.
