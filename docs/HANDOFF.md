# Handoff

> Read once at the start of a session, then stop consulting it. It is a
> snapshot and goes stale on purpose. If it disagrees with `docs/PHASES.md` or
> `docs/DECISIONS.md`, those win.

Repo: `D:\AI\Yuri` · remote `github.com/YatharthJorwal/Yuri` · branch `main`
HEAD: see `git log` (Phase 3 + layout change are one or two commits on top of `262c933`)

## State

- **Phase 0** (port + config v2 + bus + speech normalizer): done.
- **Phase 1** (router, AI Router provider, budget, audit, redaction, shadow
  mode): landed and tested. One real cloud call has been made successfully.
- **Phase 2** (voice): the path is built and green, the **measured gate is not
  done** — that needs a real microphone and speakers.
- **Phase 3** (memory v2): landed. Writer pass, supersede/forget that never delete,
  budgeted recall, decay, profile card, `python -m memory` CLI. Gate pieces that
  need a real model (50-turn persona run, ambiguous-contradiction quality) are
  unmeasured.
- **Phases 4-7**: scoped in `docs/phases/PHASE-*.md`, not started.
- Tests: **630 pytest**, **49 frontend**, `tsc --noEmit`, `vite build`,
  `cargo check` all green.
- `data/settings.json` does not exist yet: the API-model switch has never been
  flipped, so routing follows the config default (`shadow`).
- `logs/latency.jsonl` does not exist yet: no real spoken turn has run.

## Do this first

1. **Run it:** `npm run tauri dev` (venv and `node_modules` already installed).
   Ollama must be up with `qwen3.5:9b`. GPT-SoVITS is launched by the shell via
   `src-tauri/launcher.local.txt`; the Python side only checks the port first, so
   it will not double-start.
2. **Check the tray/quick-action toggle.** `+` menu → "Use API Model", and the
   tray's "Use API model (cloud)". Turn it on with the key in place: the badge
   should read `On`; remove the key temporarily and it should read `No key`
   rather than lie. This UI has not been run yet.
3. **Voice smoke test** (needs the mic): click the mic, say something, confirm
   the transcript and the spoken reply; then talk over her mid-reply and confirm
   playback stops instantly (barge-in). Report the numbers from
   `logs/latency.jsonl` (`ttft_ms`, `first_clause_ms`, `first_audio_ms`,
   `turn_ms`).
4. If barge-in false-fires more than once in 20 turns, set
   `voice.barge_in.enabled: false` and record the measurement — D-016 already
   authorises the half-duplex fallback.

## What is actually built (file map)

| Piece | Where | Notes |
| --- | --- | --- |
| Router + decision | `orchestrator/router/` | pure `decide()`, Ollama-native and AI Router providers, catalog, spend ledger, audit, redaction |
| Speech-safe text | `orchestrator/audio/speech.py` | enforced in `app._send_speak` — the single chokepoint |
| Pipelined TTS | `orchestrator/audio/speech_stream.py` | clause queue, one-clause lookahead, abort |
| Latency rows | `orchestrator/audio/latency.py` | one JSONL row per turn, written even on a stop |
| VAD | `orchestrator/audio/vad.py` | energy + adaptive floor, 200 ms pre-roll, voiced-frame minimum |
| Voice input | `orchestrator/audio/voice.py` | frames → WAV for faster-whisper |
| Arbiter | `orchestrator/audio/arbiter.py` | one mouth; ambient lines are gated by it |
| Runtime switch | `orchestrator/settings.py` | persisted API-model toggle, `data/settings.json` |
| Health/CLI | `orchestrator/check.py` | config, key, catalog, spend |
| Streaming mic | `src/mic-stream.ts` | 20 ms PCM16 @16 kHz, AEC requested |
| Persona | `orchestrator/persona.py` | the user's new prompt, verbatim apart from one clarified contraction rule |

## Gotchas worth knowing

- **The API-model toggle only affects `mind`, and nothing calls `mind` yet.**
  That role belongs to the Phase 4 job runner. Turning the switch on arms the
  gate; it does not change how a normal chat turn is answered. If you want chat
  to escalate, that is a scope decision (allow `reflex` to be cloud-eligible),
  not a bug.
- **`_send_speak` is the only place text becomes audio.** Anything new that
  synthesizes speech must go through it or `speech.prepare()`, or the dash and
  quote guarantee stops applying.
- **VAD frames must be exactly 640 bytes** (320 samples, 20 ms at 16 kHz). The
  frontend resamples; a partial frame is ignored server-side.
- **Transcription is awaited inside the WS receive loop**, so about a second of
  frames can queue up while STT runs. Acceptable today; move it to a task if it
  ever swallows the start of the next sentence.
- **Tests stub `llm.<fn>` by module attribute** and the router calls through the
  module, which is why the ported suite stayed green through the wiring. Do not
  change that to `from llm import x`.
- **Memory never deletes.** Forget and supersede set `superseded_by`; use
  `python -m memory --facts --all` to see everything and `--restore ID` to undo.
  Temp Chat turns are logged with `temp=1` and the writer skips them; anything
  that reads `transcript_log` for memory must do the same.
- **Imports after the layout change:** `from audio import tts`, `from senses
  import camera`; tests are in `orchestrator/tests/` and `orchestrator/conftest.py`
  puts `orchestrator/` on the path.
- **Do not bulk-edit source with naive string replacement** — one attempt
  replaced every "a" with "w" in five files. `git checkout --` fixed it; do
  targeted edits.
- Secrets live in `config/secrets.local.yaml` (gitignored) and `.env`
  (gitignored). The key resolves today; never commit either file.

## Next, in order

1. Finish Phase 2's gate: the measurements above, recorded in `DECISIONS.md`.
2. **Phase 3 follow-ups** (see D-025): make the Ollama-native provider honour the
   role's model and a JSON `format` so `memory_writer` really runs on the 4B;
   run the 50-turn persona check; try a real correction ("actually I...") and
   confirm the old fact stops being recalled (`python -m memory --facts --all`).
3. **Phase 4, agent core** (`docs/phases/PHASE-4-AGENT-CORE.md`): tool tiers,
   confirmation, Mind job runner. This is what makes the API-model switch
   actually do something.
4. Then P5 affect, P6 proactivity (+ owner ping), P7 evals.

## Open questions for the user

- Should a normal chat turn ever use the cloud (`reflex` escalation), or does
  `mind`-only stay the rule? Latency versus capability; currently the rule is
  mind-only.
- The owner-ping channel is still unchosen (ntfy is the recommendation).
- VRM and VRMA assets are still not in `public/`; the app shows an instruction
  panel without them.

## Commands

```powershell
cd D:\AI\Yuri
.\.venv\Scripts\python.exe -m pytest orchestrator -q      # 630
npm run test:fe                                            # 49
npm run build                                              # tsc + vite
cargo check --manifest-path src-tauri\Cargo.toml
.\.venv\Scripts\python.exe orchestrator\check.py --refresh # config, key, catalog, spend
```

Commit and push when the tree is clean: `git add -A; git commit -m "..."; git push`.
