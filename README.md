# Yuri

Local-first, agent-first companion for a single Windows PC. Successor to Luna.

**Status: Phase 0 complete.** Luna's shell and orchestrator are ported and
green, the config surface is v2, the typed event bus exists, and the speech
normalizer that keeps em dashes and quotes away from TTS is working. The model
router is designed (`docs/ROUTING.md`) and is the next build step.

Verified in this environment: `386` pytest, `49` frontend tests, `tsc --noEmit`,
`vite build`, `cargo check`. Not yet verified: a real desktop launch (needs the
VRM file, Ollama warm, and the Tauri window) and any actual audio output.

## Quickstart

```powershell
# 1. Python side
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r orchestrator\requirements-dev.txt

# 2. Shell dependencies
npm install

# 3. Config (optional -- the example is used verbatim when this is absent)
Copy-Item config\config.example.yaml config\config.yaml

# 4. Run
npm run tauri dev
```

The orchestrator starts with the shell; it is not a separate command. It needs
Ollama running with `qwen3.5:9b` (and `qwen3.5:4b`, `nomic-embed-text` for the
classifier and memory). GPT-SoVITS is launched by the orchestrator once the
Phase 2 launcher lands; until then TTS falls back to `pyttsx3` and says so in
the log.

Check what the config resolved to without starting the app:

```powershell
cd orchestrator
..\.venv\Scripts\python.exe -c "import config; print(config.print_summary())"
```

## Your avatar and animations

Neither is committed. Drop them in:

| Asset | Path |
| --- | --- |
| VRM model | `public/vrm/yuri.vrm` |
| VRMA clips | `public/vrm-animations/*.vrma` |

Both folders have a `README.txt` with the details. If the VRM is missing, the
window shows an instruction panel instead of a blank canvas. Without VRMA clips
the avatar still breathes, blinks and glances, because those are procedural.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest orchestrator -q   # 386
npm run test:fe                                        # 49
npx tsc --noEmit
npm run build
```

No GPU, network, or API key is required for the suite. Tests that touch models
use fakes.

## What was built in Phase 0

- **Port**: Luna `2015462b` shell (Tauri, three.js/VRM, tray, hotkeys, Job
  Object child management) plus its orchestrator, its memory store, its tools
  and all 314 of its tests.
- **`orchestrator/config.py` v2**: `providers` / `models` / `roles` / `routing` /
  `privacy` / `budget` / `voice` / `speech`, with every key defaulted and both
  the v2 and Luna-era shapes accepted. A cloud provider in a local-only role is
  a startup error, not a warning.
- **`orchestrator/bus.py`**: typed events with a `trust` flag and a bounded
  replay log, in-process on purpose (D-011).
- **`orchestrator/speech.py`**: deterministic speech-safe text. Em/en dashes,
  quotes, ellipses, markdown, emoji, URLs and code fences never reach TTS.
  53 tests, including a golden corpus and an idempotence invariant.
- **Rebrand**: window title, product name, identifier, persona prompt, model
  path, asset placeholders.

## Docs

| Doc | What it decides |
| --- | --- |
| [docs/SPEC-v0.1.md](docs/SPEC-v0.1.md) | The starting spec, kept verbatim for provenance |
| [docs/SCOPE.md](docs/SCOPE.md) | What Yuri is, what changes from the spec, what is out of scope |
| [docs/ROUTING.md](docs/ROUTING.md) | When a call goes local and when it goes to AI Router |
| [docs/PHASES.md](docs/PHASES.md) | Phase 0-7 with gates and non-goals |
| [docs/PHASE-2-FAST-PATH.md](docs/PHASE-2-FAST-PATH.md) | Streaming voice, barge-in, latency budget |
| [docs/DECISIONS.md](docs/DECISIONS.md) | Decision log (D-001...) |
| [config/config.example.yaml](config/config.example.yaml) | The config surface, annotated |

## Layout

```
orchestrator/     Python: FastAPI/WebSocket server, agent loop, memory, tools
  config.py       config v2 + derived v1 views
  bus.py          typed event bus
  speech.py       speech-safe text
  llm.py          Ollama/OpenAI-compatible streaming client
  memory/         SQLite + sqlite-vec: facts, episodes, recall, consolidation
  tools/          screen, camera, clipboard, vision
src-tauri/        Rust core: window, tray, hotkeys, child-process Job Object
src/              three.js + @pixiv/three-vrm, lipsync, mic, idle motion
config/           config.example.yaml, secrets.example.yaml
docs/             spec, scope, routing, phases, decisions
public/vrm/       your avatar (gitignored)
public/vrm-animations/  your clips (gitignored)
```
