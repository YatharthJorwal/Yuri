# Yuri

Local-first, agent-first companion for a single Windows PC. Successor to Luna.

**Status: Phase 0 complete, Phase 1 router landed, Phase 2 in progress (speech
path).** The port and config v2 are green; every model call goes through the
router (decision function, local + AI Router providers, spend caps, redaction,
audit); and the voice path now normalizes every spoken line in code, pipelines
TTS one clause ahead of the model, and writes a latency row per turn. Microphone
input, VAD, the arbiter and barge-in are the rest of Phase 2.

Verified in this environment: `512` pytest, `49` frontend tests, `tsc --noEmit`,
`vite build`, `cargo check`, a live `/v1/models` fetch (341 models), a real cloud
completion through the router (`mind` -> `deepseek-v4-flash`, 10/3 tokens,
$0.000002, audit row and spend ledger written), and a real GPT-SoVITS synthesis
(227 KB WAV from the running server). Not yet verified: a real desktop launch
(needs the VRM file), the tray toggle in a running build, and anything that needs
a microphone.

## Quickstart

```powershell
# 1. Python side
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r orchestrator\requirements-dev.txt

# 2. Shell dependencies
npm install

# 3. Your AI Router key (optional -- without it every role stays local)
#    Paste it after AIROUTER_API_KEY= in .env (or config/secrets.local.yaml).
#    Works while Yuri is running: flip the API-model switch and it is picked up.

# 4. Config (optional -- the example is used verbatim when this is absent)
Copy-Item config\config.example.yaml config\config.yaml

# 5. Run
npm run tauri dev
```

The orchestrator starts with the shell; it is not a separate command. It needs
Ollama running with `qwen3.5:9b` (and `qwen3.5:4b`, `nomic-embed-text` for the
classifier and memory). GPT-SoVITS is launched by the orchestrator once the
Phase 2 launcher lands; until then TTS falls back to `pyttsx3` and says so in
the log.

Check config, key resolution, catalog and spend without starting the app:

```powershell
.\.venv\Scripts\python.exe orchestrator\check.py            # cached, no network
.\.venv\Scripts\python.exe orchestrator\check.py --refresh  # fetch /v1/models
```

That is the command to run when "why is she not using the cloud" is the question.

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
.\.venv\Scripts\python.exe -m pytest orchestrator -q   # 512
npm run test:fe                                        # 49
npx tsc --noEmit
npm run build
```

No GPU, network, or API key is required for the suite. Tests that touch models
use fakes.

## What was built in Phase 1 (router)

- **`orchestrator/router/`**: `decide()` is a pure function over (role, request,
  config, state) implementing the ten rules in `ROUTING.md` in order; providers
  for Ollama-native and AI Router; a `/v1/models` catalog with cache and
  fail-closed behavior; a SQLite spend ledger; a JSONL audit log; redaction for
  keys, paths, emails and long digit runs.
- **One entry point**: `app.py`, `task_guide.py`, `ocr_watch.py`, `camera.py`,
  `tools/vision.py` and both memory jobs now ask for a *role*
  (`reflex`/`vision`/`classifier`/`memory_writer`) instead of a model.
- **Shadow by default**: decisions are computed and audited, execution stays
  local. `routing.mode: enforce` + `enforce_roles` is the opt-in; nothing spends
  money until then.
- **`orchestrator/check.py`**: config, key resolution, catalog and spend in one
  screen.
- **77 new tests** for the decision table, budget, audit, redaction, discovery,
  the SSE provider and the shadow/enforce/fallback paths.

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

## Cloud routing in one minute

Put the key in `.env` (`AIROUTER_API_KEY=`). With it, `check.py` should say
`cloud: available` and show a catalog size. Without it, Yuri still runs: every
role is local and `mind` falls back to `qwen3.5:9b`.

The router only ever escalates the `mind` role, and only for a turn that is
`trusted`, non-`secret`, has budget left, and meets one of the escalation
conditions (background job, explicit "think hard", or a large/multi-tool task).
Pixels, audio, raw transcripts and memory writes are structurally local: the
decision function refuses them before a provider is chosen.

To actually let Mind call out, either flip the switch in the app or set it in
`config/config.yaml`:

**The switch.** Open the `+` menu and use **Use API Model**, or tick
**Use API model (cloud)** in the tray menu. Both drive the same server state:

- `Off` — cloud decisions are recorded but never executed (shadow).
- `On` — `mind` may execute cloud decisions, still subject to privacy, trust and
  budget rules.
- `On` + `No key` — you asked for it but there is no resolvable
  `AIROUTER_API_KEY` (or no catalog), so everything stays local. The menu says
  so; it does not pretend.

The choice persists in `data/settings.json` and applies immediately, no restart.
The config default still exists for a fresh machine:

```yaml
routing:
  mode: enforce
  enforce_roles: [mind]
```

Either way the audit log (`logs/audit.jsonl`) records every decision, including
the ones that stayed local.

## Docs

| Doc | What it decides |
| --- | --- |
| [docs/SPEC-v0.1.md](docs/SPEC-v0.1.md) | The starting spec, kept verbatim for provenance |
| [docs/SCOPE.md](docs/SCOPE.md) | What Yuri is, what changes from the spec, what is out of scope |
| [docs/ROUTING.md](docs/ROUTING.md) | When a call goes local and when it goes to AI Router |
| [docs/PHASES.md](docs/PHASES.md) | Phase status, gates, and the roadmap index |
| [docs/DECISIONS.md](docs/DECISIONS.md) | Decision log (D-001...) |
| [config/config.example.yaml](config/config.example.yaml) | The config surface, annotated |

Per-phase plans (the contract for each phase: components, config, tests, gate,
risks):

| Phase | Plan |
| --- | --- |
| 1 — foundation + router | [docs/PHASE-1-ROUTER.md](docs/PHASE-1-ROUTER.md) |
| 2 — fast path (streaming voice, barge-in) | [docs/PHASE-2-FAST-PATH.md](docs/PHASE-2-FAST-PATH.md) |
| 3 — memory v2 | [docs/PHASE-3-MEMORY.md](docs/PHASE-3-MEMORY.md) |
| 4 — agent core (tools, tiers, Mind jobs) | [docs/PHASE-4-AGENT-CORE.md](docs/PHASE-4-AGENT-CORE.md) |
| 5 — affect + avatar life | [docs/PHASE-5-AFFECT.md](docs/PHASE-5-AFFECT.md) |
| 6 — proactivity + owner ping | [docs/PHASE-6-PROACTIVITY.md](docs/PHASE-6-PROACTIVITY.md) |
| 7 — evals + docs | [docs/PHASE-7-EVALS.md](docs/PHASE-7-EVALS.md) |

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
