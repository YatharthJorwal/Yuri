# Yuri — Phases

Phase 0 is new. Phases 1-7 map onto `SPEC-v0.1.md` §11 with the router folded
into Phase 1. Every phase has a gate that can be checked by running something,
not by opinion. Sizes are relative (S < 2 days, M ~ a week, L > a week).

```
P0 bootstrap ──> P1 foundation + router ──> P2 fast path ──> P3 memory v2
                                      └──> P4 agent core ──> P5 affect/avatar
                                                          └──> P6 proactivity
                                                                     └──> P7 evals/docs
```

P3 and P4 are independent after P2 and can swap order. P6 depends on P4 (it
needs tiers) and P5 (it needs affect to sound right).

---

## Phase 0 — Bootstrap (M)

**Goal:** a running Yuri repo that is provably the same app as Luna at
`2015462b`, plus a config schema that everything after this depends on.

Deliverables:

- New repo, shell copied from Luna (`src-tauri/`, `src/` minus sandbox, `public/`,
  vite/tsconfig/package.json).
- Orchestrator files copied into a service-oriented layout, imports fixed, still
  one process.
- `config.py` v2: every key defaulted, startup validation (see `ROUTING.md` §11).
- The Windows Job Object child supervisor ported and used for every child
  process, not just the orchestrator.
- Test harness: Luna's pytest suite and 13 frontend tests runnable from Yuri.

Gate:

- `pytest orchestrator/` passes with the same count as Luna (314).
- `npm run test:fe` passes (13), `npx tsc --noEmit`, `npx vite build`.
- App launches, VRM renders, a typed message gets a reply, TTS plays.
- `nvidia-smi` shows one model resident, not two.
- Kill the supervisor with the window: no orphan Python, SoVITS, or Ollama child.

Non-goals: no router, no bus refactor, no new behavior. If something behaves
differently from Luna here, it is a porting bug.

---

## Phase 1 — Foundation + router (L)

**Goal:** typed services on one bus, and the router in shadow mode. Behavior of
the conversation is still exactly Luna's.

Deliverables:

- `bus.py`: the envelope in `SCOPE.md` §6, per-type validation, an in-memory
  replayable log for tests.
- Logical service extraction from `app.py`: `turn`, `voice`, `mind`, `memory`,
  `perception`, `scheduler`, `tools`. Same code, different seams.
- `router/`: `providers/ollama_native.py`, `providers/airouter.py`, decision
  function, budget meter, audit writer, discovery + cache.
- `/v1/models` discovery, `zdr` preference, fail-closed behavior.
- `shadow` mode fully wired, `enforce` available per role but off.
- `speech.py` skeleton with the rule table (full behavior lands in P2).
- Startup validation for illegal configs (cloud in a local-only role).

Gate:

- All of `ROUTING.md` §12 passes.
- A week of shadow logs (or 50 replayed turns) shows decisions with reason codes
  and zero cloud calls.
- Luna's test suite still passes through the new seams.
- `python -m yuri --config config/config.yaml --check` validates a config and
  prints the resolved chains without starting the app.

Non-goals: no prompt changes, no persona changes, no memory changes, no
streaming.

---

## Phase 2 — Fast path (L)

**Goal:** talking feels instant, and she can be interrupted. Details are in
`PHASE-2-FAST-PATH.md`; this is the summary.

Deliverables: streaming reflex replies, `speech.py` enforced on every spoken
string, streaming GPT-SoVITS with clause pipelining, VAD + faster-whisper input,
one speech arbiter, barge-in behind a measured gate, per-stage latency logging,
managed SoVITS launcher.

Gate:

- First audio p50 <= 1.2 s, p95 <= 1.8 s over a 20-turn scripted run, or the
  revised target from `PHASE-2-FAST-PATH.md` §2.5 if the CPU TTS measurement
  forces one (the revision is recorded in `DECISIONS.md`, not silently adopted).
- Zero banned characters reach TTS across the golden corpus and the scripted run.
- Barge-in stops playback within 150 ms, false triggers <= 1/20 turns, or the
  feature is off and half-duplex is on.
- Typed chat still works with the mic unplugged and with SoVITS down (pyttsx3).

Non-goals: no cloud, no proactivity, no affect-driven voice style.

---

## Phase 3 — Memory v2 (M)

**Goal:** memory that survives a hard kill and corrects itself.

Deliverables: facts with source + confidence, episode summaries + embeddings
(Luna's store extended), writer pass on a timer and at session end, contradiction
detection with supersede (no "forget" keyword needed), explicit forget, salience
decay, always-in-context profile card sized by token budget.

Gate:

- Kill -9 during a session loses at most the last writer interval (default
  10 min) of memory, proven by a test that kills the process.
- 10 seeded facts, 5 corrections, 3 forgets, 20 recalls: all correct in a real
  SQLite file.
- A 50-turn persona-consistency run still names facts from turn 5 at turn 45.
- Recall adds <= 300 tokens to a reflex turn and <= 1200 to a mind turn.

Non-goals: no cloud memory, no multi-store, no graph DB.

---

## Phase 4 — Agent core (L)

**Goal:** Mind runs jobs, tools are typed and tiered, confirmation cannot be
bypassed.

Deliverables: tool registry with JSON schema + tier + audit fields; Tier 0/1
auto, Tier 2 confirmation surface in the shell, Tier 3 allowlist with revocation,
Tier 4 hard-blocked in code; Mind job runner with step limit, time limit, and
cancel; audit log per tool call; `trust` propagation from perception into tool
policy; the owner-ping channel as a single Tier 3 entry point (feature-flagged).

Gate:

- Tool-safety suite: every tier attempted from an injected-content context is
  blocked or confirmed (`SPEC-v0.1.md` §10). 100% pass, no skips.
- A 6-step Mind job runs, is cancelable mid-step, and leaves an audit trail with
  one row per step.
- Tier 4 tools cannot be reached even by editing config (a code-level deny list
  and a test that asserts it).

Non-goals: no computer control, no mouse/keyboard, no third-party messaging.

---

## Phase 5 — Affect + avatar (M)

**Goal:** she looks alive and her mood is a state, not a random loop.

Deliverables: affect vector (mood, energy, affection, irritation) updated by
events and decaying to baseline; the vector feeds the prompt, VRM expressions,
and TTS style; Luna's `idle-motion.ts`, `eyes.ts`, `expression-life.ts` ported and
driven by it instead of timers; expression holds and fades.

Gate:

- A scripted 20-event scenario produces the expected affect trajectory in a unit
  test, and visibly different VRM states on screen (three screenshots).
- Idle life has no visible loop seam in a 60 s capture.
- Persona prompt shows the affect line and the tag round-trips to the shell.

Non-goals: no animation pack expansion, no sandbox apartment, no new VRM assets.

---

## Phase 6 — Proactivity (M)

**Goal:** she can start something, and she knows when not to.

Deliverables: triggers (idle, time of day, window change, long absence, drift);
proactivity budget (N unprompted per hour, backoff when ignored); the shared
arbiter gates every unprompted line; Task Guide becomes one trigger-driven
behavior; owner ping goes live behind its own flag with a rate limit.

Gate:

- 72-hour soak: unprompted interactions never exceed the configured rate, and
  backoff is observable in the log.
- An unprompted line never interrupts the user mid-sentence or mid-reply.
- Task Guide catches drift in a scripted screen sequence without an extra model
  call per tick.
- The owner ping fires at most once per absence event, and is off by default.

Non-goals: no social feeds, no other recipients, no calendar integration.

---

## Phase 7 — Evals and docs (S, continuous)

**Goal:** the suite that keeps the previous six phases honest.

Deliverables: replayable conversation fixtures against the real resident model;
persona consistency at 50 turns; memory write/correct/forget/retrieve against a
real DB; the tool-safety suite from P4; router evals; speech-normalizer corpus;
latency budget report per turn; docs trimmed to this spec plus `DECISIONS.md`.

Gate:

- `pytest` green from a clean checkout with no network.
- One command produces `logs/eval-report.json` with every gate from P0-P6
  re-checked.
- Docs: `SCOPE.md`, `ROUTING.md`, `PHASES.md`, `DECISIONS.md` are the only design
  docs left, and none of them contradict each other.

Non-goals: no benchmark theater, no leaderboard, no model churn without a reason
written down.

---

## Cross-phase rules

- Every phase ends with an entry in `DECISIONS.md` for anything non-obvious.
- Every config key added has a code default and a line in
  `config/config.example.yaml`.
- Every new service gets its own logger tag; one `logs/yuri.log`, rotated.
- No phase starts before the previous gate passes on this machine, not in a
  sandbox. Luna's handoff recorded that Rust, VRAM, and timing only count when
  measured on the real box; keep that discipline.
