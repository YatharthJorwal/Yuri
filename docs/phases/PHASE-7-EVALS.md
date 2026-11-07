# Phase 7 — Evals, regression gates, and release

Every earlier phase ends with a gate that was checked once, by hand, on this
machine. P7 turns those gates into one command that can be re-run after any
change, and fixes the thing that actually rots a personal project: docs that
describe a version that no longer exists.

Dependencies: all of them. It starts earlier than the end, though (each phase
lands its own subset); this document is the target state.

---

## 0. Goal

`python -m evals.run` produces `logs/eval-report.json` and a one-screen summary.
Green means every phase's gate still holds. No network by default; a `--live`
flag opts into the real Ollama model (and, with a key, the cloud model).

---

## 1. Layout

```
evals/
  __init__.py
  run.py               CLI: --suite all|conv|memory|tools|router|voice|latency --live
  fixtures/
    conversation.jsonl     replayable turns + expected properties (not exact text)
    memory.jsonl           seed facts, corrections, forgets, recall probes
    tools.jsonl            tier attempts from trusted and untrusted contexts
    router.jsonl           requests + expected decision reason codes
    speech.jsonl           hostile model outputs + expected spoken text
    persona_anchors.json   the "does she still sound like Yuri" reference
  suites/
    conv.py, memory.py, tools.py, router.py, voice.py, latency.py, persona.py
  report.py            JSON + markdown, thresholds, exit code
```

Fixtures are data. A suite that needs a new case adds a JSONL line, not code.

---

## 2. What each suite asserts

### conversation (needs `--live`)

- Replay 20 fixed turns; assert properties, not strings: reply is speakable,
  within the token cap, ends with a valid emotion tag, contains no banned
  characters (`speech.BANNED_CHARS`).
- A 50-turn persona run: at turn 45, still uses the speech rules, still answers
  in the tsundere register, and still remembers facts seeded at turn 5.
- Persona drift: embed each assistant reply and the persona anchor text; mean
  cosine similarity must stay above the baseline recorded when the persona was
  accepted (a drop of 0.08 fails). Cheap, deterministic, and much harder to
  game than a vibe check.

### memory (real SQLite, no model where possible)

- Write 10 facts, apply 5 corrections, forget 3, then 20 probes: every probe
  returns the right fact or none, never a superseded one.
- Recall budget: <= 300 tokens for a reflex turn, <= 1200 for a mind turn.
- Kill test: the writer subprocess is SIGKILLed mid-pass; restart; assert the
  facts from before the last checkpoint are present and nothing is duplicated.

### tools

- Every line in `tools.jsonl` is attempted twice: trusted context and
  injected-content context. Assert blocked or confirmed, per tier.
- Hard-deny list is unreachable even with a grant. Path traversal cases.
- Confirmation hash mismatch and expiry are rejected.
- **This suite is the release blocker.** It must be 100% with no skips.

### router

- Decision table from `ROUTING.md` §4, one case per reason code, precedence.
- Budget: no call after the cap; ledger survives a restart.
- Shadow: cloud fake never called.
- Redaction: a seeded key/path/email never appears in the payload sent to the
  fake cloud provider.

### voice (fake TTS/STT, real pipeline logic)

- Clause pipelining order, abort on barge-in, prefetch depth, one failed clause
  retried alone.
- The speech corpus runs through the same normalizer and asserts
  `banned_chars_in(spoken) == set()` plus idempotence.
- Latency: 20 recorded WAV fixtures replayed through the pipeline, report p50 /
  p95 per stage from `logs/latency.jsonl`.

### latency / cost (live, opt-in)

- Same 20 turns against the real model; report time-to-first-token, time to
  first audio, VRAM peak (`nvidia-smi` sampled), cost per turn.
- Compare against `config/config.example.yaml`'s `latency.targets_ms`; a
  regression over 20% fails the run.

---

## 3. Report and thresholds

`eval-report.json`:

```json
{"ts": "...", "live": false, "commit": "abc1234",
 "suites": {"conv": {"status": "skipped"}, "tools": {"status": "pass", "cases": 48}}},
 "metrics": {"persona_similarity": 0.81, "recall_tokens_p95": 287,
             "ttfa_p50_ms": null, "cost_today_usd": 0.0},
 "failed": []}
```

Exit code 0 only when every non-skipped suite passes and no threshold regressed.
Thresholds live in `evals/thresholds.json` with the baseline commit, so a
deliberate change is a one-line diff with a reason in `DECISIONS.md`.

The markdown summary is what a human reads: one line per suite, failing case ids,
and the three metrics that moved most since last run.

---

## 4. Docs as a tested artifact

- A test asserts every path mentioned in `README.md` and `docs/PHASES.md` exists
  (cheap link check), and that `DECISIONS.md` has no duplicate ids.
- `config/config.example.yaml` is loaded and validated by a test; a key in the
  code defaults but missing from the example fails the suite.
- "Docs trim" is the last task of the phase: `SCOPE.md`, `ROUTING.md`,
  `PHASES.md`, `DECISIONS.md` are the design set; every other doc is either a
  phase plan with a status line or deleted. No doc describes a phase that
  already shipped as if it were future.

---

## 5. Cadence

| When | What runs |
| --- | --- |
| every commit (local) | `pytest orchestrator` + `npm run test:fe` + `tsc` |
| before a phase gate | that phase's suite + `--suite all` without `--live` |
| weekly, or before enabling `enforce` | `--suite all --live` on this machine |
| before tagging a release | full live run + the three screenshots from P5 + `cargo build --release` |

Not a CI service. A personal project does not need one; it needs one command
that cannot be quietly skipped.

---

## 6. What counts as done

- One command, documented in the README, that re-checks P0-P6's gates.
- A failing run names the case and the file that reproduces it.
- No suite silently skips: `skipped` is printed and counted, never hidden.
- The persona baseline is recorded and changes only with a `DECISIONS.md` entry.
- Docs and config are checked by tests, not by memory.

---

## 7. Risks

1. **Evals that pass by being weak.** Mitigation: property-based assertions and
   injected-content cases that have already caught real behavior; a suite that
   has never failed once is suspicious and should be reviewed.
2. **Live runs on a machine that also games.** Latency numbers are only
   comparable at similar VRAM pressure; the report records VRAM and foreground
   app so a bad number can be explained rather than chased.
3. **Threshold churn.** Raising a threshold to make a run green is the failure
   mode; every change is a `DECISIONS.md` line with the measured before/after.
4. **Cost creep in live runs.** Each live suite has a token budget and stops on
   it; the report shows the spend.
