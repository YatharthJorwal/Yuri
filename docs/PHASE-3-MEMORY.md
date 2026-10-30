# Phase 3 — Memory v2

Luna already has a working memory: facts, episode summaries with embeddings,
recall, consolidation at session end, and a forget path. P3 keeps that store and
fixes the four things that make it feel unreliable over weeks: facts have no
provenance, contradictions overwrite nothing, nothing is written until the
session ends, and nothing decays.

Everything stays local (D-015). The only model that touches memory content is
the local `memory_writer` role.

---

## 0. What exists today (do not rebuild)

| Piece | Where | State |
| --- | --- | --- |
| SQLite + sqlite-vec, WAL | `memory/db.py` | works, dimension check, 768-dim |
| Facts (id, content) | `memory/store.py` | flat strings, no source/confidence |
| Episodes (summary, embedding) | `memory/store.py` | vector search, nearest-first |
| Transcript log | `memory/store.py` | survives restart, clearable |
| Recall per turn | `memory/recall.py` | facts + top-k episodes, degrades safely |
| Consolidation | `memory/consolidation.py` | session end only, JSON parse is defensive |
| Forget | `memory/forget.py` | LLM picks fact indices, trigger-word gated |

The tests for all of that already pass. P3 extends; it does not replace.

---

## 1. Schema v2

Migration is additive, versioned in a `meta(key, value)` row, and idempotent.
An existing Luna `memory.db` is read only if the user runs the importer; Yuri
starts fresh otherwise (the repository is new, the old DB belongs to the old app).

```sql
meta(key TEXT PRIMARY KEY, value TEXT)          -- schema_version, last_writer_at

facts(
  id INTEGER PRIMARY KEY,
  content TEXT NOT NULL,
  subject TEXT,                  -- normalized "user" / "machine" / free tag
  source TEXT NOT NULL,          -- "user_stated" | "writer_pass" | "correction" | "import"
  confidence REAL NOT NULL,      -- 0..1; user-stated starts at 1.0
  pinned INTEGER NOT NULL DEFAULT 0,
  created_at REAL, last_seen_at REAL,
  superseded_by INTEGER          -- points at the fact that replaced this one
)
facts_vec USING vec0(embedding float[768])      -- one row per fact id

episodes(
  id INTEGER PRIMARY KEY,
  summary TEXT NOT NULL,
  salience REAL NOT NULL,        -- 0..1, decayed over time
  created_at REAL, last_used_at REAL,
  session_id TEXT, turns INTEGER, tokens INTEGER
)
episode_vectors USING vec0(embedding float[768])

profile_card(id INTEGER PRIMARY KEY CHECK(id=1), text TEXT, updated_at REAL)

events(                          -- append-only, for the eval suite and debugging
  id INTEGER PRIMARY KEY, ts REAL, kind TEXT, actor TEXT, detail TEXT
)
```

Rules: `superseded_by` rows are never returned by recall but are never deleted
either, so "what did I used to think" stays answerable. `pinned` facts never
decay and never auto-supersede (only an explicit correction does).

---

## 2. Writer pass

Today consolidation runs at session end. A hard kill loses the session. P3 adds
a periodic writer that is idempotent and cheap.

- **Scratch buffer**: every completed turn appends to the existing transcript log
  (already durable). The writer tracks `last_writer_at` and processes only turns
  after that point.
- **Trigger**: every `memory.writer.interval_s` (600) while the app runs, on
  session end, and on a clean shutdown. A kill -9 loses at most one interval.
- **Idempotence**: writer writes facts/episodes, then advances `last_writer_at`
  in the same transaction. A crash mid-pass reprocesses turns that produced
  facts already deduped by the contradiction rule (§3), so a double run is safe.
- **Model**: local `memory_writer` role, JSON out (`format` schema), one call
  per pass, capped input (`memory.writer.max_tokens`, transcript window).
- **Failure**: if the model is unreachable, the checkpoint does not advance and
  the next pass retries; nothing is lost, nothing is duplicated.

---

## 3. Contradiction and correction

The user says "actually I like X now" and the old fact must stop being true
without anyone saying "forget".

- Candidate retrieval: embed the new fact, take the 5 nearest live facts.
- Deterministic first pass: normalized-equality and negation-flip checks
  ("don't like" vs "like") settle the obvious cases without a model call.
- Model second pass: the writer is shown the candidates and asked for
  `{action: "add"|"supersede"|"duplicate", target_id}`; constrained JSON.
- Apply: `add` inserts; `duplicate` bumps `last_seen_at` and confidence;
  `supersede` sets `superseded_by` on the old row and inserts the new one with
  `source: "correction"`, `confidence: 1.0`.
- A superseded fact is excluded from recall, but its text remains for the eval
  suite and for "you used to think that" style callbacks.
- Pinned facts are never auto-superseded; the model may only propose, and the
  proposal is logged.

Explicit forget keeps Luna's trigger-word gate, but operates on the same
supersede column (`superseded_by = -1` meaning "forgotten by the user") instead
of deleting rows, so a mistake is recoverable from the DB.

---

## 4. Retrieval

Per turn, budgeted, never wholesale:

| Consumer | Budget | Contents |
| --- | --- | --- |
| `reflex` turn | <= 300 tokens | profile card (always) + up to 3 relevant facts + 2 episodes |
| `mind` job | <= 1200 tokens | profile card + up to 8 facts + 5 episodes |

- Relevance: facts via `facts_vec` nearest-neighbour, filtered by a similarity
  floor and by subject; episodes by `similarity * salience * recency`.
- If the embedding model is down, degrade to keyword overlap over facts and skip
  episodes (Luna already degrades; keep that behavior and log which path ran).
- Every recall writes a `recall` event with ids and token count so P7 can check
  the budget from real traffic.

---

## 5. Decay and compression

- `salience` starts at 0.5 for a normal episode, 0.8 if the session had a
  correction, 1.0 if the user explicitly asked to remember.
- Decay: `salience *= 0.5 ** (days / half_life_days)` (30), applied lazily on
  read and swept in the writer pass.
- Below 0.15: an episode is a compression candidate. The writer summarizes
  clusters of low-salience episodes (10 at a time) into one, keeping the date
  range, then marks the originals `superseded_by = compression_id`.
- Below 0.05 and unused for 90 days: dropped from search (row kept in `events`
  for audit). Facts decay `confidence`, never disappear, unless superseded.

---

## 6. Profile card

A small always-in-context block (<= 800 chars) the writer regenerates when facts
change materially: name, what they are building, stack, preferences that alter
answers, current projects. It is derived, stored in `profile_card`, and rebuilt
from live facts, so it can never drift from the store.

---

## 7. Failure and corruption

- DB locked/corrupt on open: rename to `memory.db.corrupt-<ts>`, create a fresh
  DB, log loudly, keep the old file for inspection. Never start with a broken
  store silently (sqlite-vec errors are otherwise cryptic).
- Disk full: writer aborts, checkpoint unchanged, one warning; recall still works.
- Embedding dimension mismatch: already detected by `db.py`; P3 adds a
  `PRAGMA`-based check before any write and refuses with a clear message.

---

## 8. Migration

`memory/migrate.py`: `schema_version` 1 -> 2, run once at startup inside a
transaction. `--import-luna <path>` is a separate opt-in command that reads
Luna's `memory.db` (facts as `source: "import"`, confidence 0.6, episodes with
fresh ids and salience 0.4). Not run automatically.

---

## 9. Tests

- `test_memory_schema.py` — fresh create, migrate from v1 fixture, idempotent.
- `test_writer.py` — interval pass, session-end pass, kill -9 subprocess test
  (spawn a writer loop, SIGKILL, restart, assert loss <= one interval), double-run
  dedupe, model-down retry.
- `test_contradiction.py` — 20 seeded pairs (correction, negation, duplicate,
  unrelated, pinned); deterministic and model paths.
- `test_recall_budget.py` — token caps for reflex and mind, relevance floor,
  embedding-down degradation.
- `test_decay.py` — half-life math, compression, drop threshold, pinned immunity.
- `test_forget.py` — existing suite plus superseded-not-deleted assertions.
- `test_corruption.py` — truncated DB is quarantined and recreated.

Gate: `PHASES.md` P3 gate, plus every supersede/forget reversible from the DB.

---

## 10. Risks

1. **Writer cost on CPU.** One 4B call every 10 minutes is ~1-2 s of CPU; fine,
   but the pass must be cancellable if a voice turn starts. Mitigation: writer
   runs in a task that yields; no model call while `speaking`.
2. **False contradictions.** A wrong supersede silently loses a true fact.
   Mitigation: confidence floor, pinned immunity, `events` audit, and the eval
   corpus in P7.
3. **Context creep.** Recall is the easiest place to blur the token budget.
   Mitigation: hard caps in code, measured per turn, asserted in tests.
4. **Two writers.** Timer and session-end can overlap. Mitigation: one asyncio
   lock, checkpoint compare inside the transaction.
