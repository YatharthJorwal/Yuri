"""
Persistent memory (Phase 3, v2). SQLite plus sqlite-vec for semantic recall.

- `facts`    -- durable statements about the user, each with a source, a
                confidence, an optional subject and a pinned flag. A fact that
                was replaced or forgotten stays in the table (`superseded_by`).
- `episodes` -- rolling summaries of past conversation (never raw transcripts),
                each with an embedding and a salience that decays over time.
- `profile_card` -- a short always-on summary derived from the live facts.
- `events`   -- an append-only audit trail of every change and every recall.
- `transcript_log` -- the durable per-turn buffer the writer reads from.

Submodules:
- `db.py`            connection, corruption quarantine, locked transactions.
- `migrate.py`       schema v1 -> v2 and the opt-in Luna importer.
- `store.py`         all reads and writes; nothing here deletes a fact.
- `writer.py`        the writer pass: transcript -> facts + episode, atomically.
- `consolidation.py` the distillation prompt and its forgiving parser.
- `contradiction.py` add / duplicate / supersede decisions and their vetoes.
- `recall.py`        budgeted, relevance-floored memory block for a turn.
- `decay.py`         salience and confidence fade, drop-from-search, compression.
- `profile.py`       the derived profile card.
- `forget.py`        explicit "forget that" (soft: reversible).
- `embeddings.py`    Ollama's embedding endpoint.
- `settings.py`      memory config sections with defaults filled in.
- `__main__.py`      `python -m memory` maintenance CLI.

Every piece degrades instead of failing a turn when something is unreachable
(embedder down, model down, DB locked). See each module's docstring.
"""
