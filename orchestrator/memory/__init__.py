"""
Phase 3 -- persistent memory. SQLite (+ sqlite-vec for semantic recall),
per docs/ARCHITECTURE.md's memory design:

- `facts`   -- durable statements about the user (preferences, current
              project, stack, games played). Small in number, always
              loaded in full -- no semantic search needed for a handful
              of rows.
- `episodes` -- rolling **summaries** of past sessions (never raw
              transcripts -- distill, don't transcribe), each with an
              embedding so a new turn can semantically recall "that bug
              we fixed last week" instead of only ever seeing the most
              recent N.

Submodules:
- `db.py`           -- connection + schema (loads the sqlite-vec extension).
- `embeddings.py`   -- talks to Ollama's embedding endpoint.
- `store.py`        -- CRUD + the actual vec0 nearest-neighbor query.
- `recall.py`       -- turns a user message into a memory block for the
                        system prompt (facts + top-K relevant episodes).
- `forget.py`       -- explicit "forget that" handling: a user directly
                        asking to remove something already stored, not
                        automatic contradiction detection (see its own
                        docstring for that distinction).
- `consolidation.py` -- runs at session end: LLM distills the session's
                        history into candidate facts + an episode summary.

Every piece degrades gracefully if something's unreachable (Ollama down
for embeddings, DB file locked, etc.) -- memory is a nice-to-have context
booster, not something a turn should ever hang or crash on. See each
submodule's own docstring for its specific failure handling.
"""
