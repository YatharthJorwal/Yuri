"""
Embedding client for semantic episode recall. Talks to Ollama's own
`/api/embed` endpoint -- deliberately not the older `/api/embeddings`
(singular), which current Ollama docs mark legacy and which is a common
source of silent 404s on newer installs. Confirmed the current shape
before writing this (same standard docs/DECISIONS.md already held the
`think` flag investigation to -- don't assume, check):

    POST {base_url}/api/embed
    {"model": "nomic-embed-text", "input": ["text one", "text two"]}
    -> {"embeddings": [[0.1, 0.2, ...], [0.3, ...]]}

`input` takes either a single string or a list (for batching); this
module always sends a one-element list and reads back `embeddings[0]`,
since Luna only ever embeds one piece of text at a time (a user message,
or one episode summary at consolidation time) -- no batching need yet.

Model: `nomic-embed-text` (768-dim, Apache-2.0, ~274MB) is the default in
config.yaml -- small enough to sit in VRAM alongside qwen3.5:9b with room
to spare (see docs/MODELS.md's VRAM budget), and since Ollama's already
running as the LLM server, this adds zero new processes -- `ollama pull
nomic-embed-text` is the only new setup step. Swappable via config same
as the LLM model; `memory.embedding.dimension` in config.yaml has to be
updated to match if the model's swapped for one with a different output
width (see db.py's dimension-mismatch check).

Reuses `llm.base_url` by default (`memory.embedding.base_url` in
config.yaml) since it's the same Ollama instance -- overridable if
embeddings ever move to a separate server.
"""

from __future__ import annotations

import httpx

from config import CONFIG


class EmbeddingUnreachableError(Exception):
    """The configured embedding server couldn't be reached, errored out, or
    returned something that doesn't parse as expected -- most likely
    Ollama isn't running, or the embedding model hasn't been pulled yet
    (`ollama pull nomic-embed-text`). Kept distinct so recall.py/
    consolidation.py can catch it and degrade gracefully (skip semantic
    recall for this turn / skip storing this episode's embedding) instead
    of the turn or session-end consolidation failing outright -- same
    pattern as llm.py's LLMUnreachableError."""


async def embed(text: str) -> list[float]:
    """Returns the embedding vector for `text`. Raises
    EmbeddingUnreachableError on any failure -- unreachable server, non-200
    response, or a response shape that doesn't match what's documented
    above."""
    cfg = CONFIG.memory.embedding
    url = f"{cfg.base_url.rstrip('/')}/api/embed"
    payload = {"model": cfg.model, "input": [text]}

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=5.0)) as client:
            response = await client.post(url, json=payload)
            if response.status_code >= 400:
                raise EmbeddingUnreachableError(
                    f"{response.status_code} from {url}: {response.text[:300]!r}"
                )
            data = response.json()
    except httpx.RequestError as exc:
        raise EmbeddingUnreachableError(f"couldn't reach {url}: {exc}") from exc

    try:
        vector = data["embeddings"][0]
    except (KeyError, IndexError, TypeError) as exc:
        raise EmbeddingUnreachableError(
            f"unexpected response shape from {url}: {data!r}"
        ) from exc

    if len(vector) != cfg.dimension:
        raise EmbeddingUnreachableError(
            f"{cfg.model} returned a {len(vector)}-dim vector, but "
            f"config.yaml's memory.embedding.dimension is {cfg.dimension} -- "
            "update the config to match (see db.py's dimension-mismatch "
            "check for what happens if this drifts silently instead)."
        )
    return vector
