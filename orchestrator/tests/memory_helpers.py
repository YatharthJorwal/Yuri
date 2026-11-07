"""Shared helpers for the memory tests: tiny deterministic vectors, a scriptable
chat-model stub (patched at llm.stream_reply, the seam the router's local
provider calls), and an embedder stub patched into every module that embeds."""

from __future__ import annotations

import llm
from memory import contradiction, decay, profile, recall, writer  # noqa: F401 -- patched below


def fake_vector(seed: float, dim: int = 4) -> list[float]:
    return [seed, 1.0 - seed, 0.0, 0.0]


UNRELATED = [0.0, 0.0, 1.0, 0.0]  # orthogonal to every fake_vector


def stub_model(monkeypatch, responder):
    """`responder` is a string, or a callable (messages) -> string so one test
    can answer the distiller, the contradiction check and the card differently.
    Returns the list of message-lists the model was called with."""
    calls: list[list[dict]] = []

    async def fake_stream_reply(messages):
        calls.append(messages)
        text = responder(messages) if callable(responder) else responder
        for ch in text:
            yield ch

    monkeypatch.setattr(llm, "stream_reply", fake_stream_reply)
    return calls


def stub_model_down(monkeypatch):
    async def failing(messages):
        raise llm.LLMUnreachableError("no server")
        yield  # pragma: no cover

    monkeypatch.setattr(llm, "stream_reply", failing)


def stub_embed(monkeypatch, fn=None):
    """Every embedding call returns fake_vector(0.2) unless `fn(text)` says otherwise."""

    async def fake_embed(text: str) -> list[float]:
        return fn(text) if fn else fake_vector(0.2)

    for module in (writer, recall, decay):
        monkeypatch.setattr(module, "embed", fake_embed)


def stub_embed_down(monkeypatch):
    from memory import embeddings

    async def failing(text: str):
        raise embeddings.EmbeddingUnreachableError("Ollama not running")

    for module in (writer, recall, decay):
        monkeypatch.setattr(module, "embed", failing)


def system_prompt(messages: list[dict]) -> str:
    return messages[0]["content"]
