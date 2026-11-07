"""
Decides what a newly extracted fact does to the facts already stored:
add it, count it as a repeat of one that exists, or supersede one it contradicts.

Two passes, cheapest first:

1. Deterministic. Normalised text equal -> duplicate. Same words once negation
   is stripped but opposite polarity ("likes pizza" / "doesn't like pizza") ->
   supersede. No model call, so the common cases cost nothing and cannot
   hallucinate.
2. Model. Only for a new fact that has a *close* neighbour (similarity floor)
   and was not settled above. It must answer with the id of one of the
   candidates it was shown; anything else is treated as "add".

Whatever the pass decided, three rules can still veto a supersede and turn it
into a plain add, logged so the owner can see it was proposed:
  - a pinned fact is only replaced by an explicit correction;
  - a lower-confidence fact never replaces a higher-confidence one, unless the
    new fact is an explicit correction;
  - a model decision naming an id that was not a candidate is ignored.
Nothing is ever deleted here; the writer applies a supersede by pointing the old
row at the new one, which `store.restore_fact` can undo.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

import router

ADD = "add"
DUPLICATE = "duplicate"
SUPERSEDE = "supersede"

_NEGATION_WORDS = {"not", "no", "never", "longer", "anymore", "cannot", "cant", "wont", "dont", "doesnt", "didnt"}
_FILLER_WORDS = {"do", "does", "did", "now", "actually", "still", "really", "currently", "the", "a", "an"}


@dataclass
class Decision:
    action: str = ADD
    target_id: int | None = None
    via: str = "none"  # deterministic | model | none
    blocked: str | None = None  # pinned | lower_confidence, when a supersede was vetoed
    proposed_target_id: int | None = None
    notes: list[str] = field(default_factory=list)


def _tokens(text: str) -> list[str]:
    text = text.lower().replace("\u2019", "'")
    text = re.sub(r"n't\b", " not", text)
    text = text.replace("can't", "can not").replace("won't", "will not")
    return re.findall(r"[a-z0-9]+", text)


def _stem(word: str) -> str:
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def normalize(text: str) -> str:
    return " ".join(_stem(w) for w in _tokens(text) if w not in _FILLER_WORDS)


def _core_and_polarity(text: str) -> tuple[tuple[str, ...], bool]:
    toks = _tokens(text)
    negated = any(t in _NEGATION_WORDS for t in toks)
    core = tuple(sorted(_stem(t) for t in toks if t not in _NEGATION_WORDS and t not in _FILLER_WORDS))
    return core, negated


def keyword_overlap(a: str, b: str) -> float:
    """Jaccard overlap of content words. The candidate finder when no embedding
    is available."""
    skip = _FILLER_WORDS | _NEGATION_WORDS
    wa = {_stem(t) for t in _tokens(a) if t not in skip}
    wb = {_stem(t) for t in _tokens(b) if t not in skip}
    return len(wa & wb) / len(wa | wb) if wa and wb else 0.0


def deterministic(new_text: str, candidates: list[dict[str, Any]]) -> Decision | None:
    new_norm = normalize(new_text)
    new_core, new_neg = _core_and_polarity(new_text)
    for cand in candidates:
        if normalize(cand["content"]) == new_norm:
            return Decision(DUPLICATE, cand["id"], "deterministic")
    for cand in candidates:
        core, neg = _core_and_polarity(cand["content"])
        if core == new_core and neg != new_neg:
            return Decision(SUPERSEDE, cand["id"], "deterministic")
    return None


_SYSTEM_PROMPT = (
    "You compare ONE new fact about a person against a short numbered list of facts already on file. "
    "Reply with a single JSON object and nothing else: "
    '{"action": "add" | "duplicate" | "supersede", "target_id": <id from the list or null>}. '
    '"duplicate" = the new fact says the same thing as a listed one. '
    '"supersede" = the new fact replaces or contradicts a listed one (it changed, ended, or was wrong). '
    '"add" = it is a different fact and both can be true. When unsure, answer "add". '
    "target_id must be one of the listed ids, or null for add."
)


def _parse_model_decision(raw: str, valid_ids: set[int]) -> Decision | None:
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    action = data.get("action")
    target = data.get("target_id")
    if action == ADD:
        return Decision(ADD, None, "model")
    if action in (DUPLICATE, SUPERSEDE) and isinstance(target, int) and target in valid_ids:
        return Decision(action, target, "model")
    return None  # unknown action or an id the model made up


async def _ask_model(new_text: str, candidates: list[dict[str, Any]]) -> Decision | None:
    listing = "\n".join(f"{c['id']}: {c['content']}" for c in candidates)
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": f"Facts on file:\n{listing}\n\nNew fact: {new_text}"},
    ]
    try:
        raw = "".join([d async for d in router.stream_text("memory_writer", messages)])
    except Exception:  # noqa: BLE001 -- any model failure just means "treat as add"
        return None
    return _parse_model_decision(raw, {c["id"] for c in candidates})


def _apply_vetoes(decision: Decision, candidates: list[dict[str, Any]], new_confidence: float, explicit: bool) -> Decision:
    """An explicit correction overrides both vetoes. Otherwise a pinned target
    or a higher-confidence target turns the supersede into a plain add."""
    if decision.action != SUPERSEDE:
        return decision
    target = next((c for c in candidates if c["id"] == decision.target_id), None)
    if target is None:
        return Decision(ADD, via=decision.via)
    if explicit:
        return decision
    if target.get("pinned"):
        return Decision(ADD, None, decision.via, "pinned", target["id"])
    if new_confidence < target.get("confidence", 0.0):
        return Decision(ADD, None, decision.via, "lower_confidence", target["id"])
    return decision


async def decide(
    new_text: str,
    candidates: list[dict[str, Any]],
    *,
    new_confidence: float,
    explicit_correction: bool = False,
    model_similarity_floor: float = 0.55,
) -> Decision:
    """`candidates` are live facts as dicts (id, content, confidence, pinned,
    similarity). `similarity` is cosine from the vector search, or keyword
    overlap when no embedding was available."""
    if not candidates:
        return Decision(ADD)
    decision = deterministic(new_text, candidates)
    if decision is None:
        close = [c for c in candidates if c.get("similarity", 0.0) >= model_similarity_floor]
        decision = (await _ask_model(new_text, close)) if close else None
    if decision is None:
        return Decision(ADD)
    return _apply_vetoes(decision, candidates, new_confidence, explicit_correction)
