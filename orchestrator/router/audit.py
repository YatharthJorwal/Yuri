"""Audit trail for model calls.

One JSONL row when a call starts (decision, candidates, redaction counts) and a
short second row when it ends (usage, real cost, latency, error). A crash
mid-call leaves a `started` row with no finish, which is a visible fact rather
than silence.

Content is never logged by default: counts, hashes and token numbers only.
"""

from __future__ import annotations

import json
import pathlib
import threading
import time
import uuid
from typing import Any


class AuditWriter:
    def __init__(self, path: str | pathlib.Path, *, enabled: bool = True) -> None:
        self.path = pathlib.Path(path)
        self.enabled = enabled
        self._lock = threading.Lock()
        if self.enabled:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def _write(self, payload: dict[str, Any]) -> None:
        if not self.enabled:
            return
        line = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as handle:
                handle.write(line + "\n")

    def start(
        self,
        *,
        decision: Any,
        privacy_class: str,
        trust: str,
        redactions: dict[str, int] | None = None,
        mode: str = "shadow",
        job_id: str | None = None,
    ) -> str:
        trace_id = uuid.uuid4().hex
        self._write(
            {
                "ts": time.time(),
                "event": "model.call.started",
                "trace_id": trace_id,
                "role": decision.role,
                "mode": mode,
                "reason": decision.reason,
                "candidates": list(decision.candidates),
                "chosen": decision.chosen,
                "shadow": decision.shadow,
                "privacy_class": privacy_class,
                "trust": trust,
                "redactions": redactions or {},
                "est_prompt_tokens": decision.est_prompt_tokens,
                "est_cost_usd": round(decision.est_cost_usd, 8),
                "job_id": job_id,
            }
        )
        return trace_id

    def finish(
        self,
        trace_id: str,
        *,
        role: str,
        latency_ms: float,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        cost_usd: float | None = None,
        estimated: bool = False,
        fallback_used: bool = False,
        error: str | None = None,
    ) -> None:
        self._write(
            {
                "ts": time.time(),
                "event": "model.call.finished",
                "trace_id": trace_id,
                "role": role,
                "latency_ms": round(latency_ms, 1),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "cost_usd": None if cost_usd is None else round(cost_usd, 8),
                "estimated": estimated,
                "fallback_used": fallback_used,
                "error": error,
            }
        )
