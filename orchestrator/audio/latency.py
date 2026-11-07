"""
Per-turn latency records.

One JSONL row per turn with the marks the conversation actually has: when the
turn started, when the first model token arrived, when the first clause was
handed to the voice, when the first audio was actually sent, and when the turn
ended. `stt` and playback marks are absent for typed turns.

The point is that "she feels slow" becomes a number with a stage attached. The
file is gitignored (`logs/latency.jsonl`) and content-free.
"""

from __future__ import annotations

import json
import pathlib
import time
import uuid
from dataclasses import dataclass, field

from config import CONFIG
from router.discovery import repo_path


@dataclass
class TurnTimings:
    trace_id: str
    marks: dict[str, float] = field(default_factory=dict)
    meta: dict[str, object] = field(default_factory=dict)

    def mark(self, name: str) -> None:
        self.marks.setdefault(name, time.monotonic())

    def ms(self, start: str, end: str) -> float | None:
        if start in self.marks and end in self.marks:
            return round((self.marks[end] - self.marks[start]) * 1000.0, 1)
        return None


class LatencyRecorder:
    def __init__(self, path: str | pathlib.Path | None = None) -> None:
        self.path = pathlib.Path(path) if path else repo_path(CONFIG.latency.log)

    def begin(self, **meta: object) -> TurnTimings:
        timings = TurnTimings(trace_id=uuid.uuid4().hex, meta=dict(meta))
        timings.mark("turn_start")
        return timings

    def finish(self, timings: TurnTimings, **extra: object) -> dict[str, object]:
        timings.mark("turn_end")
        row: dict[str, object] = {
            "trace_id": timings.trace_id,
            "ts": time.time(),
            **timings.meta,
            **extra,
            "ttft_ms": timings.ms("turn_start", "first_token"),
            "first_clause_ms": timings.ms("turn_start", "first_clause"),
            "first_audio_ms": timings.ms("turn_start", "first_audio"),
            "turn_ms": timings.ms("turn_start", "turn_end"),
            "stt_ms": timings.ms("speech_end", "transcript_ready"),
        }
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, separators=(",", ":")) + "\n")
        except OSError as exc:  # never let telemetry break a conversation
            print(f"[yuri] latency: could not write {self.path}: {exc}", flush=True)
        return row
