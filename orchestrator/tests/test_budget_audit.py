"""Spend ledger and audit writer."""

import json
import time

import pytest

from router.audit import AuditWriter
from router.budget import SpendLedger, actual_cost, estimate_cost
from router.types import RouteDecision


def test_estimate_cost_math():
    # 1000 prompt at $1/1M + 500 completion at $2/1M
    assert estimate_cost(1000, 500, 1.0, 2.0) == pytest.approx(0.002)


def test_actual_cost_missing_usage_is_zero():
    assert actual_cost(None, None, 1.0, 2.0) == 0.0
    assert actual_cost(1000, None, 1.0, 2.0) == pytest.approx(0.001)


def test_ledger_totals_windows(tmp_path):
    ledger = SpendLedger(tmp_path / "spend.db")
    now = time.time()
    ledger.record("airouter", "m", 0.01, ts=now - 10)
    ledger.record("airouter", "m", 0.01, ts=now - 2 * 3600)      # in day, not hour
    ledger.record("airouter", "m", 0.01, ts=now - 3 * 86400)     # in month only

    hour, day, month = ledger.totals(now)
    assert hour == pytest.approx(0.01)
    assert day == pytest.approx(0.02)
    assert month == pytest.approx(0.03)


def test_ledger_survives_reopen(tmp_path):
    path = tmp_path / "spend.db"
    SpendLedger(path).record("airouter", "m", 0.05)
    reopened = SpendLedger(path)
    assert reopened.totals()[2] == pytest.approx(0.05)


def test_ledger_records_estimated_flag(tmp_path):
    path = tmp_path / "spend.db"
    SpendLedger(path).record("airouter", "m", 0.01, estimated=True)
    import sqlite3

    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT estimated FROM spend").fetchone()[0] == 1


def _decision():
    return RouteDecision(
        role="mind",
        reason="COMPLEXITY_THRESHOLD",
        candidates=("airouter:model", "local_ollama:qwen3.5:9b"),
        chosen="airouter:model",
        shadow=True,
        est_prompt_tokens=120,
        est_cost_usd=0.0005,
    )


def test_audit_writes_start_and_finish(tmp_path):
    path = tmp_path / "audit.jsonl"
    writer = AuditWriter(path)
    trace = writer.start(decision=_decision(), privacy_class="internal", trust="trusted", redactions={"secret": 2})
    writer.finish(trace, role="mind", latency_ms=12.5, prompt_tokens=100, completion_tokens=20, cost_usd=0.001)

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [row["event"] for row in rows] == ["model.call.started", "model.call.finished"]
    assert rows[0]["shadow"] is True
    assert rows[0]["redactions"] == {"secret": 2}
    assert rows[0]["chosen"] == "airouter:model"
    assert rows[0]["candidates"][1] == "local_ollama:qwen3.5:9b"
    assert rows[1]["trace_id"] == trace
    assert rows[1]["cost_usd"] == pytest.approx(0.001)


def test_audit_never_contains_message_content(tmp_path):
    path = tmp_path / "audit.jsonl"
    writer = AuditWriter(path)
    writer.start(decision=_decision(), privacy_class="internal", trust="trusted")
    text = path.read_text(encoding="utf-8")
    assert "content" not in text
    assert "sk-" not in text


def test_audit_can_be_disabled(tmp_path):
    path = tmp_path / "audit.jsonl"
    writer = AuditWriter(path, enabled=False)
    trace = writer.start(decision=_decision(), privacy_class="internal", trust="trusted")
    writer.finish(trace, role="mind", latency_ms=1.0)
    assert not path.exists()
