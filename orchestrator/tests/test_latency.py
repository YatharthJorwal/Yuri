"""Latency records: marks, derived numbers, JSONL output."""

import json

import latency


def test_marks_are_first_wins():
    t = latency.TurnTimings(trace_id="x")
    t.mark("turn_start")
    first = t.marks["turn_start"]
    t.mark("turn_start")
    assert t.marks["turn_start"] == first


def test_derived_milliseconds():
    t = latency.TurnTimings(trace_id="x")
    t.marks.update({"turn_start": 10.0, "first_token": 10.4, "first_audio": 11.2})
    assert t.ms("turn_start", "first_token") == 400.0
    assert t.ms("turn_start", "first_audio") == 1200.0
    assert t.ms("turn_start", "missing") is None


def test_finish_writes_one_jsonl_row(tmp_path):
    recorder = latency.LatencyRecorder(tmp_path / "latency.jsonl")
    t = recorder.begin(surface="shell")
    t.marks.update({"first_token": t.marks["turn_start"] + 0.4, "first_clause": t.marks["turn_start"] + 0.6,
                    "first_audio": t.marks["turn_start"] + 1.2})
    row = recorder.finish(t, clauses=2)

    assert row["ttft_ms"] == 400.0
    assert row["first_clause_ms"] == 600.0
    assert row["first_audio_ms"] == 1200.0
    assert row["surface"] == "shell"
    assert row["clauses"] == 2
    assert row["turn_ms"] is not None

    lines = (tmp_path / "latency.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["trace_id"] == t.trace_id


def test_missing_marks_are_null(tmp_path):
    recorder = latency.LatencyRecorder(tmp_path / "latency.jsonl")
    row = recorder.finish(recorder.begin())
    assert row["ttft_ms"] is None
    assert row["first_audio_ms"] is None
    assert row["turn_ms"] is not None      # turn_start/turn_end always exist


def test_write_failure_never_raises(tmp_path):
    # A directory where the file should be: open() fails with an OSError.
    target = tmp_path / "latency.jsonl"
    target.mkdir()
    recorder = latency.LatencyRecorder(target)
    row = recorder.finish(recorder.begin())
    assert row["trace_id"]
