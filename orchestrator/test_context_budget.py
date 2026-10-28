"""
Tests for context_budget.py and for the config defaults that keep an older
config.yaml loading. The scenario they pin is the real one: replies cut off
after 24-132 characters (done_reason='length') once ambient comments had
filled the context window.
"""

from __future__ import annotations

import dataclasses
import pathlib

import pytest
import yaml

import config
import context_budget as cb

SYSTEM = {"role": "system", "content": "p" * 9000}  # ~ the persona prompt
TOOLS = [{"type": "function", "function": {"name": "t", "description": "d" * 2600}}]


def _hist(n, size=300):
    msgs = []
    for i in range(n):
        msgs.append({"role": "user" if i % 2 == 0 else "assistant", "content": f"{i}:" + "x" * size})
    return msgs


def test_fit_messages_noop_when_it_already_fits():
    messages = [SYSTEM] + _hist(4)
    fitted, dropped, _, _ = cb.fit_messages(messages, num_ctx=8192, reply_reserve=512, tools=TOOLS)
    assert dropped == 0 and fitted == messages


def test_fit_messages_drops_oldest_history_first_and_keeps_system_and_last():
    messages = [SYSTEM] + _hist(60, size=400)
    fitted, dropped, est, budget = cb.fit_messages(messages, num_ctx=4096, reply_reserve=512, tools=TOOLS)
    assert dropped > 0
    assert fitted[0] is SYSTEM  # persona prompt never dropped
    assert fitted[-1] is messages[-1]  # the message being answered never dropped
    # what survives is a contiguous newest-first suffix of the history
    survivors = [m for m in fitted if m is not SYSTEM]
    assert survivors == messages[-len(survivors):]


def test_fit_messages_keeps_a_mid_list_system_block():
    # _run_turn splices the memory/hint block just before the last message.
    hint = {"role": "system", "content": "recall " * 50}
    messages = [SYSTEM] + _hist(60, size=400)[:-1] + [hint, {"role": "user", "content": "now?"}]
    fitted, dropped, _, _ = cb.fit_messages(messages, num_ctx=4096, reply_reserve=512, tools=TOOLS)
    assert dropped > 0 and hint in fitted


def test_fit_messages_result_fits_the_budget_when_history_alone_is_the_problem():
    messages = [SYSTEM] + _hist(80, size=500)
    fitted, _, est, budget = cb.fit_messages(messages, num_ctx=8192, reply_reserve=512, tools=TOOLS)
    assert est <= budget
    assert sum(cb.message_tokens(m) for m in fitted) <= budget


def test_fit_messages_counts_tool_round_messages_against_the_budget():
    messages = [SYSTEM] + _hist(40, size=400)
    extra = [{"role": "tool", "content": "t" * 8000}]
    _, dropped_without, _, _ = cb.fit_messages(messages, num_ctx=8192, reply_reserve=512, tools=TOOLS)
    _, dropped_with, _, _ = cb.fit_messages(messages, num_ctx=8192, reply_reserve=512, tools=TOOLS, extra=extra)
    assert dropped_with > dropped_without


def test_fit_messages_num_ctx_none_is_a_noop():
    messages = [SYSTEM] + _hist(200, size=500)
    fitted, dropped, _, budget = cb.fit_messages(messages, num_ctx=None, reply_reserve=512, tools=TOOLS)
    assert dropped == 0 and len(fitted) == len(messages) and budget == 0


def test_fit_messages_does_not_mutate_the_input():
    messages = [SYSTEM] + _hist(60, size=400)
    before = list(messages)
    cb.fit_messages(messages, num_ctx=4096, reply_reserve=512, tools=TOOLS)
    assert messages == before


def test_fit_messages_oversized_fixed_part_returns_everything_it_can_not_drop():
    giant = {"role": "system", "content": "p" * 40000}
    fitted, dropped, est, budget = cb.fit_messages(
        [giant, {"role": "user", "content": "hi"}], num_ctx=4096, reply_reserve=512, tools=TOOLS
    )
    assert dropped == 0 and est > budget  # caller logs the warning


# --- cap_unprompted ---------------------------------------------------------


def _with_unprompted(n_user_turns=2, n_unprompted=6):
    history = [SYSTEM]
    flags: list[bool] = []
    unprompted: list[dict] = []
    for i in range(n_user_turns):
        history.append({"role": "user", "content": f"u{i}"})
        flags.append(False)
        history.append({"role": "assistant", "content": f"a{i}"})
        flags.append(False)
    for i in range(n_unprompted):
        m = {"role": "assistant", "content": f"ambient{i}"}
        history.append(m)
        flags.append(i % 2 == 0)  # distinguishable so lockstep is checkable
        unprompted.append(m)
    return history, flags, unprompted


def test_cap_unprompted_keeps_only_the_newest():
    history, flags, unprompted = _with_unprompted()
    removed = cb.cap_unprompted(history, flags, unprompted, 3)
    assert removed == 3
    assert [m["content"] for m in history if m["content"].startswith("ambient")] == [
        "ambient3", "ambient4", "ambient5",
    ]
    # real conversation untouched
    assert [m["content"] for m in history[1:5]] == ["u0", "a0", "u1", "a1"]


def test_cap_unprompted_keeps_temp_flags_in_lockstep():
    history, flags, unprompted = _with_unprompted()
    cb.cap_unprompted(history, flags, unprompted, 3)
    assert len(flags) == len(history) - 1
    # ambient3/4/5 had flags False/True/False (i % 2 == 0 -> True for 0,2,4)
    assert flags[-3:] == [False, True, False]


def test_cap_unprompted_never_touches_the_system_prompt_or_real_turns():
    history, flags, unprompted = _with_unprompted(n_unprompted=10)
    cb.cap_unprompted(history, flags, unprompted, 0)
    assert history[0] is SYSTEM
    assert len(history) == 1 + 4


def test_cap_unprompted_forgets_messages_trim_already_dropped():
    history, flags, unprompted = _with_unprompted()
    # simulate _trim_history having dropped the two oldest ambient ones
    del history[5:7]
    del flags[4:6]
    removed = cb.cap_unprompted(history, flags, unprompted, 3)
    assert removed == 1  # 4 survivors -> 3
    assert len(unprompted) == 3


def test_cap_unprompted_below_the_cap_is_a_noop():
    history, flags, unprompted = _with_unprompted(n_unprompted=2)
    assert cb.cap_unprompted(history, flags, unprompted, 3) == 0
    assert len(history) == 1 + 4 + 2


def test_cap_unprompted_works_without_temp_flags():
    history, _, unprompted = _with_unprompted()
    assert cb.cap_unprompted(history, None, unprompted, 1) == 5


# --- a realistic long session (the actual bug) ------------------------------


def test_long_ambient_session_never_exceeds_the_window():
    """25 unprompted comments of ~350 chars piled onto a normal chat, as in
    the real session, must still fit an 8192 window with room for a reply --
    both with the cap and, as a backstop, with only fit_messages."""
    history = [SYSTEM] + _hist(12, size=250)
    unprompted: list[dict] = []
    for i in range(25):
        m = {"role": "assistant", "content": "c" * 350}
        history.append(m)
        unprompted.append(m)
    # backstop alone (cap disabled):
    fitted, _, est, budget = cb.fit_messages(history, num_ctx=8192, reply_reserve=512, tools=TOOLS)
    assert est <= budget
    # cap alone leaves a small history:
    cb.cap_unprompted(history, None, unprompted, 3)
    assert sum(cb.message_tokens(m) for m in history) < 4500


# --- config defaults ---------------------------------------------------------


def test_older_config_without_the_new_keys_still_loads(tmp_path: pathlib.Path):
    raw = yaml.safe_load(config.EXAMPLE_PATH.read_text(encoding="utf-8"))
    raw["llm"].pop("num_ctx", None)
    raw["session"].pop("max_unprompted_in_history", None)
    for key in ("min_comment_gap_seconds", "same_screen_similarity", "pause_during_task"):
        raw["ocr_watch"].pop(key, None)
    path = tmp_path / "old.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    cfg = config.load_config(path)
    assert cfg.llm.num_ctx == 8192  # the fix applies even without editing config.yaml
    assert cfg.session.max_unprompted_in_history == 3
    assert cfg.ocr_watch.min_comment_gap_seconds == 120
    assert cfg.ocr_watch.same_screen_similarity == 0.6
    assert cfg.ocr_watch.pause_during_task is True


def test_example_config_documents_every_new_key():
    text = config.EXAMPLE_PATH.read_text(encoding="utf-8")
    for key in ("num_ctx", "max_unprompted_in_history", "min_comment_gap_seconds",
                "same_screen_similarity", "pause_during_task"):
        assert key in text


def test_older_config_without_vision_settings_still_loads(tmp_path: pathlib.Path):
    raw = yaml.safe_load(config.EXAMPLE_PATH.read_text(encoding="utf-8"))
    raw["llm"].pop("vision_model", None)
    raw.pop("vision", None)
    path = tmp_path / "old.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    cfg = config.load_config(path)
    assert cfg.llm.vision_model is None
    assert cfg.vision.ambient_max_long_edge == 1024
    assert cfg.vision.look_max_long_edge == 1600
    assert cfg.vision.jpeg_quality == 85


def test_example_config_documents_the_vision_keys():
    text = config.EXAMPLE_PATH.read_text(encoding="utf-8")
    for key in ("vision_model", "ambient_max_long_edge", "look_max_long_edge", "jpeg_quality"):
        assert key in text


# --- self-calibrating estimate ------------------------------------------------
# The raw chars/3.5 estimate over-counted this model (real replies measured
# ~4.1-4.4 chars/token), so the fit dropped history while the real prompt was
# only ~66% of the window. Ollama's real prompt_eval_count corrects it.


@pytest.fixture
def fresh_calibration():
    cb.reset_calibration()
    yield
    cb.reset_calibration()


def test_calibration_starts_neutral(fresh_calibration):
    assert cb.calibration_ratio() == 1.0


def test_observe_pulls_the_ratio_toward_the_real_count(fresh_calibration):
    cb.observe(6500, 5431)  # a real pair from the user's log: est ~6.5k, actual 5431
    assert 0.8 < cb.calibration_ratio() < 0.86
    for _ in range(10):
        cb.observe(6500, 5431)
    assert abs(cb.calibration_ratio() - 5431 / 6500) < 0.01


def test_observe_is_clamped_both_ways(fresh_calibration):
    cb.observe(10_000, 1_000)  # absurdly low actual
    assert cb.calibration_ratio() == cb.MIN_RATIO
    cb.reset_calibration()
    cb.observe(1_000, 50_000)  # absurdly high actual (e.g. a big image in the prompt)
    assert cb.calibration_ratio() == cb.MAX_RATIO


def test_observe_ignores_missing_or_tiny_samples(fresh_calibration):
    cb.observe(5000, None)
    cb.observe(5000, 0)
    cb.observe(50, 40)
    assert cb.calibration_ratio() == 1.0


def test_calibration_keeps_more_history_but_still_fits(fresh_calibration):
    messages = [SYSTEM] + _hist(60, size=400)
    _, dropped_raw, _, _ = cb.fit_messages(messages, num_ctx=8192, reply_reserve=512, tools=TOOLS)
    cb.observe(6500, 5431)
    fitted, dropped_cal, est, budget = cb.fit_messages(messages, num_ctx=8192, reply_reserve=512, tools=TOOLS)
    assert dropped_cal < dropped_raw  # the over-count no longer evicts history needlessly
    assert est <= budget  # and the calibrated estimate still fits the window


def test_estimate_prompt_is_raw_and_ignores_calibration(fresh_calibration):
    messages = [SYSTEM] + _hist(4)
    before = cb.estimate_prompt(messages, TOOLS)
    cb.observe(6500, 5431)
    assert cb.estimate_prompt(messages, TOOLS) == before  # observe() compares like with like


def test_older_config_without_warm_up_still_loads_and_defaults_on(tmp_path: pathlib.Path):
    raw = yaml.safe_load(config.EXAMPLE_PATH.read_text(encoding="utf-8"))
    raw["llm"].pop("warm_up_on_start", None)
    path = tmp_path / "old.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    assert config.load_config(path).llm.warm_up_on_start is True


def test_older_config_without_sampling_or_task_tool_keys_still_loads(tmp_path: pathlib.Path):
    raw = yaml.safe_load(config.EXAMPLE_PATH.read_text(encoding="utf-8"))
    for key in ("top_p", "top_k", "presence_penalty"):
        raw["llm"].pop(key, None)
    raw["task_guide"].pop("model_can_start_tasks", None)
    path = tmp_path / "old.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    cfg = config.load_config(path)
    # defaults equal the stock qwen3.5 tag's own baked-in values
    assert (cfg.llm.top_p, cfg.llm.top_k, cfg.llm.presence_penalty) == (0.95, 20, 1.5)
    assert cfg.task_guide.model_can_start_tasks is False
