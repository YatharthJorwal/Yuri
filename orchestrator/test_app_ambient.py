"""
App-level tests for the unprompted-comment path (Continuous OCR and Task
Guide) with every outside dependency faked: the screen capture, the VLM, the
LLM and TTS. They pin the behavior the real session got wrong -- the model
commenting on an unchanged screen over and over, and those comments piling up
in history until the context window was full.
"""

from __future__ import annotations

import pytest

import app
import ocr_watch
from config import CONFIG


class FakeWebSocket:
    def __init__(self):
        self.sent: list[dict] = []

    async def send_json(self, payload):
        self.sent.append(payload)


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    ocr_watch.set_active(True)
    spoken: list[str] = []

    async def fake_send_speak(_ws, text):
        spoken.append(text)

    async def fake_log(*_a, **_k):
        return None

    async def fake_stream(_messages):
        yield "Look at that mess. [annoyed]"

    monkeypatch.setattr(app, "_send_speak", fake_send_speak)
    monkeypatch.setattr(app, "_log_transcript_turn_safely", fake_log)
    monkeypatch.setattr(app.llm, "stream_reply", fake_stream)
    monkeypatch.setattr(app.vision, "capture_screen", lambda *_a, **_k: "b64")
    yield spoken
    ocr_watch.set_active(False)


def _vlm(monkeypatch, summaries):
    """Makes the 'VLM' answer comment_worthy=True with the next summary in
    the list every call -- the over-eager behavior seen in the real log."""
    queue = list(summaries)

    async def fake_check(_prev, _img):
        s = queue.pop(0)
        return {"comment_worthy": True, "summary": s, "note": f"they are looking at {s}"}

    monkeypatch.setattr(app.ocr_watch, "check_for_comment", fake_check)


@pytest.mark.asyncio
async def test_unchanged_screen_is_not_commented_on_again(monkeypatch, _isolate):
    spoken = _isolate
    _vlm(monkeypatch, [
        "debugging a Python application with an anime avatar",
        "Debugging a Python app with an anime avatar",
        "Debugging a Python app with an anime avatar",
    ])
    ws, history, unprompted = FakeWebSocket(), [{"role": "system", "content": "persona"}], []
    for _ in range(3):
        await app._run_ocr_watch_check(ws, history, False, [], unprompted)
    assert len(spoken) == 1  # the first glance only; the two repeats stay quiet


@pytest.mark.asyncio
async def test_a_real_change_still_gets_a_comment_once_the_gap_has_passed(monkeypatch, _isolate):
    spoken = _isolate
    _vlm(monkeypatch, ["player mining wood in Minecraft", "Canva design interface with laptop mockup"])
    ws, history, unprompted = FakeWebSocket(), [{"role": "system", "content": "persona"}], []
    await app._run_ocr_watch_check(ws, history, False, [], unprompted)
    # right after speaking, the minimum gap holds the next one back...
    await app._run_ocr_watch_check(ws, history, False, [], unprompted)
    assert len(spoken) == 1
    # ...and once it has passed, a genuinely different screen goes through.
    ocr_watch._state = ocr_watch.replace(  # noqa: SLF001
        ocr_watch._state, last_comment_at=ocr_watch._state.last_comment_at - 10_000  # noqa: SLF001
    )
    _vlm(monkeypatch, ["File explorer showing Unity project folders"])
    await app._run_ocr_watch_check(ws, history, False, [], unprompted)
    assert len(spoken) == 2


@pytest.mark.asyncio
async def test_suppression_reason_is_logged(monkeypatch, capsys):
    _vlm(monkeypatch, ["Minecraft gameplay with anime avatar overlay"] * 2)
    ws, history = FakeWebSocket(), [{"role": "system", "content": "persona"}]
    await app._run_ocr_watch_check(ws, history, False, [], [])
    await app._run_ocr_watch_check(ws, history, False, [], [])
    assert "[no comment: same screen as the last check]" in capsys.readouterr().err


@pytest.mark.asyncio
async def test_history_keeps_only_the_newest_unprompted_comments(monkeypatch):
    # Many different screens, gap forced open each time: history must still
    # hold at most max_unprompted_in_history of them.
    ws, history, flags, unprompted = FakeWebSocket(), [{"role": "system", "content": "persona"}], [], []
    for i in range(12):
        _vlm(monkeypatch, [f"unique screen number{i} topic{i} subject{i} thing{i}"])
        ocr_watch._state = ocr_watch.replace(ocr_watch._state, last_comment_at=0.0)  # noqa: SLF001
        await app._run_ocr_watch_check(ws, history, False, flags, unprompted)
    ambient = [m for m in history if m["role"] == "assistant"]
    assert len(ambient) == CONFIG.session.max_unprompted_in_history
    assert len(flags) == len(history) - 1  # temp flags stayed in lockstep


def test_fit_for_llm_leaves_a_short_conversation_alone():
    messages = [{"role": "system", "content": "persona"}, {"role": "user", "content": "hi"}]
    assert app._fit_for_llm(messages) == messages


def test_fit_for_llm_trims_a_flooded_history_and_says_so(capsys):
    messages = [{"role": "system", "content": "p" * 9000}]
    messages += [{"role": "assistant", "content": "c" * 400} for _ in range(120)]
    messages.append({"role": "user", "content": "what should I do next?"})
    fitted = app._fit_for_llm(messages, app.tools.TOOL_SCHEMAS)
    assert fitted[0] is messages[0] and fitted[-1] is messages[-1]
    assert len(fitted) < len(messages)
    assert "left out the" in capsys.readouterr().err


# --- what she says when the LLM call fails ------------------------------------


def test_each_llm_failure_gets_its_own_line():
    import llm

    assert app.llm_failure_line(llm.LLMTimeoutError("slow")) == app.LLM_TIMEOUT_LINE
    assert app.llm_failure_line(llm.LLMServerError("500 boom")) == app.LLM_SERVER_ERROR_LINE
    assert app.llm_failure_line(llm.LLMUnreachableError("refused")) == app.LLM_UNREACHABLE_LINE


def test_only_a_real_connection_failure_blames_the_server_not_running():
    # The original bug: a server that was up (and had the model loaded) still
    # got "is the model server even running?".
    assert "even running" in app.LLM_UNREACHABLE_LINE
    assert "even running" not in app.LLM_SERVER_ERROR_LINE
    assert "even running" not in app.LLM_TIMEOUT_LINE
