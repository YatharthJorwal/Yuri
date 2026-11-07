"""
Tests for camera.py. The round trip itself (request_frame) is tested with
a fake websocket that records what was sent and lets the test control
when/how resolve_pending_frame answers it -- there's no real browser or
network involved, same "no display/no real browser in this sandbox" limit
as everything else vision-related in this project.
"""

from __future__ import annotations

import asyncio

import pytest

import camera


class _FakeWebSocket:
    """Records every send_json call; nothing else. Real enough for
    request_frame(), which only ever calls this one method."""

    def __init__(self) -> None:
        self.sent: list[dict] = []

    async def send_json(self, data: dict) -> None:
        self.sent.append(data)


@pytest.fixture(autouse=True)
def _reset_pending_frame():
    """camera._pending_frame is module-level (see camera.py's own
    docstring on why) -- reset it before and after every test so tests
    don't leak state into each other."""
    camera._pending_frame = None  # noqa: SLF001 -- direct reset, test-only
    yield
    camera._pending_frame = None  # noqa: SLF001


# ---------------------------------------------------------------------------
# request_frame -- the round trip itself
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_request_frame_sends_request_message():
    ws = _FakeWebSocket()

    async def answer_shortly():
        await asyncio.sleep(0)
        camera.resolve_pending_frame("fake_base64_jpeg")

    asyncio.create_task(answer_shortly())
    result = await camera.request_frame(ws)

    assert result == "fake_base64_jpeg"
    assert ws.sent == [{"type": "request_camera_frame"}]


@pytest.mark.asyncio
async def test_request_frame_raises_on_explicit_none_result():
    ws = _FakeWebSocket()

    async def answer_with_error():
        await asyncio.sleep(0)
        camera.resolve_pending_frame(None)

    asyncio.create_task(answer_with_error())
    with pytest.raises(camera.ToolUnavailableError, match="isn't armed"):
        await camera.request_frame(ws)


@pytest.mark.asyncio
async def test_request_frame_raises_on_timeout(monkeypatch):
    monkeypatch.setattr(camera, "_FRAME_TIMEOUT_SECONDS", 0.01)
    ws = _FakeWebSocket()
    # Nobody ever calls resolve_pending_frame -- simulates a browser that
    # never responds (camera toggle actually off client-side, tab lost
    # focus, etc.).
    with pytest.raises(camera.ToolUnavailableError, match="didn't respond in time"):
        await camera.request_frame(ws)


@pytest.mark.asyncio
async def test_request_frame_clears_pending_after_completion():
    ws = _FakeWebSocket()

    async def answer_shortly():
        await asyncio.sleep(0)
        camera.resolve_pending_frame("frame1")

    asyncio.create_task(answer_shortly())
    await camera.request_frame(ws)
    assert camera._pending_frame is None  # noqa: SLF001


@pytest.mark.asyncio
async def test_request_frame_rejects_concurrent_calls(monkeypatch):
    monkeypatch.setattr(camera, "_FRAME_TIMEOUT_SECONDS", 5.0)
    ws = _FakeWebSocket()

    # First call left deliberately unresolved/in-flight.
    first_call = asyncio.create_task(camera.request_frame(ws))
    await asyncio.sleep(0)  # let it actually start and set _pending_frame

    with pytest.raises(camera.ToolUnavailableError, match="already in progress"):
        await camera.request_frame(ws)

    # Clean up the still-pending first call so the test doesn't leak a
    # background task.
    camera.resolve_pending_frame("frame1")
    await first_call


# ---------------------------------------------------------------------------
# resolve_pending_frame -- no-op when nothing's waiting
# ---------------------------------------------------------------------------


def test_resolve_pending_frame_noop_when_nothing_waiting():
    # Should not raise even though nothing called request_frame first.
    camera.resolve_pending_frame("stray_frame")
    assert camera._pending_frame is None  # noqa: SLF001


# ---------------------------------------------------------------------------
# describe_camera -- request_frame + llm.describe_image glued together
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_describe_camera_success(monkeypatch):
    ws = _FakeWebSocket()

    async def fake_describe_image(prompt, image_b64, **_kw):
        assert image_b64 == "fake_frame"
        return "A person sitting at a desk."

    monkeypatch.setattr(camera.llm, "describe_image", fake_describe_image)

    async def answer_shortly():
        await asyncio.sleep(0)
        camera.resolve_pending_frame("fake_frame")

    asyncio.create_task(answer_shortly())
    result = await camera.describe_camera(ws)
    assert result == "A person sitting at a desk."


@pytest.mark.asyncio
async def test_describe_camera_propagates_tool_unavailable():
    ws = _FakeWebSocket()

    async def answer_with_error():
        await asyncio.sleep(0)
        camera.resolve_pending_frame(None)

    asyncio.create_task(answer_with_error())
    with pytest.raises(camera.ToolUnavailableError):
        await camera.describe_camera(ws)
