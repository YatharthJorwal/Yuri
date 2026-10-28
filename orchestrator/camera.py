"""
Phase 5 -- webcam vision, following the exact same "pull, not push" model
capture_screen() does (docs/ARCHITECTURE.md's "Vision tools" section):
every capture only ever happens because the model chose to call the tool,
never on a timer or a continuous feed.

The one real architectural difference from capture_screen(): there is no
server-side equivalent of PIL.ImageGrab for a webcam. getUserMedia is a
browser API with no Python analog reachable from this process, so the
actual pixels can only come from the frontend. request_frame() below is a
round trip over the websocket instead of a direct OS call -- send a
request, wait (bounded by a timeout) for the frontend's response -- and
the result feeds the same llm.describe_image() entry point
vision.describe_screen()/task_guide.py's check_task_progress() already
use, so from the tool-calling loop's point of view this is still just
another text-in/text-out tool.

Single-slot, module-level pending-frame state -- same reasoning as
task_guide.py's single-active-task state: this is a single-driver-
connection app (see app.py's driver/observer comment), so there's never a
genuine need for more than one in-flight camera request at a time.
"""

from __future__ import annotations

import asyncio

from fastapi import WebSocket

import llm
from tools.vision import ToolUnavailableError

# Bounded so a browser that never responds (camera toggle actually off,
# the window closed, whatever) can't hang a turn indefinitely -- same
# "never let a single tool call block forever" spirit as every other
# network call in this codebase having a real timeout somewhere.
_FRAME_TIMEOUT_SECONDS = 10.0

_pending_frame: asyncio.Future | None = None


async def request_frame(websocket: WebSocket) -> str:
    """Round-trips to the frontend for one camera frame: sends
    request_camera_frame, then waits (bounded) for the matching
    camera_frame response app.py's message dispatch resolves via
    resolve_pending_frame() below.

    Raises ToolUnavailableError -- the same exception capture_screen()
    itself raises, so tools/__init__.py's dispatch_tool_call needs no
    special case for this tool -- on a timeout, an explicit frontend-
    reported error (camera not armed, getUserMedia permission denied),
    or a second call arriving while one's already in flight (shouldn't
    happen given MAX_TOOL_ROUNDS caps a turn to a handful of tool calls
    run one at a time, but guarded rather than silently overwriting a
    request already waiting).

    **Not verified against a real browser/websocket round trip** -- same
    caveat as every other "logically checked, not confirmed" piece of
    this project; the request/response message shapes are new this
    round, not something an existing working path already exercises.
    """
    global _pending_frame
    if _pending_frame is not None and not _pending_frame.done():
        raise ToolUnavailableError("a camera capture is already in progress")

    loop = asyncio.get_running_loop()
    _pending_frame = loop.create_future()
    try:
        await websocket.send_json({"type": "request_camera_frame"})
        try:
            result = await asyncio.wait_for(_pending_frame, timeout=_FRAME_TIMEOUT_SECONDS)
        except asyncio.TimeoutError as exc:
            raise ToolUnavailableError(
                "camera didn't respond in time -- is it armed?"
            ) from exc
    finally:
        _pending_frame = None

    if result is None:
        raise ToolUnavailableError("camera isn't armed right now")
    return result


def resolve_pending_frame(image_b64: str | None) -> None:
    """Called from app.py's message dispatch when a camera_frame message
    arrives. image_b64 is None for an explicit frontend-reported error
    (not armed, permission denied, capture failed) and a real base64 JPEG
    string otherwise -- request_frame() above turns a None into the
    ToolUnavailableError message the model sees.

    A no-op, not an error, if nothing's actually waiting -- a stray or
    late response arriving after request_frame() already timed out and
    moved on shouldn't raise or log anything, just be discarded.
    """
    if _pending_frame is not None and not _pending_frame.done():
        _pending_frame.set_result(image_b64)


async def describe_camera(websocket: WebSocket) -> str:
    """The actual tool `capture_camera` dispatches to (see
    tools/__init__.py) -- mirrors vision.py's describe_screen() exactly,
    just with a browser round trip standing in for capture_screen()'s
    direct OS call."""
    image_b64 = await request_frame(websocket)
    prompt = (
        "Describe what you can see through this camera right now, in a "
        "few sentences. Focus on what's actually visible -- the person, "
        "their expression or what they seem to be doing, the room, "
        "anything notable -- based only on the image, not a generic "
        "guess."
    )
    return await llm.describe_image(prompt, image_b64)
