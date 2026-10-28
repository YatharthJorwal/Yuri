"""
Phase 4's tool registry -- what app.py's main conversation loop hands to
llm.stream_reply_with_tools() as available tools, and how it dispatches
a tool_calls event it gets back. One place so adding a new tool later
(Phase 4 Round 2's ocr_region, Phase 11's browser-automation tool, etc.)
means adding one schema + one handler here, not touching app.py's
turn-handling loop itself.
"""

from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable

import camera
import task_guide
from config import CONFIG
from tools import vision

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "capture_screen",
            "description": (
                "Take a screenshot of the user's screen right now and get "
                "back a text description of what's on it. Use this when "
                "the user asks you to look at their screen or check their "
                "progress on something, or when you need current visual "
                "context you don't already have. Don't use this if the "
                "user already pasted text or described something in "
                "words -- only when you actually need to see the screen."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_clipboard",
            "description": (
                "Read whatever text is currently on the user's clipboard "
                "-- e.g. a stack trace or code snippet they just copied. "
                "More reliable than asking them to retype it or reading "
                "it off a screenshot. Use this when the user says they've "
                "copied something, or asks you to look at what they just "
                "copied."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_active_task",
            "description": (
                "Start, update, or stop tracking a task the user is "
                "currently working on, so their screen can be checked "
                "periodically and they can be nudged back if they drift. "
                "Call this with active=true and a short, concrete "
                "description of the current step (e.g. 'writing the game "
                "loop in main.py for the Flappy Bird clone', not just "
                "'Flappy Bird game') whenever the user states a task or "
                "you infer one and they confirm it, and call it again "
                "with an updated description whenever the current step "
                "changes. Call it with active=false once the task is "
                "finished, the user says to drop or pause it, or they "
                "ask you to stop watching. Do not call this for small "
                "one-off questions that are not really an ongoing task."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "active": {
                        "type": "boolean",
                        "description": "true to start or update tracking, false to stop.",
                    },
                    "description": {
                        "type": "string",
                        "description": (
                            "Short, concrete description of the current "
                            "step. Required when active is true; ignored "
                            "when active is false."
                        ),
                    },
                },
                "required": ["active"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "capture_camera",
            "description": (
                "Look through the user's webcam right now and get back a "
                "text description of what's visible -- the user "
                "themselves, their expression, their surroundings. Only "
                "works if the user has armed the camera from their "
                "quick-action menu; if they haven't, this will fail and "
                "you should just tell them to turn it on first. Use this "
                "when the user asks you to look at them or their room "
                "through the camera -- not for the screen, that's "
                "capture_screen."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]


async def _capture_screen_tool(**_ignored: Any) -> str:
    return await vision.describe_screen()


async def _read_clipboard_tool(**_ignored: Any) -> str:
    # read_clipboard() itself is a plain blocking sync call (OS clipboard
    # access) -- to_thread here so it doesn't stall the event loop, same
    # reasoning as recall.py/store.py's own asyncio.to_thread use around
    # blocking sqlite calls.
    return await asyncio.to_thread(vision.read_clipboard)


_LOOK_TOOLS = frozenset({"capture_screen", "capture_camera"})


def offered_schemas(already_looked: bool = False) -> list[dict[str, Any]]:
    """The tool schemas the chat model is actually offered. set_active_task is
    withheld unless task_guide.model_can_start_tasks is on: the classifier
    (task_guide.detect_task_change) decides when a task starts or stops, and a
    model that calls the tool on its own started a phantom task from a
    remembered conversation on a bare "hi".

    already_looked: the regex gate (look_intent) has already captured the
    screen or camera this turn and put the description in her context. The
    screen/camera tools are withheld then, because a model that is told to
    "call the tool every time" by the persona prompt calls it anyway -- two
    camera frames for one request, and a second description to contradict the
    first. (A look the gate didn't catch still goes through the tools.)"""
    schemas = list(TOOL_SCHEMAS)
    if not CONFIG.task_guide.model_can_start_tasks:
        schemas = [s for s in schemas if s["function"]["name"] != "set_active_task"]
    if already_looked:
        schemas = [s for s in schemas if s["function"]["name"] not in _LOOK_TOOLS]
    return schemas


async def _set_active_task_tool(active: bool = False, description: str = "", **_ignored: Any) -> str:
    if not CONFIG.task_guide.model_can_start_tasks:
        # Not offered, but a model can still emit a tool it wasn't given.
        return "(task tracking is started and stopped automatically from what the user says; it isn't a tool you can call)"
    # Plain in-memory dict update (task_guide.py) -- no blocking I/O, so
    # unlike read_clipboard above this doesn't need asyncio.to_thread.
    return task_guide.set_active_task(bool(active), str(description or ""))


async def _capture_camera_tool(*, websocket: Any = None, **_ignored: Any) -> str:
    # websocket is required for this one tool (see camera.py's own
    # docstring on why: there's no server-side equivalent of a webcam
    # grab, only a browser round trip) -- but the signature still accepts
    # it as a keyword with a default so this function has the exact same
    # shape as every other handler in _HANDLERS below, and dispatch_tool_
    # call can call all four identically rather than special-casing one.
    # None only if dispatch_tool_call is ever called without a websocket,
    # which shouldn't happen in practice (every real call site has one).
    if websocket is None:
        return "(tool unavailable: no active connection to request a camera frame over)"
    return await camera.describe_camera(websocket)


# name -> async callable, every one of which always returns a plain
# string result and never raises anything but vision.ToolUnavailableError
# (dispatch_tool_call below turns even that into a safe string instead of
# propagating). **kwargs on every handler absorbs any argument the model
# passes even though most tools take none today -- a local 9B model
# occasionally hallucinating a parameter shouldn't be a crash.
_HANDLERS: dict[str, Callable[..., Awaitable[str]]] = {
    "capture_screen": _capture_screen_tool,
    "read_clipboard": _read_clipboard_tool,
    "set_active_task": _set_active_task_tool,
    "capture_camera": _capture_camera_tool,
}


async def dispatch_tool_call(name: str, arguments: dict[str, Any], websocket: Any = None) -> str:
    """Runs one tool call by name and always returns a plain string --
    either the real result, the tool's own failure message
    (ToolUnavailableError), or a note that the model asked for a tool
    that doesn't exist (a hallucinated name, which -- being a local 9B
    model -- is real enough to handle rather than crash the turn over).

    websocket is threaded through to every handler as a keyword
    argument (harmlessly absorbed by **_ignored on the three that don't
    need it) rather than given its own special-cased call path, purely
    so capture_camera -- the one tool that genuinely needs it, see
    camera.py -- doesn't need different wiring than the rest.
    """
    handler = _HANDLERS.get(name)
    if handler is None:
        return f"(no such tool: {name!r})"
    try:
        return await handler(**arguments, websocket=websocket)
    except vision.ToolUnavailableError as exc:
        return f"(tool unavailable: {exc})"
    except TypeError as exc:
        # Neither tool takes required arguments, so this only fires on a
        # genuinely malformed call -- degrade gracefully rather than
        # crash the turn over it.
        return f"(tool call failed: {exc})"
