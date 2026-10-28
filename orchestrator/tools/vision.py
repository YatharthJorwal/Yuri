"""
Phase 4 vision tools -- "pull, not push" (docs/ARCHITECTURE.md): every
function here only ever runs because the main model chose to call it via
tool-calling (see llm.stream_reply_with_tools and tools/__init__.py's
TOOL_SCHEMAS/dispatch_tool_call), never on a timer or a continuous feed
by itself. Task Guide Mode's own scheduled capture_screen calls
(ARCHITECTURE.md's "Task Guide Mode" section) are Phase 4 Round 2 -- not
built yet, see docs/ROADMAP.md.

describe_screen() does its own small internal VLM call (llm.describe_image)
to turn a screenshot into a text description before returning -- so from
the main tool-calling loop's point of view, this is just another
text-in/text-out tool, same as read_clipboard(). Keeps app.py's dispatch
loop uniform regardless of which tool got called, and matches
ARCHITECTURE.md's "returns a textual analysis (not raw pixels) to the
reasoning pass" line directly -- raw screenshot bytes never leave this
module.
"""

from __future__ import annotations

import base64
import io
from typing import TYPE_CHECKING

import llm
from config import CONFIG

if TYPE_CHECKING:
    from PIL import Image


class ToolUnavailableError(Exception):
    """A tool couldn't do its job this call -- missing system dependency,
    OS denied access, empty/no display, etc. Distinct from
    llm.LLMUnreachableError (that's the LLM server itself, not a tool).
    Caught by tools/__init__.py's dispatch_tool_call and turned into a
    short text result the model can react to in character, rather than
    crashing the turn."""


def shrink_for_model(image: "Image.Image", max_long_edge: int) -> "Image.Image":
    """Returns `image` scaled down so its longest side is at most
    `max_long_edge` pixels (aspect ratio kept), or `image` itself if it is
    already small enough or max_long_edge is 0/negative. Vision-language
    models pay per pixel (a full 1080p capture is ~2.6k image tokens, and
    1440p/4K far more -- enough to crowd a small context window), so
    unconditionally sending native resolution was the main reason a screen
    read took seconds. LANCZOS keeps small text as legible as a downscale can."""
    from PIL import Image

    longest = max(image.size)
    if max_long_edge <= 0 or longest <= max_long_edge:
        return image
    scale = max_long_edge / longest
    new_size = (max(1, round(image.size[0] * scale)), max(1, round(image.size[1] * scale)))
    return image.resize(new_size, Image.Resampling.LANCZOS)


def capture_screen(max_long_edge: int | None = None) -> str:
    """Takes a screenshot right now and returns it base64-encoded -- the
    *raw* capture, not the text description (that's
    describe_screen() below, which is what's actually registered as a
    tool). Split out on its own so the pure-capture part stays plain and
    synchronously unit-testable (mock PIL.ImageGrab, no event loop or LLM
    stub needed) -- same reasoning as memory/store.py keeping its
    functions plain sync.

    **Not verified on the user's actual machine**: PIL.ImageGrab.grab()
    is Windows/macOS-native (no extra system dependency beyond Pillow
    itself on those two platforms -- Linux needs an X11/Wayland grab
    tool underneath, irrelevant here since this project targets
    Windows), but this has never been run against Luna's real
    screen/multi-monitor setup. Flagged the same as every other
    "logically checked, not confirmed" item in this project. This
    sandbox has no display at all, so even the happy path below is
    necessarily exercised through a mocked ImageGrab in tests, not a
    real screen.
    """
    try:
        from PIL import ImageGrab
    except ImportError as exc:
        raise ToolUnavailableError(f"Pillow isn't installed: {exc}") from exc

    try:
        image = ImageGrab.grab()
    except Exception as exc:
        # Deliberately broad: ImageGrab can fail in several OS/display-
        # specific ways (no display, permission denied, a remote-desktop
        # session with no real framebuffer) and all of them should degrade
        # the same way -- tell the model the tool failed, don't crash the
        # turn over it.
        raise ToolUnavailableError(f"screen capture failed: {exc}") from exc

    buffer = io.BytesIO()
    if max_long_edge is None:
        # Unscaled, lossless: the original behavior (and what the unit tests
        # pin). Callers that feed a model pass a size -- see the config's
        # `vision:` section.
        image.save(buffer, format="PNG")
    else:
        image = shrink_for_model(image, max_long_edge).convert("RGB")
        image.save(buffer, format="JPEG", quality=CONFIG.vision.jpeg_quality)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


async def describe_screen() -> str:
    """The actual tool `capture_screen` dispatches to (see
    tools/__init__.py) -- wraps capture_screen()'s raw base64 PNG with
    the internal VLM call that turns it into text. Kept separate from
    capture_screen() itself precisely so that function can stay
    synchronously unit-testable without needing to stub the LLM too."""
    image_b64 = capture_screen(CONFIG.vision.look_max_long_edge)
    prompt = (
        "Describe what's currently on this computer screen. Focus on: "
        "what application or website is open, any visible text that "
        "looks important (code, error messages, chat, a game's UI), and "
        "anything that looks like it needs the user's attention. Be "
        "concise -- a few sentences, not a wall of text."
    )
    return await llm.describe_image(prompt, image_b64)


def read_clipboard() -> str:
    """Whatever text is currently on the system clipboard -- "the more
    reliable path for coding help (paste a stack trace / snippet instead
    of relying on OCR of a screenshot)" per ARCHITECTURE.md. Returns an
    empty string (not an error) if the clipboard is empty or holds
    non-text content -- an empty clipboard is a normal, expected state,
    not a tool failure.
    """
    try:
        import pyperclip
    except ImportError as exc:
        raise ToolUnavailableError(f"pyperclip isn't installed: {exc}") from exc

    try:
        return pyperclip.paste() or ""
    except Exception as exc:
        # Same broad-catch reasoning as capture_screen() above -- clipboard
        # access can fail in OS-specific ways (no clipboard owner, a
        # locked-down remote session) that should all degrade the same way.
        raise ToolUnavailableError(f"clipboard read failed: {exc}") from exc
