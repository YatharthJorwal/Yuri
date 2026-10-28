"""
Luna's orchestrator -- Phase 2.

Real LLM conversation now: user text is appended to a per-connection
session history (system prompt from persona.py + prior turns), sent to the
configured OpenAI-compatible chat endpoint (config.yaml / llm.py), and the
reply streams back token-by-token. Tokens are chunked into sentence-ish
pieces (chunking.py) so TTS/lipsync can start on the first sentence
instead of waiting for the whole reply -- each chunk goes to the shell as
its own `speak` message (src/main.ts queues and plays them back to back,
not overlapping).

`history` itself is still session-only: a plain Python list tied to the
websocket connection, gone when it disconnects. Phase 3 adds durable
memory *alongside* that, not instead of it: each turn gets a fresh recall
block (facts + semantically relevant past-episode summaries, see
memory/recall.py) injected into that turn's LLM call only, and once the
connection actually ends, memory/consolidation.py distills the whole
session into new facts + one episode summary for next time. No tool-calling
yet either way -- that's Phase 4.

Phase 2.5 adds voice input: a `user_audio` message (base64 WAV/WebM bytes
from the frontend's MediaRecorder) is transcribed via faster-whisper
(stt.py) and fed into the exact same turn-handling path as `user_text` --
see `_run_turn()` below, shared by both message types.

Binds to 127.0.0.1 only, on purpose -- never expose this beyond localhost,
per the project's local-only, non-negotiable constraint (see CLAUDE.md).

Graceful shutdown: `/shutdown` (see its own docstring below) exists
because the Tauri side's tray-Quit handler used to just hard-kill this
process (`TerminateProcess` on Windows, via Rust's `Child::kill()`) --
which meant `ws_endpoint`'s `finally` block, and therefore Phase 3's
`consolidate_session()`, never ran on ANY quit path (tray Quit, Ctrl+C,
Task Manager). Every session's memory was silently lost. Found while
answering the user's own "what's the right way to close this" question,
not anticipated up front -- see docs/DECISIONS.md.
"""

import asyncio
import base64
import contextlib
import os
import signal
import subprocess
import sys
import time
import traceback
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

import camera
import context_budget
import llm
import look_intent
import ocr_watch
import stt
import task_guide
import tools
from chunking import extract_ready_chunks, flush
from config import CONFIG
from memory import consolidation, forget, recall, store
from memory.db import MemoryUnavailableError
from persona import SYSTEM_PROMPT, apply_persona_pass, extract_emotion_tag
from tools import vision
from tts import synthesize

async def _startup_model_checks() -> None:
    """Capability report first (cheap, answers fast), then the warm-up load."""
    await llm.report_model_capabilities()
    await llm.warm_up_model()


@contextlib.asynccontextmanager
async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """One diagnostic at boot: what Ollama says the configured chat/vision
    models can do, with a plain warning when e.g. the model has no vision.
    Runs as a background task so a slow or absent Ollama never delays
    startup; never raises."""
    asyncio.create_task(_startup_model_checks())
    yield


app = FastAPI(lifespan=_lifespan)

HOST = "127.0.0.1"
PORT = 8765

# Set by /shutdown, watched by every open connection's main loop (see
# ws_endpoint) so it can close itself gracefully -- letting `finally` run
# consolidation -- instead of the process just being killed out from under
# it. A plain module-level asyncio.Event is enough here: one orchestrator
# process, one event loop, no multi-worker uvicorn setup to worry about.
_shutdown_event = asyncio.Event()

# Tracked so /shutdown knows when it's actually safe to exit the process --
# once every currently-open connection has torn itself down (and therefore
# already run consolidation), there's nothing left to wait for.
_active_connections: set[WebSocket] = set()

# Full-body sandbox round: two different windows (the Tauri shell,
# src/main.ts, and the browser-tab sandbox, src/sandbox.ts) can now both
# hold an open connection to this same orchestrator at once. Only one of
# them should ever actually be allowed to start a turn -- otherwise both
# could end up generating a reply and playing TTS audio at the same time,
# which would be audibly broken, not just untidy. Whichever connects
# first becomes the "driver"; anything else connecting while the driver
# is still open is an "observer" (see ws_endpoint's connect/disconnect
# handling below and ws-client.ts's protocol comment for the client side
# of this). `_driver` is `None` whenever nobody's connected at all.
_driver: WebSocket | None = None
_connection_surface: dict[WebSocket, str] = {}

# Latest ambient description of where Luna is in the apartment, pushed by
# the sandbox's scene-state channel (see sandbox.ts's pushSceneState and
# ws-client.ts's sendSceneState). Held as a single latest-wins string rather
# than a queue: this is standing context, not events, so an older one is
# never worth replaying. Deliberately module-level and *not* part of
# `history` -- same reasoning as the memory blocks in _run_turn below: it
# would otherwise go stale, repeat every turn, and get fed to
# consolidation.py as though someone had said it out loud.
_scene_state: str | None = None

# Sent (in character, same as LLM_UNREACHABLE_LINE/STT_UNREACHABLE_LINE
# below) if an observer connection tries to start a turn anyway -- a
# stray click on a UI that should already have disabled itself (see
# main.ts's/sandbox.ts's own onSurfaceStatus handling), or a second
# window opened after all. Declining audibly, not just silently dropping
# the message, so it's obvious what happened rather than looking like a
# hang.
SURFACE_BUSY_LINE = "I'm already talking to you somewhere else right now -- finish up there first, hm?"
EMOTION_SURFACE_BUSY = "teasing"

# Said if the configured LLM server can't be reached at all -- most likely
# it just isn't running yet. In character on purpose so a first-run miss
# (forgot to start Ollama) feels like her, not a crash.
LLM_UNREACHABLE_LINE = (
    "H-hey -- I can't reach my own brain right now. Is the model server "
    "even running? Check the orchestrator terminal and try again."
)

# The server answered, but with an HTTP error (out of memory, an unsupported
# parameter, a crashed runner...). The status and body are in the log.
LLM_SERVER_ERROR_LINE = (
    "H-hey -- my brain answered, but with an error instead of words. The "
    "details are in the orchestrator terminal. Try again, and if it keeps "
    "happening, tell Yatharth."
)

# The server was reached but didn't answer in time -- usually the first
# message after startup while a big model is still loading.
LLM_TIMEOUT_LINE = (
    "Ugh -- my brain took too long to wake up. Give it a few seconds and say "
    "that again."
)

# Said if STT itself throws -- most likely stt.device: "cuda" in
# config.yaml but the CUDA cuBLAS/cuDNN DLLs aren't on PATH (see README's
# troubleshooting section), or a corrupted first-time model download.
# Without this, a failure here used to kill the websocket connection
# outright with nothing visible on the user's side at all -- see
# stt.STTError's docstring.
STT_UNREACHABLE_LINE = (
    "I-I can't hear anything right now, something's wrong with my ears. "
    "Check the orchestrator terminal?"
)

# Phase 5.1 -- same reasoning as STT_UNREACHABLE_LINE above: canned,
# in-character text for when describing an uploaded image fails (Ollama
# unreachable, same failure class llm.LLMUnreachableError already covers
# elsewhere) -- sent instead of ever reaching _run_turn, since there's no
# description to build a synthetic user_text out of.
UPLOAD_VISION_UNREACHABLE_LINE = (
    "I-I can't actually look at that right now, something's wrong on my "
    "end. Check the orchestrator terminal?"
)

# Phase 8 -- hardcoded emotion for the two error-fallback lines above,
# rather than trying to get the LLM to tag them: both LLM_UNREACHABLE_LINE
# and STT_UNREACHABLE_LINE are canned text sent *instead of* a real LLM
# call (the LLM is either unreachable, or was never consulted for the STT
# failure path), so there's no model-generated [tag] to extract either
# way -- this is app-level knowledge, not something to fake a tag for.
EMOTION_LLM_UNREACHABLE = "sad"


def llm_failure_line(exc: Exception) -> str:
    """The in-character line for an LLM failure, by what actually went wrong.
    Previously every failure -- including a server that was up and had the
    model loaded -- said "is the model server even running?", which sent the
    user to check the wrong thing."""
    if isinstance(exc, llm.LLMTimeoutError):
        return LLM_TIMEOUT_LINE
    if isinstance(exc, llm.LLMServerError):
        return LLM_SERVER_ERROR_LINE
    return LLM_UNREACHABLE_LINE
EMOTION_STT_UNREACHABLE = "sad"
EMOTION_UPLOAD_VISION_UNREACHABLE = "sad"

# Phase 4 -- how many rounds of "model calls a tool, gets a result, tries
# again" one turn allows before giving up. 3 is generous for the two
# tools that exist today (capture_screen, read_clipboard chained at most
# once or twice in a row is a realistic real request; anything beyond
# that is almost certainly a model stuck re-calling a tool rather than
# answering) and small enough to guarantee a turn can't hang forever on a
# local 9B model that won't stop. Said if that cap is actually hit --
# should be rare; **not verified** how often qwen3.5:9b actually needs
# this in practice, since real tool-calling behavior against this
# project's own Ollama instance hasn't been tested yet (see
# docs/DECISIONS.md).
MAX_TOOL_ROUNDS = 3
TOOL_STUCK_LINE = "...tch, I got stuck trying to do that. Ask me again?"

# Phase 4 Round 2 -- how often ws_endpoint's task-guide background loop
# wakes up to check whether a screen check or an idle-clear is actually
# due (see _task_guide_loop). This is just a cheap dict read most of the
# time (task_guide.due_for_check/is_idle), not the interval between real
# screen checks themselves -- that's CONFIG.task_guide.capture_interval_
# seconds, checked against task_guide's own last_check_at timestamp each
# time this fires. Short enough that the real interval (90s by default)
# isn't meaningfully delayed by polling granularity, long enough not to
# spin the event loop for no reason while no task is even active.
TASK_GUIDE_POLL_SECONDS = 15

# Continuous OCR -- same reasoning as TASK_GUIDE_POLL_SECONDS above, just
# coarser: this feature's own real interval
# (CONFIG.ocr_watch.comment_interval_seconds, 240s by default) is far
# longer than task_guide's, so there's no need to poll as tightly to stay
# close to it.
OCR_WATCH_POLL_SECONDS = 30


async def _send_speak(websocket: WebSocket, text: str) -> None:
    audio_bytes = await synthesize(text)
    await websocket.send_json(
        {
            "type": "speak",
            "text": text,
            "audio_b64": base64.b64encode(audio_bytes).decode("ascii"),
            "mime": "audio/wav",
        }
    )


async def _log_transcript_turn_safely(user_text: str, assistant_text: str) -> None:
    """Best-effort -- a DB hiccup here shouldn't take down the turn itself
    (turn_end still needs to reach the frontend either way), same
    degrade-gracefully pattern recall.py/forget.py already use for the
    same underlying DB."""
    try:
        await asyncio.to_thread(store.add_transcript_turn, user_text, assistant_text)
    except MemoryUnavailableError as exc:
        print(f"[luna] transcript log: write failed, skipping: {exc}", file=sys.stderr)


async def _get_transcript_log_safely() -> list[dict[str, str]]:
    try:
        return await asyncio.to_thread(store.get_transcript_log)
    except MemoryUnavailableError as exc:
        print(f"[luna] transcript log: read failed, returning empty: {exc}", file=sys.stderr)
        return []


async def _clear_transcript_log_safely() -> None:
    try:
        await asyncio.to_thread(store.clear_transcript_log)
    except MemoryUnavailableError as exc:
        print(f"[luna] transcript log: clear failed: {exc}", file=sys.stderr)


def _fit_for_llm(
    messages: list[dict[str, Any]],
    tools_list: list[dict[str, Any]] | None = None,
    extra: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """The messages actually sent to the model: `messages` with the oldest
    history dropped (from this copy only) until prompt + reserved reply fit
    llm.num_ctx. See context_budget.py for why -- replies were being cut off
    with done_reason 'length' because nothing bounded the prompt by size."""
    fitted, dropped, est_tokens, budget = context_budget.fit_messages(
        messages,
        num_ctx=CONFIG.llm.num_ctx,
        reply_reserve=CONFIG.llm.max_tokens,
        tools=tools_list,
        extra=extra,
    )
    if dropped:
        print(
            f"[luna] context: left out the {dropped} oldest history message(s) this call "
            f"to fit num_ctx={CONFIG.llm.num_ctx} (estimated prompt {est_tokens} tokens, "
            f"budget {budget}).",
            file=sys.stderr,
        )
    if budget and est_tokens > budget:
        print(
            f"[luna] WARNING: the persona prompt + memory block alone (~{est_tokens} tokens) "
            f"exceed the prompt budget ({budget}) for num_ctx={CONFIG.llm.num_ctx}; raise "
            "llm.num_ctx.",
            file=sys.stderr,
        )
    return fitted


def _record_unprompted(
    history: list[dict[str, str]],
    text: str,
    temp_mode: bool,
    temp_turn_flags: list[bool] | None,
    unprompted: list[dict[str, str]] | None,
) -> None:
    """Appends one of her unprompted comments (a Task Guide chide or an OCR
    remark -- an assistant turn with no user turn before it) to history,
    keeps temp_turn_flags in lockstep, then caps how many such comments stay
    in history (session.max_unprompted_in_history). `unprompted` None =
    don't track or cap (direct callers/tests)."""
    message = {"role": "assistant", "content": text}
    history.append(message)
    if temp_turn_flags is not None:
        temp_turn_flags.append(temp_mode)
    if unprompted is not None:
        unprompted.append(message)
        context_budget.cap_unprompted(
            history, temp_turn_flags, unprompted, CONFIG.session.max_unprompted_in_history
        )
    _trim_history(history, temp_turn_flags)


def _trim_history(history: list[dict[str, str]], temp_turn_flags: list[bool] | None = None) -> None:
    """Keeps the system prompt plus the last N (user, assistant) turns.
    Session memory only (Phase 3 adds durable memory), but still needs a
    cap so a long session doesn't grow the LLM's context unbounded.

    temp_turn_flags, when given, is trimmed the same way and by the same
    amount -- it's kept in lockstep with history[1:] (see _run_turn's own
    docstring), so whatever gets dropped from the front of `turns` here
    has to be dropped from its front too, or every flag after the cut
    would silently point at the wrong history entry."""
    system, turns = history[0], history[1:]
    max_messages = CONFIG.session.max_history_turns * 2
    if len(turns) > max_messages:
        turns = turns[-max_messages:]
        if temp_turn_flags is not None:
            temp_turn_flags[:] = temp_turn_flags[-max_messages:]
    history[:] = [system] + turns


async def _run_turn(
    websocket: WebSocket,
    history: list[dict[str, str]],
    user_text: str,
    temp_mode: bool = False,
    temp_turn_flags: list[bool] | None = None,
) -> None:
    """The actual agent turn: append user_text to history, stream the LLM
    reply, speak each sentence chunk as it's ready, commit (or roll back)
    history. Shared by both `user_text` (typed) and `user_audio`
    (transcribed) messages -- by the time this runs, there's no difference
    between the two.

    Also writes one row to Phase 9's persistent transcript_log table
    (memory/store.py) per call, regardless of how the turn ends (normal
    completion, LLM-unreachable fallback, or stopped mid-sentence) -- see
    spoken_parts below for what that does and doesn't include, and why
    it's tracked separately from reply_parts/history.

    temp_mode/temp_turn_flags (quick-action menu's Temp Chat toggle):
    temp_turn_flags is a list kept in lockstep with history[1:] (one
    entry per history.append(), True if temp_mode was active for that
    append) -- see ws_endpoint's own comment on why this is tracked
    per-turn rather than as a single per-session flag, and consolidation
    time (also in ws_endpoint) for where it actually gets used. None
    means "don't bother" (there's nowhere to track it, or this call site
    doesn't care) -- every real call site passes a real list."""
    history.append({"role": "user", "content": user_text})
    if temp_turn_flags is not None:
        temp_turn_flags.append(temp_mode)
    # Phase 4 Round 2: any real user turn counts as "they're still here,"
    # regardless of whether it's about the tracked task at all -- resets
    # Task Guide Mode's idle clock (see task_guide.is_idle/_task_guide_loop
    # below). A no-op if nothing's actually being tracked.
    task_guide.mark_interaction()

    # Phase 3: forget first, then recall -- so a fact just removed this
    # turn can't immediately resurface in the same turn's recall block.
    # Both are ephemeral system messages built fresh per turn, spliced
    # into the LLM call only -- never written into `history` itself (see
    # memory/recall.py's docstring for why: history is the real
    # conversation, and this would otherwise go stale, double up every
    # turn, and get fed back into consolidation.py as if it were something
    # someone actually said). Falls back to plain `history` unchanged if
    # neither has anything to add -- a turn should never fail or even
    # look different structurally just because memory had nothing to do.
    # Skipped entirely in temp_mode -- Temp Chat's whole point is "read
    # nothing from persistent memory, write nothing to it," and
    # maybe_update_task counts here too even though task state isn't
    # SQLite-persisted: it's still a live, carry-forward side effect
    # (a task tracked from a "this doesn't count" conversation chiding
    # the user later in a normal one would defeat the point).
    if temp_mode:
        forget_hint = None
        task_hint = None
        memory_block = None
    else:
        forget_hint = await forget.maybe_forget(user_text)
        # Phase 4 Round 2 fix -- see task_guide.py's own comment on why this
        # is a dedicated classification call rather than left to the model's
        # own set_active_task tool call: real testing showed the latter
        # doesn't fire reliably for this model. Same ephemeral-hint treatment
        # as forget_hint right above.
        task_hint = await task_guide.maybe_update_task(user_text)
        memory_block = await recall.build_recall_context(user_text, CONFIG.memory.recall_top_k)
    # Explicit "look at my screen / camera" requests, handled here in Python
    # instead of trusting the model to call capture_screen/capture_camera
    # itself -- see look_intent.py for the real-log evidence (2 of ~9
    # explicit requests actually fired the tool). Deliberately NOT gated on
    # temp_mode: this is live sensing that gets spliced in for this one
    # turn, not a read from or write to persistent memory.
    look_hint = await look_intent.maybe_look(user_text, websocket)
    memory_parts = [part for part in (forget_hint, task_hint, look_hint, memory_block) if part]

    # Phase 10: where she physically is, if the sandbox has told us. Same
    # ephemeral-system-message treatment as the memory blocks above, and
    # placed alongside them so there's only ever one spliced system turn.
    # This is what lets her say "I'm in the kitchen" because it's true,
    # rather than because the model picked a room at random.
    if _scene_state:
        memory_parts.append(
            f"Context about your surroundings right now: {_scene_state} "
            "Mention it only if it is relevant to what is being said -- "
            "do not narrate your location unprompted."
        )

    if memory_parts:
        combined = " ".join(memory_parts)
        messages_for_llm = history[:-1] + [{"role": "system", "content": combined}] + history[-1:]
    else:
        messages_for_llm = history

    buffer = ""
    reply_parts: list[str] = []
    # Everything actually sent to _send_speak this turn, in order --
    # tracked separately from reply_parts because the two lists serve
    # different purposes and can legitimately diverge: reply_parts is what
    # goes back into `history` for the LLM's own context (and deliberately
    # excludes the LLM-unreachable fallback line, see below), while
    # spoken_parts is Phase 9's transcript-log panel's record of what the
    # user actually heard/saw this turn, unconditionally -- including that
    # same fallback line, since it genuinely was spoken to them even though
    # it shouldn't be fed back into the LLM as if she'd said it.
    spoken_parts: list[str] = []
    detected_emotion: str | None = None
    tool_messages: list[dict[str, Any]] = []
    offered_tools = tools.offered_schemas(already_looked=bool(look_hint))
    try:
        for _ in range(MAX_TOOL_ROUNDS):
            saw_tool_call = False
            async for event in llm.stream_reply_with_tools(
                _fit_for_llm(messages_for_llm, offered_tools, tool_messages) + tool_messages,
                offered_tools,
            ):
                if event["type"] == "tool_calls":
                    saw_tool_call = True
                    # Dispatched and appended one at a time, in order --
                    # if the model asked for more than one tool in the
                    # same turn (untested whether qwen3.5:9b actually
                    # does this), each one's result needs to already be
                    # in tool_messages before the next round re-calls the
                    # model, same as a real multi-tool exchange would.
                    for call in event["calls"]:
                        result_text = await tools.dispatch_tool_call(
                            call["name"], call["arguments"], websocket=websocket
                        )
                        tool_messages.append(
                            {
                                "role": "assistant",
                                "content": "",
                                "tool_calls": [
                                    {
                                        "function": {
                                            "name": call["name"],
                                            "arguments": call["arguments"],
                                        }
                                    }
                                ],
                            }
                        )
                        tool_messages.append({"role": "tool", "content": result_text})
                    break  # re-enter the outer for-loop with results appended
                buffer += event["text"]
                chunks, buffer = extract_ready_chunks(buffer)
                for chunk in chunks:
                    reply_parts.append(chunk)
                    styled = apply_persona_pass(chunk)
                    spoken_parts.append(styled)
                    await _send_speak(websocket, styled)
            if not saw_tool_call:
                # A normal reply, not another tool call -- stream's fully
                # done now, not mid-flight, so this is the only safe point
                # to check for a trailing [emotion] tag (see
                # extract_emotion_tag()'s own docstring on why mid-stream
                # would risk a false match).
                buffer, detected_emotion = extract_emotion_tag(buffer)
                for chunk in flush(buffer):
                    reply_parts.append(chunk)
                    styled = apply_persona_pass(chunk)
                    spoken_parts.append(styled)
                    await _send_speak(websocket, styled)
                break
        else:
            # Used up every round still calling tools, never landed on an
            # actual reply -- a model stuck re-calling a tool (or hitting
            # one that keeps failing) shouldn't hang the turn forever.
            # Distinct from LLMUnreachableError below: the server itself
            # is fine, she just never got to a real answer.
            await _send_speak(websocket, TOOL_STUCK_LINE)
            spoken_parts.append(TOOL_STUCK_LINE)
    except llm.LLMUnreachableError as exc:
        # Logged: this branch used to swallow the exception completely, so a
        # 400/500 or a timeout from a server that was running looked exactly
        # like "server not running" and left nothing to diagnose from.
        print(f"[luna] LLM call failed ({type(exc).__name__}): {exc}", file=sys.stderr)
        line = llm_failure_line(exc)
        await _send_speak(websocket, line)
        spoken_parts.append(line)
        detected_emotion = EMOTION_LLM_UNREACHABLE
    except Exception:  # noqa: BLE001 -- deliberate catch-all, see below
        # Anything else (a TTS failure, a socket error, a plain bug) used to
        # kill this task silently: the finally block below still recorded
        # whatever had been spoken so far -- which reads exactly like a
        # reply that cut itself off -- and then the exception escaped
        # without turn_end ever being sent, leaving the frontend's stop
        # button stuck on. The traceback goes to the log so the cause is
        # finally visible, and the flow falls through to the normal
        # turn_end. CancelledError (the stop button) is a BaseException and
        # deliberately NOT caught here -- stop must still cancel the turn.
        print("[luna] turn failed with an unexpected error (reply cut off):", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
    finally:
        # In `finally`, not just after the try block, so a stop-button
        # cancellation (asyncio.CancelledError, raised into whichever
        # await this task happens to be sitting on when ws_endpoint's
        # "stop" handler cancels it) still records whatever was actually
        # said up to that point -- matching reality (she got cut off
        # mid-sentence) instead of silently dropping the whole turn.
        # CancelledError propagates on through after this, same as any
        # other exception would through a finally block. The
        # LLM-unreachable fallback line above is deliberately NOT added to
        # reply_parts -- it was never really said by her, so it shouldn't
        # end up looking like a real reply in history either.
        if reply_parts:
            history.append({"role": "assistant", "content": " ".join(reply_parts)})
            if temp_turn_flags is not None:
                temp_turn_flags.append(temp_mode)
        else:
            # Nothing usable came back (LLM unreachable and no fallback
            # line ended up in reply_parts either, or stopped before a
            # single chunk arrived) -- drop the dangling user turn rather
            # than leave a one-sided exchange in context for next time.
            history.pop()
            if temp_turn_flags is not None:
                temp_turn_flags.pop()
        _trim_history(history, temp_turn_flags)
        # Awaiting here, inside `finally`, after a possible cancellation
        # above is safe: the cancellation that got us here already fired
        # once (at whichever `await` this task was sitting on), and
        # nothing cancels it a second time while this runs -- ws_endpoint's
        # "stop" handler only calls current_turn_task.cancel() once, then
        # awaits the task's own unwinding. Logged unconditionally (not
        # just on normal completion) so the log panel matches reality even
        # for a stopped-mid-sentence turn, same reasoning as the
        # history.append() above.
        await _log_transcript_turn_safely(user_text, " ".join(spoken_parts))

    # Tells the frontend nothing more is coming for this turn (so it can
    # hide the stop button, re-enable input, etc.) -- only reached on a
    # *normal* completion (finished generating, or hit the LLM-unreachable
    # fallback). A stop-button cancellation skips this entirely (the
    # CancelledError above propagates straight out of the function instead
    # of reaching this line) -- ws_endpoint's "stop" handler sends its own
    # turn_end right after successfully cancelling, once it's actually
    # safe to send anything further over the socket. That stopped-turn
    # case is also why detected_emotion can genuinely be None here even
    # on a real reply -- either the tag-extraction step was never reached
    # (LLM unreachable), or the model just didn't produce a recognized
    # tag this time (see extract_emotion_tag()'s own docstring) -- the
    # frontend leaves the current expression alone rather than guessing
    # when this is omitted/None.
    await websocket.send_json({"type": "turn_end", "emotion": detected_emotion})


# Phase 4 Round 2 -- how the drift observation from task_guide.check_task_
# progress() gets turned into an actual in-character line. Framed as an
# ephemeral one-off user-role message spliced onto the end of `history`
# for this call only (never written into `history` itself) -- same
# "ephemeral system context, not a real turn" treatment as recall.py's
# memory block and _scene_state above, just user-role here since it's
# instructing her to react to something specific rather than describing
# ambient world state.
def _drift_prompt(task_description: str, observation: str) -> str:
    return (
        f"You are currently tracking this task for the user: "
        f"{task_description!r}. You just glanced at their screen on "
        f"your own, without them saying anything, and it looks like "
        f"they have drifted off it: {observation!r}. React to this "
        "right now, in character, in one short spoken line -- call it "
        "out and steer them back to the step. Do not mention that you "
        "were checking, do not mention a timer or a tool or that this "
        "was automatic, just react like you noticed it yourself. End "
        "with your usual [emotion] tag on its own line, same as always."
    )


async def _run_task_guide_check(
    websocket: WebSocket,
    history: list[dict[str, str]],
    task_description: str,
    temp_mode: bool = False,
    temp_turn_flags: list[bool] | None = None,
    unprompted: list[dict[str, str]] | None = None,
) -> None:
    """The scheduled half of Task Guide Mode actually doing its job: grab
    a screenshot, ask the VLM whether it still matches the tracked step
    (task_guide.check_task_progress), and if it clearly doesn't, have her
    react in character over the same speak pipeline _run_turn uses for a
    real reply -- just triggered by _task_guide_loop's timer instead of
    the user typing something.

    temp_mode/temp_turn_flags: read live off the connection at call time
    (see _task_guide_loop's own comment on late-binding closures) so a
    chide that happens to land while Temp Chat is toggled on still gets
    excluded from end-of-session consolidation, same as everything else
    said during that window -- this does NOT suppress the chide itself,
    just how its history entry gets treated afterward. Task tracking
    itself already can't *start* during temp_mode (see _run_turn), so
    this only matters for a task that was already active before Temp
    Chat got toggled on mid-session.

    Always calls task_guide.mark_checked() up front (whether or not this
    ends up chiding) so due_for_check() measures from "last time we
    actually looked," not "last time we found drift" -- an on-task
    result still counts as a completed check.

    Deliberately narrow about what's worth logging vs. silently skipping:
    a failed capture, an unreachable LLM, or an unparseable VLM response
    all just mean "try again next interval" -- none of these should ever
    surface to the user as more than, at most, a missed check. Not
    wrapped in a broader try/except beyond that: an unexpected exception
    here should still show up in the orchestrator's own terminal rather
    than vanish, same as any other real bug would.
    """
    task_guide.mark_checked()
    try:
        image_b64 = await asyncio.to_thread(vision.capture_screen, CONFIG.vision.ambient_max_long_edge)
    except vision.ToolUnavailableError as exc:
        print(f"[luna] task guide: capture failed, skipping this check: {exc}", file=sys.stderr)
        return

    result = await task_guide.check_task_progress(task_description, image_b64)
    if result is None:
        print("[luna] task guide: check result didn't parse, skipping this check", file=sys.stderr)
        return
    # Logged unconditionally (not just on drift) so the loop's actual
    # cadence is visible in orchestrator.log without needing to wait for
    # a real chide -- useful for confirming it's alive at all during
    # testing, or for tuning capture_interval_seconds against how often
    # checks are really landing.
    print(
        f"[luna] task guide: checked '{task_description}' -> "
        f"on_task={result['on_task']} ({result['note'] or 'no note'})",
        file=sys.stderr,
    )
    if result["on_task"]:
        return

    prompt = _drift_prompt(task_description, result["note"] or "something unrelated is on screen")
    try:
        buffer = ""
        async for delta in llm.stream_reply(
            _fit_for_llm(history + [{"role": "user", "content": prompt}])
        ):
            buffer += delta
    except llm.LLMUnreachableError as exc:
        print(f"[luna] task guide: LLM unreachable for chide, skipping: {exc}", file=sys.stderr)
        return

    buffer, detected_emotion = extract_emotion_tag(buffer)
    reply_parts = flush(buffer)
    if not reply_parts:
        return

    for chunk in reply_parts:
        await _send_speak(websocket, apply_persona_pass(chunk))

    # Recorded as her own assistant turn, same as a normal reply, so a
    # later real turn sees she already commented on the drift and Phase
    # 9's transcript-log panel shows it happened. Deliberately no
    # matching "user" turn precedes it -- nobody said anything -- which
    # is a legitimate shape for an unprompted comment, not a bug.
    _record_unprompted(history, " ".join(reply_parts), temp_mode, temp_turn_flags, unprompted)
    await _log_transcript_turn_safely("", " ".join(apply_persona_pass(p) for p in reply_parts))
    await websocket.send_json({"type": "turn_end", "emotion": detected_emotion})


def _ambient_comment_prompt(observation: str) -> str:
    return (
        f"You just glanced at the user's screen on your own, without "
        f"them saying anything, and noticed something worth remarking "
        f"on: {observation!r}. React to this right now, in character, "
        "in one short spoken line -- a genuine, in-the-moment remark, "
        "not a report. Do not mention that you were checking, do not "
        "mention a timer or a tool or that this was automatic, just "
        "react like you noticed it yourself. End with your usual "
        "[emotion] tag on its own line, same as always."
    )


async def _run_ocr_watch_check(
    websocket: WebSocket,
    history: list[dict[str, str]],
    temp_mode: bool = False,
    temp_turn_flags: list[bool] | None = None,
    unprompted: list[dict[str, str]] | None = None,
) -> None:
    """Continuous OCR's own scheduled check -- structurally the same
    shape as _run_task_guide_check above (grab a screenshot, ask the VLM
    a judgment question, react in character if it says yes), but judging
    "is this worth an unprompted remark at all" (ocr_watch.
    check_for_comment) rather than "does this match a tracked task."
    Unlike Task Guide Mode, there is no task description to compare
    against -- ocr_watch.py's own WatchState.last_seen_summary is the only
    context carried between checks, so the judgment is "changed/notable
    since last glance," not "off the goal."

    temp_mode/temp_turn_flags: same treatment as _run_task_guide_check's
    own -- read live at call time so a comment landing while Temp Chat is
    on still gets excluded from end-of-session consolidation, without
    suppressing the comment itself.

    Always calls ocr_watch.mark_checked() (with the fresh summary, so the
    next check has something to compare against) whether or not this
    ends up commenting -- due_for_check() measures from "last time we
    actually looked," not "last time we said something."

    Same "narrow about what's worth logging vs. silently skipping" policy
    as _run_task_guide_check: a failed capture, an unreachable LLM, or an
    unparseable VLM response all just mean "try again next interval,"
    never more than a missed check.
    """
    try:
        image_b64 = await asyncio.to_thread(vision.capture_screen, CONFIG.vision.ambient_max_long_edge)
    except vision.ToolUnavailableError as exc:
        print(f"[luna] ocr watch: capture failed, skipping this check: {exc}", file=sys.stderr)
        return

    previous_summary = ocr_watch.get_state().last_seen_summary
    last_comment_at = ocr_watch.get_state().last_comment_at
    result = await ocr_watch.check_for_comment(previous_summary, image_b64)
    if result is None:
        print("[luna] ocr watch: check result didn't parse, skipping this check", file=sys.stderr)
        return
    ocr_watch.mark_checked(result["summary"])
    # The model's comment_worthy is necessary, not sufficient: code applies
    # the quiet rules the prompt could not enforce (same screen as last
    # time, minimum gap since she last spoke, an actual note to react to).
    # The line below is logged unconditionally, same reasoning as
    # task_guide's own check line: it confirms the loop is alive without
    # waiting for a real comment, and says why a "yes" was overruled.
    speak, reason = ocr_watch.decide_comment(
        result,
        previous_summary,
        last_comment_at,
        CONFIG.ocr_watch.min_comment_gap_seconds,
        CONFIG.ocr_watch.same_screen_similarity,
    )
    suppressed = f" [no comment: {reason}]" if result["comment_worthy"] and not speak else ""
    print(
        f"[luna] ocr watch: checked -> comment_worthy={result['comment_worthy']} "
        f"({result['summary'] or 'no summary'}){suppressed}",
        file=sys.stderr,
    )
    if not speak:
        return

    prompt = _ambient_comment_prompt(result["note"])
    try:
        buffer = ""
        async for delta in llm.stream_reply(
            _fit_for_llm(history + [{"role": "user", "content": prompt}])
        ):
            buffer += delta
    except llm.LLMUnreachableError as exc:
        print(f"[luna] ocr watch: LLM unreachable for comment, skipping: {exc}", file=sys.stderr)
        return

    buffer, detected_emotion = extract_emotion_tag(buffer)
    reply_parts = flush(buffer)
    if not reply_parts:
        return

    for chunk in reply_parts:
        await _send_speak(websocket, apply_persona_pass(chunk))

    # Same "unprompted comment, no preceding user turn" shape
    # _run_task_guide_check's own chide uses -- see its comment for why
    # that's legitimate here too, not a bug.
    ocr_watch.mark_commented()
    _record_unprompted(history, " ".join(reply_parts), temp_mode, temp_turn_flags, unprompted)
    await _log_transcript_turn_safely("", " ".join(apply_persona_pass(p) for p in reply_parts))
    await websocket.send_json({"type": "turn_end", "emotion": detected_emotion})


@app.post("/shutdown")
async def shutdown_endpoint() -> dict:
    """Requested by src-tauri/src/lib.rs's tray-Quit handler as a graceful
    shutdown attempt *before* falling back to a hard process kill -- see
    this module's own top docstring for why that fallback alone was
    silently losing every session's Phase 3 memory.

    Sets `_shutdown_event`, which every open connection's main loop is
    already watching (see `ws_endpoint`) -- each one notices, breaks out
    of its receive loop, and runs its normal `finally` teardown (including
    `consolidate_session()`) same as a real disconnect would. This handler
    then waits a bounded amount of time for that to actually finish before
    exiting the process itself, so Rust's fallback `.kill()` only ever
    fires if this doesn't finish in time -- not as the normal path.

    Deliberately `os._exit(0)` rather than uvicorn's own graceful-shutdown
    machinery (`server.should_exit`) -- this is a single-user local
    desktop app, not a server that needs a zero-downtime drain; explicit,
    self-bounded teardown here is simpler to reason about and test than
    coordinating with uvicorn's own shutdown sequence on top of it.
    """
    _shutdown_event.set()

    # Bounded wait for every currently-open connection to actually finish
    # tearing down (and therefore finish consolidating) -- 50 x 100ms = 5s,
    # matched to the grace period src-tauri/src/lib.rs's quit handler waits
    # before falling back to a hard kill; if this window's ever too short
    # for a slow consolidation LLM call, that's a mismatch to fix on both
    # sides together, not something to silently paper over on just one.
    for _ in range(50):
        if not _active_connections:
            break
        await asyncio.sleep(0.1)

    sys.stdout.flush()
    sys.stderr.flush()
    # Scheduled slightly after this handler returns, not called inline --
    # os._exit() would otherwise tear down the process before uvicorn gets
    # a chance to actually send this response back.
    asyncio.get_event_loop().call_later(0.1, lambda: os._exit(0))
    return {"status": "shutting down"}


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    _active_connections.add(websocket)

    # Driver/observer assignment -- see _driver's own module-level comment
    # above. Decided once, right here, at connect time: "whoever's already
    # open wins" is a simple, honest rule that doesn't need to know
    # anything about *which* surface either window is (shell vs sandbox);
    # `surface` itself is only carried through for logging/debugging, not
    # used in this decision.
    global _driver
    surface = websocket.query_params.get("surface", "unknown")
    _connection_surface[websocket] = surface
    is_driver = _driver is None
    if is_driver:
        _driver = websocket
    await websocket.send_json({"type": "surface_status", "role": "driver" if is_driver else "observer"})

    history: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    # Quick-action menu's Temp Chat toggle -- per-connection, unlike
    # task_guide's module-level state, since it's about this window's own
    # conversation rather than anything shared. temp_turn_flags is kept
    # in lockstep with history[1:], one bool per history.append() (see
    # _run_turn's own docstring for the exact bookkeeping) -- read at
    # session end (this function's own `finally` below) to exclude
    # anything said while temp_mode was on from consolidation, without
    # having to throw away memorable content from the rest of a session
    # that only briefly touched temp mode.
    temp_mode = False
    temp_turn_flags: list[bool] = []
    # The unprompted comments (Task Guide chides, OCR remarks) currently in
    # `history`, by identity, so context_budget.cap_unprompted can keep only
    # the newest few. Per connection, like history itself.
    unprompted_msgs: list[dict[str, str]] = []
    # The in-flight turn, if any -- run as its own task (not just awaited
    # inline) specifically so the main loop below can keep concurrently
    # watching for a "stop" message (or /shutdown) while generation is
    # still happening. Awaiting _run_turn() directly, the way earlier
    # phases did, meant the loop couldn't read anything else off the
    # socket until a turn finished -- a "stop" message would just sit
    # unread in the transport buffer until it was too late to matter.
    current_turn_task: asyncio.Task | None = None

    # Phase 4 Round 2 -- Task Guide Mode's scheduled-capture loop, one
    # per *driver* connection only (an observer window never starts
    # turns at all, so it has no business triggering screen checks or
    # speaking chides either -- see the driver/observer comment above).
    # Nested here (rather than a module-level function taking extra
    # arguments) specifically so it can read `current_turn_task` by
    # closure and skip a check while a real turn is already mid-reply --
    # Python's late-binding closures see the *current* value of an
    # enclosing-scope variable at call time, not a snapshot from when the
    # closure was created, which is exactly the "is a turn running right
    # now" read this needs on every wake.
    #
    # ambient_speak_lock is shared with _ocr_watch_loop below -- without
    # it, Task Guide Mode's chide and Continuous OCR's comment could both
    # decide to fire around the same moment (two independent asyncio
    # tasks, polling on different schedules) and race on the same
    # `history` list and the same websocket's _send_speak calls. Held
    # only around the actual check+speak call, not the sleep/poll --
    # locked() is checked first so a loop that's not due doesn't even
    # attempt to acquire it.
    ambient_speak_lock = asyncio.Lock()

    async def _task_guide_loop() -> None:
        while True:
            await asyncio.sleep(TASK_GUIDE_POLL_SECONDS)
            state = task_guide.get_state()
            if not state.active:
                continue
            if task_guide.is_idle(state, CONFIG.task_guide.idle_timeout_seconds):
                # Stepped away -- drop the task silently, no chide.
                # Per ARCHITECTURE.md: "the task stays active until ...
                # an idle timeout is hit." Nobody's there to hear a nudge.
                task_guide.clear_active_task()
                continue
            if not task_guide.due_for_check(state, CONFIG.task_guide.capture_interval_seconds):
                continue
            if current_turn_task is not None and not current_turn_task.done():
                # A real turn is already in flight -- skip this wake
                # rather than run a screen check and a chide concurrently
                # with it (both would call _send_speak on the same
                # websocket and mutate the same `history` list). Just
                # skipped, not deferred: due_for_check() will be true
                # again next poll if the turn's still running then, and
                # mark_checked() is only called once a check actually
                # runs, so nothing here silently resets the interval.
                continue
            if ambient_speak_lock.locked():
                # Continuous OCR's own check is currently speaking --
                # same "skip, don't defer" policy as the current_turn_task
                # guard just above.
                continue
            async with ambient_speak_lock:
                await _run_task_guide_check(
                    websocket, history, state.description, temp_mode, temp_turn_flags, unprompted_msgs
                )

    async def _ocr_watch_loop() -> None:
        pause_announced = False
        while True:
            await asyncio.sleep(OCR_WATCH_POLL_SECONDS)
            state = ocr_watch.get_state()
            if not state.active:
                pause_announced = False
                continue
            paused = CONFIG.ocr_watch.pause_during_task and task_guide.get_state().active
            if paused and not pause_announced:
                # Said once, because "Continuous OCR does nothing" was
                # reported when it was really paused by a (phantom) task.
                print(
                    f"[luna] ocr watch: paused while Task Guide tracks "
                    f"'{task_guide.get_state().description}' (ocr_watch.pause_during_task); "
                    "it resumes when that task ends.",
                    file=sys.stderr,
                )
                pause_announced = True
            elif not paused and pause_announced:
                print("[luna] ocr watch: resumed (no task being tracked)", file=sys.stderr)
                pause_announced = False
            if paused:
                # Task Guide is already watching the screen; ambient remarks
                # on top of its chides were redundant (both commented on the
                # same Yahtzee screen within a minute in real use) and
                # doubled the vision calls. Restart the clock so the first
                # ambient check after the task ends is a full interval away.
                ocr_watch.defer_check()
                continue
            if not ocr_watch.due_for_check(state, CONFIG.ocr_watch.comment_interval_seconds):
                continue
            if current_turn_task is not None and not current_turn_task.done():
                continue
            if ambient_speak_lock.locked():
                continue
            async with ambient_speak_lock:
                await _run_ocr_watch_check(
                    websocket, history, temp_mode, temp_turn_flags, unprompted_msgs
                )

    task_guide_task: asyncio.Task | None = (
        asyncio.create_task(_task_guide_loop()) if is_driver else None
    )
    ocr_watch_task: asyncio.Task | None = (
        asyncio.create_task(_ocr_watch_loop()) if is_driver else None
    )

    try:
        while True:
            # Raced against _shutdown_event instead of a plain
            # `await websocket.receive_json()` -- otherwise a connection
            # sitting idle (nobody's typed anything) would just block
            # forever, never noticing /shutdown was called at all, and
            # this process would still need the hard-kill fallback every
            # time regardless of the endpoint above existing.
            receive_task = asyncio.create_task(websocket.receive_json())
            shutdown_task = asyncio.create_task(_shutdown_event.wait())
            done, pending = await asyncio.wait(
                {receive_task, shutdown_task}, return_when=asyncio.FIRST_COMPLETED
            )

            if shutdown_task in done:
                receive_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await receive_task
                break  # -> finally below runs, same as a real disconnect

            shutdown_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await shutdown_task
            data = receive_task.result()
            msg_type = data.get("type")

            if msg_type == "stop":
                # No-op if nothing's actually running -- a stray/late
                # click after a reply already finished shouldn't do
                # anything or send anything back.
                if current_turn_task is not None and not current_turn_task.done():
                    current_turn_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await current_turn_task
                    # _run_turn's own turn_end send is skipped on a
                    # cancellation (see its own comment) -- send it here
                    # instead, now that the task has actually finished
                    # unwinding and it's safe to use the socket again.
                    await websocket.send_json({"type": "turn_end"})
                continue

            # get_log/clear_log deliberately come before the driver/observer
            # gate below -- the transcript log is one shared, persistent
            # record (memory/store.py's transcript_log table), not part of
            # either connection's own in-memory `history`, so there's no
            # reason an observer window shouldn't be able to read or clear
            # it same as the driver can.
            if msg_type == "get_log":
                turns = await _get_transcript_log_safely()
                await websocket.send_json(
                    {"type": "log", "turns": turns, "user_name": CONFIG.session.user_name}
                )
                continue

            if msg_type == "clear_log":
                await _clear_transcript_log_safely()
                await websocket.send_json(
                    {"type": "log", "turns": [], "user_name": CONFIG.session.user_name}
                )
                continue

            if msg_type == "scene_state":
                # Ambient context only: never starts a turn, never replies,
                # and accepted from any connection (observer or driver) since
                # it describes the world rather than requesting anything.
                global _scene_state
                text = (data.get("text") or "").strip()
                _scene_state = text[:400] or None
                continue

            if msg_type == "set_temp_mode":
                # Quick-action menu's Temp Chat toggle -- per-connection
                # (see temp_mode's own declaration above), so no driver
                # gate needed: an observer toggling it only affects a
                # variable that observer's own connection never actually
                # reads (observers can't start turns at all, see the
                # driver-only check right below). No reply sent -- the
                # frontend already reflects its own toggle state
                # optimistically the moment it's clicked.
                temp_mode = bool(data.get("enabled"))
                continue

            if msg_type == "set_ocr_watch":
                # Quick-action menu's Continuous OCR toggle. Unlike
                # temp_mode, this genuinely is module-level, shared state
                # (ocr_watch.py, same reasoning task_guide's own state
                # has for being module-level rather than per-connection)
                # -- but in practice only the driver's own menu can ever
                # send this, since it's an observer-window UI element
                # this build doesn't even show, so no explicit gate is
                # needed here either.
                ocr_watch.set_active(bool(data.get("enabled")))
                # Logged because the shell used to drop this message silently
                # when its socket wasn't open yet, which looked exactly like
                # "Continuous OCR does nothing"; a receipt line settles it.
                print(
                    f"[luna] ocr watch: {'enabled' if data.get('enabled') else 'disabled'} "
                    f"by the shell (interval {CONFIG.ocr_watch.comment_interval_seconds}s)",
                    file=sys.stderr,
                )
                continue

            if msg_type == "camera_frame":
                # Phase 5 -- the frontend's answer to a camera.py
                # request_camera_frame it sent earlier this turn (see
                # camera.py's own docstring for the full round trip).
                # image_b64 missing/None means the frontend is reporting
                # a failure (camera not armed, getUserMedia permission
                # denied, capture threw) -- resolve_pending_frame treats
                # that the same as a real frame, just with None instead
                # of a string; camera.request_frame turns a None result
                # into the ToolUnavailableError the model actually sees.
                # No driver gate -- same reasoning as set_temp_mode
                # above, and in practice only the driver's own camera
                # module would ever be holding a pending request to
                # answer.
                camera.resolve_pending_frame(data.get("image_b64"))
                continue

            if msg_type in ("user_text", "user_audio", "user_file") and websocket is not _driver:
                # Observer window trying to start a turn -- the frontend
                # should already have disabled its own input for this
                # (see onSurfaceStatus in main.ts/sandbox.ts), so this is
                # defense in depth, not the primary guard. Declined
                # audibly rather than silently dropped, and still sends
                # turn_end so that window's own turnActive-style flag
                # (set optimistically the moment it sent something)
                # doesn't get stuck waiting forever for a reply that's
                # never coming.
                await _send_speak(websocket, apply_persona_pass(SURFACE_BUSY_LINE))
                await websocket.send_json({"type": "turn_end", "emotion": EMOTION_SURFACE_BUSY})
                continue

            if msg_type == "user_text":
                user_text = (data.get("text") or "").strip()
                if not user_text:
                    continue
                if current_turn_task is not None and not current_turn_task.done():
                    # Already mid-reply -- ignore rather than overlap two
                    # turns concurrently mutating the same `history` list.
                    # The frontend should disable input while a reply is
                    # in flight; this is just defense in depth regardless.
                    continue
                current_turn_task = asyncio.create_task(
                    _run_turn(websocket, history, user_text, temp_mode, temp_turn_flags)
                )

            elif msg_type == "user_audio":
                audio_b64 = data.get("audio_b64") or ""
                if not audio_b64:
                    continue
                try:
                    audio_bytes = base64.b64decode(audio_b64)
                except ValueError:
                    # Malformed base64 -- nothing recoverable, drop it.
                    continue

                print(f"[luna] received {len(audio_bytes)} bytes of audio", file=sys.stderr, flush=True)

                if current_turn_task is not None and not current_turn_task.done():
                    # Already mid-reply -- drop this clip without even
                    # transcribing it, same "ignore, don't overlap turns"
                    # policy as user_text above, just applied before
                    # spending STT compute on something we're going to
                    # ignore anyway.
                    continue

                try:
                    user_text = (await stt.transcribe(audio_bytes)).strip()
                except stt.STTError as exc:
                    print(f"[luna] STT failed: {exc}", file=sys.stderr)
                    await websocket.send_json({"type": "transcript", "text": ""})
                    await _send_speak(websocket, apply_persona_pass(STT_UNREACHABLE_LINE))
                    # This path never reaches _run_turn (there's no
                    # transcript to reply to), so it's the one other place
                    # besides _run_turn/the stop handler that has to send
                    # turn_end itself -- without it, the frontend's
                    # turnActive flag (set optimistically the moment audio
                    # was sent, before knowing whether STT would even
                    # succeed) would stay stuck true forever: input
                    # disabled, stop button stuck visible, no further
                    # turn_end ever coming to clear it. Found while adding
                    # Phase 8's emotion tagging, not something anticipated
                    # up front -- see docs/DECISIONS.md.
                    await websocket.send_json({"type": "turn_end", "emotion": EMOTION_STT_UNREACHABLE})
                    continue

                # Always echo the transcript back, even empty, so the
                # frontend can clear its "listening" indicator either way.
                await websocket.send_json({"type": "transcript", "text": user_text})
                if not user_text:
                    # Silence, noise, or nothing intelligible -- nothing to
                    # reply to, and nothing worth adding to history.
                    continue
                current_turn_task = asyncio.create_task(
                    _run_turn(websocket, history, user_text, temp_mode, temp_turn_flags)
                )
            elif msg_type == "user_file":
                # Quick-action menu's Upload Image/File item
                # (src/file-upload.ts classifies + reads/resizes
                # client-side; this just turns the result into a
                # synthetic user_text and runs it through the exact same
                # _run_turn every other kind of turn already uses --
                # recall/forget/task-detection, temp_mode, history,
                # logging, all of it for free, no special-casing needed
                # anywhere else in this file). caption is whatever the
                # user typed in the normal input box alongside the
                # attachment (attach-then-send flow, see main.ts's
                # submitTurn) -- optional; a bare upload with no comment
                # is a real, common case, not a degraded one.
                filename = (data.get("filename") or "uploaded file").strip()
                kind = data.get("kind")
                content = data.get("content") or ""
                caption = (data.get("caption") or "").strip()
                if not content or kind not in ("image", "text"):
                    continue
                if current_turn_task is not None and not current_turn_task.done():
                    # Same "already mid-reply, ignore" policy as
                    # user_text/user_audio above -- the frontend already
                    # guards this (turnActive disables the upload
                    # button), this is defense in depth.
                    continue

                if kind == "text":
                    # Length is already capped client-side
                    # (file-upload.ts's MAX_TEXT_CHARS) -- re-capped here
                    # too, defensively, rather than trusting the client.
                    shared_line = f'(shared a file, "{filename}") {content[:6000]}'
                else:
                    prompt = (
                        "Describe what's in this image in a few "
                        "sentences -- focus on what's actually visible, "
                        "not a generic guess."
                    )
                    try:
                        description = await llm.describe_image(prompt, content)
                    except llm.LLMUnreachableError as exc:
                        print(f"[luna] describe_image failed for upload: {exc}", file=sys.stderr)
                        await _send_speak(websocket, apply_persona_pass(UPLOAD_VISION_UNREACHABLE_LINE))
                        await websocket.send_json(
                            {"type": "turn_end", "emotion": EMOTION_UPLOAD_VISION_UNREACHABLE}
                        )
                        continue
                    shared_line = f'(shared an image, "{filename}") {description}'

                # caption goes on its own line, after the shared-file
                # marker, so a real accompanying comment ("look how
                # clean my desk is") reads as the user's own words next
                # to -- not folded into -- the file's own
                # description/content.
                user_text = f"{shared_line}\n\n{caption}" if caption else shared_line

                current_turn_task = asyncio.create_task(
                    _run_turn(websocket, history, user_text, temp_mode, temp_turn_flags)
                )
            else:
                continue
    except WebSocketDisconnect:
        pass
    finally:
        _active_connections.discard(websocket)
        _connection_surface.pop(websocket, None)
        # This connection was the driver -- promote whichever other
        # connection is still open, if any, so a still-open second window
        # (or a third one that connected while both were up) doesn't stay
        # locked out just because the original driver happened to close
        # first. Note this does *not* hand the promoted connection the old
        # driver's `history` -- each connection keeps its own (unchanged
        # since Phase 2) -- so a promoted observer starts a fresh
        # conversation rather than continuing the old one mid-thread. A
        # real shared-session handoff would need restructuring `history`
        # to live per-character rather than per-connection; not done here
        # -- see docs/DECISIONS.md.
        if websocket is _driver:
            _driver = next(iter(_active_connections), None)
            if _driver is not None:
                with contextlib.suppress(Exception):
                    await _driver.send_json({"type": "surface_status", "role": "driver"})
        # A turn still running when the connection itself goes away (not
        # just a "stop" click) -- cancel it too, same reasoning as
        # above, so consolidation below reads a `history` that's finished
        # settling instead of one an abandoned task might still be
        # mutating mid-append.
        if current_turn_task is not None and not current_turn_task.done():
            current_turn_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await current_turn_task
        # Phase 4 Round 2 -- same reasoning as current_turn_task above,
        # for the task-guide background loop: it's an infinite loop by
        # design (see _task_guide_loop), so it never finishes on its own
        # and always needs an explicit cancel on the way out. None for an
        # observer connection (never started one in the first place).
        # Deliberately does NOT clear task_guide's own module-level
        # TaskState here -- that's independent of any one connection's
        # lifetime, same as _scene_state; a driver reconnecting (e.g. app
        # restart) should still see whatever task was being tracked, not
        # have it silently wiped just because the old socket closed.
        if task_guide_task is not None and not task_guide_task.done():
            task_guide_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task_guide_task
        # Same reasoning as task_guide_task right above, for Continuous
        # OCR's own loop -- also does NOT clear ocr_watch's module-level
        # WatchState, same "independent of any one connection's
        # lifetime" reasoning.
        if ocr_watch_task is not None and not ocr_watch_task.done():
            ocr_watch_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await ocr_watch_task
        # Phase 3: distill this session into durable memory (facts +
        # one episode summary) once it's actually over. In `finally`, not
        # just the `except WebSocketDisconnect` branch, so it also runs on
        # a clean/unexpected exit either way (including the shutdown-event
        # `break` above). Wrapped defensively even though
        # consolidate_session() already catches its own known failure
        # modes internally (LLM/embedding unreachable) -- a session ending
        # should never be blocked by memory work, and an unforeseen bug
        # here shouldn't take down connection teardown.
        #
        # Temp Chat turns are filtered out first -- temp_turn_flags is
        # kept in lockstep with history[1:] (one bool per history.append,
        # see _run_turn's docstring), so this excludes exactly what was
        # said while the toggle was on, not the whole session just
        # because it was touched at some point. history[0] is always the
        # system prompt, never flag-tracked, so it's kept unconditionally.
        history_to_consolidate = [history[0]] + [
            msg for msg, is_temp in zip(history[1:], temp_turn_flags) if not is_temp
        ]
        try:
            await consolidation.consolidate_session(history_to_consolidate)
        except Exception as exc:  # noqa: BLE001 -- see comment above
            print(f"[luna] consolidation failed unexpectedly: {exc}", file=sys.stderr)
        # Only reachable for the shutdown-event break above -- a real
        # WebSocketDisconnect means the socket's already gone. Harmless
        # either way; already-closed is a no-op error we don't care about.
        with contextlib.suppress(Exception):
            await websocket.close()


def _pid_is_alive(pid: int) -> bool:
    """Cross-platform "is this PID still running" check -- the only thing
    _take_over_stale_orchestrator() below actually needs from the OS."""
    if sys.platform == "win32":
        import ctypes

        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        handle = ctypes.windll.kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return False
        ctypes.windll.kernel32.CloseHandle(handle)
        return True
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _take_over_stale_orchestrator(pid_file: Path) -> None:
    """Makes sure this process ends up as the *only* orchestrator running,
    no matter what launched it or how the previous one died.

    Found from the user's own terminal log, not anticipated up front:
    Ctrl+C on `npm run tauri dev` kills the Tauri parent via Windows'
    STATUS_CONTROL_C_EXIT before any of Tauri's own event handlers run --
    graceful_shutdown_then_kill() in src-tauri/src/lib.rs (see this
    module's own top docstring) only ever fires from a tray-Quit click or
    a window-close event, neither of which happens on a raw terminal
    Ctrl+C. The orchestrator child process spawned underneath it is left
    running, still bound to PORT, with whatever history/state it had --
    so the next launch (however it happens: Tauri spawning a fresh child,
    a manually-run `python app.py`, or the sandbox launcher in
    scripts/dev-sandbox.mjs) would otherwise either fail to bind the port
    at all, or -- more confusingly -- silently succeed while the frontend
    keeps talking to the same stale orphan the whole time, which is
    exactly what looked like "orchestrator.log showed old data" from the
    outside.

    Deliberately fixed here rather than on the Rust/Windows side:
    reliably intercepting a raw console Ctrl+C (as opposed to a
    window/tray event) is real, fiddly, hard-to-verify-without-a-Windows-
    machine territory -- see docs/DECISIONS.md. A plain PID-file takeover
    here is portable, was actually tested (simulating a stale orphaned
    process and confirming this detects and kills it -- see
    docs/DECISIONS.md), and is correct regardless of which of the three
    launch paths above is used, rather than needing each one to
    separately get shutdown-signal-handling right.
    """
    our_pid = os.getpid()
    if pid_file.exists():
        try:
            old_pid = int(pid_file.read_text().strip())
        except (ValueError, OSError):
            old_pid = None
        if old_pid is not None and old_pid != our_pid and _pid_is_alive(old_pid):
            print(f"[luna] found a stale orchestrator still running (pid {old_pid}) -- taking over", flush=True)
            if sys.platform == "win32":
                subprocess.run(["taskkill", "/PID", str(old_pid), "/F"], capture_output=True)
            else:
                with contextlib.suppress(OSError):
                    os.kill(old_pid, signal.SIGTERM)
            # Give the OS a moment to actually free the port before we try
            # to bind it ourselves below.
            time.sleep(0.5)
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    pid_file.write_text(str(our_pid))


if __name__ == "__main__":
    import uvicorn

    from memory import db as memory_db

    # Repo-root-relative regardless of cwd (Tauri's spawn sets cwd to
    # orchestrator/ via current_dir("../orchestrator"); a manually-run
    # `python app.py` from inside orchestrator/ lands at the same cwd;
    # __file__-relative means this doesn't depend on either being true).
    _take_over_stale_orchestrator(Path(__file__).resolve().parent.parent / "logs" / "orchestrator.pid")

    # Cheap check, once per process start -- see
    # check_embedding_dimension_matches()'s docstring for why this is
    # worth doing proactively instead of waiting for a mid-turn error the
    # first time someone swaps the embedding model after episodes already
    # exist.
    memory_db.check_embedding_dimension_matches()

    uvicorn.run(app, host=HOST, port=PORT)
