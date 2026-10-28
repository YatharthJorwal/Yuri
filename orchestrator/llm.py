"""
Async streaming client for the configured local LLM server. Supports two
wire protocols, picked by config.yaml's llm.api_style:

- "openai": the generic OpenAI-compatible /v1/chat/completions shape --
  SSE `data: {...}` lines, `[DONE]` sentinel. Works against Ollama or a
  llama.cpp server, since both speak it.
- "ollama_native": Ollama's own /api/chat -- NDJSON (one JSON object per
  line, no `data:` prefix), a `"done": true` flag on the final line
  instead of a sentinel. Ollama-only.

Why bother with two: Ollama's /v1 endpoint unreliably forwards the
"disable thinking" flag for hybrid-thinking models (confirmed via several
open Ollama issues as of mid-2026 -- see docs/DECISIONS.md), where the
native /api/chat is confirmed to honor it. config.yaml's llm.think is only
sent -- and only reliably honored -- in "ollama_native" mode.

Phase 4 added two more entry points, both ollama_native-only (see their
own docstrings for why): stream_reply_with_tools() for the main
conversation's tool-calling loop, and describe_image() for vision.py's
internal screenshot-to-text call. Neither touches stream_reply() above,
which consolidation.py/forget.py's background LLM calls still use
unchanged.

Context window: every native-Ollama request goes through _ollama_options(),
which stamps the same options.num_ctx (config.yaml's llm.num_ctx) on all of
them -- chat, tool-calling, vision, the background classifiers. Left unset,
the window is whatever the server defaults to, and once persona prompt +
tool schemas + memory block + history outgrew it, Ollama ended replies after
a handful of tokens with done_reason 'length' (docs/DECISIONS.md). Keeping
the value identical across calls also matters on its own: Ollama reloads the
model whenever two requests disagree on it. The OpenAI-compatible path has
no per-request way to set it (use OLLAMA_CONTEXT_LENGTH on the server).
"""

from __future__ import annotations

import json
import sys
import time
from typing import Any, AsyncIterator

import httpx

import context_budget
from config import CONFIG


def _ollama_options(**overrides: Any) -> dict[str, Any]:
    """The `options` block for every native-Ollama request. `overrides` may
    change temperature/num_predict (the vision call wants its own), but
    never num_ctx: that one value must be identical across all calls."""
    options: dict[str, Any] = {
        "temperature": CONFIG.llm.temperature,
        "num_predict": CONFIG.llm.max_tokens,
    }
    for key in ("top_p", "top_k", "presence_penalty"):
        value = getattr(CONFIG.llm, key)
        if value is not None:
            options[key] = value
    options.update(overrides)
    if CONFIG.llm.num_ctx is not None:
        options["num_ctx"] = CONFIG.llm.num_ctx
    return options


def explain_length_stop(prompt_tokens: int | None, gen_tokens: int | None) -> str:
    """Why a done_reason='length' reply stopped, from Ollama's own token
    counts. Pure so it can be tested: a reply that ended far below
    max_tokens can only have run out of context window."""
    max_tokens = CONFIG.llm.max_tokens
    num_ctx = CONFIG.llm.num_ctx
    if gen_tokens is not None and gen_tokens >= max_tokens:
        return f"hit the reply cap (llm.max_tokens={max_tokens})"
    if prompt_tokens is not None and num_ctx and prompt_tokens + (gen_tokens or 0) >= num_ctx * 0.95:
        return (
            f"the context window filled (prompt {prompt_tokens} + reply "
            f"{gen_tokens or 0} of num_ctx {num_ctx}) -- history is too big for it"
        )
    return "unknown -- stopped well under both max_tokens and num_ctx"


class LLMUnreachableError(Exception):
    """The configured LLM server couldn't be reached or returned an error
    response -- most likely it just isn't running yet. Kept distinct from
    other exceptions so app.py can give an in-character, specific reply
    instead of the connection crashing silently.

    The two subclasses below say what actually happened, because "I can't
    reach my own brain, is the server even running?" was being said for a
    server that was up and had the model loaded (an HTTP error or a slow
    cold load), with nothing in the log to tell which. Everything that
    catches this base class keeps working."""


class LLMServerError(LLMUnreachableError):
    """The server was reached and answered with an HTTP error (a 400/500 --
    out of memory, an unsupported parameter, a crashed runner...). The body
    is in the message."""


class LLMTimeoutError(LLMUnreachableError):
    """The server was reached but didn't answer in time -- typically the
    first request after startup, while an 8 GB model is still loading."""


def fold_system_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Returns `messages` with every system message after the first folded
    into a user message, so the only system message is the one at index 0.

    Why: app.py splices the per-turn context (memory recall, the screen
    description from a look, task / forget hints, her room) in as a system
    message just before the user's latest message. Ollama's own template for
    the stock qwen3.5 tolerates that, but a GGUF pulled from Hugging Face
    brings the model authors' Jinja chat template, and Qwen3.5's raises
    "System message must be at the beginning" -- an HTTP 500 on every turn
    that had any context (observed on hf.co/Abiray/Qwen3.5-9B-abliterated-
    GGUF). Folding is valid for every template and changes only the copy
    being sent, never `history`.

    A late system message joins the next user message after it (that is where
    app.py puts it: right before the user's turn), else the nearest earlier
    one; with no user message at all it is appended to the leading system
    message. Input is not mutated; a conversation that is already valid is
    returned as an equal copy.
    """
    anchor = 0 if messages and messages[0].get("role") == "system" else None
    late = [i for i, m in enumerate(messages) if m.get("role") == "system" and i != anchor]
    if not late:
        return list(messages)

    out = [dict(m) for m in messages]
    drop: set[int] = set()
    notes: dict[int, list[str]] = {}
    for i in late:
        text = str(out[i].get("content") or "").strip()
        target = next((k for k in range(i + 1, len(out)) if out[k].get("role") == "user"), None)
        if target is None:
            target = next((k for k in range(i - 1, -1, -1) if out[k].get("role") == "user"), None)
        if target is not None:
            drop.add(i)
            if text:
                notes.setdefault(target, []).append(text)
        elif anchor is not None:
            drop.add(i)
            if text:
                out[anchor]["content"] = f"{out[anchor].get('content') or ''}\n\n{text}"
        else:
            out[i]["role"] = "user"  # a lone system message and nothing else to attach it to
    for target, texts in notes.items():
        block = " ".join(texts)
        out[target]["content"] = (
            f"[Context for this reply, not part of what the user said: {block}]\n\n"
            f"{out[target].get('content') or ''}"
        )
    return [m for k, m in enumerate(out) if k not in drop]


def _request_error(url: str, exc: httpx.RequestError) -> LLMUnreachableError:
    if isinstance(exc, httpx.TimeoutException):
        return LLMTimeoutError(f"timed out waiting for {url}: {type(exc).__name__}")
    return LLMUnreachableError(f"couldn't reach {url}: {exc}")


async def stream_reply(messages: list[dict[str, str]]) -> AsyncIterator[str]:
    """Streams content-delta strings from the configured LLM server as they
    arrive. Raises LLMUnreachableError if the server can't be reached or
    errors out."""
    if CONFIG.llm.api_style == "ollama_native":
        gen = _stream_ollama_native(messages)
    else:
        gen = _stream_openai(messages)

    async for delta in gen:
        yield delta


async def stream_reply_with_tools(
    messages: list[dict[str, Any]], tools: list[dict[str, Any]]
) -> AsyncIterator[dict[str, Any]]:
    """Like stream_reply() above, but for turns that have tools available
    -- currently only app.py's main conversation loop. consolidation.py
    and forget.py's internal background calls don't need tools and keep
    using the original stream_reply() completely unchanged, so this is a
    new function rather than a change to that one's signature -- no risk
    of touching their already-tested behavior.

    Yields {"type": "content", "text": str} for ordinary reply text (same
    text stream_reply() would hand back, just wrapped), or, if/when the
    model decides to call a tool instead of replying directly,
    {"type": "tool_calls", "calls": [{"name": str, "arguments": dict}]}
    -- at which point the generator ends (the caller is expected to run
    the tools and start a fresh call with the results appended, see
    app.py's _run_turn).

    ollama_native only -- tool-calling isn't wired for the openai-
    compatible path (Phase 4 was built and tested against Ollama; add
    openai-path support later if that's ever actually needed).

    **Not verified against a real server.** Whether qwen3.5:9b actually
    emits tool_calls reliably through Ollama's *streaming* endpoint (as
    opposed to only in non-streaming responses), and what shape/timing
    those calls arrive in mid-stream, is genuine unknown territory --
    Ollama's own tool-calling-while-streaming behavior has varied across
    versions and models, and this has never run against a real Ollama
    instance. Written defensively because of that: any chunk carrying a
    tool_calls array is treated as the deciding chunk, whatever else is
    in it (see _normalize_tool_calls' own docstring for the parsing side
    of that same defensiveness). If this doesn't hold up in practice, the
    fallback is a non-streaming detect-then-stream two-call shape instead
    (slower, but a much better-trodden path for tool-calling models
    generally) -- see docs/DECISIONS.md.
    """
    if CONFIG.llm.api_style != "ollama_native":
        raise LLMUnreachableError(
            "tool-calling needs api_style: ollama_native -- not wired for "
            "the openai-compatible path"
        )

    messages = fold_system_messages(messages)
    url = f"{CONFIG.llm.base_url.rstrip('/')}/api/chat"
    payload: dict[str, Any] = {
        "model": CONFIG.llm.model,
        "messages": messages,
        "tools": tools,
        "stream": True,
        "options": _ollama_options(),
    }
    if CONFIG.llm.think is not None:
        payload["think"] = CONFIG.llm.think
    if CONFIG.llm.model in _NO_TOOLS_MODELS:
        payload.pop("tools", None)

    # Debug visibility while this is still unverified against a real
    # server (see this function's own docstring) -- one line per call,
    # not per chunk, so cheap enough to leave in rather than strip out
    # once this is confirmed working. Directly answers the question "did
    # Ollama even attempt this" the next time she doesn't use a tool she
    # should have -- print(s) go to the orchestrator's own terminal, not
    # anywhere the user would see them mid-conversation.
    tool_names = [
        t["function"]["name"]
        for t in tools
        if isinstance(t, dict) and isinstance(t.get("function"), dict)
    ]
    saw_tool_calls_key = False
    saw_any_content = False
    # Tracked for the truncation diagnostic printed below: replies were
    # observed ending mid-sentence on the user's real machine with no user
    # action and no error anywhere in the log, and a stream that just *ends*
    # (Ollama aborting a generation) used to be indistinguishable here from
    # one that finished normally.
    done_seen = False
    done_reason: Any = None
    chars_streamed = 0
    # Ollama reports real token counts on the final chunk; logging them is
    # what turns "she cut off" from a guess into a diagnosis.
    prompt_tokens: int | None = None
    gen_tokens: int | None = None
    for attempt in range(2):
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=5.0)) as client:
                async with client.stream("POST", url, json=payload) as response:
                    if response.status_code >= 400:
                        body = await response.aread()
                        if attempt == 0 and "tools" in payload and _is_no_tools_error(body):
                            _NO_TOOLS_MODELS.add(CONFIG.llm.model)
                            payload.pop("tools", None)
                            tool_names = []
                            print(
                                f"[luna] WARNING: '{CONFIG.llm.model}' doesn't support tool-calling "
                                "(Ollama said so); retrying this turn without tools and not "
                                "offering them again. Look/camera requests still work via the "
                                "regex gate.",
                                file=sys.stderr,
                            )
                            continue
                        raise LLMServerError(
                            f"{response.status_code} from {url}: {body[:300]!r}"
                        )
                    async for line in response.aiter_lines():
                        line = line.strip()
                        if not line:
                            continue
                        chunk = _parse_json(line)
                        if chunk is None:
                            continue
                        message = chunk.get("message")
                        if isinstance(message, dict):
                            if "tool_calls" in message:
                                saw_tool_calls_key = True
                            calls = _normalize_tool_calls(message.get("tool_calls"))
                            if calls:
                                print(
                                    f"[luna] tool-calling: model called "
                                    f"{[c['name'] for c in calls]}",
                                    file=sys.stderr,
                                )
                                yield {"type": "tool_calls", "calls": calls}
                                return
                            content = message.get("content") or ""
                            if content:
                                saw_any_content = True
                                chars_streamed += len(content)
                                yield {"type": "content", "text": content}
                        if chunk.get("done"):
                            done_seen = True
                            done_reason = chunk.get("done_reason")
                            prompt_tokens = _int_or_none(chunk.get("prompt_eval_count"))
                            gen_tokens = _int_or_none(chunk.get("eval_count"))
                            break
        except httpx.RequestError as exc:
            raise _request_error(url, exc) from exc
        break  # finished normally; only the no-tools retry above `continue`s
    # Raw estimate for exactly what was sent, logged beside Ollama's real
    # count and fed to the calibration (context_budget.observe).
    sent_tools = payload.get("tools")
    raw_estimate = context_budget.estimate_prompt(messages, sent_tools)
    context_budget.observe(raw_estimate, prompt_tokens)
    print(
        f"[luna] tool-calling: offered {tool_names}, replied directly "
        f"(saw_content={saw_any_content}, "
        f"'tool_calls' key ever present={saw_tool_calls_key}, "
        f"done={done_seen}, done_reason={done_reason!r}, chars={chars_streamed}, "
        f"prompt_tokens={prompt_tokens}, gen_tokens={gen_tokens}, "
        f"num_ctx={CONFIG.llm.num_ctx}, est_prompt_tokens={raw_estimate}, "
        f"calibration={context_budget.calibration_ratio():.2f})",
        file=sys.stderr,
    )
    if done_reason == "length":
        print(
            "[luna] WARNING: reply stopped by 'length' -- "
            f"{explain_length_stop(prompt_tokens, gen_tokens)}",
            file=sys.stderr,
        )
    elif (
        CONFIG.llm.num_ctx
        and prompt_tokens is not None
        and prompt_tokens >= CONFIG.llm.num_ctx * 0.85
    ):
        print(
            f"[luna] WARNING: context nearly full (prompt {prompt_tokens} of "
            f"num_ctx {CONFIG.llm.num_ctx}) -- the next reply may be cut short.",
            file=sys.stderr,
        )
    if not done_seen or done_reason not in (None, "stop"):
        # The smoking gun for a reply that was cut off by the *model side*
        # rather than by the user or a bug in this app: either the stream
        # ended with no done marker at all (Ollama aborted mid-generation),
        # or it ended for a reason other than a natural stop (e.g.
        # "length" = hit the max_tokens cap).
        print(
            f"[luna] WARNING: LLM reply ended abnormally (done={done_seen}, "
            f"done_reason={done_reason!r}) after {chars_streamed} chars -- "
            "if she cut off mid-sentence, this is why.",
            file=sys.stderr,
        )


# Models Ollama has told us can't do tool-calling (HTTP 400 "does not support
# tools" -- typical of a community GGUF pulled with `hf.co/...`, whose chat
# template Ollama can't map to its tools format). Without this, such a model
# makes EVERY chat turn fail with an LLM-unreachable error. Remembered per
# model for the process so only the first turn pays for the retry.
_NO_TOOLS_MODELS: set[str] = set()

# Same idea for structured output (the `format` JSON schema on image checks):
# if a model/server rejects it, the call is retried once without and the
# model remembered, so the old best-effort prompt-only behavior is the floor.
_NO_FORMAT_MODELS: set[str] = set()


def _is_no_tools_error(body: bytes | str) -> bool:
    text = body.decode("utf-8", "replace") if isinstance(body, bytes) else body
    return "does not support tools" in text.lower()


def _vision_timing_summary(model: str, image_b64: str, elapsed_s: float, data: dict[str, Any]) -> str:
    """One log line for a vision call, from Ollama's own timing fields
    (nanoseconds). Separates the three possible reasons a look is slow:
    `load` = the model had to be (re)loaded into VRAM, `prompt_tokens` /
    `prompt_eval` = how much the image cost to read (image tokens grow with
    pixel count), `gen` = writing the answer."""

    def secs(key: str) -> str:
        value = data.get(key)
        return f"{value / 1e9:.1f}s" if isinstance(value, (int, float)) else "?"

    return (
        f"[luna] vision: model={model} image={len(image_b64) // 1024}KB total={elapsed_s:.1f}s "
        f"load={secs('load_duration')} prompt_tokens={data.get('prompt_eval_count')} "
        f"prompt_eval={secs('prompt_eval_duration')} gen_tokens={data.get('eval_count')} "
        f"gen={secs('eval_duration')}"
    )


def vision_model_name() -> str:
    """The model that reads images: llm.vision_model when set, else the chat
    model (the original single-model setup)."""
    return CONFIG.llm.vision_model or CONFIG.llm.model


def capability_warnings(
    chat_model: str,
    chat_caps: list[str] | None,
    vision_model: str,
    vision_caps: list[str] | None,
) -> list[str]:
    """Human-readable problems from `ollama show` capability lists (None =
    unknown, e.g. an older Ollama that doesn't report them: say nothing)."""
    out: list[str] = []
    if vision_caps is not None and "vision" not in vision_caps:
        out.append(
            f"'{vision_model}' reports no 'vision' capability, so looking at the screen, "
            "Continuous OCR, Task Guide, the camera and image uploads will fail. Set "
            "llm.vision_model in config.yaml to a model that can read images "
            "(e.g. qwen3.5:4b), or use a build of your model that includes vision."
        )
    if chat_caps is not None and "tools" not in chat_caps:
        out.append(
            f"'{chat_model}' reports no 'tools' capability. Chat still works (Luna retries "
            "without tools) and look/camera requests are handled by the regex gate, but the "
            "model can't call tools on its own."
        )
    return out


async def model_capabilities(model: str) -> list[str] | None:
    """Ollama's capability list for `model` (e.g. completion, vision, tools,
    thinking) via POST /api/show, or None if the server is unreachable or too
    old to report them. Diagnostic only -- never raises."""
    if CONFIG.llm.api_style != "ollama_native":
        return None
    url = f"{CONFIG.llm.base_url.rstrip('/')}/api/show"
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(10.0, connect=3.0)) as client:
            response = await client.post(url, json={"model": model})
            if response.status_code >= 400:
                return None
            caps = response.json().get("capabilities")
    except (httpx.RequestError, ValueError):
        return None
    return [str(c) for c in caps] if isinstance(caps, list) else None


async def warm_up_model() -> None:
    """Loads the chat model into VRAM at startup (an empty-prompt
    /api/generate is Ollama's documented way to do that) with the SAME
    num_ctx every real request uses, so the first message doesn't pay for an
    8 GB cold load -- which is long enough to hit the request timeout and was
    the likely cause of the first message after startup answering "I can't
    reach my own brain" while the model finished loading in the background.
    Best-effort and silent on success apart from one log line; never raises."""
    if CONFIG.llm.api_style != "ollama_native" or not CONFIG.llm.warm_up_on_start:
        return
    url = f"{CONFIG.llm.base_url.rstrip('/')}/api/generate"
    payload = {
        "model": CONFIG.llm.model,
        "prompt": "",
        "stream": False,
        "options": _ollama_options(),
    }
    started = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=5.0)) as client:
            response = await client.post(url, json=payload)
        if response.status_code >= 400:
            print(
                f"[luna] warm-up: {response.status_code} from {url}: {response.text[:300]!r}",
                file=sys.stderr,
            )
            return
        print(
            f"[luna] warm-up: '{CONFIG.llm.model}' loaded in {time.monotonic() - started:.1f}s "
            f"(num_ctx {CONFIG.llm.num_ctx})",
            file=sys.stderr,
        )
    except Exception as exc:  # noqa: BLE001 -- best-effort; the first real turn will surface a real problem
        print(f"[luna] warm-up skipped: {type(exc).__name__}: {exc}", file=sys.stderr)


async def report_model_capabilities() -> None:
    """Startup check: prints what Ollama says the configured model(s) can do
    and warns about the mismatches that would otherwise only show up as
    mysterious failures mid-conversation. Never raises."""
    try:
        chat = CONFIG.llm.model
        vision = vision_model_name()
        chat_caps = await model_capabilities(chat)
        vision_caps = chat_caps if vision == chat else await model_capabilities(vision)
        print(
            f"[luna] model capabilities: chat '{chat}' = {chat_caps if chat_caps is not None else 'unknown'}"
            + ("" if vision == chat else f"; vision '{vision}' = {vision_caps if vision_caps is not None else 'unknown'}"),
            file=sys.stderr,
        )
        for warning in capability_warnings(chat, chat_caps, vision, vision_caps):
            print(f"[luna] WARNING: {warning}", file=sys.stderr)
    except Exception as exc:  # noqa: BLE001 -- diagnostics must never stop startup
        print(f"[luna] model capability check skipped: {exc}", file=sys.stderr)


def _int_or_none(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _normalize_tool_calls(raw_calls: Any) -> list[dict[str, Any]]:
    """Defensive parsing on purpose -- see stream_reply_with_tools'
    docstring for why this is genuinely untested against a real server.
    A malformed entry is skipped rather than crashing the turn; a chunk
    with only malformed entries just yields no tool call at all, and the
    caller falls through to treating it as ordinary content (there won't
    be any text either in that case, so it's effectively a no-op round --
    covered by app.py's MAX_TOOL_ROUNDS cap). Expected shape per entry,
    per Ollama's/OpenAI's tool-calling convention both converged on:
    {"function": {"name": str, "arguments": dict}}."""
    if not isinstance(raw_calls, list):
        return []
    calls = []
    for raw in raw_calls:
        if not isinstance(raw, dict):
            continue
        function = raw.get("function")
        if not isinstance(function, dict):
            continue
        name = function.get("name")
        if not isinstance(name, str) or not name:
            continue
        arguments = function.get("arguments")
        if not isinstance(arguments, dict):
            arguments = {}
        calls.append({"name": name, "arguments": arguments})
    return calls


async def describe_image(
    prompt: str, image_b64: str, json_schema: dict[str, Any] | None = None
) -> str:
    """One-shot, non-streaming multimodal call -- used internally by
    tools/vision.py's describe_screen() to turn a screenshot into the
    text description ARCHITECTURE.md's vision-tools section calls for
    ("returns a textual analysis (not raw pixels) to the reasoning
    pass"). Not part of the main conversation's own streamed reply, and
    not visible to app.py's tool-calling loop at all -- from that loop's
    point of view, capture_screen is just a tool that returns a string,
    same as read_clipboard.

    ollama_native only, same reasoning as stream_reply_with_tools above
    -- image support here specifically relies on Ollama's native
    "images": [base64...] field on a message, which isn't part of the
    generic OpenAI-compatible shape this project also supports.

    Uses llm.vision_model when set, else llm.model (see vision_model_name).
    Every image path -- screen looks, Continuous OCR, Task Guide, camera,
    image uploads -- goes through here; the main chat model never receives
    pixels, so it can be a text-only model. Logs a `[luna] vision:` timing
    line per call (model, image size, load / prompt-eval / generation time).

    json_schema: when given (the Task Guide and Continuous OCR checks, which
    must come back as one small JSON object), it is sent as Ollama's
    structured-output `format`, which constrains decoding so the reply parses
    whatever the model's habits are -- the abliterated GGUF ignored "reply
    with ONLY a JSON object" on 15 of 16 checks. Sampling is also made
    stricter for these calls (low temperature, no presence penalty, which
    punishes the repeated quotes and braces JSON is made of). If the server
    rejects `format`, the call is retried once without it and the model
    remembered; the old prompt-only behavior is the floor.
    """
    if CONFIG.llm.api_style != "ollama_native":
        raise LLMUnreachableError(
            "vision tools need api_style: ollama_native -- image support "
            "isn't wired for the openai-compatible path"
        )

    url = f"{CONFIG.llm.base_url.rstrip('/')}/api/chat"
    model = vision_model_name()
    use_schema = json_schema is not None and model not in _NO_FORMAT_MODELS
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt, "images": [image_b64]}],
        "stream": False,
        "options": (
            _ollama_options(temperature=0.2, num_predict=400, presence_penalty=0.0)
            if json_schema is not None
            else _ollama_options(temperature=0.4, num_predict=400, presence_penalty=0.0)
        ),
    }
    if use_schema:
        payload["format"] = json_schema
    if CONFIG.llm.think is not None:
        payload["think"] = CONFIG.llm.think

    started = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=5.0)) as client:
            response = await client.post(url, json=payload)
            if response.status_code >= 400 and use_schema:
                first_error = f"{response.status_code}: {response.text[:200]!r}"
                payload.pop("format", None)
                response = await client.post(url, json=payload)
                if response.status_code < 400:
                    # Only blame (and stop sending) the schema if dropping it
                    # is what made the call work; a different failure, e.g.
                    # a model with no vision, fails the retry too and is
                    # reported below as itself.
                    _NO_FORMAT_MODELS.add(model)
                    print(
                        f"[luna] WARNING: '{model}' rejected the structured-output schema "
                        f"({first_error}); it worked without it, so the schema won't be sent "
                        "again. The prompt alone has to produce the JSON.",
                        file=sys.stderr,
                    )
            if response.status_code >= 400:
                hint = ""
                lowered = response.text.lower()
                if "image" in lowered or "vision" in lowered or "mmproj" in lowered:
                    hint = (
                        f" -- '{model}' probably has no vision support; set llm.vision_model "
                        "in config.yaml to a model that can read images"
                    )
                raise LLMServerError(
                    f"{response.status_code} from {url}: {response.text[:300]!r}{hint}"
                )
            data = response.json()
    except httpx.RequestError as exc:
        raise _request_error(url, exc) from exc
    print(_vision_timing_summary(model, image_b64, time.monotonic() - started, data), file=sys.stderr)

    message = data.get("message")
    if not isinstance(message, dict):
        return ""
    return (message.get("content") or "").strip()


async def _stream_openai(messages: list[dict[str, str]]) -> AsyncIterator[str]:
    messages = fold_system_messages(messages)
    url = f"{CONFIG.llm.base_url.rstrip('/')}/v1/chat/completions"
    payload = {
        "model": CONFIG.llm.model,
        "messages": messages,
        "temperature": CONFIG.llm.temperature,
        "max_tokens": CONFIG.llm.max_tokens,
        "stream": True,
    }
    headers = {"Authorization": f"Bearer {CONFIG.llm.api_key}"}

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=5.0)) as client:
            async with client.stream("POST", url, json=payload, headers=headers) as response:
                if response.status_code >= 400:
                    body = await response.aread()
                    raise LLMServerError(
                        f"{response.status_code} from {url}: {body[:300]!r}"
                    )
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = line[len("data: "):].strip()
                    if data == "[DONE]":
                        break
                    chunk = _parse_json(data)
                    delta = _extract_openai_delta(chunk)
                    if delta:
                        yield delta
    except httpx.RequestError as exc:
        raise _request_error(url, exc) from exc


async def _stream_ollama_native(messages: list[dict[str, str]]) -> AsyncIterator[str]:
    messages = fold_system_messages(messages)
    url = f"{CONFIG.llm.base_url.rstrip('/')}/api/chat"
    payload: dict[str, Any] = {
        "model": CONFIG.llm.model,
        "messages": messages,
        "stream": True,
        "options": _ollama_options(),
    }
    if CONFIG.llm.think is not None:
        payload["think"] = CONFIG.llm.think

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=5.0)) as client:
            async with client.stream("POST", url, json=payload) as response:
                if response.status_code >= 400:
                    body = await response.aread()
                    raise LLMServerError(
                        f"{response.status_code} from {url}: {body[:300]!r}"
                    )
                async for line in response.aiter_lines():
                    line = line.strip()
                    if not line:
                        continue
                    chunk = _parse_json(line)
                    if chunk is None:
                        continue
                    delta = _extract_ollama_native_delta(chunk)
                    if delta:
                        yield delta
                    if chunk.get("done"):
                        break
    except httpx.RequestError as exc:
        raise _request_error(url, exc) from exc


def _parse_json(data: str) -> dict[str, Any] | None:
    try:
        return json.loads(data)
    except json.JSONDecodeError:
        return None


def _extract_openai_delta(chunk: dict[str, Any] | None) -> str:
    if not chunk:
        return ""
    try:
        return chunk["choices"][0]["delta"].get("content") or ""
    except (KeyError, IndexError, TypeError):
        return ""


def _extract_ollama_native_delta(chunk: dict[str, Any]) -> str:
    # Deliberately only "content", never "thinking" -- with think:false
    # this should be empty/absent anyway, but even if a model ignores the
    # setting, we never want to speak the reasoning phase aloud.
    message = chunk.get("message")
    if not isinstance(message, dict):
        return ""
    return message.get("content") or ""
