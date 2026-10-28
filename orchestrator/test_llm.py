"""
Tests for llm.py's Phase 4 additions. Only _normalize_tool_calls is
covered here -- it's the one piece of stream_reply_with_tools/
describe_image that's pure logic with no network/event-loop involved, so
it's the one piece that can be genuinely tested rather than just reasoned
about in this sandbox (no real Ollama server to talk to -- see both
functions' own docstrings for what's still unverified as a result).
"""

from llm import _normalize_tool_calls


def test_normalize_tool_calls_happy_path():
    raw = [{"function": {"name": "capture_screen", "arguments": {}}}]
    assert _normalize_tool_calls(raw) == [{"name": "capture_screen", "arguments": {}}]


def test_normalize_tool_calls_multiple():
    raw = [
        {"function": {"name": "capture_screen", "arguments": {}}},
        {"function": {"name": "read_clipboard", "arguments": {}}},
    ]
    assert _normalize_tool_calls(raw) == [
        {"name": "capture_screen", "arguments": {}},
        {"name": "read_clipboard", "arguments": {}},
    ]


def test_normalize_tool_calls_missing_arguments_defaults_to_empty_dict():
    raw = [{"function": {"name": "capture_screen"}}]
    assert _normalize_tool_calls(raw) == [{"name": "capture_screen", "arguments": {}}]


def test_normalize_tool_calls_arguments_wrong_type_defaults_to_empty_dict():
    # A model occasionally emitting arguments as a JSON string instead of
    # an object is a real enough failure mode for a local 9B model to
    # guard against, not just a theoretical one.
    raw = [{"function": {"name": "capture_screen", "arguments": "{}"}}]
    assert _normalize_tool_calls(raw) == [{"name": "capture_screen", "arguments": {}}]


def test_normalize_tool_calls_skips_entries_missing_name():
    raw = [{"function": {"arguments": {}}}]
    assert _normalize_tool_calls(raw) == []


def test_normalize_tool_calls_skips_entries_with_blank_name():
    raw = [{"function": {"name": "", "arguments": {}}}]
    assert _normalize_tool_calls(raw) == []


def test_normalize_tool_calls_skips_malformed_entries_keeps_valid_ones():
    raw = [
        "not even a dict",
        {"no_function_key": True},
        {"function": "not a dict either"},
        {"function": {"name": "read_clipboard", "arguments": {}}},
    ]
    assert _normalize_tool_calls(raw) == [{"name": "read_clipboard", "arguments": {}}]


def test_normalize_tool_calls_none_input():
    assert _normalize_tool_calls(None) == []


def test_normalize_tool_calls_not_a_list():
    assert _normalize_tool_calls({"function": {"name": "x"}}) == []


def test_normalize_tool_calls_empty_list():
    assert _normalize_tool_calls([]) == []


# --- context window (num_ctx) ---------------------------------------------
# Replies were being cut off with done_reason='length' far below max_tokens
# because nothing set or bounded the context window. These pin the pieces:
# one options helper stamps the same num_ctx on every native request, and
# the token counts Ollama reports are what the log and the diagnosis use.

import dataclasses
import json

import httpx
import pytest

import llm


def _with_num_ctx(monkeypatch, value):
    cfg = dataclasses.replace(llm.CONFIG, llm=dataclasses.replace(llm.CONFIG.llm, num_ctx=value))
    monkeypatch.setattr(llm, "CONFIG", cfg)


def test_ollama_options_stamps_num_ctx(monkeypatch):
    _with_num_ctx(monkeypatch, 8192)
    options = llm._ollama_options()
    assert options["num_ctx"] == 8192
    assert options["num_predict"] == llm.CONFIG.llm.max_tokens


def test_ollama_options_vision_overrides_keep_same_num_ctx(monkeypatch):
    # The vision call tunes temperature/num_predict but must NOT differ on
    # num_ctx, or Ollama reloads the model between a chat and a screen read.
    _with_num_ctx(monkeypatch, 8192)
    chat = llm._ollama_options()
    vision = llm._ollama_options(temperature=0.4, num_predict=400)
    assert vision["temperature"] == 0.4 and vision["num_predict"] == 400
    assert vision["num_ctx"] == chat["num_ctx"] == 8192


def test_ollama_options_null_num_ctx_is_omitted(monkeypatch):
    _with_num_ctx(monkeypatch, None)
    assert "num_ctx" not in llm._ollama_options()


def test_explain_length_stop_context_full(monkeypatch):
    _with_num_ctx(monkeypatch, 4096)
    assert "context window filled" in llm.explain_length_stop(4080, 12)


def test_explain_length_stop_reply_cap(monkeypatch):
    _with_num_ctx(monkeypatch, 8192)
    assert "reply cap" in llm.explain_length_stop(1500, llm.CONFIG.llm.max_tokens)


def test_explain_length_stop_unknown(monkeypatch):
    _with_num_ctx(monkeypatch, 8192)
    assert "unknown" in llm.explain_length_stop(1000, 20)


def _ndjson(*chunks):
    return "\n".join(json.dumps(c) for c in chunks).encode()


@pytest.mark.asyncio
async def test_stream_reply_with_tools_sends_num_ctx_and_logs_token_counts(monkeypatch, capsys):
    _with_num_ctx(monkeypatch, 8192)
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["payload"] = json.loads(request.content)
        return httpx.Response(
            200,
            content=_ndjson(
                {"message": {"content": "Hi."}, "done": False},
                {
                    "message": {"content": ""},
                    "done": True,
                    "done_reason": "length",
                    "prompt_eval_count": 8100,
                    "eval_count": 9,
                },
            ),
        )

    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        llm.httpx,
        "AsyncClient",
        lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw),
    )
    events = [e async for e in llm.stream_reply_with_tools([{"role": "user", "content": "x"}], [])]
    assert events == [{"type": "content", "text": "Hi."}]
    assert seen["payload"]["options"]["num_ctx"] == 8192

    err = capsys.readouterr().err
    assert "prompt_tokens=8100" in err and "gen_tokens=9" in err and "num_ctx=8192" in err
    assert "reply stopped by 'length'" in err and "context window filled" in err


@pytest.mark.asyncio
async def test_stream_reply_with_tools_warns_when_context_nearly_full(monkeypatch, capsys):
    _with_num_ctx(monkeypatch, 8192)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=_ndjson(
                {"message": {"content": "Fine."}, "done": False},
                {"done": True, "done_reason": "stop", "prompt_eval_count": 7400, "eval_count": 5},
            ),
        )

    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        llm.httpx,
        "AsyncClient",
        lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw),
    )
    _ = [e async for e in llm.stream_reply_with_tools([{"role": "user", "content": "x"}], [])]
    assert "context nearly full" in capsys.readouterr().err


# --- separate vision model, no-tools fallback, capability report ------------


def _patch_llm_config(monkeypatch, **llm_overrides):
    cfg = dataclasses.replace(
        llm.CONFIG, llm=dataclasses.replace(llm.CONFIG.llm, **llm_overrides)
    )
    monkeypatch.setattr(llm, "CONFIG", cfg)


_REAL_ASYNC_CLIENT = httpx.AsyncClient  # captured once: a second patch must not wrap the first


def _mock_http(monkeypatch, handler):
    monkeypatch.setattr(
        llm.httpx,
        "AsyncClient",
        lambda **kw: _REAL_ASYNC_CLIENT(transport=httpx.MockTransport(handler), **kw),
    )


def test_vision_model_name_falls_back_to_the_chat_model(monkeypatch):
    _patch_llm_config(monkeypatch, vision_model=None, model="chat-model")
    assert llm.vision_model_name() == "chat-model"
    _patch_llm_config(monkeypatch, vision_model="small-vlm", model="chat-model")
    assert llm.vision_model_name() == "small-vlm"


@pytest.mark.asyncio
async def test_describe_image_sends_the_vision_model_and_logs_timing(monkeypatch, capsys):
    _patch_llm_config(monkeypatch, vision_model="small-vlm", model="chat-model")
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["payload"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "message": {"content": " a cake "},
                "load_duration": 2_500_000_000,
                "prompt_eval_count": 2650,
                "prompt_eval_duration": 3_100_000_000,
                "eval_count": 40,
                "eval_duration": 900_000_000,
            },
        )

    _mock_http(monkeypatch, handler)
    assert await llm.describe_image("what is this", "aGVsbG8=") == "a cake"
    assert seen["payload"]["model"] == "small-vlm"
    err = capsys.readouterr().err
    assert "[luna] vision: model=small-vlm" in err
    assert "load=2.5s" in err and "prompt_tokens=2650" in err and "prompt_eval=3.1s" in err


@pytest.mark.asyncio
async def test_describe_image_explains_a_model_without_vision(monkeypatch):
    _patch_llm_config(monkeypatch, vision_model=None, model="text-only")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, text='{"error":"this model is missing data required for image input"}')

    _mock_http(monkeypatch, handler)
    with pytest.raises(llm.LLMUnreachableError) as err:
        await llm.describe_image("x", "aGk=")
    assert "llm.vision_model" in str(err.value)


@pytest.mark.asyncio
async def test_tool_calling_retries_without_tools_when_the_model_has_none(monkeypatch, capsys):
    _patch_llm_config(monkeypatch, model="community-gguf")
    llm._NO_TOOLS_MODELS.clear()
    payloads = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        payloads.append(body)
        if "tools" in body:
            return httpx.Response(400, text='{"error":"registry.ollama.ai/x does not support tools"}')
        return httpx.Response(
            200,
            content=_ndjson(
                {"message": {"content": "Hello."}, "done": False},
                {"done": True, "done_reason": "stop", "prompt_eval_count": 100, "eval_count": 3},
            ),
        )

    _mock_http(monkeypatch, handler)
    tools = [{"type": "function", "function": {"name": "t", "description": "d", "parameters": {}}}]
    first = [e async for e in llm.stream_reply_with_tools([{"role": "user", "content": "hi"}], tools)]
    assert first == [{"type": "content", "text": "Hello."}]
    assert "tools" in payloads[0] and "tools" not in payloads[1]  # retried without them
    assert "doesn't support tool-calling" in capsys.readouterr().err

    # remembered: the next turn doesn't pay for the failed attempt again
    payloads.clear()
    second = [e async for e in llm.stream_reply_with_tools([{"role": "user", "content": "again"}], tools)]
    assert second == [{"type": "content", "text": "Hello."}]
    assert len(payloads) == 1 and "tools" not in payloads[0]
    llm._NO_TOOLS_MODELS.clear()


@pytest.mark.asyncio
async def test_other_400s_are_still_errors_not_silent_retries(monkeypatch):
    _patch_llm_config(monkeypatch, model="community-gguf")
    llm._NO_TOOLS_MODELS.clear()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, text='{"error":"something else is wrong"}')

    _mock_http(monkeypatch, handler)
    with pytest.raises(llm.LLMUnreachableError):
        _ = [e async for e in llm.stream_reply_with_tools([{"role": "user", "content": "x"}], [{"type": "function", "function": {"name": "t"}}])]
    assert "community-gguf" not in llm._NO_TOOLS_MODELS


def test_capability_warnings_flag_missing_vision_and_tools():
    warnings = llm.capability_warnings("chat", ["completion"], "chat", ["completion"])
    assert any("no 'vision' capability" in w and "llm.vision_model" in w for w in warnings)
    assert any("no 'tools' capability" in w for w in warnings)


def test_capability_warnings_quiet_when_all_is_present_or_unknown():
    assert llm.capability_warnings("m", ["completion", "vision", "tools"], "m", ["completion", "vision", "tools"]) == []
    assert llm.capability_warnings("m", None, "m", None) == []  # old Ollama: say nothing


@pytest.mark.asyncio
async def test_model_capabilities_reads_api_show(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/show"
        return httpx.Response(200, json={"capabilities": ["completion", "vision", "tools"]})

    _mock_http(monkeypatch, handler)
    assert await llm.model_capabilities("qwen3.5:9b") == ["completion", "vision", "tools"]


@pytest.mark.asyncio
async def test_model_capabilities_unreachable_or_old_server_is_none(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"license": "x"})  # older Ollama: no field

    _mock_http(monkeypatch, handler)
    assert await llm.model_capabilities("m") is None

    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    _mock_http(monkeypatch, boom)
    assert await llm.model_capabilities("m") is None


def test_vision_timing_summary_tolerates_missing_fields():
    line = llm._vision_timing_summary("m", "a" * 2048, 4.2, {})
    assert "model=m" in line and "image=2KB" in line and "load=?" in line


@pytest.mark.asyncio
async def test_turn_log_pairs_the_estimate_with_the_real_count_and_calibrates(monkeypatch, capsys):
    import context_budget

    context_budget.reset_calibration()
    _with_num_ctx(monkeypatch, 8192)
    big = [{"role": "system", "content": "p" * 9000}, {"role": "user", "content": "u" * 6000}]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=_ndjson(
                {"message": {"content": "ok"}, "done": False},
                {"done": True, "done_reason": "stop", "prompt_eval_count": 3300, "eval_count": 2},
            ),
        )

    _mock_http(monkeypatch, handler)
    _ = [e async for e in llm.stream_reply_with_tools(big, [])]
    err = capsys.readouterr().err
    assert "est_prompt_tokens=" in err and "prompt_tokens=3300" in err and "calibration=0.77" in err
    assert context_budget.calibration_ratio() < 1.0
    context_budget.reset_calibration()


# --- failure classes, so "can't reach my own brain" stops lying -------------


@pytest.mark.asyncio
async def test_http_error_from_a_running_server_is_a_server_error_with_the_body(monkeypatch):
    _patch_llm_config(monkeypatch, model="m")
    llm._NO_TOOLS_MODELS.clear()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text='{"error":"CUDA error: out of memory"}')

    _mock_http(monkeypatch, handler)
    with pytest.raises(llm.LLMServerError) as err:
        _ = [e async for e in llm.stream_reply_with_tools([{"role": "user", "content": "x"}], [])]
    assert "500" in str(err.value) and "out of memory" in str(err.value)
    assert isinstance(err.value, llm.LLMUnreachableError)  # existing handlers still catch it


@pytest.mark.asyncio
async def test_a_slow_cold_load_is_a_timeout_error_not_unreachable(monkeypatch):
    _patch_llm_config(monkeypatch, model="m")

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("model still loading")

    _mock_http(monkeypatch, handler)
    with pytest.raises(llm.LLMTimeoutError):
        _ = [e async for e in llm.stream_reply_with_tools([{"role": "user", "content": "x"}], [])]


@pytest.mark.asyncio
async def test_a_refused_connection_stays_plain_unreachable(monkeypatch):
    _patch_llm_config(monkeypatch, model="m")

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    _mock_http(monkeypatch, handler)
    with pytest.raises(llm.LLMUnreachableError) as err:
        _ = [e async for e in llm.stream_reply_with_tools([{"role": "user", "content": "x"}], [])]
    assert not isinstance(err.value, (llm.LLMServerError, llm.LLMTimeoutError))


@pytest.mark.asyncio
async def test_describe_image_http_error_is_a_server_error(monkeypatch):
    _patch_llm_config(monkeypatch, vision_model=None, model="m")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    _mock_http(monkeypatch, handler)
    with pytest.raises(llm.LLMServerError):
        await llm.describe_image("x", "aGk=")


# --- warm-up ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_warm_up_loads_the_model_with_the_same_num_ctx(monkeypatch, capsys):
    _patch_llm_config(monkeypatch, model="big-model", num_ctx=8192, warm_up_on_start=True)
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["payload"] = json.loads(request.content)
        return httpx.Response(200, json={"done": True})

    _mock_http(monkeypatch, handler)
    await llm.warm_up_model()
    assert seen["path"] == "/api/generate"
    assert seen["payload"]["prompt"] == ""  # empty prompt = just load it
    assert seen["payload"]["model"] == "big-model"
    assert seen["payload"]["options"]["num_ctx"] == 8192  # a different value would force a reload
    assert "warm-up: 'big-model' loaded" in capsys.readouterr().err


@pytest.mark.asyncio
async def test_warm_up_can_be_disabled(monkeypatch):
    _patch_llm_config(monkeypatch, warm_up_on_start=False)

    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("must not call the server")

    _mock_http(monkeypatch, handler)
    await llm.warm_up_model()


@pytest.mark.asyncio
async def test_warm_up_never_raises(monkeypatch, capsys):
    _patch_llm_config(monkeypatch, warm_up_on_start=True)

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("ollama not started yet")

    _mock_http(monkeypatch, handler)
    await llm.warm_up_model()
    assert "warm-up skipped" in capsys.readouterr().err

    def handler500(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="no memory")

    _mock_http(monkeypatch, handler500)
    await llm.warm_up_model()
    assert "warm-up: 500" in capsys.readouterr().err


# --- strict chat templates: only ONE system message, at the start ------------
# Real failure (hf.co/Abiray/Qwen3.5-9B-abliterated-GGUF, whose Jinja template
# says "System message must be at the beginning"): app.py splices the per-turn
# context in as a system message before the user's turn, so EVERY turn with
# any context came back HTTP 500 and she said "my brain answered, but with an
# error". The stock qwen3.5's Go template tolerated it.

STRICT_TEMPLATE_ERROR = (
    '{"error":"{\\"error\\":{\\"code\\":500,\\"message\\":\\"While executing CallExpression: '
    "raise_exception('System message must be at the beginning.')\\\"}}\"}"
)


def _strict_template_server(seen: list):
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body)
        roles = [m["role"] for m in body["messages"]]
        if "system" in roles[1:] or ("system" in roles and roles[0] != "system"):
            return httpx.Response(500, text=STRICT_TEMPLATE_ERROR)
        return httpx.Response(
            200,
            content=_ndjson(
                {"message": {"content": "Hi."}, "done": False},
                {"done": True, "done_reason": "stop", "prompt_eval_count": 500, "eval_count": 2},
            ),
        )

    return handler


SPLICED = [
    {"role": "system", "content": "PERSONA"},
    {"role": "user", "content": "earlier"},
    {"role": "assistant", "content": "reply"},
    {"role": "system", "content": "You remember: likes tea. The screen shows a cake."},
    {"role": "user", "content": "hii"},
]


def test_fold_moves_a_late_system_message_into_the_next_user_message():
    out = llm.fold_system_messages(SPLICED)
    assert [m["role"] for m in out] == ["system", "user", "assistant", "user"]
    assert out[0]["content"] == "PERSONA"
    assert out[-1]["content"].endswith("\n\nhii")
    assert "likes tea. The screen shows a cake." in out[-1]["content"]
    assert out[1]["content"] == "earlier"  # other turns untouched


def test_fold_does_not_mutate_its_input_and_leaves_valid_input_alone():
    before = json.loads(json.dumps(SPLICED))
    llm.fold_system_messages(SPLICED)
    assert SPLICED == before  # history itself must never be rewritten
    valid = [{"role": "system", "content": "p"}, {"role": "user", "content": "x"}]
    assert llm.fold_system_messages(valid) == valid


def test_fold_joins_several_late_system_messages_in_order():
    msgs = [
        {"role": "system", "content": "P"},
        {"role": "system", "content": "first"},
        {"role": "system", "content": "second"},
        {"role": "user", "content": "q"},
    ]
    out = llm.fold_system_messages(msgs)
    assert [m["role"] for m in out] == ["system", "user"]
    assert "first second" in out[1]["content"] and out[1]["content"].endswith("\n\nq")


def test_fold_attaches_to_the_user_turn_even_when_tool_messages_follow():
    msgs = SPLICED + [
        {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "capture_screen", "arguments": {}}}]},
        {"role": "tool", "content": "a desktop", "tool_name": "capture_screen"},
    ]
    out = llm.fold_system_messages(msgs)
    assert [m["role"] for m in out] == ["system", "user", "assistant", "user", "assistant", "tool"]
    assert "cake" in out[3]["content"]  # the user turn, not the tool result
    assert out[4]["tool_calls"] == msgs[5]["tool_calls"] and out[5]["tool_name"] == "capture_screen"


def test_fold_keeps_other_keys_such_as_images():
    msgs = [
        {"role": "system", "content": "P"},
        {"role": "system", "content": "ctx"},
        {"role": "user", "content": "look", "images": ["aGk="]},
    ]
    out = llm.fold_system_messages(msgs)
    assert out[1]["images"] == ["aGk="] and "ctx" in out[1]["content"]


def test_fold_edge_cases_never_lose_the_text():
    only_system = [{"role": "system", "content": "P"}, {"role": "system", "content": "extra"}]
    out = llm.fold_system_messages(only_system)
    assert [m["role"] for m in out] == ["system"] and "extra" in out[0]["content"]

    no_lead = [{"role": "user", "content": "q"}, {"role": "system", "content": "late"}]
    out = llm.fold_system_messages(no_lead)
    assert [m["role"] for m in out] == ["user"] and "late" in out[0]["content"]

    assert llm.fold_system_messages([]) == []


@pytest.mark.asyncio
async def test_tool_calling_request_survives_a_strict_template(monkeypatch):
    _patch_llm_config(monkeypatch, model="hf.co/some/strict-gguf")
    llm._NO_TOOLS_MODELS.clear()
    seen = []
    _mock_http(monkeypatch, _strict_template_server(seen))
    events = [e async for e in llm.stream_reply_with_tools(SPLICED, [])]
    assert events == [{"type": "content", "text": "Hi."}]
    roles = [m["role"] for m in seen[0]["messages"]]
    assert roles.count("system") == 1 and roles[0] == "system"
    assert "cake" in seen[0]["messages"][-1]["content"]


@pytest.mark.asyncio
async def test_plain_streaming_request_survives_a_strict_template(monkeypatch):
    _patch_llm_config(monkeypatch, model="hf.co/some/strict-gguf", api_style="ollama_native")
    seen = []
    _mock_http(monkeypatch, _strict_template_server(seen))
    text = "".join([d async for d in llm.stream_reply(SPLICED)])
    assert text == "Hi."
    assert [m["role"] for m in seen[0]["messages"]].count("system") == 1


@pytest.mark.asyncio
async def test_token_estimate_is_taken_from_what_was_actually_sent(monkeypatch, capsys):
    import context_budget

    context_budget.reset_calibration()
    _patch_llm_config(monkeypatch, model="m")
    _mock_http(monkeypatch, _strict_template_server([]))
    _ = [e async for e in llm.stream_reply_with_tools(SPLICED, [])]
    folded_estimate = context_budget.estimate_prompt(llm.fold_system_messages(SPLICED), [])
    assert f"est_prompt_tokens={folded_estimate}" in capsys.readouterr().err
    context_budget.reset_calibration()


# --- sampling parity, structured output for the image checks ------------------
# The stock qwen3.5:9b tag bakes in presence_penalty 1.5 / top_k 20 / top_p 0.95
# (its Ollama params file); a Hugging Face GGUF gets none of it (`ollama show`
# has no Parameters section), so the same code sampled the two differently.


def test_sampling_options_are_sent_explicitly(monkeypatch):
    _patch_llm_config(monkeypatch, top_p=0.95, top_k=20, presence_penalty=1.5)
    o = llm._ollama_options()
    assert (o["top_p"], o["top_k"], o["presence_penalty"]) == (0.95, 20, 1.5)


def test_sampling_options_can_be_left_to_the_server(monkeypatch):
    _patch_llm_config(monkeypatch, top_p=None, top_k=None, presence_penalty=None)
    o = llm._ollama_options()
    assert not {"top_p", "top_k", "presence_penalty"} & set(o)


def test_per_call_overrides_beat_the_configured_sampling(monkeypatch):
    _patch_llm_config(monkeypatch, presence_penalty=1.5)
    assert llm._ollama_options(presence_penalty=0.0)["presence_penalty"] == 0.0


SCHEMA = {"type": "object", "properties": {"on_task": {"type": "boolean"}}, "required": ["on_task"]}


@pytest.mark.asyncio
async def test_image_check_sends_the_schema_and_strict_sampling(monkeypatch):
    _patch_llm_config(monkeypatch, vision_model=None, model="m", presence_penalty=1.5)
    llm._NO_FORMAT_MODELS.clear()
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["p"] = json.loads(request.content)
        return httpx.Response(200, json={"message": {"content": '{"on_task": true}'}})

    _mock_http(monkeypatch, handler)
    assert await llm.describe_image("check", "aGk=", json_schema=SCHEMA) == '{"on_task": true}'
    assert seen["p"]["format"] == SCHEMA
    assert seen["p"]["options"]["temperature"] == 0.2
    assert seen["p"]["options"]["presence_penalty"] == 0.0  # would punish JSON's repeated quotes/braces


@pytest.mark.asyncio
async def test_plain_look_sends_no_schema(monkeypatch):
    _patch_llm_config(monkeypatch, vision_model=None, model="m")
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["p"] = json.loads(request.content)
        return httpx.Response(200, json={"message": {"content": "a desktop"}})

    _mock_http(monkeypatch, handler)
    await llm.describe_image("what is this", "aGk=")
    assert "format" not in seen["p"] and seen["p"]["options"]["temperature"] == 0.4


@pytest.mark.asyncio
async def test_schema_rejected_by_the_server_is_dropped_and_remembered(monkeypatch, capsys):
    _patch_llm_config(monkeypatch, vision_model=None, model="picky")
    llm._NO_FORMAT_MODELS.clear()
    payloads = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        payloads.append(body)
        if "format" in body:
            return httpx.Response(500, text="grammar error")
        return httpx.Response(200, json={"message": {"content": "ok"}})

    _mock_http(monkeypatch, handler)
    assert await llm.describe_image("c", "aGk=", json_schema=SCHEMA) == "ok"
    assert "format" in payloads[0] and "format" not in payloads[1]
    assert "rejected the structured-output schema" in capsys.readouterr().err

    payloads.clear()
    await llm.describe_image("c", "aGk=", json_schema=SCHEMA)
    assert len(payloads) == 1 and "format" not in payloads[0]  # remembered
    llm._NO_FORMAT_MODELS.clear()


@pytest.mark.asyncio
async def test_a_failure_that_is_not_the_schema_is_not_blamed_on_it(monkeypatch):
    _patch_llm_config(monkeypatch, vision_model=None, model="text-only")
    llm._NO_FORMAT_MODELS.clear()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, text="this model is missing data required for image input")

    _mock_http(monkeypatch, handler)
    with pytest.raises(llm.LLMServerError) as err:
        await llm.describe_image("c", "aGk=", json_schema=SCHEMA)
    assert "llm.vision_model" in str(err.value)
    assert "text-only" not in llm._NO_FORMAT_MODELS  # the retry failed too: schema wasn't the problem
