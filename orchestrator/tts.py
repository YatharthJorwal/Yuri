"""
TTS backends. `synthesize(text) -> bytes` (WAV) is the only thing app.py
calls, regardless of which engine is active -- picked by config.yaml's
tts.engine, same pattern as the LLM's config-driven backend choice.

- "pyttsx3": Phase 1's placeholder. Drives whatever TTS engine the OS
  already has (SAPI5 on Windows) -- no model download, works immediately,
  sounds robotic. Also used as an automatic fallback if the configured
  engine is "gpt_sovits" but its API server can't be reached, so a
  turn never goes silent just because that server isn't running -- the
  real error is printed to the orchestrator's terminal when this happens,
  so "wrong voice came out" is diagnosable instead of a silent mystery --
  see synthesize() below.
- "gpt_sovits": real cloned voice, via GPT-SoVITS's own API server
  (https://github.com/RVC-Boss/GPT-SoVITS) running as a separate local
  process -- not embedded here. Request/response shape confirmed by
  reading a real reference implementation (rayenfeng/riko_project) rather
  than assumed: POST {api_url} with
  {text, text_lang, ref_audio_path, prompt_text, prompt_lang}, raw WAV
  bytes back. ref_audio_path/prompt_text point at your reference clip and
  its exact transcript -- fill those into config.yaml's tts.gpt_sovits
  block once you have a clip you're happy with (see docs/MODELS.md).
"""

import asyncio
import os
import sys
import tempfile

import httpx
import pyttsx3

from config import CONFIG


class TTSUnreachableError(Exception):
    """The configured TTS backend's server couldn't be reached or errored
    out. synthesize() catches this itself for the gpt_sovits engine and
    falls back to pyttsx3 rather than letting a turn go silent -- callers
    in app.py don't need to handle this directly."""


def _synthesize_pyttsx3_sync(text: str) -> bytes:
    # Re-initializing per call is a little wasteful but far more reliable
    # than reusing one pyttsx3 engine across calls, which is known to hang
    # on some SAPI5 setups after the first use.
    engine = pyttsx3.init()
    engine.setProperty("rate", 175)

    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        engine.save_to_file(text, path)
        engine.runAndWait()
        with open(path, "rb") as f:
            return f.read()
    finally:
        engine.stop()
        try:
            os.remove(path)
        except OSError:
            pass


async def _synthesize_gpt_sovits(text: str) -> bytes:
    cfg = CONFIG.tts.gpt_sovits
    payload = {
        "text": text,
        "text_lang": cfg.text_lang,
        "ref_audio_path": cfg.ref_audio_path,
        "prompt_text": cfg.prompt_text,
        "prompt_lang": cfg.prompt_lang,
    }
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=5.0)) as client:
            response = await client.post(cfg.api_url, json=payload)
            content_type = response.headers.get("content-type", "")
            print(
                f"[luna] gpt_sovits response: {response.status_code}, "
                f"content-type={content_type!r}, {len(response.content)} bytes",
                file=sys.stderr,
            )
            if response.status_code >= 400:
                raise TTSUnreachableError(
                    f"{response.status_code} from {cfg.api_url}: {response.text[:300]!r}"
                )
            # A 200 with an empty body, or a body that isn't actually
            # audio (e.g. a JSON error message the server returned with
            # the wrong status code), used to get handed to the frontend
            # as if it were real audio -- caught here instead, so it goes
            # through the same pyttsx3 fallback an unreachable server
            # already does, rather than the frontend trying to play back
            # garbage. "audio" in content_type (not startswith) also
            # treats a missing/empty content-type as non-audio -- stricter
            # than allowing it through by default, confirmed correct
            # against GPT-SoVITS's real responses on the user's machine.
            if not response.content:
                raise TTSUnreachableError(f"empty response body from {cfg.api_url}")
            if "audio" not in content_type.lower():
                body_preview = response.text[:1000]
                print(f"[luna] gpt_sovits non-audio body: {body_preview!r}", file=sys.stderr)
                raise TTSUnreachableError(
                    f"expected audio from {cfg.api_url}, got content-type "
                    f"{content_type!r}: {body_preview[:300]!r}"
                )
            return response.content
    except httpx.RequestError as exc:
        raise TTSUnreachableError(f"couldn't reach {cfg.api_url}: {exc}") from exc


async def synthesize(text: str) -> bytes:
    """Returns WAV bytes for `text` from whichever engine config.yaml's
    tts.engine selects. Runs pyttsx3's blocking engine in a thread so it
    doesn't stall the websocket event loop."""
    if CONFIG.tts.engine == "gpt_sovits":
        try:
            return await _synthesize_gpt_sovits(text)
        except TTSUnreachableError as exc:
            # Fall back to pyttsx3 for this one line rather than letting the
            # turn go silent -- she still says the actual reply, just in the
            # placeholder voice for that line. This WAS silent before, which
            # meant "wrong voice came out" had no visible cause anywhere --
            # printing it here is the whole difference between "check the
            # terminal" and "guess blindly."
            print(f"[luna] gpt_sovits unreachable, falling back to pyttsx3: {exc}", file=sys.stderr)

    return await asyncio.to_thread(_synthesize_pyttsx3_sync, text)
