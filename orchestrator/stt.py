"""
Speech-to-text via faster-whisper. `transcribe(audio_bytes) -> str` is the
only thing app.py calls -- mirrors tts.py's `synthesize()` shape: one
function, config-driven, doesn't leak the underlying library upward.

API shape (model_size/device/compute_type, `.transcribe()` returning
segments to join) confirmed against a real reference implementation, not
assumed -- see docs/DECISIONS.md and docs/MODELS.md's STT section.

Model loading is lazy (first call to transcribe(), not import time like
config.py's CONFIG) -- WhisperModel() pulls weights from Hugging Face on
first-ever run and takes a real moment to load onto CPU/GPU after that, and
unlike the LLM/TTS configs there's no reason to pay that cost at orchestrator
startup for a feature the user might not use in a given session.
"""

from __future__ import annotations

import asyncio
import io
import sys
import time

from faster_whisper import WhisperModel

from config import CONFIG

_model: WhisperModel | None = None

# Ceiling on how long a single transcription attempt (including a
# first-ever model load, which downloads from Hugging Face) is allowed to
# take before we give up and report it as a failure instead of blocking
# the connection's whole message loop forever. Diagnostic instrumentation,
# not a guessed fix -- added after a real report of the mic (and, because
# this blocks the same connection's receive loop, subsequently typed chat
# too) hanging indefinitely with nothing in the log. This turns "hangs
# forever, no idea why" into either a clean, loggable timeout, or -- if
# the prints below show up but transcription is just slow -- clear
# evidence it's not a true hang at all.
_TIMEOUT_SECONDS = 90


class STTError(Exception):
    """Wraps any failure from model construction or transcription -- a
    missing CUDA DLL (cuBLAS/cuDNN), a corrupted/incomplete first-time
    model download, or anything else faster-whisper/CTranslate2 can throw.
    One exception type for app.py to catch, same pattern llm.py's
    LLMUnreachableError and tts.py's TTSUnreachableError already use.
    Without this, an exception here was propagating all the way up through
    the websocket's receive loop in app.py and killing the connection
    outright -- silently, from the user's side: the mic would record fine,
    nothing would come back, and there was no error anywhere they'd
    actually see it."""


def _get_model() -> WhisperModel:
    global _model
    if _model is None:
        cfg = CONFIG.stt
        # Prints unconditionally (not just on error) so a slow first load
        # is visible in the log as it's happening, not just inferred after
        # the fact from a timeout. If this line never appears despite
        # actually recording something, the problem is upstream of stt.py
        # entirely (audio never reached the backend) -- see app.py's own
        # "received N bytes" log line right before this gets called.
        print(
            f"[luna] loading faster-whisper model '{cfg.model_size}' on "
            f"{cfg.device} ({cfg.compute_type}) -- first mic use this run "
            "only. Downloads from Hugging Face on a genuinely first-ever "
            "run, which can take a while depending on your connection.",
            file=sys.stderr,
            flush=True,
        )
        start = time.monotonic()
        _model = WhisperModel(cfg.model_size, device=cfg.device, compute_type=cfg.compute_type)
        print(
            f"[luna] faster-whisper model loaded in {time.monotonic() - start:.1f}s",
            file=sys.stderr,
            flush=True,
        )
    return _model


def _transcribe_sync(audio_bytes: bytes) -> str:
    model = _get_model()
    # A file-like object is enough -- faster-whisper decodes whatever
    # container/codec it's given (via PyAV, which bundles its own ffmpeg)
    # and resamples to the 16kHz mono its feature extractor wants, so the
    # WebM/Opus blob MediaRecorder produces in the frontend doesn't need any
    # conversion on this end. No temp file needed either.
    print(f"[luna] transcribing {len(audio_bytes)} bytes of audio...", file=sys.stderr, flush=True)
    start = time.monotonic()
    # model.transcribe() itself returns near-instantly -- `segments` is a
    # lazy generator, the actual GPU/CPU work happens while iterating it
    # below (the join). Timing wraps both so a slow join doesn't look like
    # a fast transcribe() call followed by a mysterious pause.
    segments, _info = model.transcribe(
        io.BytesIO(audio_bytes),
        language=CONFIG.stt.language,
    )
    # segments is a generator; each segment's .text already has its own
    # leading space from faster-whisper's tokenizer, so join without adding
    # more, then collapse the whole thing's outer whitespace once.
    result = "".join(segment.text for segment in segments).strip()
    print(
        f"[luna] transcribe() finished in {time.monotonic() - start:.1f}s: {result!r}",
        file=sys.stderr,
        flush=True,
    )
    return result


async def transcribe(audio_bytes: bytes) -> str:
    """Runs faster-whisper's blocking transcribe() in a thread so it doesn't
    stall the websocket event loop -- same pattern as tts.py's pyttsx3
    path. Returns "" (not an error) for silence/unintelligible audio --
    there's nothing wrong, just nothing to reply to. Raises STTError (not
    the raw underlying exception) for anything that actually went wrong,
    e.g. the model failing to load at all, or a hang against
    _TIMEOUT_SECONDS.

    On ANY failure, drops the cached model (see _get_model()) so the next
    attempt builds a fresh one instead of reusing this one -- found from a
    real case where a first transcription failed cleanly with a missing
    CUDA DLL error, and every attempt after that (still reusing the same
    now-once-failed model object) hung for the full timeout instead of
    failing the same clean way. A native library that's already thrown
    once mid-inference is in an unknown state; nothing about it is worth
    trusting for a second call."""
    global _model
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(_transcribe_sync, audio_bytes),
            timeout=_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError as exc:
        _model = None
        raise STTError(
            f"model loading and/or transcription didn't finish within "
            f"{_TIMEOUT_SECONDS}s. Check the log lines right above this "
            "one: if 'loading faster-whisper model' never printed, the "
            "problem is upstream of stt.py entirely; if it printed but "
            "'faster-whisper model loaded' or 'transcribe() finished' "
            "never did, that pinpoints which of the two actually hung. "
            "Dropping the cached model either way -- next attempt starts "
            "fresh rather than risk reusing one that just hung."
        ) from exc
    except Exception as exc:
        _model = None
        raise STTError(str(exc)) from exc
