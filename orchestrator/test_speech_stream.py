"""Pipelined speech: ordering, lookahead, abort, failure containment."""

import asyncio

import pytest

import speech
from speech_stream import SpeechStream


def make_stream(*, synth=None, send=None):
    sent: list[tuple[str, str]] = []

    async def default_send(display: str, audio: bytes) -> None:
        sent.append((display, audio.decode() if audio else ""))

    async def default_synth(text: str) -> bytes:
        return text.encode()

    return SpeechStream(send or default_send, synth=synth or default_synth), sent


@pytest.mark.asyncio
async def test_clauses_keep_their_order():
    stream, sent = make_stream()
    await stream.push("First thing. Second thing! Third thing?")
    await stream.close()

    displays = [display for display, _ in sent]
    assert displays == ["First thing.", "Second thing!", "Third thing?"]


@pytest.mark.asyncio
async def test_clauses_are_normalized_before_synthesis():
    spoken: list[str] = []

    async def synth(text: str) -> bytes:
        spoken.append(text)
        return b"x"

    stream, _ = make_stream(synth=synth)
    await stream.push("Wait\u2014what? \u201cReally\u201d")
    await stream.close()

    assert spoken
    assert all(not speech.banned_chars_in(text) for text in spoken)


@pytest.mark.asyncio
async def test_synthesis_runs_ahead_by_one_clause():
    active = 0
    peak = 0

    async def synth(text: str) -> bytes:
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.01)
        active -= 1
        return text.encode()

    async def send(display, audio):
        await asyncio.sleep(0)

    stream = SpeechStream(send, synth=synth)
    for clause in ("One one one. ", "Two two two. ", "Three three three. "):
        await stream.push(clause)
    await stream.close()

    assert peak >= 2          # synthesizing the next while sending the current


@pytest.mark.asyncio
async def test_abort_drops_what_was_not_spoken():
    started: list[str] = []

    async def synth(text: str) -> bytes:
        started.append(text)
        await asyncio.sleep(0.05)
        return b"x"

    stream, sent = make_stream(synth=synth)
    await stream.push("One one one. Two two two. Three three three.")
    await asyncio.sleep(0.01)
    stream.abort()
    await asyncio.sleep(0.12)

    assert len(sent) <= 1
    assert len(started) <= 2


@pytest.mark.asyncio
async def test_first_audio_fires_once():
    calls = []
    stream = SpeechStream(
        lambda display, audio: _noop(),
        synth=lambda text: _bytes(text),
        on_first_audio=lambda: calls.append(1),
    )
    await stream.push("One one one. Two two two.")
    await stream.close()
    assert calls == [1]


@pytest.mark.asyncio
async def test_synthesis_failure_does_not_kill_the_stream():
    calls = {"n": 0}

    async def synth(text: str) -> bytes:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("model crashed")
        return b"ok"

    sent: list[str] = []

    async def send(display, audio):
        sent.append(display)

    stream = SpeechStream(send, synth=synth)
    await stream.push("First. Second.")
    await stream.close()

    assert sent == ["Second."]        # the good clause still gets through


@pytest.mark.asyncio
async def test_send_failure_is_contained():
    async def synth(text: str) -> bytes:
        return b"x"

    async def send(display, audio):
        raise OSError("socket gone")

    stream = SpeechStream(send, synth=synth)
    await stream.push("One one one. Two two two.")
    await stream.close()               # must not raise


@pytest.mark.asyncio
async def test_close_is_idempotent():
    stream, sent = make_stream()
    await stream.push("Hello there friend.")
    await stream.close()
    await stream.close()
    assert len(sent) == 1


@pytest.mark.asyncio
async def test_push_after_close_is_ignored():
    stream, sent = make_stream()
    await stream.push("Hello there friend.")
    await stream.close()
    before = len(sent)
    await stream.push("Too late to say this.")
    assert len(sent) == before


async def _noop():
    return None


async def _bytes(text: str) -> bytes:
    return text.encode()
