"""Voice input: frames to a WAV, exactly once per utterance."""

import math
import struct
import wave
import io

import pytest

import vad
import voice

FRAME_SAMPLES = 320


def silence(frames: int = 1) -> bytes:
    return b"\x00\x00" * FRAME_SAMPLES * frames


def tone(frames: int = 1, amplitude: int = 9000) -> bytes:
    out = bytearray()
    for _ in range(frames):
        for i in range(FRAME_SAMPLES):
            out += struct.pack("<h", int(amplitude * math.sin(2 * math.pi * 220 * i / 16000)))
    return bytes(out)


def input_ready(**overrides):
    cfg = vad.VADConfig(**overrides)
    vi = voice.VoiceInput(cfg)
    for _ in range(10):
        vi.push(silence())
    return vi


def speak_and_end(vi) -> str | None:
    events = [vi.push(tone()) for _ in range(15)]
    for _ in range(30):
        events.append(vi.push(silence()))
    return vad.SPEECH_ENDED if vad.SPEECH_ENDED in events else None


def test_pcm_to_wav_header():
    wav = voice.pcm_to_wav(tone(5), 16000)
    with wave.open(io.BytesIO(wav), "rb") as handle:
        assert handle.getnchannels() == 1
        assert handle.getsampwidth() == 2
        assert handle.getframerate() == 16000
        assert handle.getnframes() == FRAME_SAMPLES * 5


def test_utterance_becomes_one_wav():
    vi = input_ready()
    assert speak_and_end(vi) == vad.SPEECH_ENDED

    wav = vi.take_wav()
    assert wav is not None
    assert wav[:4] == b"RIFF"

    with wave.open(io.BytesIO(wav), "rb") as handle:
        assert handle.getnframes() > 0

    assert vi.take_wav() is None      # consumed once


def test_flush_ends_an_utterance_early():
    vi = input_ready()
    for _ in range(15):
        vi.push(tone())
    assert vi.in_speech is True

    assert vi.flush() == vad.SPEECH_ENDED
    assert vi.take_wav() is not None
    assert vi.in_speech is False


def test_flush_without_speech_is_a_no_op():
    vi = input_ready()
    assert vi.flush() is None
    assert vi.take_wav() is None


def test_too_short_utterance_produces_no_wav():
    vi = input_ready(pre_roll_ms=0, min_utterance_ms=500)
    for _ in range(3):
        vi.push(tone())
    vi.flush()
    assert vi.take_wav() is None


def test_two_utterances_produce_two_wavs():
    vi = input_ready()
    assert speak_and_end(vi) == vad.SPEECH_ENDED
    first = vi.take_wav()
    assert speak_and_end(vi) == vad.SPEECH_ENDED
    second = vi.take_wav()
    assert first is not None and second is not None
    assert first != second
