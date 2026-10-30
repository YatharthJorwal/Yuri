"""VAD over synthetic PCM: start, end, pre-roll, minimum length, max length."""

import math
import struct

import pytest

import vad

FRAME_SAMPLES = 320          # 20 ms at 16 kHz


def silence(frames: int = 1) -> bytes:
    return b"\x00\x00" * FRAME_SAMPLES * frames


def tone(frames: int = 1, amplitude: int = 9000) -> bytes:
    out = bytearray()
    for _ in range(frames):
        for i in range(FRAME_SAMPLES):
            out += struct.pack("<h", int(amplitude * math.sin(2 * math.pi * 220 * i / 16000)))
    return bytes(out)


def detector(**overrides):
    return vad.VoiceActivityDetector(vad.VADConfig(**overrides))


def warm_up(det, frames: int = 10) -> None:
    """Let the noise floor settle on a quiet room first."""
    for _ in range(frames):
        det.feed(silence())


def test_rms_of_silence_is_very_low():
    assert vad.rms_dbfs(silence()) < -80


def test_rms_of_a_tone_is_audible():
    assert -30 < vad.rms_dbfs(tone()) < -5


def test_speech_starts_after_three_frames():
    det = detector()
    warm_up(det)
    assert det.feed(tone()) is None
    assert det.feed(tone()) is None
    assert det.feed(tone()) == vad.SPEECH_STARTED
    assert det.in_speech


def test_speech_ends_after_half_a_second_of_silence():
    det = detector()
    warm_up(det)
    for _ in range(15):
        det.feed(tone())
    for _ in range(24):
        assert det.feed(silence()) is None
    assert det.feed(silence()) == vad.SPEECH_ENDED
    assert not det.in_speech
    assert det.last_utterance is not None


def test_utterance_includes_pre_roll():
    det = detector(pre_roll_ms=200)
    warm_up(det)
    for _ in range(15):
        det.feed(tone())
    for _ in range(26):
        det.feed(silence())
    assert det.last_utterance is not None
    # 200 ms pre-roll + the utterance itself, as 640-byte frames.
    assert len(det.last_utterance) >= 200 // 20 * 640


def test_short_blip_is_discarded_when_pre_roll_is_off():
    det = detector(pre_roll_ms=0, min_utterance_ms=250)
    warm_up(det)
    for _ in range(3):
        det.feed(tone())
    for _ in range(26):
        det.feed(silence())
    assert det.last_utterance is None


def test_max_length_forces_an_end():
    det = detector(max_utterance_ms=200, min_utterance_ms=100)
    warm_up(det)
    for _ in range(3):
        det.feed(tone())
    events = [det.feed(tone()) for _ in range(20)]
    assert vad.SPEECH_ENDED in events
    assert det.last_utterance is not None


def test_partial_frames_are_ignored():
    det = detector()
    assert det.feed(b"\x00\x00\x10") is None


def test_noise_floor_adapts_to_a_loud_room():
    det = detector(absolute_floor_db=-55.0, margin_db=8.0)
    # A "quiet room" that is actually a fan: constant level well above digital
    # silence. After adaptation it must not be treated as speech.
    fan = tone(amplitude=1200)
    for _ in range(200):
        det.feed(fan)
    assert det.feed(fan) is None
    assert not det.in_speech


def test_reset_clears_state():
    det = detector()
    warm_up(det)
    det.feed(tone())
    det.reset()
    assert not det.in_speech
    assert det.last_utterance is None
