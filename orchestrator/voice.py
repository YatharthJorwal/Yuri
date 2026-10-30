"""
Streaming voice input: raw PCM frames in, finished utterances out.

The frontend sends 20 ms frames while the mic is open. This wraps the VAD with
the one thing the rest of the app cares about: a complete utterance as a WAV
file that faster-whisper can decode, produced exactly once per thing the user
said.

Nothing here touches the network or a model, so it is fully testable.
"""

from __future__ import annotations

import io
import wave

import vad

SAMPLE_RATE = 16000
SAMPLE_WIDTH = 2
CHANNELS = 1


def pcm_to_wav(pcm: bytes, sample_rate: int = SAMPLE_RATE) -> bytes:
    """Wrap mono 16-bit PCM in a WAV container (what STT actually accepts)."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(CHANNELS)
        handle.setsampwidth(SAMPLE_WIDTH)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm)
    return buffer.getvalue()


class VoiceInput:
    """One per microphone connection."""

    def __init__(self, config: vad.VADConfig | None = None) -> None:
        self.detector = vad.VoiceActivityDetector(config)
        self._pending_wav: bytes | None = None

    @property
    def in_speech(self) -> bool:
        return self.detector.in_speech

    def push(self, pcm: bytes) -> str | None:
        """Feed one frame. Returns the VAD event, and stashes a WAV when an
        utterance completes; call `take_wav()` after SPEECH_ENDED."""
        event = self.detector.feed(pcm)
        if event == vad.SPEECH_ENDED:
            self._pending_wav = self._wav_from_detector()
        return event

    def flush(self) -> str | None:
        """End the utterance immediately (mic released / audio_end received)."""
        event = self.detector.flush()
        if event == vad.SPEECH_ENDED:
            self._pending_wav = self._wav_from_detector()
        return event

    def take_wav(self) -> bytes | None:
        wav, self._pending_wav = self._pending_wav, None
        return wav

    def _wav_from_detector(self) -> bytes | None:
        pcm = self.detector.last_utterance
        if not pcm:
            return None
        self.detector.last_utterance = None
        return pcm_to_wav(pcm, self.detector.cfg.sample_rate)
