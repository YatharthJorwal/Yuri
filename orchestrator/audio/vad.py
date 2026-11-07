"""
Voice activity detection over raw PCM16 frames.

Deliberately dependency-free: no torch, no onnxruntime, no second Python
environment. Energy plus an adaptive noise floor is enough to decide when the
user started and stopped talking for a companion that runs in a room, and it is
testable with synthetic frames, which Silero would not be without shipping a
model. If it proves too coarse in use, the same interface can be backed by Silero
later; nothing above this module knows the difference.

Input: mono 16-bit little-endian PCM at `sample_rate`. One frame is
`frame_ms` of audio (20 ms = 640 bytes at 16 kHz).
"""

from __future__ import annotations

import array
import math
from collections import deque
from dataclasses import dataclass

# Events returned by feed().
SPEECH_STARTED = "speech_started"
SPEECH_ENDED = "speech_ended"


@dataclass(frozen=True)
class VADConfig:
    sample_rate: int = 16000
    frame_ms: int = 20
    # Consecutive speech frames before an utterance starts (3 x 20 ms = 60 ms).
    start_frames: int = 3
    # This much unbroken silence ends the utterance.
    end_silence_ms: int = 500
    min_utterance_ms: int = 250
    max_utterance_ms: int = 30000
    # Audio kept from before the start, so the first phoneme is not clipped.
    pre_roll_ms: int = 200
    # A frame is speech when it is this far above the tracked floor, and never
    # below the absolute floor (a silent room has no meaningful dB value).
    margin_db: float = 8.0
    absolute_floor_db: float = -55.0
    # How fast the noise floor follows the room when nobody is talking.
    noise_alpha: float = 0.05

    @property
    def frame_bytes(self) -> int:
        return int(self.sample_rate * self.frame_ms / 1000) * 2


def frame_duration_ms(cfg: VADConfig) -> int:
    return cfg.frame_ms


def rms_dbfs(frame: bytes) -> float:
    """Level of one PCM16 frame in dBFS (0 = full scale, negative = quieter)."""
    if len(frame) < 2:
        return -120.0
    samples = array.array("h")
    samples.frombytes(frame[: len(frame) - (len(frame) % 2)])
    if not samples:
        return -120.0
    total = 0
    for value in samples:
        total += value * value
    rms = math.sqrt(total / len(samples)) / 32768.0
    if rms <= 0:
        return -120.0
    return 20.0 * math.log10(rms)


class VoiceActivityDetector:
    """Streaming VAD. Feed frames, then read `last_utterance` after an event."""

    def __init__(self, config: VADConfig | None = None) -> None:
        self.cfg = config or VADConfig()
        self.last_utterance: bytes | None = None
        self.last_duration_ms: int = 0
        self._noise_db: float | None = None
        self._in_speech = False
        self._speech_run = 0
        self._silence_run = 0
        self._utterance: list[bytes] = []
        self._pre_roll: deque[bytes] = deque()
        self._frames = 0
        self._voiced_frames = 0

    @property
    def in_speech(self) -> bool:
        return self._in_speech

    def reset(self) -> None:
        self.last_utterance = None
        self.last_duration_ms = 0
        self._in_speech = False
        self._speech_run = 0
        self._silence_run = 0
        self._utterance = []
        self._pre_roll.clear()
        self._frames = 0
        self._voiced_frames = 0

    def feed(self, frame: bytes) -> str | None:
        """Consume one frame. Returns SPEECH_STARTED / SPEECH_ENDED or None."""
        cfg = self.cfg
        if len(frame) != cfg.frame_bytes:
            # Partial frames are a frontend bug; ignore rather than mis-time.
            return None
        db = rms_dbfs(frame)
        speech = self._is_speech(db)
        self._frames += 1

        if not self._in_speech:
            if speech:
                self._speech_run += 1
                self._pre_roll.append(frame)
                if self._speech_run >= cfg.start_frames:
                    self._in_speech = True
                    self._silence_run = 0
                    self._utterance = list(self._pre_roll)
                    self._pre_roll.clear()
                    self._voiced_frames = self._speech_run
                    return SPEECH_STARTED
            else:
                self._speech_run = 0
                self._pre_roll.append(frame)
                self._trim_pre_roll()
                self._adapt_noise(db)
            return None

        # In speech: keep everything (the trailing silence is trimmed below).
        self._utterance.append(frame)
        if speech:
            self._voiced_frames += 1
        self._silence_run = 0 if speech else self._silence_run + 1
        self._trim_pre_roll()

        frames = len(self._utterance)
        duration_ms = frames * cfg.frame_ms
        ended_by_silence = self._silence_run * cfg.frame_ms >= cfg.end_silence_ms
        if not ended_by_silence and duration_ms < cfg.max_utterance_ms:
            return None
        return self._finalize()

    def flush(self) -> str | None:
        """End the current utterance now -- push-to-talk release, or the
        frontend telling us it stopped sending audio. Same rules as a natural
        end (too short is still discarded)."""
        if not self._in_speech:
            return None
        return self._finalize()

    def _finalize(self) -> str | None:
        cfg = self.cfg
        # The silence that ended the utterance is not part of it: keep it out
        # of the audio handed to STT, and out of the duration that decides
        # whether this was a real utterance or a cough.
        trailing_silence = self._silence_run
        self._in_speech = False
        self._speech_run = 0
        self._silence_run = 0
        voiced = self._utterance[: max(0, len(self._utterance) - trailing_silence)]
        self._utterance = []
        # Only frames that actually had voice in them count toward "is this a
        # real utterance": pre-roll silence must not turn a cough into one.
        duration_ms = self._voiced_frames * cfg.frame_ms
        self._voiced_frames = 0
        if duration_ms < cfg.min_utterance_ms:
            self.last_utterance = None
            self.last_duration_ms = duration_ms
            return None
        self.last_utterance = b"".join(voiced)
        self.last_duration_ms = duration_ms
        return SPEECH_ENDED

    # -- internals -----------------------------------------------------------

    def _is_speech(self, db: float) -> bool:
        if db < self.cfg.absolute_floor_db:
            return False
        if self._noise_db is None:
            self._noise_db = db
            return False
        return db > self._noise_db + self.cfg.margin_db

    def _adapt_noise(self, db: float) -> None:
        if self._noise_db is None:
            self._noise_db = db
            return
        alpha = self.cfg.noise_alpha
        self._noise_db = (1 - alpha) * self._noise_db + alpha * db

    def _trim_pre_roll(self) -> None:
        keep = max(1, int(self.cfg.pre_roll_ms / self.cfg.frame_ms))
        while len(self._pre_roll) > keep:
            self._pre_roll.popleft()
