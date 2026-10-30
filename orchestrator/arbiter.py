"""
Who may make sound right now.

One state machine shared by voice turns, ambient remarks and Task Guide chides
(PHASE-2-FAST-PATH.md §4). Pure and synchronous so every transition is a unit
test; the app owns the side effects (stop playback, cancel a turn, send a line).

States: idle -> listening -> thinking -> speaking -> (listening | idle).
`interrupting` exists so a barge-in is visually distinct from a normal stop.
"""

from __future__ import annotations

IDLE = "idle"
LISTENING = "listening"
THINKING = "thinking"
SPEAKING = "speaking"
INTERRUPTING = "interrupting"

STATES = frozenset({IDLE, LISTENING, THINKING, SPEAKING, INTERRUPTING})


class Arbiter:
    def __init__(self) -> None:
        self.state = IDLE
        self.blocked_reason: str | None = None

    # -- transitions ---------------------------------------------------------

    def on_speech_started(self) -> str:
        """The user is talking. Everything else yields."""
        if self.state in (SPEAKING, THINKING):
            self.state = INTERRUPTING
        else:
            self.state = LISTENING
        self.blocked_reason = None
        return self.state

    def on_speech_ended(self) -> str:
        if self.state in (LISTENING, INTERRUPTING):
            self.state = THINKING
        return self.state

    def on_turn_started(self) -> str:
        if self.state != INTERRUPTING:
            self.state = THINKING
        return self.state

    def on_first_audio(self) -> str:
        if self.state in (THINKING, INTERRUPTING):
            self.state = SPEAKING
        return self.state

    def on_turn_finished(self) -> str:
        if self.state in (SPEAKING, THINKING, INTERRUPTING):
            self.state = IDLE
        return self.state

    def on_barge_in(self) -> str:
        """The user talked over her. Playback stops; the prompt is dropped."""
        self.state = LISTENING
        self.blocked_reason = "barge_in"
        return self.state

    # -- queries -------------------------------------------------------------

    def may_speak_ambient(self) -> tuple[bool, str | None]:
        if self.state == IDLE:
            return True, None
        return False, f"busy:{self.state}"

    def is_listening(self) -> bool:
        return self.state == LISTENING

    def snapshot(self) -> dict[str, str | None]:
        return {"state": self.state, "blocked_reason": self.blocked_reason}
