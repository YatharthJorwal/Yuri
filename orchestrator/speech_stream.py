"""
Pipelined speech: text in as it streams, audio out in order.

The old path awaited synthesis inside the token loop, so the model's next
sentence waited on the voice. This module separates the two: clauses are queued
as they arrive and a single worker synthesizes ahead by one while the previous
clause is being sent. Order is preserved, backpressure is bounded, and an abort
(stop button, barge-in) drops the queue instead of finishing a reply nobody is
listening to.

Pure enough to test without a model: `synth` and `send` are injected.
"""

from __future__ import annotations

import asyncio
import sys
from typing import Awaitable, Callable

import speech
from tts import synthesize as default_synthesize

Send = Callable[[str, bytes], Awaitable[None]]
Synth = Callable[[str], Awaitable[bytes]]


class SpeechStream:
    def __init__(
        self,
        send: Send,
        *,
        synth: Synth | None = None,
        prepare: Callable[[str], speech.PreparedSpeech] | None = None,
        on_first_audio: Callable[[], None] | None = None,
        max_queue: int = 8,
    ) -> None:
        self._send = send
        self._synth = synth or default_synthesize
        self._prepare = prepare or speech.prepare
        self._on_first_audio = on_first_audio
        self._queue: asyncio.Queue[tuple[str, str] | None] = asyncio.Queue(maxsize=max_queue)
        self._worker: asyncio.Task[None] | None = None
        self._closed = False
        self._aborted = False
        self._first_audio_done = False
        self.spoken_segments: list[str] = []

    def start(self) -> None:
        if self._worker is None:
            self._worker = asyncio.create_task(self._run(), name="speech-stream")

    async def push(self, text: str) -> None:
        """Prepare one streamed chunk and queue its clauses. Returns quickly;
        blocks only when the queue is full (the voice is behind the model)."""
        if self._closed or self._aborted:
            return
        prepared = self._prepare(text)
        if not prepared.spoken:
            return
        # One display segment per spoken segment keeps captions aligned with the
        # audio when a chunk had to be split. Both come from the same text, so
        # they stay in step.
        display_segments = speech.split_segments(prepared.display) or (prepared.display,)
        spoken_segments = prepared.segments or (prepared.spoken,)
        if len(display_segments) != len(spoken_segments):
            display_segments = (prepared.display,) * len(spoken_segments)
        self.start()
        for display, spoken in zip(display_segments, spoken_segments):
            self.spoken_segments.append(spoken)
            await self._queue.put((display, spoken))

    async def close(self) -> None:
        """Flush everything queued, then stop. Idempotent."""
        if self._aborted or self._closed:
            return
        if self._worker is None:
            self._closed = True
            return
        self._closed = True
        await self._queue.put(None)
        try:
            await self._worker
        except asyncio.CancelledError:  # pragma: no cover - shutdown path
            pass

    def abort(self) -> None:
        """Drop whatever has not been spoken. Idempotent, safe from finally."""
        if self._aborted:
            return
        self._aborted = True
        self._closed = True
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:  # pragma: no cover - race guard
                break
        if self._worker is not None and not self._worker.done():
            self._worker.cancel()

    # -- worker --------------------------------------------------------------

    async def _run(self) -> None:
        pending: tuple[str, asyncio.Task[bytes]] | None = None
        while True:
            item = await self._queue.get()
            if item is None:
                break
            display, spoken = item
            # Start the next synthesis before awaiting the previous delivery:
            # that is the one-clause lookahead the old path did not have.
            task = asyncio.create_task(self._synth(spoken))
            if pending is not None:
                await self._deliver(*pending)
            pending = (display, task)
        if pending is not None:
            await self._deliver(*pending)

    async def _deliver(self, display: str, task: asyncio.Task[bytes]) -> None:
        try:
            audio = await task
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 -- one clause must not kill the turn
            print(f"[yuri] speech: synthesis failed for a clause ({exc})", file=sys.stderr)
            return
        try:
            await self._send(display, audio)
        except Exception as exc:  # noqa: BLE001 -- a closed socket ends the send, not the app
            print(f"[yuri] speech: send failed ({exc})", file=sys.stderr)
            return
        if not self._first_audio_done:
            self._first_audio_done = True
            if self._on_first_audio is not None:
                self._on_first_audio()
