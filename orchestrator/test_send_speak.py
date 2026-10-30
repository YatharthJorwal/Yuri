"""The speak path: panel wording preserved, voice text normalized."""

import pytest

import app
import speech


class FakeWebSocket:
    def __init__(self):
        self.payloads = []

    async def send_json(self, payload):
        self.payloads.append(payload)


@pytest.mark.asyncio
async def test_send_speak_normalizes_only_the_voice(monkeypatch):
    heard = {}

    async def fake_synth(text):
        heard["text"] = text
        return b"RIFF"

    monkeypatch.setattr(app.tts, "synthesize", fake_synth)
    ws = FakeWebSocket()
    raw = "Wait\u2014what? \u201cReally\u201d \U0001F600"

    await app._send_speak(ws, raw)

    assert speech.banned_chars_in(heard["text"]) == set()
    assert ws.payloads[0]["text"] == raw          # caption keeps the original
    assert ws.payloads[0]["mime"] == "audio/wav"
    assert ws.payloads[0]["audio_b64"]


@pytest.mark.asyncio
async def test_send_speak_skips_empty_text(monkeypatch):
    called = {"n": 0}

    async def fake_synth(text):
        called["n"] += 1
        return b""

    monkeypatch.setattr(app.tts, "synthesize", fake_synth)
    ws = FakeWebSocket()

    await app._send_speak(ws, "   ")

    assert called["n"] == 0
    assert ws.payloads == []


@pytest.mark.asyncio
async def test_send_audio_payload_shape():
    ws = FakeWebSocket()
    await app._send_audio(ws, "hello", b"abc")
    payload = ws.payloads[0]
    assert payload["type"] == "speak"
    assert payload["text"] == "hello"
    assert payload["audio_b64"] == "YWJj"
