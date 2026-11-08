"""Boot behaviour: the slow startup checks run side by side, and the speech model
is loaded once, in the background, without racing a first turn."""

from __future__ import annotations

import asyncio
import dataclasses
import threading
import time

import pytest

import app
import config
from audio import stt


@pytest.mark.asyncio
async def test_startup_checks_run_concurrently_not_one_after_another(monkeypatch):
    started: list[str] = []
    release = asyncio.Event()

    def blocker(name):
        async def run():
            started.append(name)
            await release.wait()

        return run

    monkeypatch.setattr(app, "_check_chat_model", blocker("chat"))
    monkeypatch.setattr(app, "_check_tts", blocker("tts"))
    monkeypatch.setattr(app, "_check_router", blocker("router"))
    monkeypatch.setattr(app.stt, "warm_up", blocker("stt"))

    task = asyncio.create_task(app._startup_model_checks())
    await asyncio.sleep(0.05)
    # all four began while the first (the slow model load) is still blocked
    assert sorted(started) == ["chat", "router", "stt", "tts"]
    release.set()
    await task


@pytest.mark.asyncio
async def test_one_failing_startup_check_does_not_stop_the_others(monkeypatch):
    ran: list[str] = []

    async def boom():
        raise RuntimeError("ollama exploded")

    async def ok_tts():
        ran.append("tts")

    async def ok_router():
        ran.append("router")

    async def ok_stt():
        ran.append("stt")

    monkeypatch.setattr(app, "_check_chat_model", boom)
    monkeypatch.setattr(app, "_check_tts", ok_tts)
    monkeypatch.setattr(app, "_check_router", ok_router)
    monkeypatch.setattr(app.stt, "warm_up", ok_stt)
    await app._startup_model_checks()  # must not raise
    assert sorted(ran) == ["router", "stt", "tts"]


def _with_stt(**overrides):
    return dataclasses.replace(config.CONFIG, stt=dataclasses.replace(config.CONFIG.stt, **overrides))


@pytest.fixture(autouse=True)
def _fresh_model_state(monkeypatch):
    monkeypatch.setattr(stt, "_model", None)


@pytest.mark.asyncio
async def test_warm_up_loads_the_model_once(monkeypatch):
    loads = []
    monkeypatch.setattr(stt, "WhisperModel", lambda *a, **k: loads.append(1) or object())
    await stt.warm_up()
    await stt.warm_up()  # already loaded: no second load
    assert len(loads) == 1


@pytest.mark.asyncio
async def test_warm_up_can_be_switched_off(monkeypatch):
    monkeypatch.setattr(stt, "CONFIG", _with_stt(warm_up_on_start=False))
    monkeypatch.setattr(stt, "WhisperModel", lambda *a, **k: pytest.fail("must not load"))
    await stt.warm_up()


@pytest.mark.asyncio
async def test_a_failed_warm_up_never_raises_and_the_next_use_can_retry(monkeypatch, capsys):
    attempts = []

    def flaky(*a, **k):
        attempts.append(1)
        if len(attempts) == 1:
            raise OSError("no network for the download")
        return object()

    monkeypatch.setattr(stt, "WhisperModel", flaky)
    await stt.warm_up()  # swallowed
    assert "warm-up failed" in capsys.readouterr().err
    assert stt._get_model() is not None and len(attempts) == 2


def test_concurrent_first_uses_load_the_model_exactly_once(monkeypatch):
    loads = []

    def slow_load(*a, **k):
        loads.append(1)
        time.sleep(0.05)
        return object()

    monkeypatch.setattr(stt, "WhisperModel", slow_load)
    threads = [threading.Thread(target=stt._get_model) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(loads) == 1


def test_warm_up_switch_is_readable_from_voice_stt_config():
    off = config.VoiceConfig(stt={**config.VoiceConfig().stt, "warm_up_on_start": False})
    assert config._derive_stt({}, off).warm_up_on_start is False
    assert config._derive_stt({}, config.VoiceConfig()).warm_up_on_start is True
