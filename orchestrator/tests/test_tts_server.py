"""GPT-SoVITS server lifecycle: health check and managed launch."""

import dataclasses

import pytest

import config
import tts


def with_launch(**overrides):
    launch = {**dict(config.CONFIG.tts.gpt_sovits.launch), **overrides}
    sovits = dataclasses.replace(config.CONFIG.tts.gpt_sovits, launch=launch)
    return dataclasses.replace(config.CONFIG, tts=dataclasses.replace(config.CONFIG.tts, gpt_sovits=sovits))


def with_engine(engine: str):
    return dataclasses.replace(config.CONFIG, tts=dataclasses.replace(config.CONFIG.tts, engine=engine))


@pytest.mark.asyncio
async def test_health_is_false_for_pyttsx3(monkeypatch):
    monkeypatch.setattr(tts, "CONFIG", with_engine("pyttsx3"))
    assert await tts.health() is False


@pytest.mark.asyncio
async def test_health_checks_the_port(monkeypatch):
    monkeypatch.setattr(tts, "_port_open", lambda host, port, timeout=0.5: (host, port) == ("127.0.0.1", 9880))
    assert await tts.health() is True


@pytest.mark.asyncio
async def test_ensure_server_is_true_when_already_running(monkeypatch):
    monkeypatch.setattr(tts, "_port_open", lambda *a, **k: True)
    assert await tts.ensure_server() is True


@pytest.mark.asyncio
async def test_ensure_server_respects_disabled_launch(monkeypatch):
    monkeypatch.setattr(tts, "CONFIG", with_launch(enabled=False))
    monkeypatch.setattr(tts, "_port_open", lambda *a, **k: False)
    assert await tts.ensure_server() is False


@pytest.mark.asyncio
async def test_ensure_server_launches_and_waits(monkeypatch):
    opened = {"value": False}

    class FakeProcess:
        returncode = None

        def poll(self):
            return None

    def fake_launch():
        opened["value"] = True
        return FakeProcess()

    monkeypatch.setattr(tts, "_port_open", lambda *a, **k: opened["value"])
    monkeypatch.setattr(tts, "_launch_gpt_sovits", fake_launch)
    monkeypatch.setattr(tts, "CONFIG", with_launch(startup_timeout_s=5))

    assert await tts.ensure_server() is True


@pytest.mark.asyncio
async def test_ensure_server_gives_up_when_the_process_dies(monkeypatch):
    class DeadProcess:
        returncode = 1

        def poll(self):
            return 1

    monkeypatch.setattr(tts, "_port_open", lambda *a, **k: False)
    monkeypatch.setattr(tts, "_launch_gpt_sovits", lambda: DeadProcess())
    monkeypatch.setattr(tts, "CONFIG", with_launch(startup_timeout_s=5))

    assert await tts.ensure_server() is False


def test_launch_refuses_missing_paths(monkeypatch):
    monkeypatch.setattr(
        tts, "CONFIG", with_launch(python=r"C:\definitely\not\here\python.exe", enabled=True)
    )
    assert tts._launch_gpt_sovits() is None


def test_service_address_parsing():
    assert tts._service_address("http://127.0.0.1:9880/tts") == ("127.0.0.1", 9880)
    assert tts._service_address("http://localhost/tts") == ("localhost", 9880)
