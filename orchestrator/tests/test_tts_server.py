"""GPT-SoVITS server lifecycle: health check and managed launch."""

import dataclasses

import pytest

import config
from audio import tts


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


# ---- shell-managed mode: the shell starts GPT-SoVITS in parallel ---------------


@pytest.mark.asyncio
async def test_shell_managed_never_launches_and_waits_for_the_shells_server(monkeypatch):
    monkeypatch.setenv(tts.SHELL_MANAGED_ENV, "1")
    ticks = {"n": 0}

    def port_open(*a, **k):
        ticks["n"] += 1
        return ticks["n"] > 3  # closed for the first few polls, then the shell's copy is up

    def must_not_launch():
        raise AssertionError("the shell owns GPT-SoVITS; Python must not start a second one")

    monkeypatch.setattr(tts, "_port_open", port_open)
    monkeypatch.setattr(tts, "_launch_gpt_sovits", must_not_launch)
    monkeypatch.setattr(tts, "CONFIG", with_launch(enabled=True, startup_timeout_s=30))
    monkeypatch.setattr(tts.asyncio, "sleep", lambda _s: _instant())

    assert await tts.ensure_server() is True


@pytest.mark.asyncio
async def test_shell_managed_gives_up_after_the_timeout_without_launching(monkeypatch):
    monkeypatch.setenv(tts.SHELL_MANAGED_ENV, "1")
    monkeypatch.setattr(tts, "_port_open", lambda *a, **k: False)
    monkeypatch.setattr(tts, "_launch_gpt_sovits", lambda: pytest.fail("must not launch"))
    monkeypatch.setattr(tts, "CONFIG", with_launch(enabled=True, startup_timeout_s=0))
    assert await tts.ensure_server() is False


@pytest.mark.asyncio
async def test_first_lines_wait_briefly_for_the_shells_server_instead_of_the_placeholder_voice(monkeypatch):
    monkeypatch.setenv(tts.SHELL_MANAGED_ENV, "1")
    polls = {"n": 0}

    def port_open(*a, **k):
        polls["n"] += 1
        return polls["n"] > 2

    async def sovits(text):
        return b"WAV-FROM-SOVITS"

    async def never_pyttsx3(text):  # pragma: no cover
        raise AssertionError("fell back to pyttsx3")

    monkeypatch.setattr(tts, "_port_open", port_open)
    monkeypatch.setattr(tts, "_synthesize_gpt_sovits", sovits)
    monkeypatch.setattr(tts, "_synthesize_pyttsx3_sync", lambda t: pytest.fail("fell back to pyttsx3"))
    monkeypatch.setattr(tts.asyncio, "sleep", lambda _s: _instant())

    assert await tts.synthesize("hello") == b"WAV-FROM-SOVITS"


@pytest.mark.asyncio
async def test_no_grace_wait_outside_shell_managed_mode_or_after_startup(monkeypatch):
    waited = {"n": 0}

    async def fake_wait(*a, **k):
        waited["n"] += 1
        return False

    async def sovits(text):
        raise tts.TTSUnreachableError("down")

    monkeypatch.setattr(tts, "_wait_for_health", fake_wait)
    monkeypatch.setattr(tts, "_synthesize_gpt_sovits", sovits)
    monkeypatch.setattr(tts, "_synthesize_pyttsx3_sync", lambda t: b"PYTTSX3")
    monkeypatch.setattr(tts, "_port_open", lambda *a, **k: False)

    monkeypatch.delenv(tts.SHELL_MANAGED_ENV, raising=False)
    assert await tts.synthesize("a") == b"PYTTSX3" and waited["n"] == 0

    monkeypatch.setenv(tts.SHELL_MANAGED_ENV, "1")
    monkeypatch.setattr(tts, "_PROCESS_STARTED", tts.time.monotonic() - 1000)  # long after startup
    assert await tts.synthesize("b") == b"PYTTSX3" and waited["n"] == 0


async def _instant():
    return None
