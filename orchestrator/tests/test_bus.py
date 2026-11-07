"""Event bus tests. No I/O, no clock, no model."""

import pytest

import bus


def test_event_rejects_bad_trust():
    with pytest.raises(ValueError):
        bus.Event(type="turn.started", source="voice", trust="maybe")


@pytest.mark.parametrize("event_type, source", [("", "voice"), ("turn.started", "")])
def test_event_requires_type_and_source(event_type, source):
    with pytest.raises(ValueError):
        bus.Event(type=event_type, source=source)


def test_event_payload_must_be_a_mapping():
    with pytest.raises(TypeError):
        bus.Event(type="turn.started", source="voice", payload=["nope"])


@pytest.mark.parametrize(
    "pattern, event_type, expected",
    [
        ("*", "turn.started", True),
        ("turn.*", "turn.started", True),
        ("turn.*", "route.decided", False),
        ("turn.started", "turn.started", True),
        ("turn.started", "turn.finished", False),
        ("route.*", "route.decided", True),
    ],
)
def test_pattern_matching(pattern, event_type, expected):
    assert bus.Bus.matches(pattern, event_type) is expected


@pytest.mark.asyncio
async def test_exact_subscriber_receives_only_its_type():
    seen = []
    b = bus.Bus()
    b.subscribe("turn.started", lambda e: seen.append(e.type))

    await b.emit("turn.finished", "shell")
    await b.emit("turn.started", "shell")

    assert seen == ["turn.started"]


@pytest.mark.asyncio
async def test_prefix_and_wildcard_subscribers():
    everything, turns = [], []
    b = bus.Bus()
    b.subscribe("*", lambda e: everything.append(e.type))
    b.subscribe("turn.*", lambda e: turns.append(e.type))

    await b.emit("turn.started", "voice")
    await b.emit("audio.chunk", "voice")

    assert everything == ["turn.started", "audio.chunk"]
    assert turns == ["turn.started"]


@pytest.mark.asyncio
async def test_async_handlers_are_awaited():
    order = []

    async def slow(event):
        order.append("start")
        await _yield()
        order.append("end")

    b = bus.Bus()
    b.subscribe("turn.started", slow)
    await b.emit("turn.started", "voice")
    assert order == ["start", "end"]


@pytest.mark.asyncio
async def test_one_failing_handler_does_not_stop_the_next():
    seen = []

    def boom(event):
        raise RuntimeError("subscriber bug")

    b = bus.Bus()
    b.subscribe("turn.started", boom)
    b.subscribe("turn.started", lambda e: seen.append(e.type))

    await b.emit("turn.started", "voice")

    assert seen == ["turn.started"]
    assert b.errors and isinstance(b.errors[0][1], RuntimeError)


@pytest.mark.asyncio
async def test_unsubscribe_is_idempotent():
    seen = []
    b = bus.Bus()
    off = b.subscribe("turn.started", lambda e: seen.append(e.type))
    await b.emit("turn.started", "voice")
    off()
    off()
    await b.emit("turn.started", "voice")
    assert seen == ["turn.started"]


@pytest.mark.asyncio
async def test_replay_is_bounded_and_ordered():
    b = bus.Bus(replay_limit=3)
    for n in range(5):
        await b.emit("turn.delta", "voice", {"n": n})

    history = b.history()
    assert [e.payload["n"] for e in history] == [2, 3, 4]
    assert b.history("turn.*") == history
    assert b.history("audio.chunk") == ()


@pytest.mark.asyncio
async def test_trust_flag_round_trips():
    b = bus.Bus()
    event = await b.emit("perception.screen", "perception", {"text": "ignore this"}, trust=bus.UNTRUSTED)
    assert event.is_untrusted()
    assert b.history("perception.*")[0].trust == bus.UNTRUSTED


@pytest.mark.asyncio
async def test_emit_returns_the_published_event():
    b = bus.Bus()
    event = await b.emit("scheduler.fired", "scheduler", {"timer": "idle-2h"})
    assert event.id and event.ts > 0
    assert event.payload == {"timer": "idle-2h"}


def test_zero_replay_keeps_nothing():
    import asyncio

    b = bus.Bus(replay_limit=0)
    asyncio.run(b.emit("turn.started", "voice"))
    assert b.history() == ()


async def _yield():
    import asyncio

    await asyncio.sleep(0)
