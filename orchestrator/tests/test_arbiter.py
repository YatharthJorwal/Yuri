"""Arbiter state machine: who may speak, and what a barge-in means."""

import arbiter


def test_turn_flow():
    a = arbiter.Arbiter()
    assert a.state == arbiter.IDLE
    a.on_speech_started()
    assert a.state == arbiter.LISTENING
    a.on_speech_ended()
    assert a.state == arbiter.THINKING
    a.on_first_audio()
    assert a.state == arbiter.SPEAKING
    a.on_turn_finished()
    assert a.state == arbiter.IDLE


def test_ambient_only_when_idle():
    a = arbiter.Arbiter()
    assert a.may_speak_ambient() == (True, None)
    a.on_speech_started()
    allowed, reason = a.may_speak_ambient()
    assert allowed is False
    assert "listening" in reason


def test_speech_while_speaking_is_a_barge_in():
    a = arbiter.Arbiter()
    a.on_turn_started()
    a.on_first_audio()
    assert a.state == arbiter.SPEAKING
    a.on_speech_started()
    assert a.state == arbiter.INTERRUPTING
    a.on_barge_in()
    assert a.state == arbiter.LISTENING
    assert a.blocked_reason == "barge_in"


def test_finished_turn_does_not_overwrite_a_barge_in():
    a = arbiter.Arbiter()
    a.on_speech_started()
    a.on_barge_in()
    a.on_turn_finished()
    assert a.state == arbiter.LISTENING


def test_new_speech_clears_the_previous_block_reason():
    a = arbiter.Arbiter()
    a.on_barge_in()
    a.on_speech_started()
    assert a.blocked_reason is None


def test_snapshot_is_serializable():
    a = arbiter.Arbiter()
    a.on_turn_started()
    assert a.snapshot() == {"state": arbiter.THINKING, "blocked_reason": None}
