"""Redaction: what may leave, and what must never leave."""

import router.redact as redact


def test_api_keys_are_removed():
    text = "here: sk-abcdefghijklmnop1234 and ghp_ABCDEFGHIJKLMNOPQRSTUV"
    out, counts = redact.redact(text)
    assert "sk-" not in out
    assert "ghp_" not in out
    assert counts["secret"] == 2


def test_bearer_tokens_are_removed():
    out, counts = redact.redact("Authorization: Bearer abcdef1234567890abcdef")
    assert "abcdef1234567890" not in out
    assert counts["secret"] == 1


def test_windows_paths_are_removed():
    out, counts = redact.redact(r"open C:\Users\User\Documents\secret.txt now")
    assert "C:\\Users" not in out
    assert counts["path"] == 1


def test_unc_paths_are_removed():
    out, _ = redact.redact(r"copy \\server\share\file.txt")
    assert "server" not in out


def test_emails_are_removed():
    out, counts = redact.redact("mail me at name@example.com please")
    assert "name@example.com" not in out
    assert counts["email"] == 1


def test_long_digit_runs_are_removed():
    out, counts = redact.redact("card 4111111111111111 ok")
    assert "4111111111111111" not in out
    assert counts["digits"] == 1


def test_short_numbers_survive():
    out, _ = redact.redact("it took 12 seconds")
    assert "12" in out


def test_disabled_rules_leave_text_alone():
    out, counts = redact.redact("name@example.com", {"emails": False})
    assert out == "name@example.com"
    assert counts == {}


def test_redact_messages_does_not_mutate():
    messages = [{"role": "user", "content": "sk-abcdefghijklmnop1234"}]
    out, counts = redact.redact_messages(messages)
    assert "sk-" in messages[0]["content"]           # input untouched
    assert "sk-" not in out[0]["content"]
    assert counts["secret"] == 1


def test_redact_messages_leaves_structure():
    messages = [{"role": "system", "content": "a"}, {"role": "user", "content": "b"}]
    out, counts = redact.redact_messages(messages)
    assert [m["role"] for m in out] == ["system", "user"]
    assert counts == {}


def test_cap_marks_truncation():
    assert redact.cap("one two three four", 7).endswith("...")
    assert redact.cap("short", 100) == "short"
    assert redact.cap("anything", 0) == "anything"
