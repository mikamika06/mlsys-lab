"""Regression tests for API hardening and log redaction."""

from hardening.logging import sanitize_log_message


def test_log_redaction_removes_pii():
    entry = {
        "prompt": "Tell me about john.doe@example.com",
        "message": "User with email alice@test.org requested generation, SSN 000-12-3456",
        "user_metadata": {"ip": "192.168.1.1"}
    }
    config = {"log_level": "INFO", "redact_prompt": True}
    res = sanitize_log_message(entry, config)

    assert res["prompt"] == "[REDACTED_PROMPT]"
    assert "alice@test.org" not in res["message"]
    assert "000-12-3456" not in res["message"]
    assert res["user_metadata"]["ip"] == "[REDACTED_META]"


def test_log_redaction_preserves_clean_text():
    entry = {"message": "System startup complete", "prompt": "Hello"}
    config = {"log_level": "DEBUG", "redact_prompt": False}
    res = sanitize_log_message(entry, config)

    assert res["prompt"] == "Hello"
    assert res["message"] == "System startup complete"
