"""Log redaction and sanitization module for request payloads."""

import re

PII_PATTERNS = [
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
    r"\b\d{3}-\d{2}-\d{4}\b",
    r"\b(?:\d{4}[- ]?){3}\d{4}\b",
]


def sanitize_log_message(log_entry: dict, config: dict) -> dict:
    sanitized = dict(log_entry)
    log_level = config.get("log_level", "INFO").upper()
    redact_prompt = config.get("redact_prompt", True)

    if redact_prompt and "prompt" in sanitized:
        sanitized["prompt"] = "[REDACTED_PROMPT]"

    if "message" in sanitized and isinstance(sanitized["message"], str):
        msg = sanitized["message"]
        for pat in PII_PATTERNS:
            msg = re.sub(pat, "[REDACTED_PII]", msg)
        sanitized["message"] = msg

    if log_level in ("INFO", "WARNING", "ERROR"):
        if "user_metadata" in sanitized:
            sanitized["user_metadata"] = {
                k: "[REDACTED_META]" for k in sanitized["user_metadata"]
            }

    return sanitized
