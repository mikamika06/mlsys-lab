import sys

sys.path.insert(0, ".")
from vllm_hardening.sanitizer import sanitize_response, sanitize_stream_chunk


def test_sanitize_response_removes_reasoning():
    resp = {
        "choices": [
            {
                "message": {
                    "content": "test",
                    "reasoning_content": "secret thought"
                }
            }
        ],
        "reasoning_content": "top level secret"
    }
    cleaned = sanitize_response(resp, is_untrusted=True)
    assert "reasoning_content" not in cleaned
    assert "reasoning_content" not in cleaned["choices"][0]["message"]


def test_sanitize_stream_chunk_removes_delta_reasoning():
    chunk = {
        "choices": [
            {
                "delta": {
                    "content": "token",
                    "reasoning_content": "hidden"
                }
            }
        ]
    }
    cleaned = sanitize_stream_chunk(chunk, is_untrusted=True)
    assert "reasoning_content" not in cleaned["choices"][0]["delta"]


def test_trusted_caller_preserves_reasoning():
    resp = {
        "choices": [
            {
                "message": {
                    "content": "test",
                    "reasoning_content": "allowed"
                }
            }
        ]
    }
    cleaned = sanitize_response(resp, is_untrusted=False)
    assert cleaned["choices"][0]["message"]["reasoning_content"] == "allowed"
