"""Learner written regression tests."""

from compat.adapter import transform_request, validate_response


def test_request_transformation():
    req = {"model": "test", "messages": []}
    res = transform_request("stream_with_usage", req)
    assert res["_shape"] == "stream_with_usage"
    assert res["stream"] is True
    assert "stream_options" in res


def test_response_validation():
    resp = {
        "id": "1",
        "object": "chat.completion",
        "choices": [{"message": {"role": "assistant", "content": "hello"}}],
        "usage": {"total_tokens": 10}
    }
    assert validate_response("stream_with_usage", resp) is True
