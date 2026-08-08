import sys
sys.path.insert(0, ".")
from ssevall.validator import validate_sse_stream, FramingError
from ssevall.mapper import map_openai_request
from ssevall.divergence import locate_divergence_tokens


def test_validator_parses_valid_stream():
    raw = b"data: {\"id\": \"1\"}\n\ndata: {\"id\": \"2\"}\n\n"
    events = validate_sse_stream(raw)
    assert len(events) == 2
    assert events[0]["data"] == "{\"id\": \"1\"}"


def test_mapper_handles_extra_body():
    req = {"temperature": 0.7, "extra_body": {"guided_decoding": "json"}}
    params = map_openai_request(req)
    assert params["temperature"] == 0.7
    assert params["guided_decoding"] == "json"


def test_divergence_locates_tokens():
    chat = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\nHello"
    comp = "Hello"
    div = locate_divergence_tokens(chat, comp)
    assert len(div) > 0
    assert any("<|im_start|>" in t for t in div)
