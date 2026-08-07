import sys
sys.path.insert(0, ".")
from ssevall.validator import validate_sse_stream
from ssevall.mapper import map_openai_request_to_sampling_params
from ssevall.divergence import locate_divergence_tokens

def test_validator_detects_interleaving():
    chunks = [
        "id: stream_1\ndata: hello\n\n",
        "id: stream_2\ndata: world\n\n",
        "id: stream_1\ndata: duplicate\n\n"
    ]
    res = validate_sse_stream(chunks)
    assert res["interleaved"] is True

def test_mapper_includes_extra_body():
    req = {"temperature": 0.5, "extra_body": {"guided_decoding": "json"}}
    params = map_openai_request_to_sampling_params(req)
    assert params["guided_decoding"] == "json"
    assert params["temperature"] == 0.5

def test_divergence_detection():
    chat = "<|im_start|>system\nYou are helpful.<|im_end|>\nHello"
    comp = "Hello"
    res = locate_divergence_tokens(chat, comp)
    assert isinstance(res["divergence_index"], int)
