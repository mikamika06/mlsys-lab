import sys
sys.path.insert(0, ".")
from sloclassify.classify import classify_violation
from sloclassify.parser import parse_request

def test_queueing_dominant():
    req = {"request_id": 1, "arrival_time": 0, "start_time": 100, "finish_time": 120, "prompt_tokens": 10, "output_tokens": 5, "prefill_time": 5}
    parsed = parse_request(req)
    assert classify_violation(parsed, 50) == "queueing"

def test_long_prefill_dominant():
    req = {"request_id": 2, "arrival_time": 0, "start_time": 2, "finish_time": 120, "prompt_tokens": 100, "output_tokens": 5, "prefill_time": 90}
    parsed = parse_request(req)
    assert classify_violation(parsed, 50) == "long-prefill"

def test_long_output_dominant():
    req = {"request_id": 3, "arrival_time": 0, "start_time": 2, "finish_time": 200, "prompt_tokens": 10, "output_tokens": 100, "prefill_time": 5}
    parsed = parse_request(req)
    assert classify_violation(parsed, 50) == "long-output"
