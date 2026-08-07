import sys
sys.path.insert(0, ".")
from glooprof.profiler import extract_gloo_self_time
from glooprof.fingerprint import fingerprint_trace

def test_extract_positive_self_time():
    trace = {"events": [{"name": "gloo:all_reduce", "self_time": 15.0}]}
    val = extract_gloo_self_time(trace)
    assert val >= 0.0

def test_fingerprint_valid_output():
    trace = {"events": [{"name": "gloo:all_reduce", "args": {"comm_pattern": "dp"}}]}
    res = fingerprint_trace(trace)
    assert res in ("dp", "tp")
