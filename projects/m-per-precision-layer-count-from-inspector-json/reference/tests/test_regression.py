import sys
sys.path.insert(0, ".")
from engineprof.inspector import count_precisions
from engineprof.profile import aggregate_profile
from engineprof.reformat import find_reformats

def test_count_precisions_basic():
    data = {"layers": [{"precision": "FP32"}, {"precision": "FP16"}, {"precision": "FP32"}]}
    res = count_precisions(data)
    assert res["FP32"] == 2
    assert res["FP16"] == 1

def test_aggregate_profile_basic():
    data = {"records": [{"time_ms": 10.0, "invocations": 1}, {"time_ms": 20.0, "invocations": 2}]}
    res = aggregate_profile(data)
    assert res["total_time"] == 50.0
    assert res["total_invocations"] == 3
    assert res["layer_count"] == 2

def test_find_reformats_detects_boundaries():
    data = {"layers": [
        {"index": 0, "precision": "FP32", "is_reformat": False},
        {"index": 1, "precision": "FP16", "is_reformat": True}
    ]}
    res = find_reformats(data)
    assert 1 in res
    assert len(res) > 0
