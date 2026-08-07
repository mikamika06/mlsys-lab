import sys
sys.path.insert(0, ".")
from engineprof.inspector import count_precisions
from engineprof.profile import aggregate_profile
from engineprof.reformat import find_reformats

def test_count_precisions_keys():
    data = {"layers": [{"precision": "FP32"}, {"precision": "FP16"}]}
    res = count_precisions(data)
    assert "FP32" in res
    assert "FP16" in res

def test_aggregate_profile_total():
    data = {"records": [{"time_ms": 10.0}, {"time_ms": 20.0}]}
    res = aggregate_profile(data)
    assert res["total_time"] == 30.0

def test_find_reformats_detects_change():
    data = {"layers": [{"index": 0, "precision": "FP32"}, {"index": 1, "precision": "FP16"}]}
    res = find_reformats(data)
    assert 1 in res
