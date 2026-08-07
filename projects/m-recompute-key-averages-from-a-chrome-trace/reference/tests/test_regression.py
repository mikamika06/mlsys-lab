import sys
sys.path.insert(0, ".")
from tracex.parse import compute_key_averages
from tracex.metrics import compute_gpu_metrics
from tracex.diff import largest_self_time_regression

def test_key_averages_non_negative():
    trace = {"traceEvents": [{"ph": "X", "name": "matmul", "ts": 100, "dur": 50, "pid": 1, "tid": 1}]}
    res = compute_key_averages(trace)
    for k, v in res.items():
        assert v["count"] > 0
        assert v["total_duration"] >= 0
        assert v["total_self_time"] >= 0

def test_gpu_metrics_bounds():
    trace = {"traceEvents": [{"ph": "X", "name": "kernel1", "ts": 0, "dur": 100, "pid": 2, "tid": 2}]}
    res = compute_gpu_metrics(trace)
    assert 0.0 <= res["busy_fraction"] <= 1.0
    assert res["idle_gaps"] >= 0.0

def test_largest_regression_positive():
    t_a = {"traceEvents": [{"ph": "X", "name": "op1", "ts": 0, "dur": 10, "pid": 1, "tid": 1}]}
    t_b = {"traceEvents": [{"ph": "X", "name": "op1", "ts": 0, "dur": 100, "pid": 1, "tid": 1}]}
    res = largest_self_time_regression(t_a, t_b)
    assert res["name"] == "op1"
    assert res["regression"] == 90.0
