import sys
sys.path.insert(0, ".")
from perfanalysis.sweep import find_max_concurrency
from perfanalysis.latency import attribute_delta
from perfanalysis.throughput import compute_tokens_metrics

def test_find_max_concurrency_basic():
    data = [{"concurrency": 1, "p99_latency_ms": 30.0}, {"concurrency": 2, "p99_latency_ms": 50.0}]
    assert find_max_concurrency(data, 40.0) == 1

def test_attribute_delta_values():
    off = {"queue_ms": 1.0, "compute_ms": 10.0, "total_ms": 11.0}
    on = {"queue_ms": 3.0, "compute_ms": 12.0, "total_ms": 15.0}
    res = attribute_delta(off, on)
    assert abs(res["queue_delta"] - 2.0) < 1e-5
    assert abs(res["compute_delta"] - 2.0) < 1e-5

def test_compute_tokens_metrics():
    fix = {"total_tokens": 1000, "gpu_count": 1, "active_users": 10, "total_time_sec": 2.0}
    res = compute_tokens_metrics(fix)
    assert abs(res["tokens_per_sec_gpu"] - 500.0) < 1e-5
    assert abs(res["tokens_per_sec_user"] - 50.0) < 1e-5
