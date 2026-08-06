import sys

sys.path.insert(0, ".")
from kernel_analysis.bottleneck import classify_bottleneck
from kernel_analysis.metrics import compute_metrics


def test_compute_metrics_basic():
    k = {
        "id": "test",
        "flops": 1.0e8,
        "bytes_transferred": 1.0e6,
        "duration_ms": 0.1,
        "peak_gflops": 1000.0,
        "peak_bandwidth_gbps": 1000.0,
        "active_warps": 16,
        "max_warps": 32,
    }
    m = compute_metrics(k)
    assert m["gflops"] == 1000.0
    assert m["intensity"] == 100.0
    assert m["compute_pct"] == 100.0
    assert m["gap_pct"] == 0.0


def test_classification_latency_bound():
    k = {
        "id": "test_low",
        "flops": 1.0e6,
        "bytes_transferred": 1.0e5,
        "duration_ms": 0.1,
        "peak_gflops": 1000.0,
        "peak_bandwidth_gbps": 1000.0,
        "active_warps": 2,
        "max_warps": 32,
    }
    b = classify_bottleneck(k)
    assert b == "latency-bound-low-occupancy"
