import sys

sys.path.insert(0, ".")
from threadperf.classifier import classify_run, classify_runs
from threadperf.metrics import compute_performance_metrics

SAMPLE_RUNS = [
    {"run_id": "x1", "engine": "openvino", "threads_allocated": 4, "physical_cores": 8, "contention_score": 0.1, "tokens_per_sec": 50.0},
    {"run_id": "x2", "engine": "onnxruntime", "threads_allocated": 16, "physical_cores": 8, "contention_score": 0.9, "tokens_per_sec": 15.0},
]


def test_over_allocation_is_unfair():
    run = {"run_id": "test", "engine": "openvino", "threads_allocated": 12, "physical_cores": 8, "contention_score": 0.1}
    assert classify_run(run) == "unfair"


def test_high_contention_is_unfair():
    run = {"run_id": "test", "engine": "onnxruntime", "threads_allocated": 4, "physical_cores": 8, "contention_score": 0.8}
    assert classify_run(run) == "unfair"


def test_metrics_ratio_computation():
    m = compute_performance_metrics(SAMPLE_RUNS)
    assert m["throughput_ratio"] > 0
    assert m["avg_openvino"] == 50.0
    assert m["avg_onnxruntime"] == 15.0
