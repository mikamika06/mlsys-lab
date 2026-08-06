"""Regression tests."""
from workloads.classifier import classify_workload

def test_strict_privacy_self_host():
    w = {
        "id": "test_w",
        "data_privacy": "strict",
        "requires_finetuning": False,
        "avg_qps": 0.1,
        "peak_qps": 0.5,
        "monthly_tokens": 1000000
    }
    res = classify_workload(w)
    assert res["deployment"] == "self-host"

def test_high_volume_self_host():
    w = {
        "id": "test_w2",
        "data_privacy": "standard",
        "requires_finetuning": False,
        "avg_qps": 50.0,
        "peak_qps": 100.0,
        "monthly_tokens": 5000000000
    }
    res = classify_workload(w)
    assert res["deployment"] == "self-host"
