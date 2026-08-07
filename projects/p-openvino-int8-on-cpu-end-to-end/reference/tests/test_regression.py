import sys
sys.path.insert(0, ".")
from cpuopt.optimizer import optimize_pipeline

def test_optimization_within_budget():
    data = [None]
    res = optimize_pipeline("dummy_model", data, target_latency_ms=80.0)
    assert res["pipeline_ok"] is True
    assert res["latency_ms"] < 80.0
    assert res["accuracy_loss"] < 0.02
