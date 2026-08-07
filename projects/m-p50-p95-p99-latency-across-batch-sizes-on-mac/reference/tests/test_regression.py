import sys
import numpy as np

sys.path.insert(0, ".")
from latbench.tradeoff import derive_optimal_batch_sizes


def test_slo_validation():
    profile = {
        1: {"p50": 10.0, "p95": 12.0, "p99": 14.0},
        2: {"p50": 12.0, "p95": 16.0, "p99": 18.0},
        4: {"p50": 15.0, "p95": 25.0, "p99": 45.0},
        8: {"p50": 20.0, "p95": 40.0, "p99": 80.0},
    }
    slo = 30.0
    res = derive_optimal_batch_sizes(profile, slo)
    assert res["throughput_optimal_b"] is not None
    chosen_b = res["throughput_optimal_b"]
    assert profile[chosen_b]["p99"] <= slo, f"Batch size {chosen_b} violates p99 SLO of {slo}"
    assert chosen_b == 2, f"Expected batch size 2, got {chosen_b}"
