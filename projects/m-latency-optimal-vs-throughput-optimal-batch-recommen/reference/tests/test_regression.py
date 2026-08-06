import sys
import numpy as np

sys.path.insert(0, ".")
from batchopt.profile import compute_curves
from batchopt.recommender import recommend_batches


def test_latency_optimal_batch_size_is_leq_throughput_optimal():
    profile = {
        "batch_sizes": [1, 2, 4, 8, 16, 32],
        "draft_accept_rate": 0.6,
        "base_latency_ms": 12.0,
        "overhead_ms": 2.0,
    }
    res = recommend_batches(profile)
    assert res["latency_optimal_batch"] <= res["throughput_optimal_batch"]


def test_curves_non_negative():
    profile = {
        "batch_sizes": [1, 4, 16],
        "draft_accept_rate": 0.8,
        "base_latency_ms": 10.0,
        "overhead_ms": 1.0,
    }
    lat, tp = compute_curves(profile)
    assert np.all(lat > 0)
    assert np.all(tp > 0)
