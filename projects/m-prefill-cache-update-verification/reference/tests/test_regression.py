import sys
import numpy as np

sys.path.insert(0, ".")
from cacheval.verify import verify_prefill_update
from cacheval.latency import analyze_latencies
from cacheval.memory import compute_peak_memory_delta


def test_verify_identical_caches():
    k = np.zeros((1, 8, 16, 64), dtype=np.float32)
    v = np.ones((1, 8, 16, 64), dtype=np.float32)
    cache = [(k, v)]
    assert verify_prefill_update(cache, cache, 1e-5) is True


def test_analyze_latencies_basic():
    res = analyze_latencies([10.0, 12.0], [2.0, 2.5])
    assert res["valid"] is True
    assert res["stateful_mean"] < res["stateless_mean"]


def test_compute_peak_memory_delta_basic():
    res = compute_peak_memory_delta(1000, 1200)
    assert res["delta_bytes"] == 200
