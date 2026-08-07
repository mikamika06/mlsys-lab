import sys
import numpy as np

sys.path.insert(0, ".")
from ringattn.imbalance import compute_imbalance
from ringattn.simulate import ring_attention_simulate, single_process_reference


def test_imbalance_bounds():
    res = compute_imbalance(1024, 4)
    assert res["imbalance_ratio"] > 1.0
    assert len(res["workloads"]) == 4


def test_ring_matches_reference():
    np.random.seed(42)
    q = np.random.randn(64, 32)
    k = np.random.randn(64, 32)
    v = np.random.randn(64, 32)
    out_ref = single_process_reference(q, k, v)
    out_ring = ring_attention_simulate(q, k, v, 4)
    np.testing.assert_allclose(out_ref, out_ring, atol=1e-5, rtol=1e-5)
