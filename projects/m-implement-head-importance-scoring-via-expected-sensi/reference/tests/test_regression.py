import sys
import numpy as np
sys.path.insert(0, ".")
from headprune.importance import compute_importance
from headprune.greedy import compute_removal_order
from headprune.latency import measure_latency

def test_importance_shape():
    rng = np.random.default_rng(0)
    acts = rng.normal(size=(2, 4, 8, 16))
    grads = rng.normal(size=(2, 4, 8, 16))
    res = compute_importance(acts, grads)
    assert res.shape == (4, 16)

def test_removal_order_length():
    mat = np.array([[0.1, 0.5], [0.3, 0.2]])
    order = compute_removal_order(mat)
    assert len(order) == 4
    assert order[0] == (0, 0)

def test_latency_bounds():
    lat = measure_latency(100.0, 10, 20)
    assert 0.0 < lat < 100.0
