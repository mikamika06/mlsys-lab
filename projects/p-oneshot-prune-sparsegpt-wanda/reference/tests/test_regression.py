import sys
sys.path.insert(0, ".")
import numpy as np
from prune.layer import prune_unstructured, correct_bias

def test_prune_sparsity():
    w = np.random.randn(10, 20)
    scores = np.random.rand(10, 20)
    w_p, mask = prune_unstructured(w, scores, 0.5)
    assert np.mean(~mask) == 0.5
    assert np.all(w_p[mask] == w[mask])
    assert np.all(w_p[~mask] == 0)

def test_correct_bias():
    w = np.ones((5, 10))
    w_p = np.zeros((5, 10))
    x = np.ones((10, 100))
    bias = correct_bias(w, w_p, x)
    assert np.all(bias == 10.0)
