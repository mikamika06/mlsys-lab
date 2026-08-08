import sys
sys.path.insert(0, ".")
import numpy as np
from prune.methods import magnitude_prune, wanda_prune, evaluate_quality


def test_magnitude_sparsity():
    w = np.random.randn(16, 16)
    _, mask = magnitude_prune(w, 0.5)
    sparsity = 1.0 - np.mean(mask)
    assert np.isclose(sparsity, 0.5, atol=0.1)


def test_mask_is_boolean():
    w = np.random.randn(16, 16)
    _, mask = magnitude_prune(w, 0.25)
    assert mask.dtype == bool


def test_pruned_weights_shape_preserved():
    w = np.random.randn(16, 16)
    w_pruned, _ = magnitude_prune(w, 0.5)
    assert w_pruned.shape == w.shape
