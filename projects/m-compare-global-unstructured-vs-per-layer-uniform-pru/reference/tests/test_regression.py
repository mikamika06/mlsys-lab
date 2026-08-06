"""Regression tests for pruning strategies."""

import numpy as np
from prunecomp.metrics import evaluate_pruning_quality
from prunecomp.pruners import (
    compute_global_unstructured_mask,
    compute_per_layer_uniform_masks,
)


def test_pruning_invariants():
    """Verify pruning properties and behavior."""
    rng = np.random.default_rng(42)
    weights = {
        "l1": rng.normal(0, 1, size=(50, 50)),
        "l2": rng.normal(0, 10, size=(100, 10)),
    }
    ratio = 0.4

    u_masks, u_threshs = compute_per_layer_uniform_masks(weights, ratio)
    u_eval = evaluate_pruning_quality(weights, u_masks)

    g_masks, g_thresh = compute_global_unstructured_mask(weights, ratio)
    g_eval = evaluate_pruning_quality(weights, g_masks)

    for name in weights:
        expected_k = int(np.floor(ratio * weights[name].size))
        actual_pruned = u_eval["per_layer"][name]["pruned"]
        assert actual_pruned == expected_k

    g_sparsities = [g_eval["per_layer"][k]["sparsity"] for k in weights]
    assert not np.allclose(g_sparsities[0], g_sparsities[1])
