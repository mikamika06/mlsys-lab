"""Reference fixture generation and oracle implementations."""

import numpy as np


def generate_test_model(seed=123):
    """Generate synthetic weight dictionary for tests."""
    rng = np.random.default_rng(seed)
    return {
        "layer1": rng.normal(loc=0.0, scale=0.1, size=(100, 50)),
        "layer2": rng.normal(loc=0.0, scale=2.0, size=(200, 100)),
        "layer3": rng.exponential(scale=0.5, size=(50, 500)),
    }


def reference_uniform_prune(weights, ratio):
    """Compute reference uniform masks."""
    masks = {}
    thresholds = {}
    for k, w in weights.items():
        flat = np.abs(w).ravel()
        n_prune = int(np.floor(ratio * flat.size))
        if n_prune <= 0:
            thresh = 0.0
            mask = np.ones_like(w, dtype=bool)
        elif n_prune >= flat.size:
            thresh = float(np.max(flat)) if flat.size > 0 else 0.0
            mask = np.zeros_like(w, dtype=bool)
        else:
            p = np.partition(flat, n_prune - 1)
            thresh = float(p[n_prune - 1])
            mask = np.abs(w) > thresh
        masks[k] = mask
        thresholds[k] = thresh
    return masks, thresholds


def reference_global_prune(weights, ratio):
    """Compute reference global mask."""
    all_w = np.concatenate([np.abs(w).ravel() for w in weights.values()])
    n_prune = int(np.floor(ratio * all_w.size))
    if n_prune <= 0:
        thresh = 0.0
    elif n_prune >= all_w.size:
        thresh = float(np.max(all_w)) if all_w.size > 0 else 0.0
    else:
        p = np.partition(all_w, n_prune - 1)
        thresh = float(p[n_prune - 1])
    masks = {k: np.abs(w) > thresh for k, w in weights.items()}
    return masks, thresh
