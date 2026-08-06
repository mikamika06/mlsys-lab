"""Pruning mask and threshold utilities."""

import numpy as np


def compute_per_layer_uniform_masks(weights, sparsity_ratio):
    """Compute per-layer uniform pruning masks."""
    masks = {}
    thresholds = {}
    for name, w in weights.items():
        flat_abs = np.abs(w).ravel()
        k = int(np.floor(sparsity_ratio * flat_abs.size))
        if k <= 0:
            thresh = 0.0
            mask = np.ones_like(w, dtype=bool)
        elif k >= flat_abs.size:
            thresh = float(np.max(flat_abs)) if flat_abs.size > 0 else 0.0
            mask = np.zeros_like(w, dtype=bool)
        else:
            partitioned = np.partition(flat_abs, k - 1)
            thresh = float(partitioned[k - 1])
            mask = np.abs(w) > thresh
        masks[name] = mask
        thresholds[name] = thresh
    return masks, thresholds


def compute_global_unstructured_mask(weights, sparsity_ratio):
    """Compute global unstructured pruning mask."""
    all_abs = np.concatenate([np.abs(w).ravel() for w in weights.values()])
    total = all_abs.size
    k = int(np.floor(sparsity_ratio * total))
    if k <= 0:
        global_thresh = 0.0
    elif k >= total:
        global_thresh = float(np.max(all_abs)) if total > 0 else 0.0
    else:
        partitioned = np.partition(all_abs, k - 1)
        global_thresh = float(partitioned[k - 1])

    masks = {}
    for name, w in weights.items():
        masks[name] = np.abs(w) > global_thresh
    return masks, global_thresh
