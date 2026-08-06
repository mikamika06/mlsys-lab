import numpy as np
from prune.imp import magnitude_mask, apply_mask


def sparsity_sweep(eval_fn, weights, sparsities):
    results = {}
    for sp in sparsities:
        mask = magnitude_mask(weights, sp)
        pruned_weights = apply_mask(weights, mask)
        score = eval_fn(pruned_weights)
        results[sp] = score
    return results
