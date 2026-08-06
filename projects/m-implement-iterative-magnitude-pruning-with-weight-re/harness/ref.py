import numpy as np
from prune.imp import iterative_prune, magnitude_mask, apply_mask
from prune.sweep import sparsity_sweep


def dummy_model_fn(weights, masks, data):
    return [w * 0.9 + 0.1 * d for w, d in zip(weights, data)]


def dummy_eval_fn(weights):
    if isinstance(weights, list):
        return float(np.mean([np.sum(w) for w in weights]))
    return float(np.sum(weights))


def run_reference_imp(init_weights, data):
    return iterative_prune(dummy_model_fn, data, 3, 0.5, init_weights)


def run_reference_sweep(init_weights, sparsities):
    return sparsity_sweep(dummy_eval_fn, init_weights, sparsities)
