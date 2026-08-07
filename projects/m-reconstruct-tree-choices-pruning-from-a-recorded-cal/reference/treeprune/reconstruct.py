import numpy as np


def reconstruct_tree_choices(fixture, threshold=0.1):
    exps = np.exp(fixture - np.max(fixture, axis=-1, keepdims=True))
    probs = exps / np.sum(exps, axis=-1, keepdims=True)
    mask = probs > threshold
    return mask.astype(np.float64)
