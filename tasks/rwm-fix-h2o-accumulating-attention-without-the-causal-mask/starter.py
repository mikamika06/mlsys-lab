import numpy as np


def select_heavy_hitters(attn_scores: np.ndarray, budget: int) -> np.ndarray:
    # TODO: this computes importance from non-causal attention. Future tokens
    # are included in the accumulation and can incorrectly become heavy hitters.
    scores = np.asarray(attn_scores, dtype=np.float64)

    shifted = scores - np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    importance = np.sum(probs, axis=0)
    order = sorted(range(scores.shape[0]), key=lambda i: (-importance[i], i))
    return np.asarray(order[:budget], dtype=np.int64)
