import numpy as np


def select_heavy_hitters(attn_scores: np.ndarray, budget: int) -> np.ndarray:
    scores = np.asarray(attn_scores, dtype=np.float64)
    n = scores.shape[0]

    masked = scores.copy()
    masked[np.triu(np.ones((n, n), dtype=bool), k=1)] = -np.inf

    shifted = masked - np.max(masked, axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    importance = np.sum(probs, axis=0)
    order = sorted(range(n), key=lambda i: (-importance[i], i))
    return np.asarray(order[:budget], dtype=np.int64)
