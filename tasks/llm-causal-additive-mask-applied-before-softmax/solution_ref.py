import numpy as np

def causal_masked_softmax(scores: np.ndarray) -> np.ndarray:
    mask = np.tril(np.ones_like(scores, dtype=bool))
    masked_scores = scores.copy()
    masked_scores[~mask] = -np.inf
    exp_scores = np.exp(masked_scores)
    row_sums = exp_scores.sum(axis=-1, keepdims=True)
    return (exp_scores / row_sums).astype(np.float64)
