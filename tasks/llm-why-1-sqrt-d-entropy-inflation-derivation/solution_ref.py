import numpy as np


def entropy_inflation_ratio(Q: np.ndarray, K: np.ndarray) -> float:
    d = Q.shape[1]
    scores = Q @ K.T

    def mean_entropy(x):
        x = x - np.max(x, axis=1, keepdims=True)
        p = np.exp(x)
        p = p / np.sum(p, axis=1, keepdims=True)
        return float(np.mean(-np.sum(p * np.log(p + 1e-12), axis=1)))

    return mean_entropy(scores / np.sqrt(d)) / mean_entropy(scores)
