import numpy as np


def recompute_probs_from_lse(Q: np.ndarray, K: np.ndarray, lse: np.ndarray) -> np.ndarray:
    """Recompute attention probabilities from Q, K and a stored per-row
    logsumexp, without ever computing a row max or normalizing by a row sum.

    P = exp(Q @ K.T / sqrt(d) - lse[:, None])
    """
    Q = np.asarray(Q, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    lse = np.asarray(lse, dtype=np.float64)
    d = Q.shape[1]
    scores = (Q @ K.T) / np.sqrt(d)
    return np.exp(scores - lse[:, None])
