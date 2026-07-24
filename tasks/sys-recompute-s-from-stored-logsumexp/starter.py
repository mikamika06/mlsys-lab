import numpy as np


def recompute_probs_from_lse(Q: np.ndarray, K: np.ndarray, lse: np.ndarray) -> np.ndarray:
    """Recompute attention probabilities from Q, K and a stored per-row
    logsumexp, without ever computing a row max or normalizing by a row sum.

    P = exp(Q @ K.T / sqrt(d) - lse[:, None])
    """
    raise NotImplementedError('your code here')
