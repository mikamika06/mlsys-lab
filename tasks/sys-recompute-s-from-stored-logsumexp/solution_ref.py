import math
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
    inv_sqrt_d = 1.0 / math.sqrt(d)
    
    n = Q.shape[0]
    m = K.shape[0]
    
    out = np.empty((n, m), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            dot = 0.0
            for k in range(d):
                dot += Q[i, k] * K[j, k]
            score = dot * inv_sqrt_d
            out[i, j] = math.exp(score - lse[i])
            
    return out
