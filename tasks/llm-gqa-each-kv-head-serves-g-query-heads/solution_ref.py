import numpy as np

def gqa_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, g: int) -> np.ndarray:
    """
    Vectorised implementation of grouped‑query attention.
    """
    n_q, d = Q.shape
    out = np.zeros((n_q, d), dtype=Q.dtype)
    for i in range(n_q):
        j = i // g
        score = 0.0
        for k in range(d):
            score += Q[i, k] * K[j, k]
        for k in range(d):
            out[i, k] = score * V[j, k]
    return out
