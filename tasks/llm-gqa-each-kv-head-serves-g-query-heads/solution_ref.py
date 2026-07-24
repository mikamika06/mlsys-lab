import numpy as np

def gqa_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, g: int) -> np.ndarray:
    """
    Vectorised implementation of grouped‑query attention.
    """
    n_q = Q.shape[0]
    j_indices = np.arange(n_q) // g
    K_sel = K[j_indices]          # shape (n_q, d)
    V_sel = V[j_indices]          # shape (n_q, d)
    scores = np.sum(Q * K_sel, axis=1)  # dot product per query
    return scores[:, None] * V_sel
