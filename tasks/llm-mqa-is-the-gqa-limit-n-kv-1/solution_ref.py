import math
import numpy as np

def gqa_limit_nkv_1(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    """
    Compute multi‑query attention (GQA with n_kv=1).

    Parameters
    ----------
    Q : np.ndarray
        Queries of shape (B, N_q, d_k).
    K : np.ndarray
        Keys of shape (B, N_k, d_k).
    V : np.ndarray
        Values of shape (B, N_v, d_v).

    Returns
    -------
    np.ndarray
        Attention output of shape (B, N_q, d_v).
    """
    B, N_q, d_k = Q.shape
    _, N_k, _ = K.shape
    _, N_v, d_v = V.shape

    scale = math.sqrt(d_k)
    out = np.empty((B, N_q, d_v), dtype=Q.dtype)

    for b in range(B):
        for q in range(N_q):
            max_score = -float('inf')
            scores = [0.0] * N_k
            for k in range(N_k):
                dot = 0.0
                for d in range(d_k):
                    dot += Q[b, q, d] * K[b, k, d]
                score = dot / scale
                scores[k] = score
                if score > max_score:
                    max_score = score

            sum_exp = 0.0
            weights = [0.0] * N_k
            for k in range(N_k):
                w = math.exp(scores[k] - max_score)
                weights[k] = w
                sum_exp += w

            for v_dim in range(d_v):
                val = 0.0
                for k in range(N_k):
                    val += (weights[k] / sum_exp) * V[b, k, v_dim]
                out[b, q, v_dim] = val

    return out
