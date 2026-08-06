import math
import numpy as np

def mqa_single_kv_broadcast(Q: np.ndarray,
                            K: np.ndarray,
                            V: np.ndarray) -> np.ndarray:
    """
    Correct implementation of MQA with a single KV head.
    
    Parameters
    ----------
    Q : ndarray, shape (n_q, h, d_k)
        Query matrix for all heads.
    K : ndarray, shape (1, d_k)
        Shared key vector.
    V : ndarray, shape (1, d_v)
        Shared value vector.

    Returns
    -------
    out : ndarray, shape (n_q, h, d_v)
        Attention output broadcasted across all queries and heads.
    """
    n_q, h, d_k = Q.shape
    d_v = V.shape[1]
    
    scores = np.empty((n_q, h), dtype=Q.dtype)
    inv_sqrt_dk = 1.0 / math.sqrt(d_k)
    
    for q in range(n_q):
        for head in range(h):
            dot_val = 0.0
            for d in range(d_k):
                dot_val += Q[q, head, d] * K[0, d]
            scores[q, head] = dot_val * inv_sqrt_dk

    max_scores = np.empty((n_q, 1), dtype=Q.dtype)
    for q in range(n_q):
        m = scores[q, 0]
        for head in range(1, h):
            if scores[q, head] > m:
                m = scores[q, head]
        max_scores[q, 0] = m

    weights = np.empty((n_q, h), dtype=Q.dtype)
    for q in range(n_q):
        m = max_scores[q, 0]
        for head in range(h):
            weights[q, head] = math.exp(scores[q, head] - m)

    sum_weights = np.empty((n_q, 1), dtype=Q.dtype)
    for q in range(n_q):
        s = 0.0
        for head in range(h):
            s += weights[q, head]
        sum_weights[q, 0] = s

    for q in range(n_q):
        s = sum_weights[q, 0]
        for head in range(h):
            weights[q, head] /= s

    out = np.empty((n_q, h, d_v), dtype=Q.dtype)
    for q in range(n_q):
        for head in range(h):
            w = weights[q, head]
            for dv in range(d_v):
                out[q, head, dv] = w * V[0, dv]

    return out
