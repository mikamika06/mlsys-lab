import math
import numpy as np

def scaled_dot_product_attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    *,
    causal: bool = False
) -> np.ndarray:
    """
    Compute scaled dot‑product attention with optional causal masking.

    Parameters
    ----------
    Q, K : np.ndarray
        Query and key tensors of shape (B, N, d_k).
    V : np.ndarray
        Value tensor of shape (B, N, d_v).
    causal : bool, default False
        If True, apply a lower‑triangular causal mask.

    Returns
    -------
    out : np.ndarray
        Attention output of shape (B, N, d_v) with dtype float64.
    """
    Q = np.asarray(Q, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)

    B, N, d_k = Q.shape
    d_v = V.shape[-1]

    scores = np.empty((B, N, N), dtype=np.float64)
    scale = math.sqrt(d_k)

    for b in range(B):
        for i in range(N):
            for j in range(N):
                acc = 0.0
                for k in range(d_k):
                    acc += Q[b, i, k] * K[b, j, k]
                scores[b, i, j] = acc / scale

    if causal:
        for b in range(B):
            for i in range(N):
                for j in range(N):
                    if j > i:
                        scores[b, i, j] = -float("inf")

    attn_weights = np.empty((B, N, N), dtype=np.float64)
    for b in range(B):
        for i in range(N):
            max_val = scores[b, i, 0]
            for j in range(1, N):
                if scores[b, i, j] > max_val:
                    max_val = scores[b, i, j]

            sum_exp = 0.0
            for j in range(N):
                val = math.exp(scores[b, i, j] - max_val)
                attn_weights[b, i, j] = val
                sum_exp += val

            for j in range(N):
                attn_weights[b, i, j] /= sum_exp

    out = np.empty((B, N, d_v), dtype=np.float64)
    for b in range(B):
        for i in range(N):
            for j in range(d_v):
                acc = 0.0
                for k in range(N):
                    acc += attn_weights[b, i, k] * V[b, k, j]
                out[b, i, j] = acc

    return out
