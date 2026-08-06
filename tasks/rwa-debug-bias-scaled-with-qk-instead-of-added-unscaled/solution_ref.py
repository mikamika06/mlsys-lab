import math
import numpy as np


def sdpa_with_additive_bias(
    q: np.ndarray, k: np.ndarray, v: np.ndarray, bias: np.ndarray, scale: float
) -> np.ndarray:
    """Scaled dot-product attention with an additive bias (padding mask,
    ALiBi, relative position bias, ...).

    Matches the real formula used by e.g. torch.nn.functional's
    scaled_dot_product_attention with a float attn_mask: the QK^T product
    is scaled FIRST, and the bias is added AFTER scaling, unscaled. The
    bias represents a fixed logit offset (e.g. "-1e9 to mask this key" or
    an ALiBi slope*distance term) -- it must not be shrunk by `scale`.

    q: (n_q, d), k: (n_k, d), v: (n_k, d_v), bias: (n_q, n_k).
    Returns (n_q, d_v).
    """
    n_q, d = q.shape
    n_k, _ = k.shape
    _, d_v = v.shape

    logits = np.zeros((n_q, n_k), dtype=q.dtype)
    for i in range(n_q):
        for j in range(n_k):
            dot = 0.0
            for l in range(d):
                dot += q[i, l] * k[j, l]
            logits[i, j] = dot * scale + bias[i, j]

    w = np.zeros((n_q, n_k), dtype=q.dtype)
    for i in range(n_q):
        max_val = logits[i, 0]
        for j in range(1, n_k):
            if logits[i, j] > max_val:
                max_val = logits[i, j]
        
        row_sum = 0.0
        row_exp = []
        for j in range(n_k):
            val = math.exp(logits[i, j] - max_val)
            row_exp.append(val)
            row_sum += val
        
        for j in range(n_k):
            w[i, j] = row_exp[j] / row_sum

    out = np.zeros((n_q, d_v), dtype=q.dtype)
    for i in range(n_q):
        for j in range(d_v):
            val = 0.0
            for l in range(n_k):
                val += w[i, l] * v[l, j]
            out[i, j] = val

    return out
