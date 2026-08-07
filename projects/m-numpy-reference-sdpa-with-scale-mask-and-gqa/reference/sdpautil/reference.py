import numpy as np


def sdpa_reference(q, k, v, scale=None, mask=None):
    """Compute Scaled Dot-Product Attention using NumPy."""
    q = np.asarray(q, dtype=np.float64)
    k = np.asarray(k, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)

    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("Inputs q, k, v must be 4D arrays (B, H, S, D)")

    b_q, h_q, s_q, d_q = q.shape
    b_k, h_k, s_k, d_k = k.shape
    b_v, h_v, s_v, d_v = v.shape

    if (b_q != b_k) or (b_q != b_v):
        raise ValueError("Batch dimensions must match")

    if d_q != d_k or d_q != d_v:
        raise ValueError("Head dimension d must match across q, k, v")

    if s_k != s_v or h_k != h_v:
        raise ValueError("Key and Value sequence length and head counts must match")

    if h_q % h_k != 0:
        raise ValueError("Query heads must be a multiple of Key/Value heads for GQA")

    gqa_ratio = h_q // h_k
    if gqa_ratio > 1:
        k = np.repeat(k, gqa_ratio, axis=1)
        v = np.repeat(v, gqa_ratio, axis=1)

    if scale is None:
        scale = 1.0 / np.sqrt(d_q)

    scores = np.matmul(q, np.swapaxes(k, -2, -1)) * scale

    if mask is not None:
        mask = np.asarray(mask)
        if mask.dtype == bool:
            scores = np.where(mask, scores, -1e9)
        else:
            scores = scores + mask

    scores_max = np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores - scores_max)
    attn_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

    out = np.matmul(attn_weights, v)
    return out
