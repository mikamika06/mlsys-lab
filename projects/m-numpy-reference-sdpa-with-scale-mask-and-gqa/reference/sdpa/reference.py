import numpy as np


def numpy_sdpa(query, key, value, attn_mask=None, is_causal=False, scale=None):
    B, H_q, L_q, D = query.shape
    _, H_kv, L_kv, _ = key.shape

    if scale is None:
        scale = 1.0 / np.sqrt(D)

    repeats = H_q // H_kv
    k = np.repeat(key, repeats, axis=1)
    v = np.repeat(value, repeats, axis=1)

    scores = np.matmul(query, k.transpose(0, 1, 3, 2)) * scale

    if is_causal:
        mask = np.triu(np.ones((L_q, L_kv), dtype=bool), k=1)
        scores[..., mask] = -np.inf

    if attn_mask is not None:
        scores[..., ~attn_mask] = -np.inf

    scores_max = np.max(scores, axis=-1, keepdims=True)
    scores_max[scores_max == -np.inf] = 0
    exp_scores = np.exp(scores - scores_max)
    exp_scores[scores == -np.inf] = 0

    weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

    return np.matmul(weights, v)
