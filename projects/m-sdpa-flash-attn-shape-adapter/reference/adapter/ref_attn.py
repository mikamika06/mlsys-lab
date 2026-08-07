import numpy as np


def reference_attention(q, k, v, scale=None):
    b, h, n_q, d = q.shape
    if scale is None:
        scale = 1.0 / np.sqrt(d)
    scores = np.matmul(q, np.transpose(k, (0, 1, 3, 2))) * scale
    max_scores = np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores - max_scores)
    sum_exp = np.sum(exp_scores, axis=-1, keepdims=True)
    attn_weights = exp_scores / sum_exp
    out = np.matmul(attn_weights, v)
    lse = max_scores.squeeze(-1) + np.log(sum_exp.squeeze(-1))
    return out, lse


def compute_lse(q, k, scale=None):
    b, h, n_q, d = q.shape
    if scale is None:
        scale = 1.0 / np.sqrt(d)
    scores = np.matmul(q, np.transpose(k, (0, 1, 3, 2))) * scale
    max_scores = np.max(scores, axis=-1, keepdims=True)
    sum_exp = np.sum(np.exp(scores - max_scores), axis=-1, keepdims=True)
    lse = max_scores.squeeze(-1) + np.log(sum_exp.squeeze(-1))
    return lse
