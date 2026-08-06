import numpy as np


def blockwise_attention(q, k, v, mask):
    scale = 1.0 / np.sqrt(q.shape[-1])
    scores = np.matmul(q, k.transpose(0, 1, 3, 2)) * scale
    scores = np.where(mask == 0, -1e4, scores)
    row_max = np.max(scores, axis=-1, keepdims=True)
    row_max = np.where(np.isneginf(row_max) | np.isnan(row_max), 0.0, row_max)
    exp_scores = np.exp(scores - row_max)
    exp_scores = np.where(mask == 0, 0.0, exp_scores)
    sum_exp = np.sum(exp_scores, axis=-1, keepdims=True)
    sum_exp_safe = np.where(sum_exp == 0.0, 1.0, sum_exp)
    weights = exp_scores / sum_exp_safe
    return np.matmul(weights, v)
