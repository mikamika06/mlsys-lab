import numpy as np


def ring_attention(q, k, v, scale):
    seq_len, dim = q.shape
    scores = np.dot(q, k.T) * scale
    max_val = np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores - max_val)
    sum_exp = np.sum(exp_scores, axis=-1, keepdims=True)
    out = np.dot(exp_scores, v) / sum_exp
    return out
