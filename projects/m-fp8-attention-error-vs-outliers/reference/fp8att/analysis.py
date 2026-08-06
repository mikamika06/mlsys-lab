import numpy as np


def compute_attention_error(q, k, v):
    scale = 1.0 / np.sqrt(q.shape[-1])
    scores = np.matmul(q, k.transpose(0, 1, 3, 2)) * scale
    max_val = np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores - max_val)
    sum_exp = np.sum(exp_scores, axis=-1, keepdims=True)
    attn = exp_scores / sum_exp
    return np.matmul(attn, v).astype(np.float32)
