import numpy as np


def fused_attention_scores(scores, alibi, window, soft_cap):
    scores = np.asarray(scores, dtype=np.float64)
    alibi = np.asarray(alibi, dtype=np.float64)

    x = scores + alibi
    x = soft_cap * np.tanh(x / soft_cap)

    n = x.shape[0]
    rows = np.arange(n)[:, None]
    cols = np.arange(n)[None, :]
    x = x.copy()
    x[np.abs(rows - cols) > window] = -np.inf

    row_max = np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x - row_max)
    exp_x[~np.isfinite(x)] = 0.0

    return exp_x / np.sum(exp_x, axis=1, keepdims=True)
