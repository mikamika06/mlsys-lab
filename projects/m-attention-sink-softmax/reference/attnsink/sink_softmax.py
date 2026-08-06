import numpy as np


def attention_sink_softmax(Q: np.ndarray, K: np.ndarray, V: np.ndarray, sink_size: int, window_size: int):
    L, d_k = Q.shape
    _, d_v = V.shape
    scale = 1.0 / np.sqrt(d_k)

    scores = (Q @ K.T) * scale

    rows = np.arange(L)[:, None]
    cols = np.arange(L)[None, :]

    causal = cols <= rows
    sink_mask = cols < sink_size
    win_mask = cols >= (rows - window_size + 1)
    mask = causal & (sink_mask | win_mask)

    masked_scores = np.where(mask, scores, -1e9)
    max_scores = np.max(masked_scores, axis=-1, keepdims=True)
    exp_scores = np.exp(masked_scores - max_scores)
    exp_scores[~mask] = 0.0

    sum_exp = np.sum(exp_scores, axis=-1, keepdims=True)
    weights = exp_scores / np.maximum(sum_exp, 1e-12)

    out = weights @ V
    lse = np.squeeze(max_scores, axis=-1) + np.log(np.squeeze(sum_exp, axis=-1))
    return out, lse
