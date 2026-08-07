import numpy as np


def windowed_attention(q: np.ndarray, k: np.ndarray, v: np.ndarray, mask: np.ndarray) -> np.ndarray:
    d_k = q.shape[-1]
    scores = np.matmul(q, k.swapaxes(-1, -2)) / np.sqrt(d_k)

    scores = np.where(mask, scores, -1e9)
    scores -= np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores)
    probs = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

    return np.matmul(probs, v)
