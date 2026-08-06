import math
import numpy as np


def sliding_window_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, w: int) -> np.ndarray:
    """Single-head scaled dot-product attention with a Mistral sliding-window mask.

    Query i attends only to keys j with i - w < j <= i (the w most recent keys,
    including itself). Masked positions are set to -inf before the softmax.
    """
    Q = np.asarray(Q, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    n, d = Q.shape

    scale = math.sqrt(d)
    out = np.zeros((n, d), dtype=np.float64)

    for i in range(n):
        scores = [float("-inf")] * n
        max_val = float("-inf")
        for j in range(n):
            if j <= i and (i - j) < w:
                dot = 0.0
                for k in range(d):
                    dot += Q[i, k] * K[j, k]
                score = dot / scale
                scores[j] = score
                if score > max_val:
                    max_val = score

        sum_e = 0.0
        p = [0.0] * n
        for j in range(n):
            if j <= i and (i - j) < w:
                e = math.exp(scores[j] - max_val)
                p[j] = e
                sum_e += e

        for j in range(n):
            if j <= i and (i - j) < w:
                p_val = p[j] / sum_e
                for k in range(d):
                    out[i, k] += p_val * V[j, k]

    return out
