import math
import numpy as np


def causal_self_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Causal scaled dot-product self-attention.

    Q, K, V: (n, d). Row i may only attend to keys/values at position
    <= i. Masking is applied to the LOGITS (score[i, j] = -inf for j > i)
    BEFORE softmax, so every row's probabilities still sum to 1 over the
    positions it is allowed to see. Returns (n, d).
    """
    n, d = Q.shape
    sqrt_d = math.sqrt(d)
    scores = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if j > i:
                scores[i][j] = -float('inf')
            else:
                dot = 0.0
                for k in range(d):
                    dot += Q[i, k] * K[j, k]
                scores[i][j] = dot / sqrt_d

    for i in range(n):
        max_val = scores[i][0]
        for j in range(1, n):
            if scores[i][j] > max_val:
                max_val = scores[i][j]
        for j in range(n):
            scores[i][j] -= max_val

    probs = [[0.0] * n for _ in range(n)]
    for i in range(n):
        row_sum = 0.0
        for j in range(n):
            val = math.exp(scores[i][j])
            probs[i][j] = val
            row_sum += val
        for j in range(n):
            probs[i][j] /= row_sum

    result = [[0.0] * d for _ in range(n)]
    for i in range(n):
        for c in range(d):
            s = 0.0
            for j in range(n):
                s += probs[i][j] * V[j, c]
            result[i][c] = s

    return np.array(result, dtype=Q.dtype)
