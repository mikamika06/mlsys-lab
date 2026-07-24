import numpy as np


def flex_attention(Q, K, V, score_mod):
    """FlexAttention: apply score_mod before softmax."""
    Q = np.asarray(Q, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    N, d = Q.shape
    scores = Q @ K.T / np.sqrt(d)
    qi = np.arange(N).reshape(N, 1)
    ki = np.arange(N).reshape(1, N)
    scores = score_mod(scores, qi, ki)
    scores -= scores.max(axis=-1, keepdims=True)
    weights = np.exp(scores)
    row_sum = weights.sum(axis=-1, keepdims=True)
    row_sum = np.where(row_sum == 0, 1.0, row_sum)
    weights /= row_sum
    return (weights @ V).astype(np.float32)
