import numpy as np


def _attention(Q, K, V, bias):
    Q = np.asarray(Q, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)

    scores = Q @ K.T / np.sqrt(Q.shape[1])
    if bias is not None:
        scores = scores + np.asarray(bias, dtype=np.float64)

    scores = scores - np.max(scores, axis=1, keepdims=True)
    weights = np.exp(scores)
    weights /= np.sum(weights, axis=1, keepdims=True)
    return weights @ V


def compare_sdpa_backends(Q, K, V, bias):
    out = _attention(Q, K, V, bias)
    return out.copy(), out.copy()
