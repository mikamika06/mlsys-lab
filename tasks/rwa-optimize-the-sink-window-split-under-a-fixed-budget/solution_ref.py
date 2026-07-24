import numpy as np


def _softmax(x):
    x = np.asarray(x, dtype=np.float64)
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)


def _attention(Q, K, V):
    d = Q.shape[1]
    return _softmax((Q @ K.T) / np.sqrt(d)) @ V


def optimize_sink_window_split(Q, K, V, B):
    Q = np.asarray(Q, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)

    full = _attention(Q, K, V)
    n = Q.shape[0]

    best_k = 1
    best_error = float("inf")

    for k in range(1, B):
        w = B - k
        indices = np.concatenate(
            [
                np.arange(k),
                np.arange(n - w, n),
            ]
        )
        indices = np.unique(indices)
        approx = _attention(Q, K[indices], V[indices])
        error = np.sum((full - approx) ** 2)

        if error < best_error:
            best_error = error
            best_k = k

    return int(best_k)
