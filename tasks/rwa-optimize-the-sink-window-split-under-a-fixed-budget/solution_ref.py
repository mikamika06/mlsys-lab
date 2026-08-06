import math
import numpy as np


def _softmax(x):
    x = np.asarray(x, dtype=np.float64)
    n_rows, n_cols = x.shape
    out = np.zeros((n_rows, n_cols), dtype=np.float64)
    for i in range(n_rows):
        m = x[i, 0]
        for j in range(1, n_cols):
            if x[i, j] > m:
                m = x[i, j]
        s = 0.0
        for j in range(n_cols):
            val = math.exp(x[i, j] - m)
            out[i, j] = val
            s += val
        for j in range(n_cols):
            out[i, j] /= s
    return out


def _attention(Q, K, V):
    Q = np.asarray(Q, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    n_q, d = Q.shape
    n_k = K.shape[0]
    d_v = V.shape[1]
    sqrt_d = math.sqrt(d)
    scores = np.zeros((n_q, n_k), dtype=np.float64)
    for i in range(n_q):
        for j in range(n_k):
            acc = 0.0
            for l in range(d):
                acc += Q[i, l] * K[j, l]
            scores[i, j] = acc / sqrt_d
    sm = _softmax(scores)
    out = np.zeros((n_q, d_v), dtype=np.float64)
    for i in range(n_q):
        for j in range(d_v):
            acc = 0.0
            for l in range(n_k):
                acc += sm[i, l] * V[l, j]
            out[i, j] = acc
    return out


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
        idx_list = []
        for i in range(k):
            idx_list.append(i)
        for i in range(n - w, n):
            idx_list.append(i)
        indices = np.array(sorted(list(set(idx_list))), dtype=np.int64)
        approx = _attention(Q, K[indices], V[indices])
        
        error = 0.0
        n_rows, n_cols = full.shape
        for i in range(n_rows):
            for j in range(n_cols):
                diff = full[i, j] - approx[i, j]
                error += diff * diff

        if error < best_error:
            best_error = error
            best_k = k

    return int(best_k)
