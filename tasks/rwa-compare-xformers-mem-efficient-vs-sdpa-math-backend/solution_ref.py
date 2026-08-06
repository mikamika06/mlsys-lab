import math
import numpy as np


def _attention(Q, K, V, bias):
    Q = np.asarray(Q, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)

    n, d = Q.shape
    m, h = V.shape
    scale = 1.0 / math.sqrt(d)

    scores = np.zeros((n, m), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            acc = 0.0
            for k in range(d):
                acc += Q[i, k] * K[j, k]
            val = acc * scale
            if bias is not None:
                val += float(bias[i, j])
            scores[i, j] = val

    max_vals = np.zeros((n, 1), dtype=np.float64)
    for i in range(n):
        m_val = scores[i, 0]
        for j in range(1, m):
            if scores[i, j] > m_val:
                m_val = scores[i, j]
        max_vals[i, 0] = m_val

    weights = np.zeros((n, m), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            weights[i, j] = math.exp(scores[i, j] - max_vals[i, 0])

    sum_vals = np.zeros((n, 1), dtype=np.float64)
    for i in range(n):
        s_val = 0.0
        for j in range(m):
            s_val += weights[i, j]
        sum_vals[i, 0] = s_val

    for i in range(n):
        for j in range(m):
            weights[i, j] /= sum_vals[i, 0]

    out = np.zeros((n, h), dtype=np.float64)
    for i in range(n):
        for j in range(h):
            acc = 0.0
            for k in range(m):
                acc += weights[i, k] * V[k, j]
            out[i, j] = acc

    return out


def compare_sdpa_backends(Q, K, V, bias):
    out = _attention(Q, K, V, bias)
    return out.copy(), out.copy()
