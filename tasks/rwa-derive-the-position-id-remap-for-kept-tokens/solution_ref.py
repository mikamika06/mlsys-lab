import math
import numpy as np


def _rope(x, positions, theta):
    x = np.asarray(x, dtype=np.float64)
    n, d = x.shape
    half = d // 2
    freq = [0.0] * half
    for i in range(half):
        freq[i] = theta ** (-i * 2.0 / d)
    out_list = [[0.0] * d for _ in range(n)]
    for i in range(n):
        pos = float(positions[i])
        for j in range(d):
            out_list[i][j] = float(x[i, j])
        for h in range(half):
            angle = pos * freq[h]
            c = math.cos(angle)
            s = math.sin(angle)
            idx0 = 2 * h
            idx1 = 2 * h + 1
            x0 = float(x[i, idx0])
            x1 = float(x[i, idx1])
            out_list[i][idx0] = x0 * c - x1 * s
            out_list[i][idx1] = x0 * s + x1 * c
    return np.array(out_list, dtype=np.float64)


def streaming_rope_attention(q, k, v, kept_indices, theta=10000.0):
    q = np.asarray(q, dtype=np.float64)[kept_indices]
    k = np.asarray(k, dtype=np.float64)[kept_indices]
    v = np.asarray(v, dtype=np.float64)[kept_indices]
    positions = np.arange(len(kept_indices), dtype=np.float64)

    qr = _rope(q, positions, theta)
    kr = _rope(k, positions, theta)

    n_q = qr.shape[0]
    n_k = kr.shape[0]
    d = qr.shape[1]
    scale = math.sqrt(d)

    scores = [[0.0] * n_k for _ in range(n_q)]
    for i in range(n_q):
        for j in range(n_k):
            dot = 0.0
            for k_idx in range(d):
                dot += float(qr[i, k_idx]) * float(kr[j, k_idx])
            scores[i][j] = dot / scale

    for i in range(n_q):
        max_val = scores[i][0]
        for j in range(1, n_k):
            if scores[i][j] > max_val:
                max_val = scores[i][j]
        for j in range(n_k):
            scores[i][j] -= max_val

    weights = [[0.0] * n_k for _ in range(n_q)]
    for i in range(n_q):
        for j in range(n_k):
            weights[i][j] = math.exp(scores[i][j])

    for i in range(n_q):
        total = 0.0
        for j in range(n_k):
            total += weights[i][j]
        for j in range(n_k):
            weights[i][j] /= total

    v_rows = v.shape[0]
    v_cols = v.shape[1]
    out_v = [[0.0] * v_cols for _ in range(n_q)]
    for i in range(n_q):
        for j in range(v_cols):
            val = 0.0
            for k_idx in range(v_rows):
                val += weights[i][k_idx] * float(v[k_idx, j])
            out_v[i][j] = val

    return np.array(out_v, dtype=np.float64)
