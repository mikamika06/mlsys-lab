import math
import numpy as np


def _rope(x, positions):
    x = np.asarray(x, dtype=np.float64).copy()
    N, d = x.shape
    half = d // 2
    freqs = [1.0 / (10000.0 ** (2 * j / d)) for j in range(half)]
    for i in range(N):
        pos = positions[i]
        for j in range(half):
            angle = pos * freqs[j]
            c = math.cos(angle)
            s = math.sin(angle)
            even_val = x[i, 2 * j]
            odd_val = x[i, 2 * j + 1]
            x[i, 2 * j] = even_val * c - odd_val * s
            x[i, 2 * j + 1] = even_val * s + odd_val * c
    return x


def _attention(q, k, v):
    if q.shape[0] == 0:
        return np.empty_like(q)
    N, d = q.shape
    scale = 1.0 / math.sqrt(d)
    logits = np.empty((N, N), dtype=np.float64)
    for i in range(N):
        for j in range(N):
            dot = 0.0
            for k_idx in range(d):
                dot += q[i, k_idx] * k[j, k_idx]
            logits[i, j] = dot * scale
    for i in range(N):
        m = logits[i, 0]
        for j in range(1, N):
            if logits[i, j] > m:
                m = logits[i, j]
        for j in range(N):
            logits[i, j] -= m
    probs = np.empty((N, N), dtype=np.float64)
    for i in range(N):
        for j in range(N):
            probs[i, j] = math.exp(logits[i, j])
    for i in range(N):
        row_sum = 0.0
        for j in range(N):
            row_sum += probs[i, j]
        for j in range(N):
            probs[i, j] /= row_sum
    v_N, v_d = v.shape
    out = np.empty((N, v_d), dtype=np.float64)
    for i in range(N):
        for j in range(v_d):
            val = 0.0
            for k_idx in range(N):
                val += probs[i, k_idx] * v[k_idx, j]
            out[i, j] = val
    return out


def packed_rope_attention(q, k, v, cu_seqlens):
    out = np.empty_like(q, dtype=np.float64)
    for i in range(len(cu_seqlens) - 1):
        start = int(cu_seqlens[i])
        end = int(cu_seqlens[i + 1])
        if end == start:
            continue
        positions = np.arange(end - start, dtype=np.float64)
        rq = _rope(q[start:end], positions)
        rk = _rope(k[start:end], positions)
        out[start:end] = _attention(rq, rk, v[start:end])
    return out
