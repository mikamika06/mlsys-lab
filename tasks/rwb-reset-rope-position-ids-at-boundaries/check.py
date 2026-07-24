import numpy as np


def _rope(x, positions):
    x = np.asarray(x, dtype=np.float64).copy()
    d = x.shape[1]
    half = d // 2
    freqs = 1.0 / (10000.0 ** (2 * np.arange(half) / d))
    angles = positions[:, None] * freqs[None, :]
    c = np.cos(angles)
    s = np.sin(angles)
    a = x[:, 0::2].copy()
    b = x[:, 1::2].copy()
    x[:, 0::2] = a * c - b * s
    x[:, 1::2] = a * s + b * c
    return x


def _softmax_attention(q, k, v):
    if q.shape[0] == 0:
        return np.empty_like(q)
    logits = q @ k.T / np.sqrt(q.shape[1])
    logits = logits - np.max(logits, axis=1, keepdims=True)
    weights = np.exp(logits)
    weights = weights / np.sum(weights, axis=1, keepdims=True)
    return weights @ v


def _oracle(q, k, v, cu_seqlens):
    out = np.empty_like(q)
    for i in range(len(cu_seqlens) - 1):
        start = int(cu_seqlens[i])
        end = int(cu_seqlens[i + 1])
        if end == start:
            continue
        pos = np.arange(end - start, dtype=np.float64)
        rq = _rope(q[start:end], pos)
        rk = _rope(k[start:end], pos)
        out[start:end] = _softmax_attention(rq, rk, v[start:end])
    return out


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(7)
    cases = [
        (np.array([0, 3, 5]), 5, 4),
        (np.array([0, 1, 4, 7]), 7, 6),
        (np.array([0, 2, 2, 5]), 5, 8),
    ]

    worst = 0.0
    for cu, n, d in cases:
        q = rng.normal(size=(n, d))
        k = rng.normal(size=(n, d))
        v = rng.normal(size=(n, d))
        ref = _oracle(q, k, v, cu)
        try:
            got = np.asarray(sol.packed_rope_attention(q, k, v, cu), dtype=np.float64)
            err = float(np.max(np.abs(got - ref)))
        except Exception:
            err = float("inf")
        worst = max(worst, err)
    return {"max_abs_err": worst}
