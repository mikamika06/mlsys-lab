import numpy as np


def _quantize_symmetric(x, bits):
    x = np.asarray(x, dtype=np.float64)
    qmax = (1 << (bits - 1)) - 1
    scale = np.max(np.abs(x)) / qmax
    if scale == 0:
        return np.zeros_like(x)
    return np.round(x / scale) * scale


def _attention(K, V, q):
    logits = (np.asarray(K, dtype=np.float64) @ np.asarray(q, dtype=np.float64)) / np.sqrt(K.shape[1])
    logits = logits - np.max(logits)
    weights = np.exp(logits)
    weights = weights / np.sum(weights)
    return weights @ np.asarray(V, dtype=np.float64)


def _ref(K, V, q):
    base = _attention(K, V, q)
    configs = [
        (8, 8),
        (4, 4),
        (8, 4),
    ]
    out = []
    for kb, vb in configs:
        kq = _quantize_symmetric(K, kb)
        vq = _quantize_symmetric(V, vb)
        out.append(float(np.max(np.abs(_attention(kq, vq, q) - base))))
    return np.asarray(out, dtype=np.float64)


def grade(sol, fx) -> dict:
    cases = []
    rng = np.random.default_rng(42)
    for n, d in [(8, 8), (16, 8), (12, 16)]:
        K = rng.normal(size=(n, d)).astype(np.float64)
        V = rng.normal(size=(n, d)).astype(np.float64)
        q = rng.normal(size=(d,)).astype(np.float64)
        cases.append((K, V, q))

    max_err = 0.0
    for K, V, q in cases:
        try:
            got = np.asarray(sol.kv_config_attention_errors(K, V, q), dtype=np.float64)
        except Exception:
            return {"max_abs_err": float("inf")}
        ref = _ref(K, V, q)
        if got.shape != ref.shape:
            return {"max_abs_err": float("inf")}
        max_err = max(max_err, float(np.max(np.abs(got - ref))))
    return {"max_abs_err": max_err}
