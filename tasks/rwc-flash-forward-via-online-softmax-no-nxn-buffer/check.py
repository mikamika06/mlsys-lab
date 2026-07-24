import numpy as np


def _naive_attention(Q, K, V):
    """Reference: naive softmax attention in float64."""
    Q = np.asarray(Q, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    d = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d)
    scores -= scores.max(axis=-1, keepdims=True)
    weights = np.exp(scores)
    weights /= weights.sum(axis=-1, keepdims=True)
    return weights @ V


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(7)
    cases = [
        (16, 8, 4),
        (32, 16, 8),
        (64, 32, 16),
        (128, 64, 32),
    ]
    worst_err = 0.0
    for N, d, bs in cases:
        Q = rng.standard_normal((N, d)).astype(np.float32)
        K = rng.standard_normal((N, d)).astype(np.float32)
        V = rng.standard_normal((N, d)).astype(np.float32)
        ref = _naive_attention(Q, K, V).astype(np.float32)
        try:
            got = np.asarray(sol.flash_attention_forward(Q.copy(), K.copy(), V.copy(), block_size=bs), dtype=np.float32)
        except Exception:
            return {"max_abs_err": float("inf")}
        if got.shape != ref.shape:
            return {"max_abs_err": float("inf")}
        err = float(np.max(np.abs(got.astype(np.float64) - ref.astype(np.float64))))
        if err > worst_err:
            worst_err = err
    return {"max_abs_err": worst_err}
