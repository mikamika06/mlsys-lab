import numpy as np


def _rope(x, positions, theta):
    x = np.asarray(x, dtype=np.float64)
    n, d = x.shape
    half = d // 2
    freq = theta ** (-np.arange(0, half, dtype=np.float64) * 2.0 / d)
    angles = np.asarray(positions, dtype=np.float64)[:, None] * freq[None, :]
    c = np.cos(angles)
    s = np.sin(angles)
    out = x.copy()
    a = x[:, 0::2]
    b = x[:, 1::2]
    out[:, 0::2] = a * c - b * s
    out[:, 1::2] = a * s + b * c
    return out


def _oracle(q, k, v, kept_indices, theta=10000.0):
    q = np.asarray(q, dtype=np.float64)[kept_indices]
    k = np.asarray(k, dtype=np.float64)[kept_indices]
    v = np.asarray(v, dtype=np.float64)[kept_indices]
    positions = np.arange(len(kept_indices), dtype=np.float64)
    qr = _rope(q, positions, theta)
    kr = _rope(k, positions, theta)
    logits = qr @ kr.T / np.sqrt(q.shape[1])
    logits = logits - np.max(logits, axis=1, keepdims=True)
    probs = np.exp(logits)
    probs /= np.sum(probs, axis=1, keepdims=True)
    return probs @ v


def grade(sol, fx) -> dict:
    cases = [
        (
            np.array([[1., 0.], [0.5, 1.], [1., 1.], [2., -1.]]),
            np.array([[0., 1.], [1., 0.], [1., 1.], [-1., 2.]]),
            np.array([[1., 2.], [3., 4.], [5., 6.], [7., 8.]]),
            np.array([0, 2]),
        ),
        (
            np.arange(24, dtype=np.float64).reshape(4, 6) / 10.0,
            np.arange(24, dtype=np.float64).reshape(4, 6) / 7.0,
            np.arange(24, dtype=np.float64).reshape(4, 6) / 5.0,
            np.array([3, 1, 2]),
        ),
        (
            np.random.default_rng(7).normal(size=(7, 8)),
            np.random.default_rng(8).normal(size=(7, 8)),
            np.random.default_rng(9).normal(size=(7, 8)),
            np.array([0, 4, 6]),
        ),
    ]
    worst = 0.0
    for q, k, v, kept in cases:
        try:
            got = np.asarray(
                sol.streaming_rope_attention(
                    q.tolist(),
                    k.tolist(),
                    v.tolist(),
                    kept.tolist(),
                ),
                dtype=np.float64,
            )
        except Exception:
            return {"max_abs_err": float("inf")}
        ref = _oracle(q, k, v, kept)
        worst = max(worst, float(np.max(np.abs(got - ref))))
    return {"max_abs_err": worst}
