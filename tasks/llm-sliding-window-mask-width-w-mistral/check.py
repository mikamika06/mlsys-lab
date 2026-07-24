import numpy as np

from mlsys.scorers import max_abs_err


def _reference(Q, K, V, w):
    """Oracle: single-head SDPA with a Mistral sliding-window mask (i-w < j <= i)."""
    Q = np.asarray(Q, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    n, d = Q.shape
    scores = Q @ K.T / np.sqrt(d)

    rows = np.arange(n).reshape(-1, 1)
    cols = np.arange(n).reshape(1, -1)
    allowed = (cols <= rows) & (rows - cols < w)  # i - w < j <= i

    masked = np.where(allowed, scores, -np.inf)
    masked = masked - np.max(masked, axis=-1, keepdims=True)
    e = np.exp(masked)
    p = e / np.sum(e, axis=-1, keepdims=True)
    return p @ V


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(0)
    # (n, d, w): single token; window==1 (self only); mid-window; w == n; w > n.
    cases = [(1, 4, 1), (6, 3, 1), (7, 5, 3), (8, 4, 8), (9, 6, 20), (5, 2, 2)]

    worst = 0.0
    for n, d, w in cases:
        Q = rng.standard_normal((n, d))
        K = rng.standard_normal((n, d))
        V = rng.standard_normal((n, d))
        ref = _reference(Q, K, V, w)
        try:
            out = np.asarray(sol.sliding_window_attention(Q, K, V, w), dtype=np.float64)
        except Exception:
            return {"max_abs_err": float("inf")}
        if out.shape != ref.shape:
            return {"max_abs_err": float("inf")}
        worst = max(worst, max_abs_err(ref, out))

    return {"max_abs_err": float(worst)}
