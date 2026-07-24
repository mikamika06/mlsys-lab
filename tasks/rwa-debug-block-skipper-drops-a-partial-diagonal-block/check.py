import numpy as np

from mlsys import scorers


def _dense_causal_attention(q: np.ndarray, k: np.ndarray, v: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    k = np.asarray(k, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    n, d = q.shape
    scale = 1.0 / np.sqrt(d)
    scores = (q @ k.T) * scale
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    scores = np.where(mask, -np.inf, scores)
    scores = scores - np.max(scores, axis=1, keepdims=True)
    w = np.exp(scores)
    w = w / np.sum(w, axis=1, keepdims=True)
    return w @ v


def _scenarios():
    rng = np.random.default_rng(0)
    scenarios = []
    for n, d, block_size in [(8, 4, 4), (32, 8, 8), (64, 16, 16), (48, 6, 12), (16, 4, 4)]:
        q = rng.normal(size=(n, d))
        k = rng.normal(size=(n, d))
        v = rng.normal(size=(n, d))
        scenarios.append((q, k, v, block_size))
    return scenarios


def grade(sol, fx) -> dict:
    worst = 0.0
    for q, k, v, block_size in _scenarios():
        ref = _dense_causal_attention(q, k, v)
        try:
            got = sol.block_sparse_causal_attention(q.copy(), k.copy(), v.copy(), block_size)
        except Exception:
            return {"max_abs_err": float("inf")}

        try:
            got = np.asarray(got, dtype=np.float64)
        except Exception:
            return {"max_abs_err": float("inf")}

        if got.shape != ref.shape:
            return {"max_abs_err": float("inf")}

        err = scorers.max_abs_err(ref, got)
        if not np.isfinite(err):
            return {"max_abs_err": float("inf")}
        worst = max(worst, err)

    return {"max_abs_err": worst}
