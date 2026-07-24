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

    def prompt(n, d):
        q = rng.normal(size=(n, d))
        k = rng.normal(size=(n, d))
        v = rng.normal(size=(n, d))
        return q, k, v

    # (n, d, [list of chunk schedules to test against the same q,k,v])
    configs = [
        (19, 6, [[3, 5, 2, 9], [19], [1] * 19, [10, 9]]),
        (30, 5, [[10, 10, 10], [5, 5, 5, 5, 5, 5], [30], [1, 2, 27]]),
        (7, 4, [[7], [1, 1, 1, 1, 1, 1, 1], [2, 5]]),
        (16, 8, [[1, 1, 1, 1, 12], [4, 4, 4, 4], [16]]),
    ]

    for n, d, schedules in configs:
        q, k, v = prompt(n, d)
        for chunk_sizes in schedules:
            scenarios.append((q, k, v, chunk_sizes))

    return scenarios


def grade(sol, fx) -> dict:
    worst = 0.0
    for q, k, v, chunk_sizes in _scenarios():
        ref = _dense_causal_attention(q, k, v)
        try:
            got = sol.chunked_causal_prefill(q.copy(), k.copy(), v.copy(), list(chunk_sizes))
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
