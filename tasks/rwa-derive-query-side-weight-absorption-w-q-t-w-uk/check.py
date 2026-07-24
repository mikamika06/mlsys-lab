import numpy as np


def _oracle_scores(W_Q, W_UK, xs, cs):
    scores = []
    for x, c in zip(xs, cs):
        q = W_Q.astype(np.float64) @ x.astype(np.float64)
        k = W_UK.astype(np.float64) @ c.astype(np.float64)
        scores.append(float(q @ k))
    return np.asarray(scores, dtype=np.float64)


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(12345)
    W_Q = rng.normal(size=(6, 4)).astype(np.float32)
    W_UK = rng.normal(size=(6, 5)).astype(np.float32)
    xs = rng.normal(size=(8, 4)).astype(np.float32)
    cs = rng.normal(size=(8, 5)).astype(np.float32)

    try:
        absorbed = np.asarray(sol.absorb_query_weight(W_Q, W_UK), dtype=np.float64)
        got = []
        for x, c in zip(xs, cs):
            got.append(float((absorbed.T @ x.astype(np.float64)) @ c.astype(np.float64)))
        got = np.asarray(got, dtype=np.float64)
    except Exception:
        return {"max_abs_err": float("inf")}

    ref = _oracle_scores(W_Q, W_UK, xs, cs)
    return {"max_abs_err": float(np.max(np.abs(got - ref)))}
