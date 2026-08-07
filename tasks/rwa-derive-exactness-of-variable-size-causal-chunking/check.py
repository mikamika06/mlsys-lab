import numpy as np


def _oracle(Q, K, V):
    Q = np.asarray(Q, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    n, d = Q.shape
    scores = Q @ K.T / np.sqrt(d)
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    scores = np.where(mask, -np.inf, scores)
    scores = scores - np.max(scores, axis=1, keepdims=True)
    weights = np.exp(scores)
    weights = weights / np.sum(weights, axis=1, keepdims=True)
    return weights @ V


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(7)
    cases = [
        (9, 4, 3, [1, 2, 6]),
        (8, 3, 2, [3, 1, 4]),
        (11, 5, 4, [2, 2, 1, 6]),
    ]
    worst = 0.0
    for n, d, m, chunks in cases:
        Q = rng.normal(size=(n, d))
        K = rng.normal(size=(n, d))
        V = rng.normal(size=(n, m))
        ref = _oracle(Q, K, V)
        try:
            got = np.asarray(sol.causal_chunk_attention(Q.tolist(), K.tolist(), V.tolist(), chunks), dtype=np.float64)
            err = float(np.max(np.abs(got - ref)))
        except Exception:
            return {"max_abs_err": float("inf")}
        worst = max(worst, err)
    return {"max_abs_err": worst}
