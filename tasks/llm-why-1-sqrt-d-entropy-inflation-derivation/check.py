import numpy as np


def _oracle_ratio(Q, K):
    d = Q.shape[1]
    scores = Q @ K.T

    def row_entropy(x):
        x = x - np.max(x, axis=1, keepdims=True)
        p = np.exp(x)
        p = p / np.sum(p, axis=1, keepdims=True)
        return float(np.mean(-np.sum(p * np.log(p + 1e-12), axis=1)))

    unscaled = row_entropy(scores)
    scaled = row_entropy(scores / np.sqrt(d))
    return scaled / unscaled


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(12345)
    Q = rng.normal(size=(32, 64)).astype(np.float64)
    K = rng.normal(size=(32, 64)).astype(np.float64)

    ref = _oracle_ratio(Q, K)
    try:
        got = float(sol.entropy_inflation_ratio(Q, K))
    except Exception:
        return {"rel_err": float("inf")}

    err = abs(got - ref) / (abs(ref) + 1e-12)
    return {"rel_err": float(err)}
