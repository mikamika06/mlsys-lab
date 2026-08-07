import numpy as np


def _oracle(Q, K, V):
    n, d = Q.shape
    scores = (Q @ K.T) / np.sqrt(d)
    row = np.arange(n)[:, None]
    col = np.arange(n)[None, :]
    scores = np.where(col > row, -np.inf, scores)
    scores = scores - np.max(scores, axis=1, keepdims=True)
    probs = np.exp(scores)
    probs = probs / np.sum(probs, axis=1, keepdims=True)
    return probs @ V


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(3)
    worst = 0.0
    for _ in range(6):
        n = int(rng.integers(3, 20))
        d = int(rng.integers(2, 10))
        Q = rng.standard_normal((n, d))
        K = rng.standard_normal((n, d))
        V = rng.standard_normal((n, d))

        ref = _oracle(Q, K, V)
        try:
            got = np.asarray(
                sol.causal_self_attention(Q.tolist(), K.tolist(), V.tolist()),
                dtype=np.float64,
            )
        except Exception:
            return {"max_abs_err": float("inf")}
        if got.shape != ref.shape:
            return {"max_abs_err": float("inf")}
        worst = max(worst, float(np.max(np.abs(got - ref))))
    return {"max_abs_err": worst}
