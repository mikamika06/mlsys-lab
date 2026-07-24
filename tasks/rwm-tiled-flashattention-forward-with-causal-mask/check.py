import numpy as np


def _oracle(Q, K, V):
    n, d = Q.shape
    scores = (Q @ K.T) / np.sqrt(float(d))
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    scores = scores.copy()
    scores[mask] = -np.inf
    scores = scores - np.max(scores, axis=1, keepdims=True)
    probs = np.exp(scores)
    probs = probs / np.sum(probs, axis=1, keepdims=True)
    return probs @ V


def grade(sol, fx) -> dict:
    cases = [
        (5, 3, 2),
        (8, 4, 3),
        (7, 5, 4),
    ]
    worst = 0.0
    rng = np.random.default_rng(12345)

    for n, d, block in cases:
        Q = rng.normal(size=(n, d)).astype(np.float64)
        K = rng.normal(size=(n, d)).astype(np.float64)
        V = rng.normal(size=(n, d)).astype(np.float64)

        ref = _oracle(Q, K, V)
        try:
            got = np.asarray(
                sol.flash_attention_forward(Q, K, V, block_size=block),
                dtype=np.float64,
            )
        except Exception:
            return {"max_abs_err": float("inf")}

        if got.shape != ref.shape:
            return {"max_abs_err": float("inf")}

        worst = max(worst, float(np.max(np.abs(got - ref))))

    return {"max_abs_err": worst}
