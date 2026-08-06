import numpy as np


def _oracle_attention(Q, K, V):
    scores = (Q @ K.T) / np.sqrt(Q.shape[1])
    scores = scores - np.max(scores, axis=1, keepdims=True)
    weights = np.exp(scores)
    weights = weights / np.sum(weights, axis=1, keepdims=True)
    return weights @ V


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(7)
    cases = [
        (5, 3, 2),
        (8, 4, 3),
        (13, 6, 5),
    ]

    best = 0.0
    max_err = 0.0
    for n, d, m in cases:
        Q = rng.normal(size=(n, d)).astype(np.float64)
        K = rng.normal(size=(n, d)).astype(np.float64)
        V = rng.normal(size=(n, m)).astype(np.float64)

        ref = _oracle_attention(Q, K, V)
        try:
            one = np.asarray(
                sol.streaming_attention(Q.tolist(), K.tolist(), V.tolist(), 1),
                dtype=np.float64,
            )
            all_at_once = np.asarray(
                sol.streaming_attention(Q.tolist(), K.tolist(), V.tolist(), n),
                dtype=np.float64,
            )
        except Exception:
            return {"max_abs_err": float("inf")}

        if one.shape != ref.shape or all_at_once.shape != ref.shape:
            return {"max_abs_err": float("inf")}

        err = max(
            float(np.max(np.abs(one - ref))),
            float(np.max(np.abs(all_at_once - ref))),
            float(np.max(np.abs(one - all_at_once))),
        )
        max_err = max(max_err, err)

    return {"max_abs_err": max_err}
