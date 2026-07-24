import numpy as np


def _oracle(Q, K, V, bias):
    Q = np.asarray(Q, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    scores = Q @ K.T / np.sqrt(Q.shape[1])
    if bias is not None:
        scores = scores + np.asarray(bias, dtype=np.float64)
    scores = scores - np.max(scores, axis=1, keepdims=True)
    weights = np.exp(scores)
    weights = weights / np.sum(weights, axis=1, keepdims=True)
    return weights @ V


def grade(sol, fx) -> dict:
    cases = []
    rng = np.random.default_rng(42)
    for n, m, d, h in [(3, 4, 5, 2), (8, 6, 4, 3), (2, 7, 3, 5)]:
        Q = rng.normal(size=(n, d))
        K = rng.normal(size=(m, d))
        V = rng.normal(size=(m, h))
        cases.append((Q, K, V, None))
        cases.append((Q, K, V, rng.normal(size=(n, m))))

    worst = 0.0
    for Q, K, V, bias in cases:
        try:
            mem, math = sol.compare_sdpa_backends(Q, K, V, bias)
            mem = np.asarray(mem, dtype=np.float64)
            math = np.asarray(math, dtype=np.float64)
        except Exception:
            return {"max_abs_err": float("inf")}

        ref = _oracle(Q, K, V, bias)
        err = max(
            float(np.max(np.abs(mem - ref))),
            float(np.max(np.abs(math - ref))),
            float(np.max(np.abs(mem - math))),
        )
        worst = max(worst, err)

    return {"max_abs_err": worst}
