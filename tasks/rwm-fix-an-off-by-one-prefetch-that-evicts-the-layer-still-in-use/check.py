import numpy as np


def _attention(q, k, v):
    q = np.asarray(q, dtype=np.float64)
    k = np.asarray(k, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    scores = q @ k.T / np.sqrt(q.shape[-1])
    scores = scores - np.max(scores, axis=-1, keepdims=True)
    weights = np.exp(scores)
    weights = weights / np.sum(weights, axis=-1, keepdims=True)
    return weights @ v


def _oracle(layers, qs):
    return [
        _attention(qs[i], layers[i][0], layers[i][1])
        for i in range(len(layers))
    ]


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(7)
    cases = []

    for n, d, m in [(4, 3, 5), (6, 4, 3), (8, 5, 4)]:
        layers = []
        qs = []
        for i in range(n):
            base = rng.normal(size=(m, d))
            value = rng.normal(size=(m, d))
            layers.append((base, value))
            qs.append(rng.normal(size=(2, d)))
        cases.append((layers, qs))

    worst = 0.0
    for layers, qs in cases:
        try:
            got = sol.scheduled_attention(layers, qs, None, None)
        except Exception:
            return {"max_abs_err": float("inf")}

        ref = _oracle(layers, qs)
        if len(got) != len(ref):
            return {"max_abs_err": float("inf")}

        for a, b in zip(got, ref):
            err = float(np.max(np.abs(np.asarray(a, dtype=np.float64) - b)))
            worst = max(worst, err)

    return {"max_abs_err": worst}
