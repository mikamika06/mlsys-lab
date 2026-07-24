import numpy as np


def _oracle(Q, K, V, slopes):
    Q = np.asarray(Q, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    slopes = np.asarray(slopes, dtype=np.float64)

    H, n, d = Q.shape
    m = K.shape[1]

    scores = np.einsum("hnd,hmd->hnm", Q, K) / np.sqrt(d)

    q_idx = np.arange(n, dtype=np.float64)[:, None]
    kv_idx = np.arange(m, dtype=np.float64)[None, :]
    bias = kv_idx - q_idx  # (n, m)

    scores = scores + slopes[:, None, None] * bias[None, :, :]

    scores = scores - np.max(scores, axis=-1, keepdims=True)
    weights = np.exp(scores)
    weights = weights / np.sum(weights, axis=-1, keepdims=True)

    return np.einsum("hnm,hmv->hnv", weights, V)


def _cases(rng):
    cases = []
    shapes = [
        (1, 3, 3, 2, 2),
        (2, 4, 5, 3, 2),
        (3, 6, 4, 4, 5),
        (4, 5, 5, 8, 1),
        (2, 1, 7, 3, 3),
    ]
    for H, n, m, d, dv in shapes:
        Q = rng.normal(size=(H, n, d))
        K = rng.normal(size=(H, m, d))
        V = rng.normal(size=(H, m, dv))
        slopes = rng.uniform(0.01, 1.0, size=(H,))
        cases.append((Q, K, V, slopes))
    return cases


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(0)
    max_err = 0.0
    for Q, K, V, slopes in _cases(rng):
        try:
            got = sol.alibi_score_mod_attention(Q, K, V, slopes)
        except Exception:
            return {"max_abs_err": float("inf")}

        ref = _oracle(Q, K, V, slopes)
        got = np.asarray(got, dtype=np.float64)
        if got.shape != ref.shape:
            return {"max_abs_err": float("inf")}

        err = float(np.max(np.abs(got - ref)))
        max_err = max(max_err, err)

    return {"max_abs_err": max_err}
