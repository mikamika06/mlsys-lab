import numpy as np


def _oracle(Q, K, V):
    scores = Q @ K.T
    scores = scores - np.max(scores, axis=1, keepdims=True)
    w = np.exp(scores)
    w = w / np.sum(w, axis=1, keepdims=True)
    return w @ V


def _cases():
    rng = np.random.default_rng(0)
    cases = []

    # Ascending-magnitude K rows so later blocks raise the running max.
    n_q, n_k, d, d_v = 4, 12, 5, 3
    Q = rng.standard_normal((n_q, d))
    ramp = np.linspace(0.5, 6.0, n_k)[:, None]
    K = rng.standard_normal((n_k, d)) * ramp
    V = rng.standard_normal((n_k, d_v))
    cases.append((Q, K, V, 3))

    # Random data, several block sizes (including block_size that doesn't
    # evenly divide n_k, and block_size == n_k i.e. a single block).
    for block_size in (1, 4, 7, 20):
        n_q, n_k, d, d_v = 6, 20, 4, 2
        Q = rng.standard_normal((n_q, d))
        K = rng.standard_normal((n_k, d)) * rng.uniform(0.5, 8.0, size=(n_k, 1))
        V = rng.standard_normal((n_k, d_v))
        cases.append((Q, K, V, block_size))

    # Single query row.
    n_k, d, d_v = 9, 3, 4
    Q = rng.standard_normal((1, d))
    K = rng.standard_normal((n_k, d)) * rng.uniform(0.5, 5.0, size=(n_k, 1))
    V = rng.standard_normal((n_k, d_v))
    cases.append((Q, K, V, 2))

    return cases


def grade(sol, fx) -> dict:
    worst = 0.0
    for Q, K, V, block_size in _cases():
        ref = _oracle(Q.astype(np.float64), K.astype(np.float64), V.astype(np.float64))
        try:
            got = np.asarray(
                sol.tiled_attention_forward(Q.copy(), K.copy(), V.copy(), block_size),
                dtype=np.float64,
            )
        except Exception:
            return {"max_abs_err": float("inf")}

        if got.shape != ref.shape:
            return {"max_abs_err": float("inf")}

        worst = max(worst, float(np.max(np.abs(got - ref))))
    return {"max_abs_err": worst}
