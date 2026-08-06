import numpy as np


def _full_attention(Q, K, V):
    scores = (Q @ K.T) / np.sqrt(Q.shape[1])
    scores = scores - np.max(scores, axis=1, keepdims=True)
    probs = np.exp(scores)
    probs = probs / np.sum(probs, axis=1, keepdims=True)
    return probs @ V


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(42)
    cases = [
        (17, 8, 5, 3),
        (32, 16, 7, 4),
        (9, 4, 6, 2),
    ]
    worst = 0.0

    for n, d, dv, block_size in cases:
        Q_np = rng.normal(size=(n, d)).astype(np.float64)
        K_np = rng.normal(size=(n, d)).astype(np.float64)
        V_np = rng.normal(size=(n, dv)).astype(np.float64)

        Q = Q_np.tolist()
        K = K_np.tolist()
        V = V_np.tolist()

        ref = _full_attention(Q_np, K_np, V_np)
        try:
            got_raw = sol.streaming_softmax_attention(Q, K, V, block_size)
            got = np.asarray(got_raw, dtype=np.float64)
        except Exception:
            return {"max_abs_err": float("inf")}

        if got.shape != ref.shape:
            return {"max_abs_err": float("inf")}

        worst = max(worst, float(np.max(np.abs(got - ref))))

    return {"max_abs_err": worst}
