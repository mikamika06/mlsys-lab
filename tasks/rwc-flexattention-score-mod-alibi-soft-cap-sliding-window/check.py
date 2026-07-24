import numpy as np


def _ref_flex_attention(Q, K, V, score_mod):
    Q = np.asarray(Q, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    N, d = Q.shape
    scores = Q @ K.T / np.sqrt(d)
    qi = np.arange(N).reshape(N, 1)
    ki = np.arange(N).reshape(1, N)
    scores = score_mod(scores, qi, ki)
    scores -= scores.max(axis=-1, keepdims=True)
    weights = np.exp(scores)
    row_sum = weights.sum(axis=-1, keepdims=True)
    # handle all-inf rows (fully masked)
    row_sum = np.where(row_sum == 0, 1.0, row_sum)
    weights /= row_sum
    return (weights @ V).astype(np.float32)


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(13)
    N, d = 32, 16
    Q = rng.standard_normal((N, d)).astype(np.float32)
    K = rng.standard_normal((N, d)).astype(np.float32)
    V = rng.standard_normal((N, d)).astype(np.float32)

    def alibi_mod(scores, qi, ki):
        return scores - 0.5 * np.abs(qi - ki)

    def softcap_mod(scores, qi, ki):
        c = 50.0
        return c * np.tanh(scores / c)

    def sliding_window_mod(scores, qi, ki):
        w = 8
        mask = np.abs(qi - ki) > w
        return np.where(mask, -1e9, scores)

    mods = [alibi_mod, softcap_mod, sliding_window_mod]
    worst_err = 0.0
    for mod in mods:
        ref = _ref_flex_attention(Q, K, V, mod)
        try:
            got = np.asarray(sol.flex_attention(Q.copy(), K.copy(), V.copy(), mod), dtype=np.float32)
        except Exception:
            return {"max_abs_err": float("inf")}
        if got.shape != ref.shape:
            return {"max_abs_err": float("inf")}
        err = float(np.max(np.abs(got.astype(np.float64) - ref.astype(np.float64))))
        if err > worst_err:
            worst_err = err

    return {"max_abs_err": worst_err}
