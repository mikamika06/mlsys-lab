import numpy as np


def _dense_attention(Q, K, V):
    Q = np.asarray(Q, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    d = Q.shape[-1]
    S = Q @ K.T / np.sqrt(d)
    S = S - S.max(axis=-1, keepdims=True)
    W = np.exp(S)
    W = W / W.sum(axis=-1, keepdims=True)
    return W @ V


def _tiled_cases():
    rng = np.random.default_rng(0)
    specs = [
        (64, 8, [1, 8, 32, 64]),
        (50, 6, [1, 8, 32, 50]),   # 32, 8 don't evenly divide 50 -> ragged tail
    ]
    out = []
    for N, d, blocks in specs:
        Q = rng.standard_normal((N, d))
        K = rng.standard_normal((N, d))
        V = rng.standard_normal((N, d))
        out.append((Q, K, V, blocks))
    return out


def _oracle_stable_softmax(scores):
    scores = np.asarray(scores, dtype=np.float64)
    s = scores - scores.max(axis=-1, keepdims=True)
    w = np.exp(s)
    return w / w.sum(axis=-1, keepdims=True)


def _oracle_unstable_overflows(scores):
    scores = np.asarray(scores, dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        w = np.exp(scores)
        out = w / w.sum(axis=-1, keepdims=True)
    return bool(not np.all(np.isfinite(out)))


def _stability_cases():
    rng = np.random.default_rng(1)
    moderate = rng.uniform(-5.0, 5.0, size=(4, 6))  # ordinary scale, no overflow
    large = rng.uniform(-3.0, 3.0, size=(4, 6))
    large[:, 0] += 1000.0  # forces exp() overflow in float64 if unshifted
    return [moderate, large]


FAIL = {"max_abs_err": float("inf"), "exact_match": 0.0}


def grade(sol, fx) -> dict:
    worst_err = 0.0
    all_ok = 1.0

    for Q, K, V, blocks in _tiled_cases():
        ref = _dense_attention(Q, K, V)
        for bs in blocks:
            try:
                got = sol.tiled_attention(Q.copy(), K.copy(), V.copy(), bs)
                got = np.asarray(got, dtype=np.float64)
            except Exception:
                return dict(FAIL)
            if got.shape != ref.shape or not np.all(np.isfinite(got)):
                return dict(FAIL)
            worst_err = max(worst_err, float(np.max(np.abs(got - ref))))

    for scores in _stability_cases():
        ref_stable = _oracle_stable_softmax(scores)
        ref_overflow = _oracle_unstable_overflows(scores)

        try:
            got_stable, got_overflow = sol.softmax_stability_probe(scores.copy())
            got_stable = np.asarray(got_stable, dtype=np.float64)
            got_overflow = bool(got_overflow)
        except Exception:
            return dict(FAIL)

        if got_stable.shape != ref_stable.shape or not np.all(np.isfinite(got_stable)):
            return dict(FAIL)

        worst_err = max(worst_err, float(np.max(np.abs(got_stable - ref_stable))))
        if got_overflow != ref_overflow:
            all_ok = 0.0

    return {"max_abs_err": worst_err, "exact_match": all_ok}
