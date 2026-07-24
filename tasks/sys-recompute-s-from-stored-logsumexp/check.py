import numpy as np


def _stable_lse(scores):
    m = np.max(scores, axis=1, keepdims=True)
    return (m[:, 0] + np.log(np.sum(np.exp(scores - m), axis=1)))


def _oracle(Q, K, lse):
    d = Q.shape[1]
    scores = (Q @ K.T) / np.sqrt(d)
    return np.exp(scores - lse[:, None])


def _make_case(rng, n, m, d, scale):
    Q = rng.normal(size=(n, d)) * scale
    K = rng.normal(size=(m, d)) * scale
    return Q, K


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(3)
    cases = [
        _make_case(rng, 5, 6, 4, 1.0),
        _make_case(rng, 8, 3, 8, 2.5),
        _make_case(rng, 4, 4, 16, 30.0),   # scores reach into the hundreds
        _make_case(rng, 6, 5, 32, 60.0),   # scores overflow exp() directly (>709)
    ]

    worst = 0.0
    for Q, K, in [(c[0], c[1]) for c in cases]:
        d = Q.shape[1]
        scores = (Q @ K.T) / np.sqrt(d)
        lse = _stable_lse(scores)
        ref = _oracle(Q, K, lse)

        try:
            got = np.asarray(
                sol.recompute_probs_from_lse(Q.copy(), K.copy(), lse.copy()),
                dtype=np.float64,
            )
        except Exception:
            return {"max_abs_err": float("inf")}

        if got.shape != ref.shape or not np.all(np.isfinite(got)):
            return {"max_abs_err": float("inf")}

        err = float(np.max(np.abs(got - ref)))
        worst = max(worst, err)

    return {"max_abs_err": worst}
