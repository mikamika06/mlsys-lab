import numpy as np


def _oracle_rank(weights, grads):
    h = weights.shape[0]
    flat_w = weights.reshape(h, -1)
    flat_g = grads.reshape(h, -1)

    mag_scores = np.sqrt(np.sum(flat_w * flat_w, axis=1))
    taylor_scores = np.sum(np.abs(flat_g * flat_w), axis=1)

    mag_order = sorted(range(h), key=lambda i: (-float(mag_scores[i]), i))
    taylor_order = sorted(range(h), key=lambda i: (-float(taylor_scores[i]), i))
    return mag_order, taylor_order


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(12345)
    cases = []

    while len(cases) < 3:
        weights = rng.normal(size=(6, 2, 3))
        grads = rng.normal(size=(6, 2, 3))
        mag, tay = _oracle_rank(weights, grads)
        if mag != tay:
            cases.append((weights, grads))

    exact = 1.0
    differ = 1.0

    for weights, grads in cases:
        ref_mag, ref_taylor = _oracle_rank(weights, grads)
        if ref_mag == ref_taylor:
            differ = 0.0
        try:
            got_mag, got_taylor = sol.rank_heads_by_importance(
                weights.copy(), grads.copy()
            )
            got_mag = list(got_mag)
            got_taylor = list(got_taylor)
        except Exception:
            exact = 0.0
            break

        if got_mag != ref_mag or got_taylor != ref_taylor:
            exact = 0.0

        if got_mag == got_taylor:
            differ = 0.0

    return {
        "exact_match": exact,
        "rankings_differ": differ,
    }
