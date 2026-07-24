import numpy as np


def _oracle(scores, V):
    x = np.asarray(scores, dtype=np.float64)
    x = x - np.max(x)
    e = np.exp(x)
    w = e / np.sum(e)
    return w @ np.asarray(V, dtype=np.float64)


def _make_case(rng, n, d, loc=0.0, scale=1.0):
    scores = rng.normal(loc=loc, scale=scale, size=n)
    V = rng.normal(size=(n, d))
    return scores, V


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(0)
    cases = [
        (*_make_case(rng, 37, 5), 8),
        (*_make_case(rng, 100, 3, loc=50.0, scale=10.0), 16),
        (*_make_case(rng, 64, 4), 16),
        (*_make_case(rng, 9, 2), 4),
        (*_make_case(rng, 1, 3), 4),
    ]

    worst_err = 0.0
    blockwise_ok = 1.0
    orig_exp = np.exp

    for scores, V, block_size in cases:
        n = int(scores.shape[0])
        violation = [False]

        def traced_exp(x, *a, **kw):
            arr = np.asarray(x)
            if arr.ndim >= 1 and arr.shape[0] == n and n > block_size:
                violation[0] = True
            return orig_exp(x, *a, **kw)

        ref = _oracle(scores, V)

        np.exp = traced_exp
        try:
            got = sol.online_softmax_weighted_sum(scores.copy(), V.copy(), block_size)
            got = np.asarray(got, dtype=np.float64)
        except Exception:
            np.exp = orig_exp
            return {"max_abs_err": float("inf"), "blockwise": 0.0}
        finally:
            np.exp = orig_exp

        if violation[0]:
            blockwise_ok = 0.0

        if got.shape != ref.shape:
            return {"max_abs_err": float("inf"), "blockwise": 0.0}

        err = float(np.max(np.abs(got - ref)))
        worst_err = max(worst_err, err)

    return {"max_abs_err": worst_err, "blockwise": blockwise_ok}
