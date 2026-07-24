import numpy as np


def _oracle_bound(n: int) -> float:
    eps = np.finfo(np.float32).eps
    return float((n * eps) / (1.0 - n * eps))


def _empirical_max_rel_err():
    rng = np.random.default_rng(12345)
    max_err = 0.0
    max_n = 0
    for n in [8, 32, 128, 512, 2048]:
        a = rng.normal(size=n).astype(np.float32)
        b = rng.normal(size=n).astype(np.float32)
        got32 = np.dot(a, b).astype(np.float64)
        ref64 = np.dot(a.astype(np.float64), b.astype(np.float64))
        err = abs(got32 - ref64) / (abs(ref64) + 1e-30)
        if err > max_err:
            max_err = float(err)
            max_n = n
    return max_n, max_err


def grade(sol, fx) -> dict:
    try:
        for n in [1, 8, 32, 128, 512, 2048]:
            predicted = float(sol.dot_error_bound(n))
            oracle = _oracle_bound(n)
            if not np.isfinite(predicted):
                return {"rel_err": 0.0}
            if abs(predicted - oracle) > abs(oracle) * 1e-12 + 1e-15:
                return {"rel_err": 0.0}

        max_n, measured = _empirical_max_rel_err()
        predicted = float(sol.dot_error_bound(max_n))
        if measured > predicted * 1.0000001:
            return {"rel_err": 0.0}
    except Exception:
        return {"rel_err": 0.0}

    return {"rel_err": 1.0}
