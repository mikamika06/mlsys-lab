import numpy as np

_FAIL = {"valid_input": 0.0, "naive_breaks": 0.0, "stable_holds": 0.0}


def _true_var(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    mean = np.mean(x)
    return float(np.mean((x - mean) ** 2))


def _naive_var(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    mean = np.mean(x)
    meansq = np.mean(x ** 2)
    return float(meansq - mean * mean)


def _welford_var(x: np.ndarray) -> float:
    n = 0
    mean = 0.0
    m2 = 0.0
    for xi in np.asarray(x, dtype=np.float64):
        n += 1
        delta = xi - mean
        mean += delta / n
        delta2 = xi - mean
        m2 += delta * delta2
    return float(m2 / n)


def grade(sol, fx) -> dict:
    try:
        x = np.asarray(sol.pathological_variance_input(), dtype=np.float64).ravel()
    except Exception:
        return dict(_FAIL)

    if x.ndim != 1 or x.shape[0] < 8 or not np.all(np.isfinite(x)):
        return dict(_FAIL)

    true_var = _true_var(x)
    if not np.isfinite(true_var) or true_var <= 1e-6:
        return dict(_FAIL)

    res = {"valid_input": 1.0, "naive_breaks": 0.0, "stable_holds": 0.0}

    naive = _naive_var(x)
    stable = _welford_var(x)
    if not (np.isfinite(naive) and np.isfinite(stable)):
        return res

    naive_rel_err = abs(naive - true_var) / abs(true_var)
    stable_rel_err = abs(stable - true_var) / abs(true_var)

    res["naive_breaks"] = 1.0 if naive_rel_err > 0.5 else 0.0
    res["stable_holds"] = 1.0 if stable_rel_err < 1e-8 else 0.0
    return res
