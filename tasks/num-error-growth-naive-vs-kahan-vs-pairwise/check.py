import math
import numpy as np


def _sequence(n):
    x = np.ones(n, dtype=np.float64)
    x[0] = 1e16
    return x


def _naive(x):
    s = 0.0
    for v in x:
        s += float(v)
    return s


def _kahan(x):
    s = 0.0
    c = 0.0
    for v in x:
        y = float(v) - c
        t = s + y
        c = (t - s) - y
        s = t
    return s


def _pairwise(x):
    n = len(x)
    if n == 0:
        return 0.0
    if n == 1:
        return float(x[0])
    mid = n // 2
    return _pairwise(x[:mid]) + _pairwise(x[mid:])


def _oracle(ns):
    errors = []
    for n in ns:
        x = _sequence(int(n))
        ref = math.fsum(float(v) for v in x)
        vals = [_naive(x), _kahan(x), _pairwise(x)]
        errors.append([
            abs(v - ref) / (abs(ref) + 1e-30)
            for v in vals
        ])
    return np.asarray(errors, dtype=np.float64)


def _fit_slopes(ns, errors):
    return np.asarray([
        np.polyfit(
            np.log(ns.astype(np.float64)),
            np.log(np.maximum(errors[:, i], 1e-300)),
            1,
        )[0]
        for i in range(errors.shape[1])
    ])


def grade(sol, fx) -> dict:
    ns = np.asarray([1000, 2000, 4000, 8000, 16000], dtype=np.int64)
    try:
        got = sol.summation_error_growth()
        oracle = _oracle(ns)
    except Exception:
        return {"slope_behavior": 0.0, "final_error_quality": 0.0}

    try:
        if not np.array_equal(np.asarray(got["N"]), ns):
            return {"slope_behavior": 0.0, "final_error_quality": 0.0}

        measured = np.column_stack([
            np.asarray(got["naive"], dtype=np.float64),
            np.asarray(got["kahan"], dtype=np.float64),
            np.asarray(got["pairwise"], dtype=np.float64),
        ])

        slopes = _fit_slopes(ns, measured)
        oracle_slopes = _fit_slopes(ns, oracle)

        slope_ok = (
            slopes[0] > 0.8
            and slopes[1] < 0.5
            and slopes[2] < 0.5
            and slopes[0] > slopes[1]
            and slopes[0] > slopes[2]
        )

        final_ok = (
            np.max(
                np.abs(
                    np.log10(np.maximum(measured[-1], 1e-300))
                    - np.log10(np.maximum(oracle[-1], 1e-300))
                )
            ) < 0.25
            and np.max(np.abs(slopes - np.asarray(got["slopes"]))) < 0.15
            and np.max(np.abs(oracle_slopes - slopes)) < 0.25
        )

        return {
            "slope_behavior": 1.0 if slope_ok else 0.0,
            "final_error_quality": 1.0 if final_ok else 0.0,
        }
    except Exception:
        return {"slope_behavior": 0.0, "final_error_quality": 0.0}
