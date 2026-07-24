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
    if len(x) == 0:
        return 0.0
    if len(x) == 1:
        return float(x[0])
    mid = len(x) // 2
    return _pairwise(x[:mid]) + _pairwise(x[mid:])


def _rel_err(a, b):
    return abs(a - b) / (abs(b) + 1e-30)


def summation_error_growth():
    ns = np.asarray([1000, 2000, 4000, 8000, 16000], dtype=np.int64)
    errors = []

    for n in ns:
        x = _sequence(int(n))
        ref = math.fsum(float(v) for v in x)
        errors.append([
            _rel_err(_naive(x), ref),
            _rel_err(_kahan(x), ref),
            _rel_err(_pairwise(x), ref),
        ])

    errors = np.asarray(errors, dtype=np.float64)

    slopes = np.asarray([
        np.polyfit(
            np.log(ns.astype(np.float64)),
            np.log(np.maximum(errors[:, i], 1e-300)),
            1,
        )[0]
        for i in range(3)
    ])

    return {
        "N": ns,
        "naive": errors[:, 0],
        "kahan": errors[:, 1],
        "pairwise": errors[:, 2],
        "slopes": slopes,
    }
