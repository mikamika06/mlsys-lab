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

    log_ns = [math.log(float(n)) for n in ns]
    slopes_list = []
    for i in range(3):
        log_errors_i = [math.log(err if err > 1e-300 else 1e-300) for err in errors[:, i]]
        n_pts = len(log_ns)
        sum_x = 0.0
        sum_y = 0.0
        sum_xy = 0.0
        sum_xx = 0.0
        for j in range(n_pts):
            xi = float(log_ns[j])
            yi = float(log_errors_i[j])
            sum_x += xi
            sum_y += yi
            sum_xy += xi * yi
            sum_xx += xi * xi
        denom = n_pts * sum_xx - sum_x * sum_x
        m = (n_pts * sum_xy - sum_x * sum_y) / denom if denom != 0.0 else 0.0
        slopes_list.append(m)

    slopes = np.asarray(slopes_list, dtype=np.float64)

    return {
        "N": ns,
        "naive": errors[:, 0],
        "kahan": errors[:, 1],
        "pairwise": errors[:, 2],
        "slopes": slopes,
    }
