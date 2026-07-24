import numpy as np
from mlsys.sim import cache as cachesim


def _ftz(x):
    if 0 < abs(x) < np.finfo(np.float64).tiny:
        return 0.0
    return x


def _ref(a, b):
    acc = 0.0
    for x, y in zip(a, b):
        acc += _ftz(float(x) * float(y))
    trace = []
    for i in range(len(a)):
        trace.append(i * 8)
        trace.append(4096 + i * 8)
    return acc, trace


def grade(sol, fx) -> dict:
    cases = [
        (
            np.array([1.0, 2.0, -3.0], dtype=np.float64),
            np.array([4.0, 5.0, 6.0], dtype=np.float64),
        ),
        (
            np.array([1e-300, 1e-300, 1.0], dtype=np.float64),
            np.array([1e-10, -1e-10, 2.0], dtype=np.float64),
        ),
        (
            np.array([np.finfo(np.float64).tiny, 1.0, -1.0], dtype=np.float64),
            np.array([0.5, 2.0, 3.0], dtype=np.float64),
        ),
    ]

    max_err = 0.0
    miss_ok = 1.0
    for a, b in cases:
        try:
            value, addrs = sol.dot_ftz_trace(a, b)
        except Exception:
            return {"max_abs_err": float("inf"), "miss_count": 0.0}

        ref_value, ref_trace = _ref(a, b)
        max_err = max(max_err, abs(float(value) - ref_value))

        got_misses = cachesim.simulate(
            addrs,
            line_bytes=64,
            sets=4,
            ways=2,
        )
        ref_misses = cachesim.simulate(
            ref_trace,
            line_bytes=64,
            sets=4,
            ways=2,
        )
        if got_misses != ref_misses:
            miss_ok = 0.0

    return {
        "max_abs_err": max_err,
        "miss_count": 2 if miss_ok == 1.0 else 0,
    }
