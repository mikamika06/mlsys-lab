import numpy as np

MIN_SUBNORMAL = 2.0 ** -9
MIN_NORMAL = 2.0 ** -6
MAX_NORMAL = 448.0

UNDERFLOW_TO_ZERO = 0
SUBNORMAL = 1
NORMAL = 2
OVERFLOW_CLAMPED = 3


def _oracle(x):
    x = np.asarray(x, dtype=np.float64)
    a = np.abs(x)
    out = np.full(x.shape, OVERFLOW_CLAMPED, dtype=np.int64)
    out = np.where(a <= MAX_NORMAL, NORMAL, out)
    out = np.where(a < MIN_NORMAL, SUBNORMAL, out)
    out = np.where(a < MIN_SUBNORMAL, UNDERFLOW_TO_ZERO, out)
    return out.astype(np.int64)


def grade(sol, fx) -> dict:
    x = fx["fp8_x"]
    ref = _oracle(x)

    try:
        got = sol.classify_e4m3_regime(x.copy())
        got = np.asarray(got)
    except Exception:
        return {"exact_match": 0.0}

    if got.shape != ref.shape:
        return {"exact_match": 0.0}

    ok = float(np.array_equal(got.astype(np.int64), ref))
    return {"exact_match": ok}
