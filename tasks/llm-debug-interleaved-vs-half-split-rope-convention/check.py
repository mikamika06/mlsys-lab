import numpy as np


def _rope_oracle(x, position):
    x = np.asarray(x, dtype=np.float64)
    d = x.shape[0]
    half = d // 2
    idx = np.arange(half, dtype=np.float64)
    theta = position * (10000.0 ** (-2.0 * idx / d))
    c = np.cos(theta)
    s = np.sin(theta)

    out = np.empty(d, dtype=np.float64)
    first = x[:half]
    second = x[half:]
    out[:half] = first * c - second * s
    out[half:] = first * s + second * c
    return out


def grade(sol, fx) -> dict:
    cases = [
        (np.array([1.0, 2.0, 3.0, 4.0]), 1),
        (np.array([0.5, -2.0, 4.0, 7.0, 1.5, -3.0]), 17),
        (np.arange(8, dtype=np.float64) - 2.5, 31),
        (np.array([3.2, -1.1, 5.7, 0.0, -4.2, 8.8, 2.1, -6.5]), 123),
    ]

    worst = 0.0
    for x, pos in cases:
        ref = _rope_oracle(x, pos)
        try:
            got = np.asarray(sol.apply_rope(x.copy(), pos), dtype=np.float64)
        except Exception:
            return {"max_abs_err": float("inf")}
        if got.shape != ref.shape:
            return {"max_abs_err": float("inf")}
        worst = max(worst, float(np.max(np.abs(got - ref))))
    return {"max_abs_err": worst}
