import numpy as np


def _ref_variance(x):
    """Independent oracle: two-pass mean then centered sum of squares."""
    x = np.asarray(x, dtype=np.float64)
    mean = np.mean(x)
    centered = x - mean
    return float(np.mean(centered * centered))


def grade(sol, fx) -> dict:
    x = [float(v) for v in fx["x"]]
    ref = _ref_variance(x)

    try:
        got = float(sol.welford_variance(x))
    except Exception:
        return {"rel_err": float("inf")}

    if not np.isfinite(got):
        return {"rel_err": float("inf")}

    rel_err = abs(got - ref) / (abs(ref) + 1e-12)
    return {"rel_err": rel_err}
