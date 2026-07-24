import inspect
import math
import re

import numpy as np

FAIL = {
    "rel_err": float("inf"),
    "exact_zero_fraction": 0.0,
    "naive_rel_err": float("inf"),
}


def _uses_library_expm1(sol) -> bool:
    try:
        src = inspect.getsource(sol)
    except Exception:
        return True
    return re.search(r"expm1", src) is not None


def _inputs():
    rng = np.random.default_rng(0)
    return np.concatenate([
        np.logspace(-18.0, 0.0, 400),
        -np.logspace(-18.0, 0.0, 400),
        np.linspace(-30.0, 30.0, 401),
        np.array([0.0, 5e-324, -5e-324, 1e-300, -1e-300, 1.0, -1.0, 30.0, -30.0]),
        rng.normal(scale=5.0, size=400),
    ])


def grade(sol, fx) -> dict:
    if _uses_library_expm1(sol):
        return dict(FAIL)

    x = _inputs()
    ref = np.expm1(x)                       # real NumPy oracle

    # block the C-level shortcuts for the duration of the call
    np_saved, math_saved = np.expm1, math.expm1

    def _blocked(*a, **k):
        raise RuntimeError("expm1 is not allowed in this task")

    np.expm1 = _blocked
    math.expm1 = _blocked
    try:
        got = sol.exp_minus_one(x.copy())
    except Exception:
        return dict(FAIL)
    finally:
        np.expm1 = np_saved
        math.expm1 = math_saved

    got = np.asarray(got, dtype=np.float64)
    if got.shape != x.shape or not np.all(np.isfinite(got)):
        return dict(FAIL)

    nz = ref != 0.0
    rel = np.abs(got[nz] - ref[nz]) / np.abs(ref[nz])
    zero_ok = float(np.mean(got[~nz] == ref[~nz])) if np.any(~nz) else 1.0

    with np.errstate(over="ignore"):
        naive = np.exp(x) - 1.0
    naive_rel = np.abs(naive[nz] - ref[nz]) / np.abs(ref[nz])

    return {
        "rel_err": float(np.max(rel)),
        "exact_zero_fraction": zero_ok,
        "naive_rel_err": float(np.max(naive_rel)),
    }
