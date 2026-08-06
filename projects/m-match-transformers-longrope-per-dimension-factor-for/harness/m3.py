import importlib.util
import os
import numpy as np


def _run(path):
    spec = importlib.util.spec_from_file_location("learner_regression", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fns = [getattr(mod, n) for n in dir(mod) if n.startswith("test_") and callable(getattr(mod, n))]
    if not fns:
        return None
    for fn in fns:
        fn()
    return True


def _survives(path):
    try:
        return _run(path) is True
    except Exception:
        return False


def check(workdir):
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_uniform_scaling": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"The tests fail on a correct implementation: {type(e).__name__}: {str(e)[:120]}"
        return out

    if first is None:
        out["_note"] = "No test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import longrope.scaling as sc
    good_func = sc.compute_longrope_factors

    def buggy_uniform_scaling(head_dim, original_max_len, target_max_len, base=10000.0, short_factor=None, long_factor=None):
        scale = target_max_len / original_max_len
        return np.full(head_dim // 2, scale, dtype=np.float64)

    sc.compute_longrope_factors = buggy_uniform_scaling
    import longrope
    longrope.scaling.compute_longrope_factors = buggy_uniform_scaling

    try:
        survived = _survives(path)
        out["catches_uniform_scaling"] = 0.0 if survived else 1.0
    finally:
        sc.compute_longrope_factors = good_func
        longrope.scaling.compute_longrope_factors = good_func

    return out
