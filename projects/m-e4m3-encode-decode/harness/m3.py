import importlib.util
import os
import sys


def _run(path):
    spec = importlib.util.spec_from_file_location("learner_regression", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fns = [
        getattr(mod, n)
        for n in dir(mod)
        if n.startswith("test_") and callable(getattr(mod, n))
    ]
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
    out = {
        "has_tests": 0.0,
        "passes_on_good": 0.0,
        "catches_invalid_e4m3_max": 0.0,
    }
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"tests fail on reference code: {type(e).__name__}: {str(e)[:120]}"
        return out

    if first is None:
        out["_note"] = "no test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import fp8.descale as descale_mod

    orig_compute_scale = descale_mod.compute_scale

    def broken_compute_scale(x):
        import numpy as np

        max_val = float(np.max(np.abs(x)))
        if max_val == 0.0:
            return 1.0
        return 255.0 / max_val

    descale_mod.compute_scale = broken_compute_scale
    if "fp8.optimize" in sys.modules:
        sys.modules["fp8.optimize"].compute_scale = broken_compute_scale

    try:
        survived = _survives(path)
        out["catches_invalid_e4m3_max"] = 0.0 if survived else 1.0
    finally:
        descale_mod.compute_scale = orig_compute_scale
        if "fp8.optimize" in sys.modules:
            sys.modules["fp8.optimize"].compute_scale = orig_compute_scale

    return out
