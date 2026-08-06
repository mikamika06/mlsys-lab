import importlib.util
import os
import sys


def _run(path):
    spec = importlib.util.spec_from_file_location("learner_regression", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fns = [getattr(mod, n) for n in dir(mod)
           if n.startswith("test_") and callable(getattr(mod, n))]
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
    if workdir not in sys.path:
        sys.path.insert(0, workdir)

    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_unclamped_zp": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out
    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"the tests fail on correct code: {type(e).__name__}: {str(e)[:120]}"
        return out
    if first is None:
        out["_note"] = "no test_* functions found"
        return out
    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import quantizer.params as p
    good = p.calc_affine_params

    def broken(val_min: float, val_max: float, qmin: int = 0, qmax: int = 255):
        if val_min == val_max:
            return 1.0, qmin
        scale = (val_max - val_min) / float(qmax - qmin)
        zp = int(round(qmin - (val_min / scale)))
        return float(scale), zp

    p.calc_affine_params = broken
    import quantizer
    if hasattr(quantizer, "calc_affine_params"):
        quantizer.calc_affine_params = broken
    try:
        out["catches_unclamped_zp"] = 0.0 if _survives(path) else 1.0
    finally:
        p.calc_affine_params = good
        if hasattr(quantizer, "calc_affine_params"):
            quantizer.calc_affine_params = good
    return out
