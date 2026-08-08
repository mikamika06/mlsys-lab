import importlib.util
import os
import sys

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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_broken_mse": 0.0, "catches_broken_bias": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"the tests fail on a correct implementation: {type(e).__name__}: {str(e)[:120]}"
        return out

    if first is None:
        out["_note"] = "no test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import quant_observe.observer as obs
    import quant_observe.bias as b
    good_mse = obs.mse_observer
    good_bias = b.ignored_zp_bias

    def bad_mse(x, args):
        scale, zp = obs.minmax_observer(x, args)
        return scale * 2.0, zp

    def bad_bias(x, args, method):
        return -1.0

    obs.mse_observer = bad_mse
    b.mse_observer = bad_mse
    try:
        if not _survives(path):
            out["catches_broken_mse"] = 1.0
    finally:
        obs.mse_observer = good_mse
        b.mse_observer = good_mse

    b.ignored_zp_bias = bad_bias
    try:
        if not _survives(path):
            out["catches_broken_bias"] = 1.0
    finally:
        b.ignored_zp_bias = good_bias

    return out
