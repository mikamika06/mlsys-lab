import importlib.util
import os
import sys

def _run(path):
    spec = importlib.util.spec_from_file_location("test_regression", path)
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
    sys.path.insert(0, workdir)
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_broken_invariant": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")

    if not os.path.isfile(path):
        out["_note"] = "test_regression.py is missing"
        return out

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"fails on correct code: {type(e).__name__}: {str(e)[:120]}"
        return out

    if first is None:
        out["_note"] = "no test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    try:
        import accounting.sizes as s
    except ImportError:
        return out

    good = s.layer_bytes
    def bad_layer_bytes(shape, method):
        c_out = shape[0]
        import math
        n = math.prod(shape)
        w = n // c_out
        if method == "lut4_joint_int8_channel":
            return c_out * math.ceil(w / 2) + c_out * 32
        if method == "lut8_joint_int8_channel":
            return n + c_out * 512
        return good(shape, method)

    s.layer_bytes = bad_layer_bytes
    try:
        survived = _survives(path)
        if not survived:
            out["catches_broken_invariant"] = 1.0
    finally:
        s.layer_bytes = good

    return out
