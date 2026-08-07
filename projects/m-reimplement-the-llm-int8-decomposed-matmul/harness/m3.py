import importlib.util
import os


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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_bad_scaling": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out
    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"tests fail on correct implementation: {type(e).__name__}: {str(e)[:120]}"
        return out
    if first is None:
        out["_note"] = "no test_* functions found"
        return out
    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import int8_matmul.dequant as dq
    good_scales = dq.derive_vector_scales

    def bad_scales(tensor):
        import numpy as np
        return np.ones_like(derive_vector_scales_dummy(tensor))

    def derive_vector_scales_dummy(tensor):
        import numpy as np
        return np.max(np.abs(tensor), axis=-1, keepdims=True) / 127.0

    dq.derive_vector_scales = lambda t: np.ones_like(good_scales(t)) * 999.0
    try:
        out["catches_bad_scaling"] = 0.0 if _survives(path) else 1.0
    finally:
        dq.derive_vector_scales = good_scales
    return out
