import importlib.util
import os
import sys
import numpy as np

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
    out = {
        "has_tests": 0.0,
        "passes_on_good": 0.0,
        "catches_linear_search": 0.0,
        "catches_ignored_sanitization": 0.0,
    }
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    if workdir not in sys.path:
        sys.path.insert(0, workdir)

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"Tests fail on correct implementation: {type(e).__name__}: {str(e)[:120]}"
        return out

    if first is None:
        out["_note"] = "No test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import polyreduce.bisect as bmod
    import polyreduce.compare as cmod

    good_bisect = bmod.bisect_divergent_step
    good_classify = cmod.classify_divergence

    def linear_bisect(num_steps, check_step_fn):
        res = -1
        for i in range(num_steps):
            if not check_step_fn(i):
                if res == -1:
                    res = i
        return res

    bmod.bisect_divergent_step = linear_bisect
    try:
        out["catches_linear_search"] = 0.0 if _survives(path) else 1.0
    finally:
        bmod.bisect_divergent_step = good_bisect

    def no_san_classify(arr_a, arr_b, rtol=1e-3, atol=1e-5, denormal_threshold=1e-7):
        a = np.asarray(arr_a)
        b = np.asarray(arr_b)
        if a.shape != b.shape:
            return "REAL_BUG"
        if np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=True):
            return "MATCH"
        return "REAL_BUG"

    cmod.classify_divergence = no_san_classify
    try:
        out["catches_ignored_sanitization"] = 0.0 if _survives(path) else 1.0
    finally:
        cmod.classify_divergence = good_classify

    return out
