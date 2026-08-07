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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_broken_search": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")

    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    sys.path.insert(0, workdir)

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

    import awq_clip.quant as q
    good_search = q.search_clipping

    def bad_search(w, n_bits=4, group_size=128, n_grid=100):
        w_reshaped = w.reshape(-1, group_size)
        w_max = np.max(np.abs(w_reshaped), axis=1, keepdims=True)
        w_max = np.maximum(w_max, 1e-7)
        best_idx = np.zeros(w_reshaped.shape[0], dtype=int)
        opt_max = w_max * 0.01
        return best_idx, opt_max

    q.search_clipping = bad_search

    try:
        survived = _survives(path)
        out["catches_broken_search"] = 0.0 if survived else 1.0
        if survived:
            out["_note"] = "Test did not fail when search_clipping returned a terrible scaling factor (c=0.01)"
    finally:
        q.search_clipping = good_search

    return out
