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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_unscaled_lse": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py missing"
        return out
    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"Learner tests fail on valid code: {type(e).__name__}: {str(e)[:120]}"
        return out

    if first is None:
        out["_note"] = "No test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import ringattn.lse as lse_mod
    good_merge = lse_mod.merge_lse_pair

    def broken_unscaled_merge(out_a, max_a, sum_a, out_b, max_b, sum_b):
        new_max = np.maximum(max_a, max_b)
        alpha = np.exp(max_a - new_max)
        beta = np.exp(max_b - new_max)
        new_sum = alpha * sum_a + beta * sum_b
        new_out = (out_a + out_b) / 2.0
        return new_out, new_max, new_sum

    lse_mod.merge_lse_pair = broken_unscaled_merge
    import ringattn
    if hasattr(ringattn, "lse"):
        ringattn.lse.merge_lse_pair = broken_unscaled_merge

    try:
        out["catches_unscaled_lse"] = 0.0 if _survives(path) else 1.0
    finally:
        lse_mod.merge_lse_pair = good_merge
        if hasattr(ringattn, "lse"):
            ringattn.lse.merge_lse_pair = good_merge

    return out
