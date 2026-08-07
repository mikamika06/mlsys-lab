import importlib.util
import os
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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_broken_mask": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"tests fail on a correct implementation: {type(e).__name__}: {str(e)[:120]}"
        return out

    if first is None:
        out["_note"] = "no test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import swm.masking as sm
    good_mask = sm.generate_sliding_window_mask

    def broken_mask(seq_len, window_size):
        # Defect: Returns a full causal mask, ignoring the sliding window limit
        i = np.arange(seq_len)[:, None]
        j = np.arange(seq_len)[None, :]
        return (i - j) >= 0

    sm.generate_sliding_window_mask = broken_mask
    try:
        survived = _survives(path)
        if not survived:
            out["catches_broken_mask"] = 1.0
        else:
            out["_note"] = "Test did not fail when the sliding window mask ignored window_size"
    finally:
        sm.generate_sliding_window_mask = good_mask

    return out
