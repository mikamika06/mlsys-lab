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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_unscaled": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"tests fail on correct code: {type(e).__name__}: {str(e)[:120]}"
        return out

    if first is None:
        out["_note"] = "no test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import gradacc.accumulate as ga
    good_accumulate = ga.accumulate

    def buggy_accumulate(micro_batches, W, b, steps):
        dW_acc = np.zeros_like(W)
        db_acc = np.zeros_like(b)
        for X, Y in micro_batches:
            dW, db = ga.compute_gradients(X, Y, W, b)
            # BUG: Not scaled by accumulation_steps
            dW_acc += dW
            db_acc += db
        return dW_acc, db_acc

    ga.accumulate = buggy_accumulate
    try:
        out["catches_unscaled"] = 0.0 if _survives(path) else 1.0
    finally:
        ga.accumulate = good_accumulate

    return out
