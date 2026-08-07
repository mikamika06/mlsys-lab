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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_leaky_weights": 0.0}
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

    import qlora.train as qt
    original_train = qt.train_20_steps

    def bad_train(layer, X, target, lr=0.01):
        res = original_train(layer, X, target, lr)
        # Corrupt the frozen weights to test if regression test catches it
        layer.weight = np.zeros_like(layer.weight)
        return res

    qt.train_20_steps = bad_train

    try:
        passed = _survives(path)
    finally:
        qt.train_20_steps = original_train

    if not passed:
        out["catches_leaky_weights"] = 1.0
    else:
        out["_note"] = "test did not fail when base weights were modified"

    return out
