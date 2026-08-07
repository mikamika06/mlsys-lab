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
    path = os.path.join(workdir, "tests", "test_regression.py")
    out = {
        "has_tests": 0.0, "passes_on_good": 0.0,
        "catches_bad_sparsity": 0.0, "catches_bad_bias": 0.0,
        "faults_caught": 0.0
    }
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    import prune.layer as pl

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"tests fail: {e}"
        return out
    if first is None:
        out["_note"] = "no test functions"
        return out
    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    good_prune = pl.prune_unstructured
    def bad_prune(w, scores, sparsity):
        return good_prune(w, scores, 0.99)
    pl.prune_unstructured = bad_prune
    try:
        out["catches_bad_sparsity"] = 0.0 if _survives(path) else 1.0
    finally:
        pl.prune_unstructured = good_prune

    good_bias = pl.correct_bias
    def bad_bias(w, w_pruned, x):
        return np.zeros(w.shape[0])
    pl.correct_bias = bad_bias
    try:
        out["catches_bad_bias"] = 0.0 if _survives(path) else 1.0
    finally:
        pl.correct_bias = good_bias

    out["faults_caught"] = out["catches_bad_sparsity"] + out["catches_bad_bias"]
    return out
