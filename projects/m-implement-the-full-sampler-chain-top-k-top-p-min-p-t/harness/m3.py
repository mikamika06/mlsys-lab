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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_bad_min_p": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"tests fail on correct implementation: {e}"
        return out

    if first is None:
        out["_note"] = "no test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import sampler.chain as c
    good_min_p = c.apply_min_p

    import numpy as np

    def bad_min_p(logits, p):
        out = logits.copy()
        if p <= 0.0:
            return out
        probs = np.exp(out - np.max(out))
        probs /= np.sum(probs)
        # BUG: using a flat probability threshold 'p' instead of 'p * max_prob'
        out[probs < p] = -np.inf
        return out

    c.apply_min_p = bad_min_p
    try:
        out["catches_bad_min_p"] = 0.0 if _survives(path) else 1.0
    finally:
        c.apply_min_p = good_min_p

    return out
