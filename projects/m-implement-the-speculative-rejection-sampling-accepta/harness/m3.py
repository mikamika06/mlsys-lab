import importlib.util
import os
import sys
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
    sys.path.insert(0, workdir)
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_missing_q_norm": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")

    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        sys.path.pop(0)
        return out

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"tests fail on a correct implementation: {e}"
        sys.path.pop(0)
        return out

    if first is None:
        out["_note"] = "no test_* functions found"
        sys.path.pop(0)
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    try:
        import speculative.accept as acc
    except ImportError:
        sys.path.pop(0)
        return out

    good_eval = acc.evaluate_draft

    def bad_eval(target_p, draft_p, tokens, u):
        k = target_p.shape[0]
        for i in range(k):
            p_i = target_p[i, tokens[i]]
            if u[i] >= p_i:
                diff = np.maximum(0.0, target_p[i] - draft_p[i])
                s = np.sum(diff)
                if s > 0:
                    diff /= s
                return i, diff
        return k, None

    acc.evaluate_draft = bad_eval
    try:
        if not _survives(path):
            out["catches_missing_q_norm"] = 1.0
    finally:
        acc.evaluate_draft = good_eval
        sys.path.pop(0)

    return out
