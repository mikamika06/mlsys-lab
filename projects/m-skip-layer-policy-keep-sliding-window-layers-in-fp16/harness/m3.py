import importlib.util
import os
import sys

def _run(path):
    spec = importlib.util.spec_from_file_location("learner_regression", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["learner_regression"] = mod
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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_bad_policy": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    sys.path.insert(0, workdir)
    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"tests fail on correct implementation: {type(e).__name__} - {str(e)}"
        return out
    finally:
        sys.path.pop(0)

    if first is None:
        out["_note"] = "no test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    sys.path.insert(0, workdir)
    import policy.quant as pq
    good_policy = pq.assign_kv_dtypes

    def bad_policy(layers):
        return ["float8" for _ in layers]

    pq.assign_kv_dtypes = bad_policy
    try:
        if sys.modules.get("learner_regression"):
            del sys.modules["learner_regression"]
        survived = _survives(path)
        out["catches_bad_policy"] = 0.0 if survived else 1.0
    finally:
        pq.assign_kv_dtypes = good_policy
        sys.path.pop(0)

    return out
