import importlib.util
import os
import sys

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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_bad_merge": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    sys.path.insert(0, workdir)
    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"the tests fail on a correct implementation: {type(e).__name__}: {str(e)[:120]}"
        return out

    if first is None:
        out["_note"] = "no test_* functions found"
        sys.path.pop(0)
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import adapter_merge.numerical as num
    good = num.merge_weights

    def bad_merge(w_base, lora_a, lora_b, scale):
        return w_base + (lora_b @ lora_a)

    num.merge_weights = bad_merge
    try:
        if not _survives(path):
            out["catches_bad_merge"] = 1.0
    finally:
        num.merge_weights = good
        sys.path.pop(0)

    return out
