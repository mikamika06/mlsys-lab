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
    path = os.path.join(workdir, "tests", "test_regression.py")
    out = {"has_tests": 0.0, "passes_on_good": 0.0,
           "catches_bad_scaling": 0.0, "catches_small_sample": 0.0,
           "faults_caught": 0.0}
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    sys.path.insert(0, workdir)
    import quant.calibration as cal

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"tests fail: {e}"
        return out
    if first is None:
        out["_note"] = "no test functions found"
        return out
    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    good_eval = cal.evaluate_drop
    cal.evaluate_drop = lambda data: {"wiki": 0.1, "code": 0.2, "logs": 0.15}
    try:
        out["catches_bad_scaling"] = 0.0 if _survives(path) else 1.0
    finally:
        cal.evaluate_drop = good_eval

    good_size = cal.find_min_size
    cal.find_min_size = lambda data: 1000
    try:
        out["catches_small_sample"] = 0.0 if _survives(path) else 1.0
    finally:
        cal.find_min_size = good_size

    out["faults_caught"] = out["catches_bad_scaling"] + out["catches_small_sample"]
    return out
