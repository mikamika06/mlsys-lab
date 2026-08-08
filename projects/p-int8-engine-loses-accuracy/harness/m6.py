import importlib.util
import os
import sys

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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_bad_sensitivity": 0.0, "catches_uncalibrated": 0.0, "faults_caught": 0.0}
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    if workdir not in sys.path:
        sys.path.insert(0, workdir)

    import int8_eng.tuning as tun

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

    good_find = tun.find_sensitive_layers
    tun.find_sensitive_layers = lambda data, threshold=0.03: []
    try:
        out["catches_bad_sensitivity"] = 0.0 if _survives(path) else 1.0
    finally:
        tun.find_sensitive_layers = good_find

    good_calib = tun.calibrate
    tun.calibrate = lambda model, data: model
    try:
        out["catches_uncalibrated"] = 0.0 if _survives(path) else 1.0
    finally:
        tun.calibrate = good_calib

    out["faults_caught"] = out["catches_bad_sensitivity"] + out["catches_uncalibrated"]
    return out
