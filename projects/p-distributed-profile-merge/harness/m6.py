import importlib.util
import os

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
        "has_tests": 0.0,
        "passes_on_good": 0.0,
        "catches_clock_drift": 0.0,
        "catches_false_straggler": 0.0,
        "faults_caught": 0.0
    }
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    import profiler.merge as pm

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

    good_align = pm.align_clocks
    def faulty_align(profiles):
        return profiles
    pm.align_clocks = faulty_align
    try:
        out["catches_clock_drift"] = 0.0 if _survives(path) else 1.0
    finally:
        pm.align_clocks = good_align

    good_find = pm.find_straggler
    def faulty_find(timeline):
        return 0
    pm.find_straggler = faulty_find
    try:
        out["catches_false_straggler"] = 0.0 if _survives(path) else 1.0
    finally:
        pm.find_straggler = good_find

    out["faults_caught"] = out["catches_clock_drift"] + out["catches_false_straggler"]
    return out
