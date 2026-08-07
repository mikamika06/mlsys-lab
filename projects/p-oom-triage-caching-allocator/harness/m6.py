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
        "catches_bad_fragmentation": 0.0,
        "catches_bad_leak": 0.0,
        "faults_caught": 0.0
    }
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

    import oom_triage.analyzer as ana

    good_frag = ana.analyze_fragmentation
    def bad_frag(snap):
        return {"allocated": 0, "reserved": 0, "max_free": 0}

    ana.analyze_fragmentation = bad_frag
    try:
        out["catches_bad_fragmentation"] = 0.0 if _survives(path) else 1.0
    finally:
        ana.analyze_fragmentation = good_frag

    good_leak = ana.find_leaked_tensors
    def bad_leak(snaps):
        return []

    ana.find_leaked_tensors = bad_leak
    try:
        out["catches_bad_leak"] = 0.0 if _survives(path) else 1.0
    finally:
        ana.find_leaked_tensors = good_leak

    out["faults_caught"] = out["catches_bad_fragmentation"] + out["catches_bad_leak"]
    return out
