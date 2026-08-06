import importlib.util
import os

def _run(path):
    spec = importlib.util.spec_from_file_location("learner_regression", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fn = getattr(mod, "test_run_all", None)
    if callable(fn):
        fn()
        return True
    fns = [getattr(mod, n) for n in dir(mod) if n.startswith("test_") and callable(getattr(mod, n))]
    if fns:
        for f in fns:
            f()
        return True
    return None

def _survives(path):
    try:
        return _run(path) is True
    except Exception:
        return False

def check(workdir):
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_broken_flags": 0.0}
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
        out["_note"] = "no test functions found"
        return out
    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import backperf.cmake as cm
    orig_ef = cm.enumerate_flags

    def broken_ef(backend):
        return ["-DGGML_BROKEN_FLAG=ON"]

    cm.enumerate_flags = broken_ef
    try:
        survived = _survives(path)
        out["catches_broken_flags"] = 0.0 if survived else 1.0
    finally:
        cm.enumerate_flags = orig_ef
    return out
