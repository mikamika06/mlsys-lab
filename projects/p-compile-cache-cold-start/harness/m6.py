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
    path = os.path.join(workdir, "tests", "test_regression.py")
    out = {"has_tests": 0.0, "passes_on_good": 0.0,
           "catches_invalid_version": 0.0, "catches_stale_cache": 0.0,
           "faults_caught": 0.0}
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    import compcache.cache as cc
    import compcache.invalidator as inv

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"tests fail on correct implementation: {type(e).__name__}: {e}"
        return out
    if first is None:
        out["_note"] = "no test_* functions found"
        return out
    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    good_check = inv.check_version
    inv.check_version = lambda cache, ver: True
    try:
        out["catches_invalid_version"] = 0.0 if _survives(path) else 1.0
    finally:
        inv.check_version = good_check

    good_lookup = cc.CompilationCache.lookup
    def broken_lookup(self, key):
        return b"stale_blob"
    cc.CompilationCache.lookup = broken_lookup
    try:
        out["catches_stale_cache"] = 0.0 if _survives(path) else 1.0
    finally:
        cc.CompilationCache.lookup = good_lookup

    out["faults_caught"] = out["catches_invalid_version"] + out["catches_stale_cache"]
    return out
