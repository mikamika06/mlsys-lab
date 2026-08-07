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
           "catches_memory_overflow": 0.0, "catches_broken_cache": 0.0,
           "faults_caught": 0.0}
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    import moe.cache as c_mod

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"tests fail: {e}"
        return out

    if first is None:
        out["_note"] = "no test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    orig_init = c_mod.ExpertCache.__init__
    def broken_init(self, capacity_bytes):
        orig_init(self, capacity_bytes * 10)

    c_mod.ExpertCache.__init__ = broken_init
    try:
        out["catches_memory_overflow"] = 0.0 if _survives(path) else 1.0
    finally:
        c_mod.ExpertCache.__init__ = orig_init

    orig_access = c_mod.ExpertCache.access
    def broken_access(self, expert_id, size_bytes=1000):
        return True

    c_mod.ExpertCache.access = broken_access
    try:
        out["catches_broken_cache"] = 0.0 if _survives(path) else 1.0
    finally:
        c_mod.ExpertCache.access = orig_access

    out["faults_caught"] = out["catches_memory_overflow"] + out["catches_broken_cache"]
    return out
