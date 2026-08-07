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
           "catches_bad_threads": 0.0, "catches_disabled_arena": 0.0,
           "faults_caught": 0.0}
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    import ort_tune.config as cfg_mod

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"the tests fail on a correct implementation: {type(e).__name__}: {e}"
        return out
    if first is None:
        out["_note"] = "no test_* functions found"
        return out
    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    good_run = cfg_mod.RuntimeEngine.run
    def bad_run(self, inputs):
        return 210.0
    cfg_mod.RuntimeEngine.run = bad_run
    try:
        out["catches_bad_threads"] = 0.0 if _survives(path) else 1.0
    finally:
        cfg_mod.RuntimeEngine.run = good_run

    def bad_run_arena(self, inputs):
        if self.config.get("enable_arena"):
            return 150.0
        return 210.0
    cfg_mod.RuntimeEngine.run = bad_run_arena
    try:
        out["catches_disabled_arena"] = 0.0 if _survives(path) else 1.0
    finally:
        cfg_mod.RuntimeEngine.run = good_run

    out["faults_caught"] = out["catches_bad_threads"] + out["catches_disabled_arena"]
    return out
