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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_fault": 0.0, "faults_caught": 0.0}
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    import profiler_study.modes as modes_mod

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"tests failed: {e}"
        return out
    if first is None:
        out["_note"] = "no test functions found"
        return out
    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    good_measure = modes_mod.ProfilerEnvironment.measure

    def faulty_measure(self, mode):
        return 999.0

    modes_mod.ProfilerEnvironment.measure = faulty_measure
    try:
        out["catches_fault"] = 0.0 if _survives(path) else 1.0
    finally:

        modes_mod.ProfilerEnvironment.measure = good_measure

    out["faults_caught"] = out["catches_fault"]
    return out
