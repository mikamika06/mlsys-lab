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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "faults_caught": 0.0}
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    import sys
    if workdir not in sys.path:
        sys.path.insert(0, workdir)

    import moe_drops.router as router

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"tests fail on a correct implementation: {type(e).__name__}: {e}"
        return out

    if first is None:
        out["_note"] = "no test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    good_compute = router.compute_dropped_fraction

    def bad_compute(*args, **kwargs):
        return 0.0

    router.compute_dropped_fraction = bad_compute
    try:
        caught = not _survives(path)
        out["faults_caught"] = 1.0 if caught else 0.0
    finally:
        router.compute_dropped_fraction = good_compute

    return out
