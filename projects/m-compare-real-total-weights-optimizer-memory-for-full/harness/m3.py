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
    if workdir not in sys.path:
        sys.path.insert(0, workdir)

    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_broken_memory": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out
    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"the tests fail on a correct implementation: {type(e).__name__}: {str(e)[:120]}"
        return out
    if first is None:
        out["_note"] = "no test_* functions found"
        return out
    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    try:
        import optmem.states as s
        good_compute = s.compute_optimizer_bytes

        def broken_compute(config, mode):
            res = good_compute(config, mode)
            res["optimizer"] = 0
            res["total"] = res["weights"] + res["gradients"]
            return res

        s.compute_optimizer_bytes = broken_compute
        try:
            out["catches_broken_memory"] = 0.0 if _survives(path) else 1.0
        finally:
            s.compute_optimizer_bytes = good_compute
    except Exception as e:
        out["_note"] = f"Error setting up test injection: {type(e).__name__}: {e}"
    return out
