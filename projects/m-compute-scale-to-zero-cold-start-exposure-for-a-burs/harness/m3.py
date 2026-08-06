import importlib.util
import os
import sys

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
    sys.path.insert(0, workdir)
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_max_timeout_bug": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")

    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        sys.path.pop(0)
        return out

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"the tests fail on a correct optimizer: {type(e).__name__}: {str(e)[:120]}"
        sys.path.pop(0)
        return out

    if first is None:
        out["_note"] = "no test_* functions found"
        sys.path.pop(0)
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import scalezero.optimizer as opt
    good = opt.find_optimal_timeout

    def buggy(traffic, cold_start_latency, max_exposure_ratio):
        total_reqs = sum(traffic)
        if total_reqs == 0:
            return 1
        best = len(traffic)
        from scalezero.simulator import simulate_scale_to_zero
        for timeout in range(1, len(traffic) + 1):
            exposed, _ = simulate_scale_to_zero(traffic, timeout, cold_start_latency)
            if exposed / total_reqs <= max_exposure_ratio:
                best = timeout
        return best

    opt.find_optimal_timeout = buggy
    try:
        out["catches_max_timeout_bug"] = 0.0 if _survives(path) else 1.0
    finally:
        opt.find_optimal_timeout = good
        sys.path.pop(0)

    return out
