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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_naive_overlap": 0.0}
    sys.path.insert(0, workdir)
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

    import profile_analyzer.utilization as util_mod

    good_fn = util_mod.compute_profile_utilization

    def naive_sum_utilization(kernels, trace_start_ns, trace_end_ns):
        if trace_end_ns <= trace_start_ns:
            return 0.0
        total_time = sum((k["end_ns"] - k["start_ns"]) for k in kernels)
        return total_time / float(trace_end_ns - trace_start_ns)

    util_mod.compute_profile_utilization = naive_sum_utilization
    try:
        out["catches_naive_overlap"] = 0.0 if _survives(path) else 1.0
    finally:
        util_mod.compute_profile_utilization = good_fn

    return out
