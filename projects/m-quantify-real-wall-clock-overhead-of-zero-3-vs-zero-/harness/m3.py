import importlib.util
import os
import ref
import numpy as np

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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_broken_invariant": 0.0}
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

    import zeroperf.metrics as m
    good_oh = m.compute_overhead

    def broken_overhead(z2_times, z3_times, warmup=10):
        return -0.25

    m.compute_overhead = broken_overhead
    import zeroperf
    if hasattr(zeroperf, "compute_overhead"):
        zeroperf.compute_overhead = broken_overhead

    try:
        out["catches_broken_invariant"] = 0.0 if _survives(path) else 1.0
    finally:
        m.compute_overhead = good_oh
        if hasattr(zeroperf, "compute_overhead"):
            zeroperf.compute_overhead = good_oh
    return out
