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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_no_warmup": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    sys.path.insert(0, workdir)
    sys.modules.pop("bench.harness", None)

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"the tests fail on a correct benchmark: {type(e).__name__}: {str(e)[:120]}"
        return out

    if first is None:
        out["_note"] = "no test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import bench.harness as h
    good = h.benchmark

    def broken_benchmark(fn, warmup_iters, measure_iters, percentiles):
        import time
        import numpy as np
        times = []
        for _ in range(measure_iters):
            t0 = time.perf_counter_ns()
            fn()
            t1 = time.perf_counter_ns()
            times.append(t1 - t0)
        if not times:
            return {p: 0.0 for p in percentiles}
        res = np.atleast_1d(np.percentile(times, percentiles))
        return {p: float(val) for p, val in zip(percentiles, res)}

    h.benchmark = broken_benchmark
    sys.modules.pop("bench.harness", None) # Force test to import the patched one
    sys.modules["bench.harness"] = h

    try:
        out["catches_no_warmup"] = 0.0 if _survives(path) else 1.0
    finally:
        h.benchmark = good
        sys.path.pop(0)

    return out
