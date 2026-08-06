import importlib.util
import os
import sys
import time
import torch


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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_unsynced_timing": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out
    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"tests fail on correct bench implementation: {type(e).__name__}: {str(e)[:120]}"
        return out
    if first is None:
        out["_note"] = "no test_* functions found"
        return out
    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import mpsbench.bench as b
    good_time_execution = b.time_execution

    def broken_time_execution(fn, device_str):
        t0 = time.perf_counter()
        res = fn()
        t1 = time.perf_counter()
        return res, t1 - t0

    b.time_execution = broken_time_execution
    import mpsbench
    mpsbench.bench.time_execution = broken_time_execution

    try:
        out["catches_unsynced_timing"] = 0.0 if _survives(path) else 1.0
    finally:
        b.time_execution = good_time_execution
        mpsbench.bench.time_execution = good_time_execution

    return out
