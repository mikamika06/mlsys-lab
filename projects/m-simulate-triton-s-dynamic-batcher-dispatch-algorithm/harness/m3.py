import importlib.util
import os
import sys

def _run(path, workdir):
    sys.path.insert(0, workdir)
    try:
        spec = importlib.util.spec_from_file_location("learner_regression", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        fns = [getattr(mod, n) for n in dir(mod) if n.startswith("test_") and callable(getattr(mod, n))]
        if not fns:
            return None
        for fn in fns:
            fn()
        return True
    finally:
        sys.path.pop(0)

def _survives(path, workdir):
    try:
        return _run(path, workdir) is True
    except Exception:
        return False

def check(workdir):
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_ignored_floor": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    try:
        first = _run(path, workdir)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"the tests fail on a correct optimizer: {type(e).__name__}: {str(e)[:120]}"
        return out

    if first is None:
        out["_note"] = "no test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    sys.path.insert(0, workdir)
    try:
        import triton_batcher.optimize as opt
        good = opt.optimize_config

        def broken_optimize(arr, mx, pref_cands, delay_cands, floor, comp):
            from triton_batcher.simulate import simulate
            from triton_batcher.metrics import measure_metrics
            best_p99 = float('inf')
            best_config = None
            for p in pref_cands:
                for d in delay_cands:
                    ds = simulate(arr, mx, p, d, comp)
                    m = measure_metrics(arr, ds, comp)
                    # BUG: Ignores the throughput floor!
                    if m["p99_queue_delay_us"] < best_p99:
                        best_p99 = m["p99_queue_delay_us"]
                        best_config = {"preferred": p, "delay_us": d}
            return best_config

        opt.optimize_config = broken_optimize
        try:
            out["catches_ignored_floor"] = 0.0 if _survives(path, workdir) else 1.0
        finally:
            opt.optimize_config = good
    finally:
        sys.path.pop(0)

    return out
