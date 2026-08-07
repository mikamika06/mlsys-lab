import os
import importlib.util
import ref
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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_ignored_floor": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    sys.path.insert(0, workdir)
    try:
        try:
            first = _run(path)
        except Exception as e:
            out["has_tests"] = 1.0
            out["_note"] = f"tests fail on correct implementation: {type(e).__name__}: {str(e)[:120]}"
            return out

        if first is None:
            out["_note"] = "no test_* functions found"
            return out

        out["has_tests"] = 1.0
        out["passes_on_good"] = 1.0

        import triton.optimize as opt
        good = opt.optimize_delay

        def broken_optimize(arrivals, max_batch_size, preferred_batch_sizes, delays_to_try, throughput_floor, compute_fn):
            best_delay = None
            best_p99 = float('inf')
            for delay in sorted(delays_to_try):
                batches = opt.simulate(arrivals, max_batch_size, preferred_batch_sizes, delay, compute_fn)
                metrics = opt.calculate_metrics(arrivals, batches, compute_fn)
                if metrics["p99_queue_delay"] < best_p99:
                    best_p99 = metrics["p99_queue_delay"]
                    best_delay = delay
            return best_delay

        opt.optimize_delay = broken_optimize
        try:
            out["catches_ignored_floor"] = 0.0 if _survives(path) else 1.0
        finally:
            opt.optimize_delay = good
    finally:
        sys.path.pop(0)
    return out
