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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_unmerged_busy": 0.0}
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

    import gpuprof.busy as b
    good_busy = b.compute_gpu_busy_time

    def naive_sum_busy(x_events, stream_ids=None):
        total = 0.0
        for ev in x_events:
            if ev.get("cat") != "gpu_op":
                continue
            if stream_ids is not None:
                st = ev.get("args", {}).get("stream")
                if st not in stream_ids:
                    continue
            total += float(ev.get("dur", 0.0))
        return total

    b.compute_gpu_busy_time = naive_sum_busy
    import gpuprof
    gpuprof.busy.compute_gpu_busy_time = naive_sum_busy

    try:
        out["catches_unmerged_busy"] = 0.0 if _survives(path) else 1.0
    finally:
        b.compute_gpu_busy_time = good_busy
        gpuprof.busy.compute_gpu_busy_time = good_busy

    return out
