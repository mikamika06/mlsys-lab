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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_naive_scheduler": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"the tests fail on a correct schedule: {type(e).__name__}: {str(e)[:120]}"
        return out

    if first is None:
        out["_note"] = "no test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import loraserve.scheduler as s
    good = s.schedule_adapter_batch

    def naive_fifo_scheduler(requests, max_batch_size, max_active_adapters):
        batches = []
        for i in range(0, len(requests), max_batch_size):
            batches.append(requests[i:i + max_batch_size])
        return batches

    s.schedule_adapter_batch = naive_fifo_scheduler
    import loraserve
    loraserve.schedule_adapter_batch = naive_fifo_scheduler

    try:
        out["catches_naive_scheduler"] = 0.0 if _survives(path) else 1.0
    finally:
        s.schedule_adapter_batch = good
        loraserve.schedule_adapter_batch = good

    return out
