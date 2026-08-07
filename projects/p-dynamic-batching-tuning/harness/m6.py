import importlib.util
import os

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
    path = os.path.join(workdir, "tests", "test_regression.py")
    out = {
        "has_tests": 0.0,
        "passes_on_good": 0.0,
        "catches_bad_window": 0.0,
        "catches_unbounded_queue": 0.0,
        "faults_caught": 0.0
    }
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"tests failed: {e}"
        return out

    if first is None:
        out["_note"] = "no test functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import batching.window as bw
    good_win = bw.find_optimal_window
    bw.find_optimal_window = lambda c, s: {"optimal_window": -999.0, "max_throughput": 0.0}
    try:
        out["catches_bad_window"] = 0.0 if _survives(path) else 1.0
    finally:
        bw.find_optimal_window = good_win

    import batching.queues as bq
    good_pop = bq.TieredQueueManager.pop_batch
    def bad_pop(self, max_size):
        return []
    bq.TieredQueueManager.pop_batch = bad_pop
    try:
        out["catches_unbounded_queue"] = 0.0 if _survives(path) else 1.0
    finally:
        bq.TieredQueueManager.pop_batch = good_pop

    out["faults_caught"] = out["catches_bad_window"] + out["catches_unbounded_queue"]
    return out
