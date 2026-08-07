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
    sys.path.insert(0, workdir)
    path = os.path.join(workdir, "tests", "test_regression.py")
    out = {
        "has_tests": 0.0,
        "passes_on_good": 0.0,
        "catches_broken_eviction": 0.0,
        "catches_broken_economics": 0.0,
        "faults_caught": 0.0,
    }
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    import cache.economics as econ
    import cache.eviction as evict

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"learner tests failed on correct reference: {type(e).__name__}: {e}"
        return out

    if first is None:
        out["_note"] = "no test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    orig_access = evict.CacheSimulator.access

    def broken_access(self, key, compute_cost=1.0):
        self.hits += 1
        return True

    evict.CacheSimulator.access = broken_access
    try:
        out["catches_broken_eviction"] = 0.0 if _survives(path) else 1.0
    finally:
        evict.CacheSimulator.access = orig_access

    orig_breakeven = econ.compute_breakeven_hit_rate

    def broken_breakeven(compute_cost_per_req, total_requests, total_memory_cost):
        return -0.5

    econ.compute_breakeven_hit_rate = broken_breakeven
    try:
        out["catches_broken_economics"] = 0.0 if _survives(path) else 1.0
    finally:
        econ.compute_breakeven_hit_rate = orig_breakeven

    out["faults_caught"] = out["catches_broken_eviction"] + out["catches_broken_economics"]
    return out
