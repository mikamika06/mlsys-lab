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
    except Exception:  # noqa: BLE001
        return False


def check(workdir):
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_invalid_sla": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    sys.path.insert(0, workdir)
    try:
        first = _run(path)
    except Exception as e:  # noqa: BLE001
        out["has_tests"] = 1.0
        out["_note"] = f"Tests fail on correct implementation: {type(e).__name__}: {str(e)[:120]}"
        return out

    if first is None:
        out["_note"] = "No test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import capacity.batching as b
    good_batching = b.find_optimal_batch_size

    def broken_find_optimal_batch_size(profiles, sla_latency_ms, cost_per_node_hr):
        best_batch = None
        min_cost = float("inf")
        for p in profiles:
            cost_per_sec = cost_per_node_hr / 3600.0
            cost_per_1k = (cost_per_sec / p["tokens_per_sec"]) * 1000.0
            if cost_per_1k < min_cost:
                min_cost = cost_per_1k
                best_batch = p["batch_size"]
        return {"optimal_batch_size": best_batch, "min_cost_per_1k_tokens": min_cost}

    b.find_optimal_batch_size = broken_find_optimal_batch_size
    import capacity
    capacity.batching.find_optimal_batch_size = broken_find_optimal_batch_size

    try:
        out["catches_invalid_sla"] = 0.0 if _survives(path) else 1.0
    finally:
        b.find_optimal_batch_size = good_batching
        capacity.batching.find_optimal_batch_size = good_batching

    return out
