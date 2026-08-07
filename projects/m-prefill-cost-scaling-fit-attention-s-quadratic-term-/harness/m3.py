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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_memory_bug": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    import sys
    sys.path.insert(0, workdir)

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"fails on good: {e}"
        return out

    if first is None:
        out["_note"] = "no test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import prefill.memory as m
    good = m.cheapest_config

    def broken_config(model, gpus, ctx, budget):
        req_gb = model["weights_gb"]
        best = None
        best_cost = float('inf')
        best_mem = -1
        for g in gpus:
            for n in range(1, g["count"] + 1):
                mem = n * g["mem_gb"]
                cost = n * g["cost_per_hr"]
                if mem >= req_gb and cost <= budget:
                    if cost < best_cost or (cost == best_cost and mem > best_mem):
                        best = (g["name"], n)
                        best_cost = cost
                        best_mem = mem
        return best

    m.cheapest_config = broken_config
    import prefill
    if hasattr(prefill, "memory"):
        prefill.memory.cheapest_config = broken_config

    try:
        out["catches_memory_bug"] = 0.0 if _survives(path) else 1.0
    finally:
        m.cheapest_config = good
        if hasattr(prefill, "memory"):
            prefill.memory.cheapest_config = good

    return out
