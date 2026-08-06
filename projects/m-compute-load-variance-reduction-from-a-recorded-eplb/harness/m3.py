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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_naive_bound": 0.0}
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

    import eplb.bounds as b
    good_min_redundant = b.min_redundant_replicas

    def naive_min_redundant(expert_loads, num_ranks, target_max_load):
        total_load = sum(expert_loads)
        target_total = target_max_load * num_ranks
        if total_load <= target_total:
            return 0
        return int((total_load - target_total) / max(expert_loads))

    b.min_redundant_replicas = naive_min_redundant
    import eplb
    eplb.bounds.min_redundant_replicas = naive_min_redundant

    try:
        out["catches_naive_bound"] = 0.0 if _survives(path) else 1.0
    finally:
        b.min_redundant_replicas = good_min_redundant
        eplb.bounds.min_redundant_replicas = good_min_redundant

    return out
