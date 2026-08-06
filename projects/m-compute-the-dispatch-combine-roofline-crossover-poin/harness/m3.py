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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_naive_placement": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    try:
        first = _run(path)
    except Exception as e:  # noqa: BLE001
        out["has_tests"] = 1.0
        out["_note"] = f"tests fail on correct baseline: {type(e).__name__}: {str(e)[:120]}"
        return out

    if first is None:
        out["_note"] = "no test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import moeplan.placement as p
    good_pack = p.pack_experts

    def naive_round_robin(expert_loads, num_ranks, expert_memory_mb, rank_memory_budget_mb):
        return [i % num_ranks for i in range(len(expert_loads))]

    p.pack_experts = naive_round_robin
    import moeplan
    moeplan.placement.pack_experts = naive_round_robin

    try:
        out["catches_naive_placement"] = 0.0 if _survives(path) else 1.0
    finally:
        p.pack_experts = good_pack
        moeplan.placement.pack_experts = good_pack

    return out
