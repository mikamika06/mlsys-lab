import importlib.util
import os


def _run(path):
    spec = importlib.util.spec_from_file_location("learner_regression", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fns = [
        getattr(mod, n)
        for n in dir(mod)
        if n.startswith("test_") and callable(getattr(mod, n))
    ]
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
    out = {
        "has_tests": 0.0,
        "passes_on_good": 0.0,
        "catches_naive_greedy": 0.0,
    }
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out
    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = (
            f"tests fail on good code: {type(e).__name__}: {str(e)[:120]}"
        )
        return out
    if first is None:
        out["_note"] = "no test_* functions found"
        return out
    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import zerodp.partition as p

    good_fn = p.partition_bin_packing

    def broken_bin_packing(tensor_sizes, world_size):
        if world_size <= 0:
            return {
                "assignments": [],
                "loads": [],
                "max_load": 0,
                "min_load": 0,
                "imbalance": 0,
            }
        indexed = list(enumerate(tensor_sizes))
        loads = [0] * world_size
        assignments = [[] for _ in range(world_size)]
        for idx, sz in indexed:
            min_val = min(loads)
            target_rank = loads.index(min_val)
            assignments[target_rank].append(idx)
            loads[target_rank] += sz
        for r in range(world_size):
            assignments[r].sort()
        max_l = max(loads) if loads else 0
        min_l = min(loads) if loads else 0
        return {
            "assignments": assignments,
            "loads": loads,
            "max_load": max_l,
            "min_load": min_l,
            "imbalance": max_l - min_l,
        }

    p.partition_bin_packing = broken_bin_packing
    import zerodp

    zerodp.partition_bin_packing = broken_bin_packing

    try:
        out["catches_naive_greedy"] = 0.0 if _survives(path) else 1.0
    finally:
        p.partition_bin_packing = good_fn
        zerodp.partition_bin_packing = good_fn

    return out
