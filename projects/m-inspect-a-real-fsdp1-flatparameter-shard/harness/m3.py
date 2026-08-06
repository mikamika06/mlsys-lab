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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_unbalanced_fault": 0.0}
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

    import fsdp_shard.balance as b_mod
    good_fn = b_mod.per_rank_balance

    def broken_fn(param_sizes, world_size):
        res = good_fn(param_sizes, world_size)
        if len(res) > 0:
            res[0] += 100
        return res

    b_mod.per_rank_balance = broken_fn
    import fsdp_shard
    if hasattr(fsdp_shard, "per_rank_balance"):
        fsdp_shard.per_rank_balance = broken_fn

    try:
        out["catches_unbalanced_fault"] = 0.0 if _survives(path) else 1.0
    finally:
        b_mod.per_rank_balance = good_fn
        if hasattr(fsdp_shard, "per_rank_balance"):
            fsdp_shard.per_rank_balance = good_fn
    return out
