import importlib.util
import os
import sys


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
    sys.path.insert(0, workdir)
    out = {
        "has_tests": 0.0,
        "passes_on_good": 0.0,
        "catches_unequal_remainder_distribution": 0.0,
    }

    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py missing"
        return out

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"tests fail on valid code: {type(e).__name__}: {str(e)[:120]}"
        return out

    if first is None:
        out["_note"] = "no test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import fsdp_analyzer.sharding as sharding_mod

    good_func = sharding_mod.compute_world_shard_distribution

    def faulty_world_shard_distribution(num_params: int, world_size: int) -> list[int]:
        base = num_params // world_size
        remainder = num_params % world_size
        res = [base] * world_size
        if remainder > 0:
            res[-1] += remainder
        return res

    sharding_mod.compute_world_shard_distribution = faulty_world_shard_distribution
    try:
        out["catches_unequal_remainder_distribution"] = 0.0 if _survives(path) else 1.0
    finally:
        sharding_mod.compute_world_shard_distribution = good_func

    return out
