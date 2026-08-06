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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_broken_repair": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out
    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"the tests fail on a correct plan: {type(e).__name__}: {str(e)[:120]}"
        return out
    if first is None:
        out["_note"] = "no test_* functions found"
        return out
    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import feasibility.repair as r
    good_repair = r.repair_launch

    def broken_repair(max_model_len, max_num_seqs, gpu_memory_bytes, num_layers, kv_heads, head_dim, block_size, dtype_bytes, overhead_bytes=0):
        return max_model_len * 2, max_num_seqs * 2

    r.repair_launch = broken_repair
    import feasibility
    feasibility.repair_launch = broken_repair
    try:
        out["catches_broken_repair"] = 0.0 if _survives(path) else 1.0
    finally:
        r.repair_launch = good_repair
        feasibility.repair_launch = good_repair
    return out
