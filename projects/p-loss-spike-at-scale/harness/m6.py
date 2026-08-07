import importlib.util
import os
import numpy as np

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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_bug": 0.0}

    if not os.path.isfile(path):
        return out

    try:
        import system.distributed as dist
    except Exception:
        return out

    try:
        first = _run(path)
    except Exception:
        out["has_tests"] = 1.0
        return out

    if first is None:
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    good_reduce = dist.safe_all_reduce_sum

    def leaky_reduce(tensors):
        res = np.zeros_like(tensors[0], dtype=np.float16)
        for t in tensors:
            res += t.astype(np.float16)
        return res.astype(np.float32)

    dist.safe_all_reduce_sum = leaky_reduce
    try:
        out["catches_bug"] = 0.0 if _survives(path) else 1.0
    finally:
        dist.safe_all_reduce_sum = good_reduce

    return out
