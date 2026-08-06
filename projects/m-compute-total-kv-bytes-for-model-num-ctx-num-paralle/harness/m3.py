import importlib.util
import os
import sys

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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_missing_kv_factor": 0.0}
    sys.path.insert(0, workdir)
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
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

    import slots.memory as m
    good = m.compute_kv_bytes

    def bad_compute(num_layers, num_kv_heads, head_dim, num_ctx, num_parallel, dtype_bytes=2):
        return num_layers * num_kv_heads * head_dim * num_ctx * num_parallel * dtype_bytes

    m.compute_kv_bytes = bad_compute
    try:
        out["catches_missing_kv_factor"] = 0.0 if _survives(path) else 1.0
    finally:
        m.compute_kv_bytes = good

    return out
