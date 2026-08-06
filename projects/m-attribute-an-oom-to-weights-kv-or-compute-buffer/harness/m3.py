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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_wrong_order": 0.0}
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

    import memplan.oom as o
    good = o.attribute_oom

    def broken(ram_total, model_bytes, kv_bytes, compute_bytes, use_mmap):
        avail = ram_total
        if avail < kv_bytes:
            return "kv"
        avail -= kv_bytes
        if not use_mmap:
            if avail < model_bytes:
                return "weights"
            avail -= model_bytes
        if avail < compute_bytes:
            return "compute"
        return "none"
    
    o.attribute_oom = broken
    try:
        if not _survives(path):
            out["catches_wrong_order"] = 1.0
    finally:
        o.attribute_oom = good
    
    return out
