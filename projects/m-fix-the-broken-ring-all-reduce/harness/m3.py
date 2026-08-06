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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_forward_order": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"tests fail on correct code: {type(e).__name__}: {str(e)[:120]}"
        return out

    if first is None:
        out["_note"] = "no test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import ring.bucket as b
    good_assign = b.assign_buckets

    def broken_forward_assign(parameters, bucket_cap_mb):
        buckets = []
        current_bucket = []
        current_size = 0
        cap_bytes = bucket_cap_mb * 1024 * 1024
        for p in parameters:
            p_size = p.get("size_bytes", 4)
            if current_size + p_size > cap_bytes and current_bucket:
                buckets.append(current_bucket)
                current_bucket = [p["name"]]
                current_size = p_size
            else:
                current_bucket.append(p["name"])
                current_size += p_size
        if current_bucket:
            buckets.append(current_bucket)
        return buckets

    b.assign_buckets = broken_forward_assign
    import ring
    ring.bucket.assign_buckets = broken_forward_assign

    try:
        out["catches_forward_order"] = 0.0 if _survives(path) else 1.0
    finally:
        b.assign_buckets = good_assign
        ring.bucket.assign_buckets = good_assign

    return out
