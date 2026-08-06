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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_merged_blobs": 0.0}
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

    import partitioner.predict as p
    good = p.partition_ops

    def bad_partition(ops, allowlist):
        res = []
        for op in ops:
            if op['type'] in allowlist:
                res.append(0)
            else:
                res.append(-1)
        return res

    p.partition_ops = bad_partition
    
    import sys
    if 'partitioner.tune' in sys.modules:
        import partitioner.tune as t
        if hasattr(t, 'partition_ops'):
            t.partition_ops = bad_partition

    try:
        out["catches_merged_blobs"] = 0.0 if _survives(path) else 1.0
    finally:
        p.partition_ops = good
        if 'partitioner.tune' in sys.modules:
            t.partition_ops = good

    return out
