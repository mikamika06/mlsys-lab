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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_collisions": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"the tests fail on a correct map_tensors: {type(e).__name__}: {str(e)[:120]}"
        return out

    if first is None:
        out["_note"] = "no test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import mmproj.mapping as mapping
    good = mapping.map_tensors

    def broken_map_tensors(raw_names):
        res = good(raw_names).copy()
        for k, v in res.items():
            if "attn_q" in v:
                res[k] = v.replace("attn_q", "attn_k")
        return res

    mapping.map_tensors = broken_map_tensors
    try:
        if not _survives(path):
            out["catches_collisions"] = 1.0
        else:
            out["_note"] = "test did not catch that attn_q was incorrectly mapped to attn_k (creating a collision)"
    finally:
        mapping.map_tensors = good

    return out
