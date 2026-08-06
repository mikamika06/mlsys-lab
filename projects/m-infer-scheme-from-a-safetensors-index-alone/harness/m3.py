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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_invalid_repair": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out
    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"tests fail on valid reference: {type(e).__name__}: {str(e)[:120]}"
        return out

    if first is None:
        out["_note"] = "no test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import ctinspect.repair as r
    good_repair = r.repair_missing_weight_shape

    def broken_repair(index_data):
        res = good_repair(index_data)
        for meta in res.get("tensor_metadata", {}).values():
            if "weight_shape" in meta and len(meta["weight_shape"]) >= 2:
                meta["weight_shape"][0] = meta["weight_shape"][0] + 999
        return res

    r.repair_missing_weight_shape = broken_repair
    import ctinspect
    ctinspect.repair.repair_missing_weight_shape = broken_repair

    try:
        out["catches_invalid_repair"] = 0.0 if _survives(path) else 1.0
    finally:
        r.repair_missing_weight_shape = good_repair
        ctinspect.repair.repair_missing_weight_shape = good_repair

    return out
