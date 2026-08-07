import importlib.util
import os
import ref

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
    out = {
        "has_tests": 0.0,
        "passes_on_good": 0.0,
        "catches_broken_promotion": 0.0,
    }
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"tests fail on correct implementation: {type(e).__name__}: {str(e)[:120]}"
        return out

    if first is None:
        out["_note"] = "no test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import mlxops.promotion as p
    good_promote = p.promote_dtypes

    def broken_promote(dt1, dt2):
        s1 = str(dt1).lower()
        s2 = str(dt2).lower()
        if "int" in s1 or "int" in s2:
            return "int32"
        return "float32"

    p.promote_dtypes = broken_promote
    import mlxops
    if hasattr(mlxops, "promote_dtypes"):
        mlxops.promote_dtypes = broken_promote

    try:
        out["catches_broken_promotion"] = 0.0 if _survives(path) else 1.0
    finally:
        p.promote_dtypes = good_promote
        if hasattr(mlxops, "promote_dtypes"):
            mlxops.promote_dtypes = good_promote

    return out
