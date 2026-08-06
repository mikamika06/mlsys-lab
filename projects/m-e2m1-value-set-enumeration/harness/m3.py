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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_fault": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "missing test file"
        return out
    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"tests fail on good: {type(e).__name__}"
        return out
    if first is None:
        out["_note"] = "no test functions"
        return out
    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import e2m1.enumeration as en
    good_enum = en.enumerate_e2m1

    def broken_enum():
        res = good_enum()
        if res:
            res[0]["value"] = 999.0
        return res

    en.enumerate_e2m1 = broken_enum
    import e2m1
    if hasattr(e2m1, "enumerate_e2m1"):
        e2m1.enumerate_e2m1 = broken_enum

    try:
        out["catches_fault"] = 0.0 if _survives(path) else 1.0
    finally:
        en.enumerate_e2m1 = good_enum
        if hasattr(e2m1, "enumerate_e2m1"):
            e2m1.enumerate_e2m1 = good_enum
    return out
