import importlib.util
import os

def _run(path):
    spec = importlib.util.spec_from_file_location("learner_regression", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fns = [getattr(mod, n) for n in dir(mod)
           if n.startswith("test_") and callable(getattr(mod, n))]
    if not fns:
        test_cases = [getattr(mod, n) for n in dir(mod) if isinstance(getattr(mod, n), type) and any(m.startswith("test_") for m in dir(getattr(mod, n)))]
        if test_cases:
            import unittest
            suite = unittest.defaultTestLoader.loadTestsFromModule(mod)
            result = unittest.TextTestRunner().run(suite)
            return result.wasSuccessful()
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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_broken_opset": 0.0}
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
    if first is None or first is False:
        out["_note"] = "no valid test functions or test execution failed"
        out["has_tests"] = 1.0 if os.path.isfile(path) else 0.0
        return out
    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import exporttools.opset as o_mod
    good_func = o_mod.enumerate_opset

    def broken_enumerate(spec):
        res = good_func(spec)
        res["opset_version"] = 999
        return res

    o_mod.enumerate_opset = broken_enumerate
    import exporttools
    exporttools.opset.enumerate_opset = broken_enumerate

    try:
        out["catches_broken_opset"] = 0.0 if _survives(path) else 1.0
    finally:
        o_mod.enumerate_opset = good_func
        exporttools.opset.enumerate_opset = good_func
    return out
