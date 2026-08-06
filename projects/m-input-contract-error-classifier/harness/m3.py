import os
import importlib.util

def _run(path):
    spec = importlib.util.spec_from_file_location("learner_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fns = [getattr(mod, n) for n in dir(mod) if n.startswith("test_") and callable(getattr(mod, n))]
    if not fns:
        return None
    for fn in fns:
        fn()
    return True

def check(workdir):
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_broken_guard": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")

    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"fails on good implementation: {type(e).__name__}: {str(e)[:120]}"
        return out

    if first is None:
        out["_note"] = "no test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    try:
        import flash_contract.guard as g
        import flash_contract.classifier as c
    except ImportError:
        return out

    good_guard = g.check_contiguity

    def broken_guard(strides):
        return True

    g.check_contiguity = broken_guard
    c.check_contiguity = broken_guard

    try:
        passed = False
        try:
            _run(path)
            passed = True
        except Exception:
            pass

        if not passed:
            out["catches_broken_guard"] = 1.0
    finally:
        g.check_contiguity = good_guard
        c.check_contiguity = good_guard

    return out
