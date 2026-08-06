import importlib.util
import os
import sys


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
    sys.path.insert(0, os.path.join(workdir))
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_missing_transform": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"learner tests fail on reference code: {type(e).__name__}: {str(e)[:120]}"
        return out

    if first is None:
        out["_note"] = "no test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import compat.adapter as adapter
    good_transform = adapter.transform_request

    def broken_transform(shape_name, payload):
        res = dict(payload)
        return res

    adapter.transform_request = broken_transform
    if "compat.suite" in sys.modules:
        import compat.suite
        compat.suite.transform_request = broken_transform

    try:
        out["catches_missing_transform"] = 0.0 if _survives(path) else 1.0
    finally:
        adapter.transform_request = good_transform
        if "compat.suite" in sys.modules:
            import compat.suite
            compat.suite.transform_request = good_transform

    return out
