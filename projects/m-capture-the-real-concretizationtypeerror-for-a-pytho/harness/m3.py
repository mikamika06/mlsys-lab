import importlib.util
import os

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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_ignored_errors": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"Tests failed on reference solution: {type(e).__name__}: {str(e)[:120]}"
        return out

    if first is None:
        out["_note"] = "no test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import jaxcomp.tracer as tr
    orig_capture = tr.capture_concretization_error

    def broken_capture(fn, *args, **kwargs):
        try:
            fn(*args, **kwargs)
        except Exception as e:
            return True, e
        return False, None

    tr.capture_concretization_error = broken_capture
    import jaxcomp
    jaxcomp.tracer.capture_concretization_error = broken_capture

    try:
        out["catches_ignored_errors"] = 0.0 if _survives(path) else 1.0
    finally:
        tr.capture_concretization_error = orig_capture
        jaxcomp.tracer.capture_concretization_error = orig_capture

    return out
