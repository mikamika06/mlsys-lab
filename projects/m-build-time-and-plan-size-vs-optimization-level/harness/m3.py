import importlib.util
import os
import sys


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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_nondeterministic_roundtrip": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    sys.path.insert(0, workdir)
    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"The tests fail on correct code: {type(e).__name__}: {str(e)[:120]}"
        return out
    finally:
        if sys.path and sys.path[0] == workdir:
            sys.path.pop(0)

    if first is None:
        out["_note"] = "No test_* functions found in tests/test_regression.py"
        return out
    
    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import trtpipe.determinism as det
    good_verify = det.verify_roundtrip

    def broken_verify(engine_plan):
        return True, "fake_checksum_always_passes"

    det.verify_roundtrip = broken_verify
    import trtpipe
    trtpipe.verify_roundtrip = broken_verify

    sys.path.insert(0, workdir)
    try:
        out["catches_nondeterministic_roundtrip"] = 0.0 if _survives(path) else 1.0
    finally:
        det.verify_roundtrip = good_verify
        trtpipe.verify_roundtrip = good_verify
        if sys.path and sys.path[0] == workdir:
            sys.path.pop(0)

    return out
