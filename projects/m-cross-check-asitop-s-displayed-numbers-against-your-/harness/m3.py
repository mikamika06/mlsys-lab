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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_fault": 0.0}
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

    import edgemetrics.logger as logger_mod
    original_detect = logger_mod.detect_drop

    def broken_detect(samples, threshold_ratio=0.5):
        return -1

    logger_mod.detect_drop = broken_detect
    import edgemetrics
    if hasattr(edgemetrics, "detect_drop"):
        edgemetrics.detect_drop = broken_detect

    try:
        survives = _survives(path)
        out["catches_fault"] = 0.0 if survives else 1.0
        if survives:
            out["_note"] = "test suite survived broken detect_drop implementation"
    finally:
        logger_mod.detect_drop = original_detect
        if hasattr(edgemetrics, "detect_drop"):
            edgemetrics.detect_drop = original_detect
    return out
