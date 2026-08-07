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
    sys.path.insert(0, workdir)
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_invalid_overrides": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")

    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"tests fail on correct reference: {type(e).__name__}: {str(e)[:120]}"
        return out

    if first is None:
        out["_note"] = "no test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import fa_classifier.gating as gating

    orig_check = gating.check_hardware_gating

    def broken_check_hardware_gating(pkg_identity, hw_info):
        res = orig_check(pkg_identity, hw_info)
        res["compatible"] = True
        return res

    gating.check_hardware_gating = broken_check_hardware_gating
    import fa_classifier.dispatch as dispatch

    dispatch.check_hardware_gating = broken_check_hardware_gating

    try:
        survived = _survives(path)
        out["catches_invalid_overrides"] = 0.0 if survived else 1.0
    finally:
        gating.check_hardware_gating = orig_check
        dispatch.check_hardware_gating = orig_check

    return out
