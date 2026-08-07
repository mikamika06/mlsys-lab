import importlib.util
import os
import sys


def _run(path):
    spec = importlib.util.spec_from_file_location("learner_regression", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fns = [
        getattr(mod, n)
        for n in dir(mod)
        if n.startswith("test_") and callable(getattr(mod, n))
    ]
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
        "catches_missing_state_tracking": 0.0,
    }
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"The tests fail on a correct plan: {type(e).__name__}: {str(e)[:120]}"
        return out

    if first is None:
        out["_note"] = "No test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import optmem.state as state_mod

    good_est = state_mod.estimate_optimizer_state_bytes

    def broken_est(params, optimizer_type, initialized=True):
        return 0

    state_mod.estimate_optimizer_state_bytes = broken_est
    import optmem
    optmem.state.estimate_optimizer_state_bytes = broken_est

    try:
        out["catches_missing_state_tracking"] = 0.0 if _survives(path) else 1.0
    finally:
        state_mod.estimate_optimizer_state_bytes = good_est
        optmem.state.estimate_optimizer_state_bytes = good_est

    return out
