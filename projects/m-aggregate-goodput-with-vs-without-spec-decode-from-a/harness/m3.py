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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_zero_penalty_bug": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    try:
        first = _run(path)
    except Exception as e:  # noqa: BLE001
        out["has_tests"] = 1.0
        out["_note"] = f"The tests fail on correct code: {type(e).__name__}: {str(e)[:120]}"
        return out

    if first is None:
        out["_note"] = "No test_* functions found in tests/test_regression.py"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import specdiag.goodput as gp
    orig_compute = gp.compute_request_goodput

    def zero_penalty_bug(req, penalty_factor=0.5):
        return orig_compute(req, penalty_factor=0.0)

    gp.compute_request_goodput = zero_penalty_bug
    import specdiag
    specdiag.goodput.compute_request_goodput = zero_penalty_bug

    try:
        out["catches_zero_penalty_bug"] = 0.0 if _survives(path) else 1.0
    finally:
        gp.compute_request_goodput = orig_compute
        specdiag.goodput.compute_request_goodput = orig_compute

    return out
