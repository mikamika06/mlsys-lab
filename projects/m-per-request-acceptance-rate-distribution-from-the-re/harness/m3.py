import importlib.util
import os


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
        "catches_invalid_acceptance": 0.0,
    }
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"Tests fail on correct reference implementation: {type(e).__name__}: {str(e)[:120]}"
        return out

    if first is None:
        out["_note"] = "No test_* functions found in tests/test_regression.py"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import eagle_diag.metrics as m

    good_compute = m.compute_request_acceptance

    def buggy_compute_request_acceptance(log_records):
        res = good_compute(log_records)
        for r in res:
            r["mean_acceptance_rate"] = r["mean_acceptance_rate"] * 2.5
        return res

    m.compute_request_acceptance = buggy_compute_request_acceptance
    try:
        out["catches_invalid_acceptance"] = 0.0 if _survives(path) else 1.0
    finally:
        m.compute_request_acceptance = good_compute

    return out
