import importlib.util
import os


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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_broken_threshold": 0.0}
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
        out["_note"] = "No test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import imatrix_analysis.threshold as thresh
    good_fn = thresh.find_imatrix_convergence_bpw

    def broken_find_imatrix_convergence_bpw(tensors_data, tol=1e-3):
        return 0.0

    thresh.find_imatrix_convergence_bpw = broken_find_imatrix_convergence_bpw
    import imatrix_analysis
    imatrix_analysis.find_imatrix_convergence_bpw = broken_find_imatrix_convergence_bpw

    try:
        out["catches_broken_threshold"] = 0.0 if _survives(path) else 1.0
    finally:
        thresh.find_imatrix_convergence_bpw = good_fn
        imatrix_analysis.find_imatrix_convergence_bpw = good_fn

    return out
