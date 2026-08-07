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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_max_scale": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    sys.path.insert(0, workdir)
    try:
        try:
            first = _run(path)
        except Exception as e:
            out["has_tests"] = 1.0
            out["_note"] = f"the tests fail on a correct implementation: {type(e).__name__}: {str(e)[:120]}"
            return out

        if first is None:
            out["_note"] = "no test_* functions found"
            return out

        out["has_tests"] = 1.0
        out["passes_on_good"] = 1.0

        import quant.scale as qs
        good_find = qs.find_best_scale_mse

        def broken_find(w, num_candidates=100, qmin=-8, qmax=7):
            return qs.compute_max_scale(w, qmax=qmax)

        qs.find_best_scale_mse = broken_find
        try:
            out["catches_max_scale"] = 0.0 if _survives(path) else 1.0
        finally:
            qs.find_best_scale_mse = good_find
    finally:
        sys.path.pop(0)

    return out
