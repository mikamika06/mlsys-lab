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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_permissive_gate": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out
    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"the tests fail on reference code: {type(e).__name__}: {str(e)[:120]}"
        return out
    if first is None:
        out["_note"] = "no test_* functions found"
        return out
    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import numval.gate as g
    good_gate = g.evaluate_gate

    def permissive_gate(y_ref, y_test, min_sqnr_db=30.0, min_cos_sim=0.99, max_rel_err=1e-2, eps=1e-12):
        res = good_gate(y_ref, y_test, min_sqnr_db, min_cos_sim, max_rel_err, eps)
        res["passed"] = True
        return res

    g.evaluate_gate = permissive_gate
    import numval
    numval.evaluate_gate = permissive_gate
    try:
        out["catches_permissive_gate"] = 0.0 if _survives(path) else 1.0
    finally:
        g.evaluate_gate = good_gate
        numval.evaluate_gate = good_gate

    return out
