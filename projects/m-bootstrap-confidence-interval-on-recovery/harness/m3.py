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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_bad_ci": 0.0}
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

    import evalrep.bootstrap as b
    good_ci = b.bootstrap_recovery_ci

    def bad_ci(base_scores, quant_scores, num_samples=1000, alpha=0.05, seed=42):
        res = good_ci(base_scores, quant_scores, num_samples, alpha, seed)
        res["lower"], res["upper"] = res["upper"], res["lower"]
        return res

    b.bootstrap_recovery_ci = bad_ci
    import evalrep
    if hasattr(evalrep, "bootstrap") and hasattr(evalrep.bootstrap, "bootstrap_recovery_ci"):
        evalrep.bootstrap.bootstrap_recovery_ci = bad_ci

    try:
        out["catches_bad_ci"] = 0.0 if _survives(path) else 1.0
    finally:
        b.bootstrap_recovery_ci = good_ci
        if hasattr(evalrep, "bootstrap") and hasattr(evalrep.bootstrap, "bootstrap_recovery_ci"):
            evalrep.bootstrap.bootstrap_recovery_ci = good_ci
    return out
