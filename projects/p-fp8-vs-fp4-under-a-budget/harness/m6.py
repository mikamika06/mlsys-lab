import importlib.util
import os

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
    path = os.path.join(workdir, "tests", "test_regression.py")
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_invalid_budget": 0.0, "catches_missing_overhead": 0.0, "faults_caught": 0.0}
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    import quant.budget as budget_mod
    import quant.selector as selector_mod

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"tests failed on correct implementation: {type(e).__name__}: {e}"
        return out
    if first is None:
        out["_note"] = "no test_* functions found"
        return out
    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    good_bpw = budget_mod.compute_effective_bpw
    budget_mod.compute_effective_bpw = lambda fn, bs, sb: float(fn == "fp4" and 4.0 or 8.0)
    try:
        out["catches_missing_overhead"] = 0.0 if _survives(path) else 1.0
    finally:
        budget_mod.compute_effective_bpw = good_bpw

    good_rec = selector_mod.recommend_format
    selector_mod.recommend_format = lambda c, b, a: "fp8"
    try:
        out["catches_invalid_budget"] = 0.0 if _survives(path) else 1.0
    finally:
        selector_mod.recommend_format = good_rec

    out["faults_caught"] = out["catches_invalid_budget"] + out["catches_missing_overhead"]
    return out
