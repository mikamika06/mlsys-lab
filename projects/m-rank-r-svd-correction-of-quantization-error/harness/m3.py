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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_invalid_budget": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out
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

    import quantcorr.budget as b_mod
    good_alloc = b_mod.allocate_eora_ranks

    def broken_alloc(weights, total_budget_params, base_bits):
        return [1000 for _ in weights]

    b_mod.allocate_eora_ranks = broken_alloc
    import quantcorr
    if hasattr(quantcorr, "budget"):
        quantcorr.budget.allocate_eora_ranks = broken_alloc

    try:
        out["catches_invalid_budget"] = 0.0 if _survives(path) else 1.0
    finally:
        b_mod.allocate_eora_ranks = good_alloc
        if hasattr(quantcorr, "budget"):
            quantcorr.budget.allocate_eora_ranks = good_alloc
    return out
