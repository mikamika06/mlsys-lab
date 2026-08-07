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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_budget_violation": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out
    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"tests fail on good code: {type(e).__name__}: {str(e)[:120]}"
        return out
    if first is None:
        out["_note"] = "no test_* functions found"
        return out
    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import proftune.budget as b
    good_fn = b.select_config_for_budget

    def broken_fn(step_meta, byte_budget):
        return {"with_stack": True, "record_shapes": True, "profile_memory": True}

    b.select_config_for_budget = broken_fn
    import proftune
    proftune.budget.select_config_for_budget = broken_fn
    try:
        out["catches_budget_violation"] = 0.0 if _survives(path) else 1.0
    finally:
        b.select_config_for_budget = good_fn
        proftune.budget.select_config_for_budget = good_fn
    return out
