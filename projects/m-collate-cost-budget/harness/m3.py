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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_budget_leaks": 0.0}
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

    import collate.budget as b
    good_eval = b.evaluate_budget

    def broken_eval(profile_results, max_budget_ms_per_batch, target_throughput_samples_sec):
        res = good_eval(profile_results, max_budget_ms_per_batch, target_throughput_samples_sec)
        for k in res:
            res[k]["within_budget"] = True
            res[k]["compliant"] = res[k]["meets_throughput"]
        return res

    b.evaluate_budget = broken_eval
    import collate
    collate.budget.evaluate_budget = broken_eval

    try:
        out["catches_budget_leaks"] = 0.0 if _survives(path) else 1.0
    finally:
        b.evaluate_budget = good_eval
        collate.budget.evaluate_budget = good_eval

    return out
