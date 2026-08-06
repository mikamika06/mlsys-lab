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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_fault": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"Learner's test failed on correct code: {type(e).__name__}: {str(e)[:120]}"
        return out

    if first is None:
        out["_note"] = "No test_* functions found in test_regression.py"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import moe_analyzer.imbalance as imb
    good_compute = imb.compute_imbalance_over_time

    def broken_compute_imbalance(log_entries):
        res = good_compute(log_entries)
        res["imbalance_ratios"] = [1.0 for _ in res["imbalance_ratios"]]
        return res

    imb.compute_imbalance_over_time = broken_compute_imbalance
    import moe_analyzer
    moe_analyzer.compute_imbalance_over_time = broken_compute_imbalance

    try:
        out["catches_fault"] = 0.0 if _survives(path) else 1.0
    finally:
        imb.compute_imbalance_over_time = good_compute
        moe_analyzer.compute_imbalance_over_time = good_compute

    return out
