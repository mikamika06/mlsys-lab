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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_mixed_precision": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"Tests fail on correct implementation: {type(e).__name__}: {str(e)[:120]}"
        return out

    if first is None:
        out["_note"] = "No test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import bench.xeon_ranking as xr
    orig_fn = xr.check_precision_fairness

    def flawed_check_precision_fairness(engine_a_info, engine_b_info):
        return {
            "fair": True,
            "engine_a_prec": engine_a_info.get("precision"),
            "engine_b_prec": engine_b_info.get("precision"),
        }

    xr.check_precision_fairness = flawed_check_precision_fairness
    import bench
    bench.xeon_ranking.check_precision_fairness = flawed_check_precision_fairness

    try:
        out["catches_mixed_precision"] = 0.0 if _survives(path) else 1.0
    finally:
        xr.check_precision_fairness = orig_fn
        bench.xeon_ranking.check_precision_fairness = orig_fn

    return out
