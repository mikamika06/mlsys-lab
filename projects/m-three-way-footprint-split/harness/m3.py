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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_invalid_lifetimes": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out
    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"the tests fail on correct code: {type(e).__name__}: {str(e)[:120]}"
        return out
    if first is None:
        out["_note"] = "no test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import footprint.predictor as p
    good_predict = p.predict_peak_rss

    def broken_predict(execution_plan, alignment=64, overhead_bytes=0):
        res = good_predict(execution_plan, alignment, overhead_bytes)
        res["peak_rss_bytes"] = overhead_bytes
        return res

    p.predict_peak_rss = broken_predict
    import footprint
    footprint.predict_peak_rss = broken_predict

    try:
        out["catches_invalid_lifetimes"] = 0.0 if _survives(path) else 1.0
    finally:
        p.predict_peak_rss = good_predict
        footprint.predict_peak_rss = good_predict

    return out
