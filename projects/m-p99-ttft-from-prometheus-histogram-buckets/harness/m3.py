import importlib.util
import os


def _run(path):
    spec = importlib.util.spec_from_file_location("learner_test", path)
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
    out = {
        "has_tests": 0.0,
        "passes_on_good": 0.0,
        "catches_no_interpolation": 0.0,
        "catches_flapping_alert": 0.0,
    }
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"tests fail on reference implementation: {type(e).__name__}: {str(e)[:120]}"
        return out

    if first is None:
        out["_note"] = "no test_* functions found in tests/test_regression.py"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import kvobs.alerting as alt_mod
    import kvobs.histogram as hist_mod

    orig_hist = hist_mod.calculate_histogram_quantile
    orig_alert = alt_mod.HysteresisAlert

    def bad_quantile(q, buckets):
        if not buckets:
            return 0.0
        total = buckets[-1][1]
        rank = q * total
        for b in buckets:
            if b[1] >= rank:
                return float(b[0])
        return float(buckets[-1][0])

    class FlappingAlert:
        def __init__(self, high_threshold, low_threshold, hold_periods=3):
            self.high_threshold = high_threshold

        def process(self, value):
            return value >= self.high_threshold

    try:
        hist_mod.calculate_histogram_quantile = bad_quantile
        import kvobs
        kvobs.histogram.calculate_histogram_quantile = bad_quantile
        out["catches_no_interpolation"] = 0.0 if _survives(path) else 1.0
    finally:
        hist_mod.calculate_histogram_quantile = orig_hist
        kvobs.histogram.calculate_histogram_quantile = orig_hist

    try:
        alt_mod.HysteresisAlert = FlappingAlert
        import kvobs
        kvobs.alerting.HysteresisAlert = FlappingAlert
        out["catches_flapping_alert"] = 0.0 if _survives(path) else 1.0
    finally:
        alt_mod.HysteresisAlert = orig_alert
        kvobs.alerting.HysteresisAlert = orig_alert

    return out
