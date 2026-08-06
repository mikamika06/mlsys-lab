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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_flawed_payback": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out
    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"tests fail on reference code: {type(e).__name__}: {str(e)[:120]}"
        return out
    if first is None:
        out["_note"] = "no test_* functions found in tests/test_regression.py"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import picker.payback as p
    good_payback = p.calculate_payback_volume

    def flawed_payback(build_time_sec, base_latency_ms, target_latency_ms):
        return 0

    p.calculate_payback_volume = flawed_payback
    import picker
    picker.payback.calculate_payback_volume = flawed_payback

    try:
        out["catches_flawed_payback"] = 0.0 if _survives(path) else 1.0
    finally:
        p.calculate_payback_volume = good_payback
        picker.payback.calculate_payback_volume = good_payback

    return out
