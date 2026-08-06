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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_swapped_modes": 0.0}
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

    import quanteval.modes as m
    import quanteval.table as t

    good_eval = m.evaluate_mode_output

    def broken_eval(weights, bias, x, mode, calibration_range=None):
        if mode == "full_int8":
            return good_eval(weights, bias, x, "dynamic_int8", calibration_range)
        elif mode == "dynamic_int8":
            return good_eval(weights, bias, x, "full_int8", calibration_range)
        return good_eval(weights, bias, x, mode, calibration_range)

    m.evaluate_mode_output = broken_eval
    t.evaluate_mode_output = broken_eval
    try:
        out["catches_swapped_modes"] = 0.0 if _survives(path) else 1.0
    finally:
        m.evaluate_mode_output = good_eval
        t.evaluate_mode_output = good_eval

    return out
