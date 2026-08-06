import importlib.util
import os
import sys

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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_missing_batch_div": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out
    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"the tests fail on a correct plan: {type(e).__name__}: {str(e)[:120]}"
        return out
    if first is None:
        out["_note"] = "no test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    sys.path.insert(0, workdir)
    import specbatch.measure as measure
    good_measure = measure.measure_tokens_per_sec

    def broken_measure(trace, batch_size):
        total_accepted = sum(a for _, a, _ in trace)
        total_time = sum(l for _, _, l in trace)
        if total_time == 0:
            return 0.0
        return total_accepted / total_time

    measure.measure_tokens_per_sec = broken_measure
    try:
        survived = _survives(path)
        out["catches_missing_batch_div"] = 0.0 if survived else 1.0
    finally:
        measure.measure_tokens_per_sec = good_measure

    return out
