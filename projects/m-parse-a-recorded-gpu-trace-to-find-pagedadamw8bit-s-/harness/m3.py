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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_ignored_spikes": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"learner tests failed on reference implementation: {type(e).__name__}: {str(e)[:120]}"
        return out

    if first is None:
        out["_note"] = "no test_* functions found in tests/test_regression.py"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import gputrace.metrics as m
    orig_fn = m.find_spillover_spike

    def broken_spike_finder(events):
        if not events:
            return {"argmin_index": -1, "max_ratio": 0.0}
        return {"argmin_index": 0, "max_ratio": 1.0}

    m.find_spillover_spike = broken_spike_finder
    import gputrace
    gputrace.find_spillover_spike = broken_spike_finder

    try:
        out["catches_ignored_spikes"] = 0.0 if _survives(path) else 1.0
    finally:
        m.find_spillover_spike = orig_fn
        gputrace.find_spillover_spike = orig_fn

    return out
