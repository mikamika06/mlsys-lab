import importlib.util
import os
import sys

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
    if workdir not in sys.path:
        sys.path.insert(0, workdir)

    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_nosync_bug": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")

    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py missing"
        return out

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"tests fail on correct analyzer: {type(e).__name__}: {str(e)[:120]}"
        return out

    if first is None:
        out["_note"] = "no test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import perf.analyzer as a
    good_compute = a.compute_metrics

    def nosync_compute(events):
        t, h, d, e2e = good_compute(events)
        return t, h, d, h

    a.compute_metrics = nosync_compute

    try:
        survives = _survives(path)
        out["catches_nosync_bug"] = 0.0 if survives else 1.0
        if survives:
            out["_note"] = "tests passed even when compute_metrics returned host latency instead of end-to-end"
    finally:
        a.compute_metrics = good_compute

    return out
