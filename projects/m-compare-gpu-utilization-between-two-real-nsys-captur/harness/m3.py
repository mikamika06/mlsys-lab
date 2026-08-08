import importlib.util
import os
import sys


def _run(path):
    spec = importlib.util.spec_from_file_location("learner_regression", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fns = [
        getattr(mod, n)
        for n in dir(mod)
        if n.startswith("test_") and callable(getattr(mod, n))
    ]
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
    sys.path.insert(0, workdir)
    out = {
        "has_tests": 0.0,
        "passes_on_good": 0.0,
        "catches_unmerged_intervals": 0.0,
    }
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"tests fail on reference solution: {type(e).__name__}: {str(e)[:120]}"
        return out

    if first is None:
        out["_note"] = "no test_* functions found in test_regression.py"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import nsys_analyzer.utilization as util_mod

    good_fn = util_mod.compute_gpu_utilization

    def broken_unmerged_utilization(kernel_events, capture_window):
        start_win, end_win = capture_window
        if end_win <= start_win:
            return 0.0
        active_ns = sum(
            min(k["end_ns"], end_win) - max(k["start_ns"], start_win)
            for k in kernel_events
            if min(k["end_ns"], end_win) > max(k["start_ns"], start_win)
        )
        return (active_ns / (end_win - start_win)) * 100.0

    util_mod.compute_gpu_utilization = broken_unmerged_utilization
    try:
        out["catches_unmerged_intervals"] = 0.0 if _survives(path) else 1.0
    finally:
        util_mod.compute_gpu_utilization = good_fn

    return out
