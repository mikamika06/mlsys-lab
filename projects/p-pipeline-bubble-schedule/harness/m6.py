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
    path = os.path.join(workdir, "tests", "test_regression.py")
    out = {
        "has_tests": 0.0,
        "passes_on_good": 0.0,
        "catches_broken_activation": 0.0,
        "catches_broken_schedule": 0.0,
        "faults_caught": 0.0
    }
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    import sched.pipeline as pipe

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"tests failed: {type(e).__name__}: {e}"
        return out

    if first is None:
        out["_note"] = "no test functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    good_budget = pipe.PipelineScheduler.check_activation_budget
    pipe.PipelineScheduler.check_activation_budget = lambda self, b: False
    try:
        out["catches_broken_activation"] = 0.0 if _survives(path) else 1.0
    finally:
        pipe.PipelineScheduler.check_activation_budget = good_budget

    good_util = pipe.PipelineScheduler.gpipe_utilization
    pipe.PipelineScheduler.gpipe_utilization = lambda self: 0.0
    try:
        out["catches_broken_schedule"] = 0.0 if _survives(path) else 1.0
    finally:
        pipe.PipelineScheduler.gpipe_utilization = good_util

    out["faults_caught"] = out["catches_broken_activation"] + out["catches_broken_schedule"]
    return out
