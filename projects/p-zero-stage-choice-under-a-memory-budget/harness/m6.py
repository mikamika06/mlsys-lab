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
    path = os.path.join(workdir, "tests", "test_regression.py")
    out = {
        "has_tests": 0.0,
        "passes_on_good": 0.0,
        "catches_broken_estimator": 0.0,
        "catches_broken_planner": 0.0,
        "faults_caught": 0.0,
    }
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    sys.path.insert(0, workdir)

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"Tests fail on correct implementation: {e}"
        return out

    if first is None:
        out["_note"] = "No test_ functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import zero_planner.estimator as est_mod

    orig_zero2 = est_mod.ZeroEstimator.memory_zero2
    est_mod.ZeroEstimator.memory_zero2 = lambda self, w, act: self.memory_zero1(w, act)
    try:
        out["catches_broken_estimator"] = 0.0 if _survives(path) else 1.0
    finally:
        est_mod.ZeroEstimator.memory_zero2 = orig_zero2

    import zero_planner.planner as plan_mod

    orig_doubled = plan_mod.ZeroPlanner.predict_doubled_gpus
    plan_mod.ZeroPlanner.predict_doubled_gpus = lambda self, c, s, a: {
        "new_world_size": c,
        "memory_bytes": 0,
        "memory_saved_bytes": 0,
        "comm_bytes": 0,
    }
    try:
        out["catches_broken_planner"] = 0.0 if _survives(path) else 1.0
    finally:
        plan_mod.ZeroPlanner.predict_doubled_gpus = orig_doubled

    out["faults_caught"] = out["catches_broken_estimator"] + out["catches_broken_planner"]
    return out
