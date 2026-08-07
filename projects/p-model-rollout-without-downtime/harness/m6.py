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
    import sys
    if workdir not in sys.path:
        sys.path.insert(0, workdir)
    path = os.path.join(workdir, "tests", "test_regression.py")
    out = {"has_tests": 0.0, "passes_on_good": 0.0,
           "catches_broken_policy": 0.0, "catches_broken_rollback": 0.0,
           "faults_caught": 0.0}
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    import rollout.policy as pol

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"tests fail on correct implementation: {type(e).__name__}: {e}"
        return out
    if first is None:
        out["_note"] = "no test_* functions found"
        return out
    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    good_weight = pol.RolloutPolicy.get_weight

    def bad_weight(self, step):
        return 1.0

    pol.RolloutPolicy.get_weight = bad_weight
    try:
        out["catches_broken_policy"] = 0.0 if _survives(path) else 1.0
    finally:
        pol.RolloutPolicy.get_weight = good_weight

    good_rollback = pol.RolloutPolicy.should_rollback

    def bad_rollback(self, rate):
        return False

    pol.RolloutPolicy.should_rollback = bad_rollback
    try:
        out["catches_broken_rollback"] = 0.0 if _survives(path) else 1.0
    finally:
        pol.RolloutPolicy.should_rollback = good_rollback

    out["faults_caught"] = out["catches_broken_policy"] + out["catches_broken_rollback"]
    return out
