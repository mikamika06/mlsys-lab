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
        "catches_broken_policy": 0.0,
        "catches_broken_tracker": 0.0,
        "faults_caught": 0.0
    }

    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    import specdec.policy as pol_mod
    import specdec.tracker as tr_mod

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"the tests fail on a correct implementation: {type(e).__name__}: {e}"
        return out

    if first is None:
        out["_note"] = "no test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    orig_decide = pol_mod.AdaptivePolicy.decide
    pol_mod.AdaptivePolicy.decide = lambda self, domain, batch_size, max_gamma=8: (max_gamma, True)
    try:
        out["catches_broken_policy"] = 0.0 if _survives(path) else 1.0
    finally:
        pol_mod.AdaptivePolicy.decide = orig_decide

    orig_rate = tr_mod.AcceptanceTracker.get_acceptance_rate
    tr_mod.AcceptanceTracker.get_acceptance_rate = lambda self, domain=None: 1.0
    try:
        out["catches_broken_tracker"] = 0.0 if _survives(path) else 1.0
    finally:
        tr_mod.AcceptanceTracker.get_acceptance_rate = orig_rate

    out["faults_caught"] = out["catches_broken_policy"] + out["catches_broken_tracker"]
    return out
