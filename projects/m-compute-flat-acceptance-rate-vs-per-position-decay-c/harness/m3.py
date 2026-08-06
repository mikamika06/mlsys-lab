import importlib.util
import os
import numpy as np


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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_flat_decay_bug": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"Tests fail on correct implementation: {type(e).__name__}: {str(e)[:120]}"
        return out

    if first is None:
        out["_note"] = "No test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import spec.decay as d
    good_decay = d.compute_acceptance_metrics

    def buggy_decay(traces):
        flat_rate, decay_curve = good_decay(traces)
        buggy_curve = np.full_like(decay_curve, flat_rate)
        return flat_rate, buggy_curve

    d.compute_acceptance_metrics = buggy_decay
    import spec
    spec.decay.compute_acceptance_metrics = buggy_decay

    try:
        out["catches_flat_decay_bug"] = 0.0 if _survives(path) else 1.0
    finally:
        d.compute_acceptance_metrics = good_decay
        spec.decay.compute_acceptance_metrics = good_decay

    return out
