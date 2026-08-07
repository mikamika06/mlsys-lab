import importlib.util
import os
import numpy as np


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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_biased_cascade": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"the tests fail on correct implementation: {type(e).__name__}: {str(e)[:120]}"
        return out

    if first is None:
        out["_note"] = "no test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import cascade.sampling as samp
    import cascade.latency as lat

    orig_stage1 = samp.cascade_stage1_accept
    orig_win = lat.is_2stage_net_win

    def broken_stage1(q1, q2, x1, rng):
        prob = float(q2[x1] / q1[x1])
        if rng.uniform() < prob:
            return True, int(x1)
        x2 = int(rng.choice(len(q2)))
        return False, x2

    def broken_win(c1, gamma1, c2, gamma2, cT, alpha2, alpha_direct):
        return True

    samp.cascade_stage1_accept = broken_stage1
    lat.is_2stage_net_win = broken_win

    try:
        out["catches_biased_cascade"] = 0.0 if _survives(path) else 1.0
    finally:
        samp.cascade_stage1_accept = orig_stage1
        lat.is_2stage_net_win = orig_win

    return out
