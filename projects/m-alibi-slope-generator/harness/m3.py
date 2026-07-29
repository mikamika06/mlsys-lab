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


def _broken_softcap_backward(grad_output, x, cap):
    x = np.asarray(x, dtype=np.float64)
    grad_output = np.asarray(grad_output, dtype=np.float64)
    t = np.tanh(x / cap)
    return grad_output * (1.0 - t)


def check(workdir):
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_broken_backward": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out
    try:
        first = _run(path)
    except Exception as e:  # noqa: BLE001
        out["has_tests"] = 1.0
        out["_note"] = f"the tests fail on a correct implementation: {type(e).__name__}: {str(e)[:120]}"
        return out
    if first is None:
        out["_note"] = "no test_* functions found"
        return out
    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import scoremod
    import scoremod.softcap as sc

    good = sc.softcap_backward
    sc.softcap_backward = _broken_softcap_backward
    scoremod.softcap_backward = _broken_softcap_backward
    try:
        out["catches_broken_backward"] = 0.0 if _survives(path) else 1.0
    finally:
        sc.softcap_backward = good
        scoremod.softcap_backward = good
    return out
