import importlib.util
import os
import numpy as np


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
    out = {
        "has_tests": 0.0,
        "passes_on_good": 0.0,
        "catches_exploding_scales": 0.0,
    }
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out
    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = (
            f"tests fail on correct implementation: {type(e).__name__}: {str(e)[:120]}"
        )
        return out
    if first is None:
        out["_note"] = "no test_* functions found"
        return out
    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import awqsim.scale as sc

    good_scales = sc.compute_awq_scales

    def broken_scales(W_list, X, alpha=0.5, max_scale_ratio=5.0):
        Sx = np.mean(np.abs(X), axis=0)
        Sw_list = [np.mean(np.abs(W), axis=0) for W in W_list]
        Sw = np.mean(Sw_list, axis=0)
        s_raw = (Sx**alpha) / (Sw ** (1.0 - alpha) + 1e-8)
        return s_raw / np.mean(s_raw)

    sc.compute_awq_scales = broken_scales
    import awqsim

    awqsim.scale.compute_awq_scales = broken_scales
    try:
        out["catches_exploding_scales"] = 0.0 if _survives(path) else 1.0
    finally:
        sc.compute_awq_scales = good_scales
        awqsim.scale.compute_awq_scales = good_scales

    return out
