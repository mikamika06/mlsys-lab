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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_masked_residual": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out
    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"tests fail on correct reference: {type(e).__name__}: {str(e)[:120]}"
        return out
    if first is None:
        out["_note"] = "no test_* functions found in tests/test_regression.py"
        return out
    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import spec.sampling as samp
    good_sample = samp.sample_residual

    def flawed_sample(p_logits, q_logits, temperature, top_p_target, top_p_draft):
        p_dist = samp.apply_temperature_and_topp(p_logits, temperature, top_p_target)
        q_dist = samp.apply_temperature_and_topp(q_logits, temperature, top_p_draft)
        mask = q_dist > 0.0
        p_dist_masked = np.where(mask, p_dist, 0.0)
        diff = np.maximum(0.0, p_dist_masked - q_dist)
        s = np.sum(diff)
        if s > 0:
            res_dist = diff / s
        else:
            res_dist = p_dist
        return int(np.random.choice(len(p_dist), p=res_dist))

    samp.sample_residual = flawed_sample
    try:
        out["catches_masked_residual"] = 0.0 if _survives(path) else 1.0
    finally:
        samp.sample_residual = good_sample

    return out
