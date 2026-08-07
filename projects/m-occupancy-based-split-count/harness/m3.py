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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_unscaled_combine": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"tests fail on good code: {type(e).__name__}: {str(e)[:120]}"
        return out

    if first is None:
        out["_note"] = "no test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import splitkv.combine as comb
    good_combine = comb.combine_splits

    def unscaled_combine(p_max, p_lse, p_out):
        weights = p_lse
        denom = np.maximum(np.sum(weights, axis=-1, keepdims=True), 1e-20)
        norm_weights = weights / denom
        c_out = np.sum(p_out * norm_weights[..., None], axis=-2)
        c_lse = np.squeeze(np.max(p_max, axis=-1, keepdims=True), axis=-1)
        return c_out, c_lse

    comb.combine_splits = unscaled_combine
    import splitkv
    if hasattr(splitkv, "combine_splits"):
        splitkv.combine_splits = unscaled_combine

    try:
        out["catches_unscaled_combine"] = 0.0 if _survives(path) else 1.0
    finally:
        comb.combine_splits = good_combine
        if hasattr(splitkv, "combine_splits"):
            splitkv.combine_splits = good_combine

    return out
