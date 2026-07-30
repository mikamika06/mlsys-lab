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


def _broken_sdpa(query, key, value, is_causal=False, scale=None, enable_gqa=False):
    query = np.asarray(query, dtype=np.float64)
    key = np.asarray(key, dtype=np.float64)
    value = np.asarray(value, dtype=np.float64)
    head_dim = query.shape[-1]
    scale_factor = (1.0 / np.sqrt(head_dim)) if scale is None else scale
    if enable_gqa:
        n_rep = query.shape[-3] // key.shape[-3]
        if n_rep > 1:
            key = np.concatenate([key] * n_rep, axis=-3)
            value = np.concatenate([value] * n_rep, axis=-3)
    q_len = query.shape[-2]
    kv_len = key.shape[-2]
    if is_causal:
        keep = np.tril(np.ones((q_len, kv_len), dtype=bool))
        bias = np.where(keep, 0.0, -np.inf)
    else:
        bias = np.zeros((q_len, kv_len))
    scores = np.matmul(query, np.swapaxes(key, -1, -2)) * scale_factor
    scores = scores + bias
    scores = scores - np.max(scores, axis=-1, keepdims=True)
    weights = np.exp(scores)
    weights = weights / np.sum(weights, axis=-1, keepdims=True)
    return np.matmul(weights, value)


def check(workdir):
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_bad_grouping": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out
    try:
        first = _run(path)
    except Exception as e:  # noqa: BLE001
        out["has_tests"] = 1.0
        out["_note"] = f"the tests fail on a correct solution: {type(e).__name__}: {str(e)[:120]}"
        return out
    if first is None:
        out["_note"] = "no test_* functions found"
        return out
    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import gqa
    import gqa.core as core
    good = gqa.scaled_dot_product_attention

    gqa.scaled_dot_product_attention = _broken_sdpa
    core.scaled_dot_product_attention = _broken_sdpa
    try:
        out["catches_bad_grouping"] = 0.0 if _survives(path) else 1.0
    finally:
        gqa.scaled_dot_product_attention = good
        core.scaled_dot_product_attention = good
    return out
