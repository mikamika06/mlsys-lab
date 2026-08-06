import importlib.util
import os
import sys
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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_double_counted_sinks": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    sys.path.insert(0, workdir)
    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"the tests fail on a correct implementation: {type(e).__name__}: {str(e)[:120]}"
        return out

    if first is None:
        out["_note"] = "no test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import attnsink.sink_softmax as ssm

    good_fn = ssm.attention_sink_softmax

    def broken_sink_softmax(Q, K, V, sink_size, window_size):
        L, d_k = Q.shape
        _, d_v = V.shape
        scale = 1.0 / np.sqrt(d_k)
        out_arr = np.zeros((L, d_v), dtype=np.float64)
        lse_arr = np.zeros(L, dtype=np.float64)
        for t in range(L):
            sink_indices = list(range(0, min(sink_size, t + 1)))
            win_indices = list(range(max(0, t - window_size + 1), t + 1))
            valid_indices = sink_indices + win_indices
            K_valid = K[valid_indices]
            V_valid = V[valid_indices]
            logits = np.dot(K_valid, Q[t]) * scale
            max_logit = np.max(logits)
            exp_logits = np.exp(logits - max_logit)
            sum_exp = np.sum(exp_logits)
            lse_arr[t] = max_logit + np.log(sum_exp)
            weights = exp_logits / sum_exp
            out_arr[t] = np.dot(weights, V_valid)
        return out_arr, lse_arr

    ssm.attention_sink_softmax = broken_sink_softmax
    try:
        out["catches_double_counted_sinks"] = 0.0 if _survives(path) else 1.0
    finally:
        ssm.attention_sink_softmax = good_fn

    return out
