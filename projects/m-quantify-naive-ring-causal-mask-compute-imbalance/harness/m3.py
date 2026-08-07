import importlib.util
import os
import numpy as np
import ref

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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_missing_causal_mask": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out
    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"the tests fail on a correct plan: {type(e).__name__}: {str(e)[:120]}"
        return out
    if first is None:
        out["_note"] = "no test_* functions found"
        return out
    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import ring.simulate as s
    good = s.ring_attention_simulate

    def broken_simulate(q_shards, k_shards, v_shards):
        C = len(q_shards)
        out_shards = []
        for i in range(C):
            q = q_shards[i]
            o_unnorm = np.zeros_like(q, dtype=float)
            m = np.full((q.shape[0], 1), -np.inf, dtype=float)
            l = np.zeros((q.shape[0], 1), dtype=float)
            for step in range(C):
                j = (i - step) % C
                if j > i:
                    continue
                k = k_shards[j]
                v = v_shards[j]
                scores = q @ k.T
                # FAULT: missing causal mask for the diagonal block (j == i)
                m_curr = np.max(scores, axis=-1, keepdims=True)
                m_new = np.maximum(m, m_curr)
                exp_scores = np.exp(scores - m_new)
                exp_old = np.exp(m - m_new)
                l_new = l * exp_old + np.sum(exp_scores, axis=-1, keepdims=True)
                o_unnorm = o_unnorm * exp_old + exp_scores @ v
                m = m_new
                l = l_new
            out_shards.append(o_unnorm / l)
        return out_shards

    s.ring_attention_simulate = broken_simulate
    try:
        if _survives(path):
            out["catches_missing_causal_mask"] = 0.0
            out["_note"] = "tests passed even without a causal mask on the diagonal block"
        else:
            out["catches_missing_causal_mask"] = 1.0
    finally:
        s.ring_attention_simulate = good
    return out
