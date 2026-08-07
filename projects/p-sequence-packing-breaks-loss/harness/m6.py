import importlib.util
import os
import sys


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
        "catches_boundary_leakage": 0.0,
        "catches_bad_normalization": 0.0,
        "faults_caught": 0.0,
    }
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    sys.path.insert(0, workdir)
    import seqpack.attention as attn_mod
    import seqpack.loss as loss_mod

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"tests fail on correct implementation: {type(e).__name__}: {e}"
        return out

    if first is None:
        out["_note"] = "no test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    good_mask = attn_mod.create_block_diagonal_mask

    def leaky_mask(seq_ids):
        import numpy as np
        L = len(seq_ids)
        return np.tril(np.ones((L, L), dtype=bool))

    attn_mod.create_block_diagonal_mask = leaky_mask
    try:
        out["catches_boundary_leakage"] = 0.0 if _survives(path) else 1.0
    finally:
        attn_mod.create_block_diagonal_mask = good_mask

    good_loss = loss_mod.compute_packed_loss

    def bad_norm_loss(logits, labels, label_mask, seq_ids):
        import numpy as np
        L, V = logits.shape
        exp_lg = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = exp_lg / np.sum(exp_lg, axis=-1, keepdims=True)
        safe_lbl = np.clip(labels, 0, V - 1)
        l_per_token = -np.log(probs[np.arange(L), safe_lbl] + 1e-12)
        return float(np.sum(l_per_token * (label_mask > 0)) / float(L))

    loss_mod.compute_packed_loss = bad_norm_loss
    try:
        out["catches_bad_normalization"] = 0.0 if _survives(path) else 1.0
    finally:
        loss_mod.compute_packed_loss = good_loss

    out["faults_caught"] = out["catches_boundary_leakage"] + out["catches_bad_normalization"]
    return out
