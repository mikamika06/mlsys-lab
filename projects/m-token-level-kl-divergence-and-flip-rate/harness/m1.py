import sys
import os
import numpy as np
import ref


def check(workdir):
    sys.path.insert(0, workdir)
    out = {"kl_matched": 0.0, "flip_matched": 0.0}

    try:
        from eval_metrics.divergence import compute_kl_divergence, compute_flip_rate
    except ImportError as e:
        out["_note"] = f"Import error: {e}"
        return out

    p_logits, q_logits, _, _, _ = ref.generate_test_data()

    want_kl = ref.compute_kl_divergence(p_logits, q_logits)
    try:
        got_kl = compute_kl_divergence(p_logits, q_logits)
        if np.allclose(want_kl, got_kl, rtol=1e-4, atol=1e-5):
            out["kl_matched"] = 1.0
        else:
            out["_note"] = f"KL divergence mismatch. Max diff: {np.max(np.abs(want_kl - got_kl))}"
    except Exception as e:
        out["_note"] = f"KL computation failed: {type(e).__name__}: {e}"
        return out

    want_flip = ref.compute_flip_rate(p_logits, q_logits)
    try:
        got_flip = compute_flip_rate(p_logits, q_logits)
        if abs(want_flip - got_flip) < 1e-6:
            out["flip_matched"] = 1.0
        else:
            out["_note"] = f"Flip rate mismatch: want {want_flip}, got {got_flip}"
    except Exception as e:
        out["_note"] = f"Flip rate computation failed: {type(e).__name__}: {e}"
        return out

    return out
