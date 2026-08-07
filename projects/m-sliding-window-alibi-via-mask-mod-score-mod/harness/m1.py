import numpy as np
import ref

def check(workdir):
    out = {"max_abs_err": 1e9, "mask_match": 0.0}
    try:
        from flex.mods import alibi_score_mod, sliding_window_mask_mod
    except ImportError:
        out["_note"] = "failed to import mods"
        return out

    q, kv = np.meshgrid(np.arange(128), np.arange(128), indexing='ij')
    score = np.zeros_like(q, dtype=np.float32)

    try:
        got_score = alibi_score_mod(score, 2, q, kv, 8)
        want_score = ref.alibi_score_mod(score, 2, q, kv, 8)
        out["max_abs_err"] = float(np.max(np.abs(got_score - want_score)))
    except Exception as e:
        out["_note_score"] = f"alibi error: {e}"

    try:
        got_mask = sliding_window_mask_mod(0, 0, q, kv, 32)
        want_mask = ref.sliding_window_mask_mod(0, 0, q, kv, 32)
        if np.array_equal(got_mask, want_mask):
            out["mask_match"] = 1.0
    except Exception as e:
        out["_note_mask"] = f"mask error: {e}"

    return out
