import numpy as np
from prune.methods import wanda_prune, evaluate_quality


def debug_wanda_domain(w, X_match, X_mismatch, sparsity):
    _, mask_match = wanda_prune(w, X_match, sparsity)
    _, mask_mismatch = wanda_prune(w, X_mismatch, sparsity)
    overlap = np.mean(mask_match == mask_mismatch)
    rel_err_match = evaluate_quality(w, w * mask_match, X_match)
    rel_err_mismatch = evaluate_quality(w, w * mask_mismatch, X_match)
    degradation = float(rel_err_mismatch - rel_err_match)
    return {
        "overlap": float(overlap),
        "degradation": degradation,
        "mismatched_higher_error": bool(rel_err_mismatch > rel_err_match)
    }
