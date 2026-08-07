import numpy as np
from numval.metrics import cosine_similarity, sqnr


def evaluate_gate(y_ref, y_test, min_sqnr_db=30.0, min_cos_sim=0.99, max_rel_err=1e-2, eps=1e-12):
    """Evaluates composite accept/reject gate metrics."""
    sqnr_val = sqnr(y_ref, y_test, eps=eps)
    cos_val = cosine_similarity(y_ref, y_test, eps=eps)
    r = np.asarray(y_ref, dtype=np.float64)
    t = np.asarray(y_test, dtype=np.float64)
    diff = np.abs(r - t)
    denom = np.abs(r) + eps
    rel_err_val = float(np.max(diff / denom))

    passed = bool(
        (sqnr_val >= min_sqnr_db)
        and (cos_val >= min_cos_sim)
        and (rel_err_val <= max_rel_err)
    )
    return {
        "passed": passed,
        "sqnr_db": float(sqnr_val),
        "cos_sim": float(cos_val),
        "max_rel_err": float(rel_err_val),
    }
