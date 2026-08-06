import numpy as np


def evaluate_tolerance(error_stats, policy):
    max_atol = policy.get("atol", 1e-5)
    max_rtol = policy.get("rtol", 1e-2)

    max_abs = error_stats.get("max_abs_diff", 0.0)
    mean_rel = error_stats.get("mean_rel_diff", 0.0)

    accepted = True
    reason = "PASSED"

    if max_abs > max_atol and mean_rel > max_rtol:
        accepted = False
        reason = "REJECT_EXCEEDED_TOLERANCE"
    elif error_stats.get("has_nan", False):
        accepted = False
        reason = "REJECT_NAN"
    else:
        accepted = True
        reason = "ACCEPT_WITHIN_BOUNDS"

    return {"accepted": accepted, "reason": reason}
