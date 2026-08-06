import numpy as np

def accept_reject_prob(p_val, q_val):
    """Compute standard acceptance probability min(1, p/q)."""
    if q_val <= 0:
        return 1.0
    return float(min(1.0, p_val / q_val))

def argmin_index(values):
    """Return index of minimum value."""
    return int(np.argmin(values))

def find_correct_variant(variants, p, q):
    """Find variant index that minimizes distribution divergence or error."""
    scores = []
    for v in variants:
        err = 0.0
        for i in range(len(p)):
            sim_accept = v(i, p, q)
            true_accept = accept_reject_prob(p[i], q[i])
            err += abs(sim_accept - true_accept)
        scores.append(err)
    return argmin_index(scores)
