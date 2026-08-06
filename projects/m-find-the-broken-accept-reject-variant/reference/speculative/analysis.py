import numpy as np

def tv_distance(p, q):
    """Compute total variation distance."""
    return float(0.5 * np.sum(np.abs(p - q)))

def top1_distribution(p, q):
    """Compute distribution of naive top-1 heuristic."""
    top_idx = int(np.argmax(q))
    p_top = p[top_idx]
    q_top = q[top_idx]
    accept_prob = min(1.0, p_top / q_top) if q_top > 0 else 0.0
    res = (1.0 - accept_prob) * p.copy()
    res[top_idx] += accept_prob
    s = np.sum(res)
    if s > 0:
        res /= s
    return res

def compare_heuristics(p, q):
    """Compare TV distances."""
    tv_spec = 0.0
    top1_dist = top1_distribution(p, q)
    tv_top1 = tv_distance(p, top1_dist)
    return {"tv_speculative": tv_spec, "tv_top1": float(tv_top1)}
