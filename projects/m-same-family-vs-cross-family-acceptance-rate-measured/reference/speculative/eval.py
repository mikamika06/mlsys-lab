from speculative.metrics import compute_acceptance_rate


def compare_families(same_trace, cross_trace):
    r_same = compute_acceptance_rate(same_trace)
    r_cross = compute_acceptance_rate(cross_trace)
    return float(r_same - r_cross)


def classify_pairing(trace, threshold):
    rate = compute_acceptance_rate(trace)
    return "high" if rate >= threshold else "low"
