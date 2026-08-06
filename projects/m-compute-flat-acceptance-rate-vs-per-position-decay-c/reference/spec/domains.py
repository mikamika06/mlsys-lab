import numpy as np


def analyze_domain_acceptance(domain_decay_curves, draft_length):
    """Compute overall acceptance rates per domain for a given draft length."""
    results = {}
    for domain, curve in domain_decay_curves.items():
        k = min(draft_length, len(curve))
        if k == 0:
            results[domain] = 0.0
            continue
        sub_curve = np.asarray(curve[:k], dtype=np.float64)
        prefix_probs = np.cumprod(sub_curve)
        expected_accepted = np.sum(prefix_probs)
        results[domain] = float(expected_accepted / k)
    return results
