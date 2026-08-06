import numpy as np


def compute_acceptance_metrics(traces):
    """Compute flat acceptance rate and per-position acceptance decay curve."""
    total_accepted = 0
    total_drafted = 0
    max_len = max(len(t) for t in traces) if traces else 0
    pos_accepted = np.zeros(max_len, dtype=np.float64)
    pos_counts = np.zeros(max_len, dtype=np.float64)

    for trace in traces:
        for pos, accepted in enumerate(trace):
            total_drafted += 1
            pos_counts[pos] += 1
            if accepted:
                total_accepted += 1
                pos_accepted[pos] += 1

    flat_rate = total_accepted / total_drafted if total_drafted > 0 else 0.0
    decay_curve = np.zeros(max_len, dtype=np.float64)
    valid_mask = pos_counts > 0
    decay_curve[valid_mask] = pos_accepted[valid_mask] / pos_counts[valid_mask]

    return float(flat_rate), decay_curve
