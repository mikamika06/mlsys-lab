import numpy as np


def find_spillover_spike(events):
    """Find index of event with highest latency per page fault using argmin on negative efficiency."""
    if not events:
        return {"argmin_index": -1, "max_ratio": 0.0}

    durations = np.array([e["dur"] for e in events], dtype=np.float64)
    faults = np.array([max(1, e["page_faults"]) for e in events], dtype=np.float64)

    ratios = durations / faults
    neg_ratios = -ratios
    target_idx = int(np.argmin(neg_ratios))

    return {
        "argmin_index": target_idx,
        "max_ratio": float(ratios[target_idx])
    }
