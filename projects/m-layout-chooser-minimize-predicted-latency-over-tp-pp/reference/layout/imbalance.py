import numpy as np


def straggler_factor(routing_histogram: np.ndarray) -> float:
    """Compute expert routing straggler factor (max load / average load)."""
    arr = np.asarray(routing_histogram, dtype=np.float64)
    if arr.size == 0 or np.sum(arr) == 0:
        return 1.0
    mean_val = np.mean(arr)
    if mean_val == 0:
        return 1.0
    return float(np.max(arr) / mean_val)
