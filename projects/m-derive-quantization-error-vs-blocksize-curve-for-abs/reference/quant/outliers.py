import numpy as np

def expected_outlier_count(hidden_states, threshold=6.0):
    """Estimate outlier count exceeding threshold."""
    abs_vals = np.abs(hidden_states)
    exceed = abs_vals > threshold
    return int(np.sum(exceed))
