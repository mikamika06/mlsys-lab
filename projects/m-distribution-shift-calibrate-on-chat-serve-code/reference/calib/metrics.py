import numpy as np


def relative_error(ref_vals, target_vals):
    diff = np.abs(ref_vals - target_vals)
    denom = np.abs(ref_vals) + 1e-5
    return float(np.mean(diff / denom))
