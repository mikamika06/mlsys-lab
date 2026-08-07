import numpy as np


def compute_variance(runs):
    arr = np.array(runs, dtype=np.float64)
    mean_val = np.mean(arr, axis=0)
    var_val = np.var(arr, axis=0)
    max_rel_err = np.max(np.abs(arr - mean_val) / (np.abs(mean_val) + 1e-8))
    return {
        "mean": mean_val.tolist(),
        "variance": var_val.tolist(),
        "max_rel_err": float(max_rel_err),
    }
