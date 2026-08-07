import numpy as np

SAMPLE_RUNS = [
    [1.000, 2.000, 3.000],
    [1.001, 1.999, 3.002],
    [0.999, 2.001, 2.998],
    [1.000, 2.000, 3.001],
]

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

def deterministic_backward(grads):
    arr = np.array(grads, dtype=np.float64)
    sorted_arr = np.sort(arr, axis=0)
    accumulated = np.sum(sorted_arr, axis=0)
    return accumulated.tolist()
