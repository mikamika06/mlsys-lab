import numpy as np
import math

def _quantise(weights, grid):
    n = weights.shape[0]
    m = grid.shape[0]
    idx = np.empty(n, dtype=np.int64)
    for i in range(n):
        w = weights[i]
        min_abs_diff = math.inf
        best_j = 0
        for j in range(m):
            diff = w - grid[j]
            abs_diff = diff if diff >= 0.0 else -diff
            if abs_diff < min_abs_diff:
                min_abs_diff = abs_diff
                best_j = j
        idx[i] = best_j
    return grid[idx]

def best_grid(weights: np.ndarray) -> str:
    grids = {
        "NF4": np.linspace(-1.0, 1.0, 16),
        "FP4": np.linspace(-8.0, 8.0, 16),
        "INT4": np.arange(-8, 8, dtype=np.float64)
    }
    mses = {}
    for name, grid in grids.items():
        q = _quantise(weights, grid)
        n = weights.shape[0]
        acc = 0.0
        for i in range(n):
            diff = weights[i] - q[i]
            acc += diff * diff
        mses[name] = float(acc / n)
    
    order = ["NF4", "FP4", "INT4"]
    best_name = order[0]
    best_mse = mses[best_name]
    for name in order[1:]:
        mse = mses[name]
        if mse < best_mse:
            best_mse = mse
            best_name = name
    return best_name
