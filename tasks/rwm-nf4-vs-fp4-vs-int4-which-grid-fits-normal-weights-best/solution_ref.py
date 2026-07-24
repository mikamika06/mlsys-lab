import numpy as np

def _quantise(weights, grid):
    idx = np.argmin(np.abs(weights[:, None] - grid), axis=1)
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
        mses[name] = float(np.mean((weights - q)**2))
    best = min(mses.keys(), key=lambda k: (mses[k], ["NF4","FP4","INT4"].index(k)))
    return best
