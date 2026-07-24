import numpy as np

def _mse(weights, grid):
    idx = np.argmin(np.abs(weights[:, None] - grid), axis=1)
    q = grid[idx]
    return float(np.mean((weights - q)**2))

def grade(sol, fx) -> dict:
    rng = np.random.default_rng(0)
    weights = rng.standard_normal(1000).astype(np.float64)

    grids = {
        "NF4": np.linspace(-1.0, 1.0, 16),
        "FP4": np.linspace(-8.0, 8.0, 16),
        "INT4": np.arange(-8, 8, dtype=np.float64)
    }

    mses = {name: _mse(weights, grid) for name, grid in grids.items()}
    best = min(mses.keys(), key=lambda k: (mses[k], ["NF4","FP4","INT4"].index(k)))
    try:
        out = sol.best_grid(weights)
    except Exception:
        return {"exact_match": 0.0}
    ok = 1.0 if out == best else 0.0
    return {"exact_match": ok}
