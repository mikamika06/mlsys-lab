from __future__ import annotations
import math

def _quantise(weights, grid):
    n = len(weights)
    m = len(grid)
    q = [0.0] * n
    for i in range(n):
        w = weights[i]
        min_abs_diff = math.inf
        best_val = grid[0]
        for j in range(m):
            diff = w - grid[j]
            abs_diff = diff if diff >= 0.0 else -diff
            if abs_diff < min_abs_diff:
                min_abs_diff = abs_diff
                best_val = grid[j]
        q[i] = best_val
    return q

def best_grid(weights: list[float]) -> str:
    nf4 = [-1.0 + i * 2.0 / 15.0 for i in range(16)]
    fp4 = [-8.0 + i * 16.0 / 15.0 for i in range(16)]
    int4 = [float(i) for i in range(-8, 8)]
    grids = {
        "NF4": nf4,
        "FP4": fp4,
        "INT4": int4
    }
    mses = {}
    n = len(weights)
    for name, grid in grids.items():
        q = _quantise(weights, grid)
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
