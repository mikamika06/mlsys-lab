from quantplan.pareto import compute_pareto
import numpy as np


def best_config(weights_shape, blocksizes, double_quants, mse_budget):
    items = compute_pareto(weights_shape, blocksizes, double_quants)
    valid = [x for x in items if x["mse"] <= mse_budget]
    if not valid:
        valid = items
    mems = [x["memory_bytes"] for x in valid]
    idx = int(np.argmin(mems))
    return valid[idx]
