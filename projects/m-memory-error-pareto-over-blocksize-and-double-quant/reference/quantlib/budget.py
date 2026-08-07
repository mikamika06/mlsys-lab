from quantlib.pareto import compute_pareto_frontier

def find_smallest_config(layers, non_quantized, blocksizes, dq_blocksizes, mse_budget):
    frontier = compute_pareto_frontier(layers, non_quantized, blocksizes, dq_blocksizes)
    valid = [cfg for cfg in frontier if cfg["mse"] <= mse_budget]
    if not valid:
        return None
    valid.sort(key=lambda x: x["memory"])
    return valid[0]
