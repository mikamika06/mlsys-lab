from tpscaling.table import compute_scaling_table

def optimal_tp(config, hardware):
    table = compute_scaling_table(config, [1, 2, 4, 8])
    best = min(table, key=lambda d: d["memory_gb"] + d["comm_bytes"] / 1e8)
    return best["tp"]
