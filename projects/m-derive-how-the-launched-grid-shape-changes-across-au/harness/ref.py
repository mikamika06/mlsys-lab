CONFIGS = [
    {"BLOCK_M": 16, "BLOCK_N": 16},
    {"BLOCK_M": 32, "BLOCK_N": 32},
    {"BLOCK_M": 64, "BLOCK_N": 32},
]

def derive_grid(M, N, config):
    block_m = config.get("BLOCK_M", 16)
    block_n = config.get("BLOCK_N", 16)
    grid_m = (M + block_m - 1) // block_m
    grid_n = (N + block_n - 1) // block_n
    return (grid_m, grid_n)

def compare_grids(M, N, configs):
    results = []
    for cfg in configs:
        g = derive_grid(M, N, cfg)
        results.append({
            "config": cfg,
            "grid": g,
            "total_blocks": g[0] * g[1]
        })
    return results
