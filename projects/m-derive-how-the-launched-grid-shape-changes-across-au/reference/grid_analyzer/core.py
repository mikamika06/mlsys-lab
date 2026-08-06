def derive_grid(M, N, config):
    block_m = config.get("BLOCK_M", 16)
    block_n = config.get("BLOCK_N", 16)
    grid_m = (M + block_m - 1) // block_m
    grid_n = (N + block_n - 1) // block_n
    return (grid_m, grid_n)
