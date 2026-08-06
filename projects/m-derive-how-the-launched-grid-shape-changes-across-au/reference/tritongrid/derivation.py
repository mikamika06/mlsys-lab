def compute_grid(problem_shape, config):
    m = problem_shape.get("M", 0)
    n = problem_shape.get("N", 0)
    block_m = config.get("BLOCK_M", 128)
    block_n = config.get("BLOCK_N", 128)
    grid_x = (n + block_n - 1) // block_n
    grid_y = (m + block_m - 1) // block_m
    return (grid_x, grid_y)
