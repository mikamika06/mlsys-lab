import math

def launch_grid_config(width, height, block_w, block_h):
    grid_x = math.ceil(width / block_w)
    grid_y = math.ceil(height / block_h)

    meta_lambda = lambda meta: (grid_x, grid_y)

    executed_tiles = set()
    for gy in range(grid_y):
        for gx in range(grid_x):
            executed_tiles.add((gx, gy))

    return {
        "grid_x": grid_x,
        "grid_y": grid_y,
        "total_programs": grid_x * grid_y,
        "meta_lambda": meta_lambda,
        "executed_tiles": executed_tiles
    }
