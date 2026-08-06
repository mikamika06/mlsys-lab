import math

def derive_num_programs(width, height, block_w, block_h):
    grid_x = math.ceil(width / block_w)
    grid_y = math.ceil(height / block_h)
    return grid_x, grid_y, grid_x * grid_y
