import math

CONFIGS = [
    {"width": 100, "height": 100, "block_w": 32, "block_h": 32},
    {"width": 64, "height": 64, "block_w": 16, "block_h": 16},
    {"width": 128, "height": 60, "block_w": 32, "block_h": 32},
    {"width": 15, "height": 15, "block_w": 16, "block_h": 16},
    {"width": 257, "height": 129, "block_w": 64, "block_h": 64},
]

def derive_num_programs(width, height, block_w, block_h):
    grid_x = math.ceil(width / block_w)
    grid_y = math.ceil(height / block_h)
    return grid_x, grid_y, grid_x * grid_y
